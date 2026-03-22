# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
from pathlib import Path
from typing import Iterable, Iterator, Optional, Union
from sentencepiece import SentencePieceProcessor
from tokenizers import Tokenizer as HFTokenizer

import torch

from litgpt.utils import fix_and_load_json


class Tokenizer:
    def __init__(self, checkpoint_dir: Union[Path, str]) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise NotADirectoryError(f"The checkpoint directory does not exist: {str(checkpoint_dir)}")

        self.model_name = checkpoint_dir.stem
        self.use_bos = self.check_if_bos_token_used(checkpoint_dir)
        self.bos_id = None
        self.eos_id = None

        vocabulary_path_json = checkpoint_dir / "tokenizer.json"
        vocabulary_path_model = checkpoint_dir / "tokenizer.model"

        if vocabulary_path_json.is_file():
            from tokenizers import Tokenizer as HFTokenizer
            self.processor = HFTokenizer.from_file(str(vocabulary_path_json))
            self.backend = "huggingface"
            
            # Load additional configs if they exist
            self._load_special_token_ids(checkpoint_dir)

        elif vocabulary_path_model.is_file():
            from sentencepiece import SentencePieceProcessor
            self.processor = SentencePieceProcessor(model_file=str(vocabulary_path_model))
            self.backend = "sentencepiece"
            self.bos_id = self.processor.bos_id()
            self.eos_id = self.processor.eos_id()
        else:
            # Qwen3 Fix: If neither is found, look for tiktoken files or specialized Qwen configs
            # For now, we point specifically to the error to avoid the generic NotImplementedError
            raise FileNotFoundError(
                f"Could not find 'tokenizer.json' or 'tokenizer.model' in {checkpoint_dir}. "
                "Ensure the Qwen3 tokenizer files are present in the checkpoint directory."
            )
        

        # NOTE: A temporary fix until it's resolved on Tokenizers side.
        # LlaMA tokenizer strips leading spaces if to decode a single token at a time.
        # https://github.com/huggingface/transformers/issues/31643
        self.apply_decoding_fix = None
        if (config_path := checkpoint_dir / "tokenizer_config.json").is_file():
            with open(config_path, encoding="utf-8") as fp:
                self.apply_decoding_fix = "LlamaTokenizer" in json.load(fp)["tokenizer_class"]


    def _load_special_token_ids(self, checkpoint_dir: Path) -> None:
        """
        Helper to find BOS/EOS IDs from various config files.
        Critical for Qwen3 which often maps EOS to <|im_end|> or <|endoftext|>.
        """
        # Default fallback for Qwen
        default_eos_token = "<|endoftext|>"
        
        # 1. Try tokenizer_config.json
        config_path = checkpoint_dir / "tokenizer_config.json"
        if config_path.is_file():
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
                
            bos_token = config.get("bos_token")
            eos_token = config.get("eos_token") or config.get("pad_token")

            if isinstance(bos_token, dict): bos_token = bos_token.get("content")
            if isinstance(eos_token, dict): eos_token = eos_token.get("content")

            if bos_token: self.bos_id = self.token_to_id(bos_token)
            if eos_token: self.eos_id = self.token_to_id(eos_token)

        # 2. Final Fallback if still None
        if self.eos_id is None:
            try:
                self.eos_id = self.token_to_id(default_eos_token)
            except ValueError:
                self.eos_id = 0 # Safe default for most BPE
        
        if self.bos_id is None:
            self.bos_id = self.eos_id # Qwen often uses same token for both

    @property
    def vocab_size(self) -> int:
        if self.backend == "huggingface":
            return self.processor.get_vocab_size(with_added_tokens=False)
        if self.backend == "sentencepiece":
            return self.processor.vocab_size()
        raise RuntimeError

    def token_to_id(self, token: str) -> int:
        if self.backend == "huggingface":
            id_ = self.processor.token_to_id(token)
        elif self.backend == "sentencepiece":
            id_ = self.processor.piece_to_id(token)
        else:
            raise RuntimeError
        if id_ is None:
            raise ValueError(f"token {token!r} not found in the collection.")
        return id_

    def check_if_bos_token_used(self, checkpoint_dir: Path) -> bool:
        if not (tokenizer_config_path := checkpoint_dir / "tokenizer_config.json").is_file():
            return False
        with open(tokenizer_config_path, encoding="utf-8") as fp:
            config = json.load(fp)
        # for LlaMA-3 tokenizer there is no `add_bos_token` at all and `tokenizer_class` is only
        # `PreTrainedTokenizerFast`
        if checkpoint_dir.stem.startswith(("Meta-Llama-3", "Llama-3")):
            return True
        if checkpoint_dir.stem.startswith("SmolLM2") and checkpoint_dir.name.endswith("Instruct"):
            return True
        if "add_bos_token" in config:
            return config["add_bos_token"]
        # if `add_bos_token` isn't in the config file, but LLaMA tokenizer is used - return True.
        # ex: https://huggingface.co/stabilityai/StableBeluga2/blob/main/tokenizer_config.json#L2
        return config.get("tokenizer_class") == "LlamaTokenizer"

    def encode(
        self,
        string: str,
        device: Optional[torch.device] = None,
        bos: Optional[bool] = None,
        eos: bool = False,
        max_length: int = -1,
    ) -> torch.Tensor:
        if self.backend == "huggingface":
            tokens = self.processor.encode(string).ids
        elif self.backend == "sentencepiece":
            tokens = self.processor.encode(string)
        else:
            raise RuntimeError(f"`{self.backend}` is not supported.")
        if tokens is None:
            raise ValueError("`self.processor` returned tokens of None value.")

        if bos or (bos is None and self.use_bos):
            if self.bos_id is None:
                raise NotImplementedError("This tokenizer does not have a defined bos token.")
            if not tokens or tokens[0] != self.bos_id:
                tokens = [self.bos_id] + tokens
        # if the processor misbehaves and adds `bos` token no matter what
        elif tokens and tokens[0] == self.bos_id:
            tokens = tokens[1:]

        if eos and (not tokens or tokens[-1] != self.eos_id):
            tokens = tokens + [self.eos_id]
        # if the processor misbehaves and adds `eos` token no matter what
        elif tokens and tokens[-1] == self.eos_id:
            tokens = tokens[:-1]

        if max_length > 0:
            tokens = tokens[:max_length]
        return torch.tensor(tokens, dtype=torch.int, device=device)

    def decode(self, tensor: torch.Tensor) -> str:
        tokens = [tensor.item()] if tensor.ndim == 0 else tensor.tolist()
        if len(tokens) == 1 and self.apply_decoding_fix:
            dummy_token_id = 33  # \x1e
            dummy_token = self.processor.decode([dummy_token_id])
            if dummy_token != "\x1e":
                dummy_token_id = 165  # \x1e is different in salamandra tokenizers
                dummy_token = self.processor.decode([dummy_token_id])
            return self.processor.decode([dummy_token_id] + tokens)[len(dummy_token) :]
        return self.processor.decode(tokens)

    def decode_stream(
        self, token_stream: Iterable[torch.Tensor], device: Optional[torch.device] = None
    ) -> Iterator[str]:
        if self.backend == "huggingface":
            try:
                for token in token_stream:
                    yield self.decode(token)
            except KeyboardInterrupt:
                return
        elif self.backend == "sentencepiece":
            # TODO: Is there a way to not have to do this?
            # This may actually affect our tokens per second.

            # sentencepiece does not support decoding token-by-token because it adds spaces based on the surrounding tokens
            # meaning that we need to decode everything each time
            so_far = torch.tensor([], dtype=torch.long, device=device)
            decoded_so_far = ""
            try:
                for token in token_stream:
                    so_far = so_far.to(device=token.device)
                    so_far = torch.cat((so_far, token.view(-1)))
                    decoded_new = self.decode(so_far)
                    yield decoded_new[len(decoded_so_far) :]
                    decoded_so_far = decoded_new
            except KeyboardInterrupt:
                return
        else:
            raise NotImplementedError(self.backend)