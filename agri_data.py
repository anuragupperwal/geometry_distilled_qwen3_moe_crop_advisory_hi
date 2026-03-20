import torch
from torch.utils.data import Dataset
import pandas as pd


class AgriDataset(Dataset):
    """
    Dataset loader for ChatML formatted training data.

    Expected format:

    <|im_start|>system
    ...
    <|im_end|>
    <|im_start|>user
    ...
    <|im_end|>
    <|im_start|>assistant
    <think>
    ...
    </think>

    answer
    <|im_end|>
    """

    def __init__(self, data_path, tokenizer, max_seq_length=4096):

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.df = pd.read_parquet(data_path)

        print(f"Loaded {len(self.df)} samples from {data_path}")

        self.pad_id = getattr(tokenizer, "pad_id", tokenizer.eos_id)


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):

        text = self.df.iloc[idx]["text"]

        # --------------------------------
        # Locate assistant start
        # --------------------------------

        assistant_tag = "<|im_start|>assistant"

        if assistant_tag in text:
            split_idx = text.find(assistant_tag)
        else:
            split_idx = 0

        prompt_text = text[:split_idx]

        # --------------------------------
        # Tokenize
        # --------------------------------

        full_ids = self.tokenizer.encode(text, bos=False, eos=False)
        prompt_ids = self.tokenizer.encode(prompt_text, bos=False, eos=False)

        prompt_len = len(prompt_ids)

        # --------------------------------
        # Truncate
        # --------------------------------

        if len(full_ids) > self.max_seq_length:
            full_ids = full_ids[:self.max_seq_length]

        # --------------------------------
        # Create training mask
        # --------------------------------

        mask = torch.ones(len(full_ids), dtype=torch.float32)

        mask_boundary = min(prompt_len, self.max_seq_length)

        mask[:mask_boundary] = 0.0

        # --------------------------------
        # Convert to tensor
        # --------------------------------

        full_ids.clone().detach()

        # --------------------------------
        # Padding
        # --------------------------------

        pad_len = self.max_seq_length - len(full_ids)

        if pad_len > 0:

            pad_tensor = torch.full((pad_len,), self.pad_id, dtype=torch.long)
            mask_pad = torch.zeros(pad_len)

            full_ids = torch.cat([full_ids, pad_tensor])
            mask = torch.cat([mask, mask_pad])

        return full_ids, mask