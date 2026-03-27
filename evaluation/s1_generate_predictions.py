import os
import re
import time
import torch
import pandas as pd
from tqdm import tqdm

from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer

torch.set_float32_matmul_precision("high")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_DATA = "data/test/extracted_100_rows.parquet"
TOKENIZER_DIR = "checkpoints/Qwen/Qwen3-0.6B"

MAX_NEW_TOKENS = 1024
NUM_SAMPLES = 3
OUTPUT_DIR = "results/evaluation_outputs/mix_test_2"

# MODELS = {
#     "Teacher": (
#         "checkpoints/Qwen/Qwen3-8B/lit_model.pth",
#         "Qwen3-8B"
#     ),
#     "Base": (
#         "checkpoints/Qwen/Qwen3-0.6B/lit_model.pth",
#         "Qwen3-0.6B"
#     ),
#     "Distilled": (
#         "checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/16_03_run_test_80k_5DEB07_better_results/lit_model.pth",
#         "Qwen3-0.6B-MoE"
#     ),
# }

MODELS = {
    "16_03": (
        "checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/16_03_run_test_80k_5DEB07_better_results/lit_model.pth",
        "Qwen3-0.6B-MoE"
    ),
    "20_03": (
        "checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/20_03_run_test_80k_EC0051/epoch-2.pth",
        "Qwen3-0.6B-MoE"
    ),
    "23_03": (
        "checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/23_03_run_test_80k_1D76FE/epoch-2.pth",
        "Qwen3-0.6B-MoE"
    ),
    "Base": (
        "checkpoints/Qwen/Qwen3-0.6B/lit_model.pth",
        "Qwen3-0.6B"
    ),
    "Teacher": (
        "checkpoints/Qwen/Qwen3-8B/lit_model.pth",
        "Qwen3-8B"
    ),
}


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

def get_config(name):
    config = Config.from_name(name)

    if "8B" in name:
        config.n_embd = 4096
        config.n_head = 32
        config.intermediate_size = 12288

    return config


# ---------------------------------------------------------
# ROUTER ENTROPY
# ---------------------------------------------------------

def compute_router_entropy(model):

    loads = []

    for layer in model.transformer.h:

        if hasattr(layer.mlp, "last_indices"):

            indices = layer.mlp.last_indices

            if indices is None:
                continue

            flat = indices.view(-1)

            counts = torch.bincount(flat)

            probs = counts / counts.sum()

            entropy = -(probs * torch.log(probs + 1e-9)).sum()

            loads.append(entropy.item())

    if len(loads) == 0:
        return 0

    return sum(loads) / len(loads)


# ---------------------------------------------------------
# OUTPUT SPLITTING
# ---------------------------------------------------------

def split_thought_advisory(text):

    text = text.strip()
    lower = text.lower()

    think_start = lower.find("<think>")
    think_end = lower.find("</think>")

    thought = ""
    advisory = ""

    if think_start != -1 and think_end != -1 and think_end > think_start:
        thought = text[think_start + len("<think>"):think_end]
        advisory = text[think_end + len("</think>"):]

    elif think_end != -1:
        thought = text[:think_end]
        advisory = text[think_end + len("</think>"):]

    else:
        advisory = text

    thought = re.sub(r"</?think>", "", thought, flags=re.IGNORECASE)
    advisory = re.sub(r"</?advisory>", "", advisory, flags=re.IGNORECASE)

    thought = re.sub(r"<[^>]+>", "", thought)
    advisory = re.sub(r"<[^>]+>", "", advisory)

    thought = re.sub(r"\s+", " ", thought).strip()
    advisory = re.sub(r"\s+", " ", advisory).strip()

    return thought, advisory


# ---------------------------------------------------------
# GENERATION
# ---------------------------------------------------------

def generate(model, idx, max_new_tokens, eos_id):

    B, T = idx.shape

    model.set_kv_cache(
        batch_size=B,
        max_seq_length=T + max_new_tokens,
        device=DEVICE
    )

    input_pos = torch.arange(0, T, device=DEVICE)
    logits = model(idx, input_pos=input_pos)[:, -1, :]

    generated = []

    temperature = 0.8
    top_p = 0.9
    top_k = None

    for i in range(max_new_tokens):
        # probs = torch.softmax(logits, dim=-1)
        # deterministic generation (faster + reproducible)
        # idx_next = torch.argmax(probs, dim=-1, keepdim=True)

        logits = logits / temperature

        # Top-K filtering
        # values, _ = torch.topk(logits, top_k)
        # min_val = values[:, -1].unsqueeze(-1)
        # logits = torch.where(logits < min_val, torch.full_like(logits, -1e10), logits)

        # Top-P (nucleus sampling)
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        cutoff = cumulative_probs > top_p
        cutoff[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(cutoff, 0)

        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        idx_next = sorted_indices.gather(
            -1,
            torch.multinomial(sorted_probs, 1)
        )

        token = idx_next.item()

        if token == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)
        generated.append(idx_next)

        input_pos = torch.tensor([T + i], device=DEVICE)
        logits = model(idx_next, input_pos=input_pos)[:, -1, :]

    model.clear_kv_cache()

    if len(generated) == 0:
        return torch.tensor([[]], device=DEVICE)

    return torch.cat(generated, dim=1)


# ---------------------------------------------------------
# MAIN GENERATION PIPELINE
# ---------------------------------------------------------

def run_generation():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = Tokenizer(TOKENIZER_DIR)
    eos_id = tokenizer.eos_id

    df = pd.read_parquet(TEST_DATA)
    df = df.head(NUM_SAMPLES)


    predictions_log = []
    generation_time_log = []

    for model_name, (ckpt, config_name) in MODELS.items():

        print(f"\nRunning generation for {model_name}")

        start_model_time = time.time()

        config = get_config(config_name)

        model = GPT(config).to(DEVICE, dtype=torch.bfloat16)

        weights = torch.load(ckpt, map_location=DEVICE)

        if "model_state_dict" in weights:
            weights = weights["model_state_dict"]

        model.load_state_dict(weights, strict=False)

        model.eval()

        for _, row in tqdm(df.iterrows(), total=len(df)):

            prompt = (
                "<|im_start|>system\n"
                + row["system_instruction"]
                + "\n\n"
                "Analyze the situation carefully before answering.\n\n"
                "Instructions:\n"
                "1. First think step-by-step before answering. Let's analyze the situation carefully. Then produce the final advisory using the thinking tokens.\n"
                "2. Identify crop condition, possible risks, and recommended actions.\n"
                "3. Then produce a practical advisory.\n\n"
                "Output format (strictly follow):\n"
                "<think> \n"
                "Step-by-step reasoning about the problem.\n"
                "</think> \n"
                "Then provide the final advisory in Hindi for the farmer.\n"
                "<|im_end|>\n"

                "<|im_start|>user\n"
                + row["prompt"]
                + "\n<|im_end|>\n"

                "<|im_start|>assistant\n<think>\n"
            )

            input_ids = tokenizer.encode(prompt, bos=True, eos=False).to(DEVICE).unsqueeze(0)

            # SAMPLE TIMER
            start_sample_time = time.time()

            with torch.no_grad():
                output_ids = generate(model, input_ids, MAX_NEW_TOKENS, eos_id)

            sample_time = time.time() - start_sample_time

            output = tokenizer.decode(output_ids[0])
            output = output.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()

            thought, advisory = split_thought_advisory(output)

            # Router entropy
            entropy = compute_router_entropy(model)

            predictions_log.append({
                "id": row["custom_id"],
                "model": model_name,
                "question": row["prompt"],
                "reference": row["advisory"],
                "thought": thought,
                "prediction": advisory,
                "raw_output": output,
                "generation_time_sec": sample_time,
                "router_entropy": entropy
            })

        model_time = time.time() - start_model_time

        print(f"{model_name} total generation time: {model_time:.2f} sec")
        print(f"{model_name} avg/sample: {model_time/len(df):.2f} sec")

        generation_time_log.append({
            "model": model_name,
            "total_generation_time_sec": model_time,
            "avg_generation_time_sec": model_time/len(df)
        })

        del model
        torch.cuda.empty_cache()

    pd.DataFrame(predictions_log).to_csv(
        os.path.join(OUTPUT_DIR, "predictions.csv"),
        index=False
    )

    pd.DataFrame(generation_time_log).to_csv(
        os.path.join(OUTPUT_DIR, "generation_time_log.csv"),
        index=False
    )

    print(f"\nGeneration completed! Saved at {OUTPUT_DIR}")


if __name__ == "__main__":
    run_generation()