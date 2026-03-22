import torch
from collections import defaultdict, Counter
from pathlib import Path
from tqdm import tqdm

from torch.utils.data import DataLoader

from litgpt.tokenizer import Tokenizer
from litgpt.model import GPT
from litgpt.config import Config

from agri_data import AgriDataset


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

CHECKPOINT_PATH = "checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/12_03_run_test_80k_5C0943/lit_model.pth"
TOKENIZER_PATH = "checkpoints/Qwen/Qwen3-0.6B"

DATA_PATH = "data/train_bilingual_mixed_83k_agri65k.parquet"

MAX_SEQ_LENGTH = 512
BATCH_SIZE = 2

# number of tokens to analyze (sampling)
MAX_TOKENS_TO_ANALYZE = 20000


# --------------------------------------------------
# EXPERT TOKEN ANALYSIS
# --------------------------------------------------

def analyze_experts(model, tokenizer, dataloader):

    """
    Computes token specialization per expert.

    Structure:
    layer_id
        └ expert_id
            └ Counter(token_id -> frequency)
    """

    expert_token_map = defaultdict(lambda: defaultdict(Counter))

    device = next(model.parameters()).device

    model.eval()

    total_tokens = 0

    with torch.no_grad():

        for input_ids, _ in tqdm(dataloader, desc="Analyzing tokens"):

            input_ids = input_ids.to(device)

            # forward pass to trigger router
            model(input_ids)

            tokens = input_ids.view(-1)

            # iterate over layers
            for layer_id, layer in enumerate(model.transformer.h):

                if not hasattr(layer.mlp, "last_indices"):
                    continue

                expert_indices = layer.mlp.last_indices  # shape: [tokens, top_k]

                for token, experts in zip(tokens, expert_indices):

                    token_id = int(token)

                    # token routed to top-k experts
                    for expert in experts:

                        expert_id = int(expert)

                        expert_token_map[layer_id][expert_id][token_id] += 1

                    total_tokens += 1

                    if total_tokens >= MAX_TOKENS_TO_ANALYZE:
                        break

                if total_tokens >= MAX_TOKENS_TO_ANALYZE:
                    break

            if total_tokens >= MAX_TOKENS_TO_ANALYZE:
                break

    return expert_token_map


# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------

def save_specialization_report(expert_token_map, tokenizer):

    output_path = "outputs/12_03_run_test_80k_5C0943/expert_token_specialization.md"

    Path("outputs").mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:

        f.write("# Expert Token Specialization\n\n")

        for layer_id in sorted(expert_token_map.keys()):

            f.write(f"\n## Layer {layer_id}\n\n")

            for expert_id in sorted(expert_token_map[layer_id].keys()):

                f.write(f"### Expert {expert_id}\n")

                token_counter = expert_token_map[layer_id][expert_id]

                for token_id, count in token_counter.most_common(20):

                    token_str = tokenizer.decode(torch.tensor([token_id])).strip()
                    token_str = token_str.replace("\n", "\\n")

                    f.write(f"{token_str:<15} {count}\n")

                f.write("\n")

    print(f"\nExpert specialization report saved to:\n{output_path}")

# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nLoading tokenizer...")
    tokenizer = Tokenizer(TOKENIZER_PATH)

    print("Loading dataset...")
    dataset = AgriDataset(
        data_path=DATA_PATH,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print("Loading distilled MoE model...")

    model = GPT(Config.from_name("Qwen3-0.6B-MoE")).to(device, dtype=torch.bfloat16)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    print("\nStarting expert specialization analysis...\n")

    expert_token_map = analyze_experts(model, tokenizer, dataloader)

    save_specialization_report(expert_token_map, tokenizer)



if __name__ == "__main__":
    main()