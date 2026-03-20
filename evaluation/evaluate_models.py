import torch
from collections import defaultdict, Counter
from tqdm import tqdm
from torch.utils.data import DataLoader

from litgpt.tokenizer import Tokenizer
from litgpt.model import GPT
from litgpt.config import Config

from agri_data import AgriDataset


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

CHECKPOINT_PATH = "checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/12_03_run_test_80k_5C0943/lit_model.pth"
TOKENIZER_PATH = "checkpoints/Qwen/Qwen3-0.6B"
DATA_PATH = "data/train_bilingual_mixed_83k_agri65k.parquet"

MAX_SEQ_LENGTH = 512
BATCH_SIZE = 2

# We only need a small number of tokens
MAX_TOKENS_TO_ANALYZE = 50000


# --------------------------------------------------
# METHOD: Expert Token Attribution
# --------------------------------------------------

"""
Concept:
--------

We measure which tokens are routed to which experts.

Steps:
1. Run the model on real data
2. Read router outputs (expert indices)
3. Record token → expert mapping
4. Count most frequent tokens per expert

This reveals expert specialization.
"""


def analyze_experts(model, tokenizer, dataloader):

    expert_token_map = defaultdict(list)

    device = next(model.parameters()).device
    model.eval()

    token_counter = 0

    with torch.no_grad():

        for input_ids, _ in tqdm(dataloader):

            input_ids = input_ids.to(device)

            # Run model forward pass
            model(input_ids)

            # Iterate through MoE layers
            for layer in model.transformer.h:

                if not hasattr(layer.mlp, "last_indices"):
                    continue

                expert_indices = layer.mlp.last_indices

                # Some implementations return None
                if expert_indices is None:
                    continue

                tokens = input_ids.view(-1)
                experts = expert_indices.view(-1)

                for token, expert in zip(tokens, experts):

                    expert_token_map[int(expert)].append(int(token))
                    token_counter += 1

                    if token_counter >= MAX_TOKENS_TO_ANALYZE:
                        break

                if token_counter >= MAX_TOKENS_TO_ANALYZE:
                    break

            if token_counter >= MAX_TOKENS_TO_ANALYZE:
                break

    print("\n")
    print("===================================")
    print("EXPERT SPECIALIZATION ANALYSIS")
    print("===================================")

    for expert_id in sorted(expert_token_map.keys()):

        print("\n-----------------------------------")
        print(f"Expert {expert_id}")
        print("-----------------------------------")

        token_counts = Counter(expert_token_map[expert_id])

        for token_id, count in token_counts.most_common(20):

            token = tokenizer.decode([token_id])
            token = token.replace("\n", "\\n")

            print(f"{token:<12} {count}")


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

    print("Loading model...")

    model = GPT(Config.from_name("Qwen3-0.6B-MoE")).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    print("\nStarting expert analysis...\n")

    analyze_experts(model, tokenizer, dataloader)


if __name__ == "__main__":
    main()