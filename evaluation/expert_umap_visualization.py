import torch
import numpy as np
import umap
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader

from litgpt.tokenizer import Tokenizer
from litgpt.model import GPT
from litgpt.config import Config
from agri_data import AgriDataset


CHECKPOINT_PATH = "checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/12_03_run_test_80k_5C0943/lit_model.pth"
TOKENIZER_PATH = "checkpoints/Qwen/Qwen3-0.6B"
DATA_PATH = "data/train_bilingual_mixed_83k_agri65k.parquet"

MAX_SEQ_LENGTH = 4072
BATCH_SIZE = 2
MAX_TOKENS = 30000

TARGET_LAYER = 14


def collect_vectors(model, dataloader):

    hidden_vectors = []
    expert_ids = []

    device = next(model.parameters()).device

    total = 0

    model.eval()

    with torch.no_grad():

        for input_ids, _ in tqdm(dataloader):

            input_ids = input_ids.to(device)

            model(input_ids)

            layer = model.transformer.h[TARGET_LAYER]

            h = layer.mlp.last_hidden_input
            idx = layer.mlp.last_indices

            if h is None:
                continue

            h = h.cpu()
            idx = idx.cpu()

            for vec, experts in zip(h, idx):

                hidden_vectors.append(vec.float().numpy())
                expert_ids.append(int(experts[0]))

                total += 1

                if total > MAX_TOKENS:
                    return np.array(hidden_vectors), np.array(expert_ids)

    return np.array(hidden_vectors), np.array(expert_ids)


def plot_umap(hidden_vectors, expert_ids):

    print("Running UMAP projection...")

    # normalize vectors
    hidden_vectors = hidden_vectors / np.linalg.norm(hidden_vectors, axis=1, keepdims=True)

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        n_components=2,
        metric="cosine",
        random_state=42
    )

    embedding = reducer.fit_transform(hidden_vectors)

    plt.figure(figsize=(10,8))

    scatter = plt.scatter(
        embedding[:,0],
        embedding[:,1],
        c=expert_ids,
        cmap="tab10",
        s=4,
        alpha=0.7
    )

    plt.colorbar(scatter, label="Expert ID")

    plt.title("MoE Expert Routing UMAP Visualization (Layer 6)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    save_path = "outputs/12_03_run_test_80k_5C0943/expert_umap_layer6.png"

    plt.savefig(save_path, dpi=300)

    print(f"Saved plot to {save_path}")

    plt.show()


def main():

    device = torch.device("cuda")

    tokenizer = Tokenizer(TOKENIZER_PATH)

    dataset = AgriDataset(
        data_path=DATA_PATH,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH
    )

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = GPT(Config.from_name("Qwen3-0.6B-MoE")).to(device, dtype=torch.bfloat16)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    hidden_vectors, expert_ids = collect_vectors(model, dataloader)

    plot_umap(hidden_vectors, expert_ids)


if __name__ == "__main__":
    main()