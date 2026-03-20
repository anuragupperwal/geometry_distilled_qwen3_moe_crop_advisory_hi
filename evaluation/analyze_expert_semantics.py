import json
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm

from litgpt.tokenizer import Tokenizer
from litgpt.model import GPT
from litgpt.config import Config
from agri_data import AgriDataset


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT_PATH = "checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/12_03_run_test_80k_5C0943/lit_model.pth"
TOKENIZER_PATH = "checkpoints/Qwen/Qwen3-0.6B"
DATA_PATH = "data/train_bilingual_mixed_83k_agri65k.parquet"

MAX_SEQ_LENGTH = 4072
BATCH_SIZE = 2

MAX_TOKENS = 30000
N_CLUSTERS = 12
PCA_DIM = 64

OUTPUT_DIR = Path("outputs/12_03_run_test_80k_5C0943")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = OUTPUT_DIR / "expert_semantic_analysis.json"


# ============================================================
# COLLECTION
# ============================================================

def collect_hidden_and_experts(model, dataloader, tokenizer):
    """
    Collects:
      - hidden vectors before expert block
      - layer_id
      - top-1 routed expert_id
      - token_id
      - token_text

    Returns lists that can later be converted to numpy arrays.
    """
    hidden_vectors = []
    layer_ids = []
    expert_ids = []
    token_ids = []
    token_texts = []

    device = next(model.parameters()).device
    model.eval()

    total = 0

    with torch.no_grad():
        for input_ids, _ in tqdm(dataloader, desc="Collecting hidden states"):
            input_ids = input_ids.to(device)

            # Trigger forward pass so last_hidden_input / last_indices get populated
            model(input_ids)

            # Flatten input tokens once for alignment with last_hidden_input
            flat_tokens = input_ids.view(-1).detach().cpu()

            for layer_id, layer in enumerate(model.transformer.h):
                if not hasattr(layer.mlp, "last_hidden_input"):
                    continue

                h = layer.mlp.last_hidden_input
                idx = layer.mlp.last_indices

                if h is None or idx is None:
                    continue

                # h shape: [B*T, C]
                # idx shape: [B*T, top_k]
                h = h.detach().cpu().to(torch.float32)
                idx = idx.detach().cpu()

                # Safety check
                if h.size(0) != flat_tokens.size(0):
                    continue

                for vec, routed_experts, tok in zip(h, idx, flat_tokens):
                    # top-1 expert only for cleaner first-pass analysis
                    expert_id = int(routed_experts[0].item())
                    token_id = int(tok.item())

                    hidden_vectors.append(vec.numpy())
                    layer_ids.append(layer_id)
                    expert_ids.append(expert_id)
                    token_ids.append(token_id)

                    # tokenizer.decode expects tensor, not python list
                    tok_text = tokenizer.decode(torch.tensor([token_id]))
                    tok_text = tok_text.replace("\n", "\\n")
                    token_texts.append(tok_text)

                    total += 1
                    if total >= MAX_TOKENS:
                        return (
                            np.array(hidden_vectors, dtype=np.float32),
                            np.array(layer_ids, dtype=np.int32),
                            np.array(expert_ids, dtype=np.int32),
                            np.array(token_ids, dtype=np.int32),
                            token_texts,
                        )

    return (
        np.array(hidden_vectors, dtype=np.float32),
        np.array(layer_ids, dtype=np.int32),
        np.array(expert_ids, dtype=np.int32),
        np.array(token_ids, dtype=np.int32),
        token_texts,
    )


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_hidden_vectors(hidden_vectors):
    """
    Normalize and reduce dimensionality before clustering.
    """
    # L2 normalize for more meaningful KMeans geometry
    norms = np.linalg.norm(hidden_vectors, axis=1, keepdims=True) + 1e-8
    hidden_vectors = hidden_vectors / norms

    # PCA for denoising + faster clustering
    pca_dim = min(PCA_DIM, hidden_vectors.shape[1], hidden_vectors.shape[0] - 1)
    if pca_dim >= 2:
        pca = PCA(n_components=pca_dim, random_state=0)
        hidden_vectors = pca.fit_transform(hidden_vectors)

    return hidden_vectors


# ============================================================
# ANALYSIS
# ============================================================

def analyze_semantics(hidden_vectors, layer_ids, expert_ids, token_ids, token_texts):
    """
    Clusters hidden vectors and builds semantic summaries per (layer, expert).
    """
    print("\nClustering semantic representations...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=10)
    cluster_ids = kmeans.fit_predict(hidden_vectors)

    # Structure:
    # results[layer_id][expert_id] = {
    #   "num_tokens": ...,
    #   "clusters": Counter,
    #   "top_tokens": Counter
    # }
    results = defaultdict(lambda: defaultdict(lambda: {
        "num_tokens": 0,
        "clusters": Counter(),
        "top_tokens": Counter(),
    }))

    for layer_id, expert_id, cluster_id, token_text in zip(
        layer_ids, expert_ids, cluster_ids, token_texts
    ):
        entry = results[int(layer_id)][int(expert_id)]
        entry["num_tokens"] += 1
        entry["clusters"][int(cluster_id)] += 1
        entry["top_tokens"][token_text] += 1

    return results


# ============================================================
# PRINTING
# ============================================================

def print_results(results, top_clusters=5, top_tokens=15):
    print("\n" + "=" * 70)
    print("EXPERT SEMANTIC SPECIALIZATION")
    print("=" * 70)

    for layer_id in sorted(results.keys()):
        print(f"\n\nLAYER {layer_id}")
        print("-" * 70)

        for expert_id in sorted(results[layer_id].keys()):
            entry = results[layer_id][expert_id]
            total = entry["num_tokens"]

            print(f"\nExpert {expert_id} | tokens analyzed: {total}")

            print("  Cluster distribution:")
            for cluster_id, count in entry["clusters"].most_common(top_clusters):
                pct = 100.0 * count / max(total, 1)
                print(f"    Cluster {cluster_id}: {pct:.2f}% ({count})")

            print("  Top tokens:")
            shown = 0
            for token_str, count in entry["top_tokens"].most_common():
                # Skip empty-looking tokens for readability
                if token_str.strip() == "":
                    continue
                print(f"    {repr(token_str):<20} {count}")
                shown += 1
                if shown >= top_tokens:
                    break


# ============================================================
# SAVE
# ============================================================

def save_results(results, output_path):
    serializable = {}

    for layer_id in results:
        serializable[str(layer_id)] = {}
        for expert_id in results[layer_id]:
            entry = results[layer_id][expert_id]
            serializable[str(layer_id)][str(expert_id)] = {
                "num_tokens": entry["num_tokens"],
                "clusters": dict(entry["clusters"]),
                "top_tokens": dict(entry["top_tokens"].most_common(50)),
            }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"\nSaved semantic analysis to: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nLoading tokenizer...")
    tokenizer = Tokenizer(TOKENIZER_PATH)

    print("Loading dataset...")
    dataset = AgriDataset(
        data_path=DATA_PATH,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    print("Loading distilled MoE model...")
    model = GPT(Config.from_name("Qwen3-0.6B-MoE")).to(device, dtype=torch.bfloat16)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    print("\nCollecting expert routing data...\n")
    hidden_vectors, layer_ids, expert_ids, token_ids, token_texts = collect_hidden_and_experts(
        model, dataloader, tokenizer
    )

    print(f"\nCollected {len(hidden_vectors)} token representations.")

    processed_vectors = preprocess_hidden_vectors(hidden_vectors)

    results = analyze_semantics(
        processed_vectors,
        layer_ids,
        expert_ids,
        token_ids,
        token_texts,
    )

    print_results(results)
    save_results(results, OUTPUT_JSON)


if __name__ == "__main__":
    main()