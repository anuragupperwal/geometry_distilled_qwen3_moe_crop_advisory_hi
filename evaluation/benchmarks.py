#!/usr/bin/env python3
import re
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from datasets import load_dataset
from pathlib import Path

from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer

torch.manual_seed(42)
np.random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "results/benchmark_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOKENIZER_DIR = "checkpoints/Qwen/Qwen3-0.6B"

MODELS = {
    "Teacher": (
        "checkpoints/Qwen/Qwen3-8B/lit_model.pth",
        "Qwen3-8B"
    ),
    "Base": (
        "checkpoints/Qwen/Qwen3-0.6B/lit_model.pth",
        "Qwen3-0.6B"
    ),
    "Distilled": (
        "checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/23_03_run_test_80k_1D76FE/epoch-2.pth",
        "Qwen3-0.6B-MoE"
    )
}


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

    for i in range(max_new_tokens):

        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)

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


def evaluate_mgsm(model, tokenizer):

    ds = load_dataset("openai/gsm8k", "main", split="test[:100]")

    correct = 0
    total = 0

    for row in tqdm(ds):

        prompt = f"Question: {row['question']}\nAnswer:"

        input_ids = tokenizer.encode(prompt).to(DEVICE).unsqueeze(0)

        with torch.no_grad():
            out = generate(model, input_ids, 64, tokenizer.eos_id)

        text = tokenizer.decode(out[0])

        pred_numbers = re.findall(r"-?\d+\.?\d*", text)

        gold_numbers = re.findall(r"-?\d+\.?\d*", row["answer"])
        if len(pred_numbers) > 0 and len(gold_numbers) > 0:
            pred = float(pred_numbers[-1])
            gold = float(gold_numbers[-1])

            if abs(pred - gold) < 1e-6:
                correct += 1

        total += 1

    return correct / total


def normalize(text):
    return text.lower().strip()

def evaluate_tydiqa(model, tokenizer):

    ds = load_dataset("tydiqa", "secondary_task", split="validation")

    f1_scores = []

    for row in tqdm(ds):

        # filter only Hindi samples
        if row["language"] != "hindi":
            continue

        prompt = f"""
You are a helpful assistant.

Context:
{row['context']}

Question:
{row['question']}

Answer the question using only the context.

Answer:
"""

        input_ids = tokenizer.encode(prompt).to(DEVICE).unsqueeze(0)

        with torch.no_grad():
            out = generate(model, input_ids, 32, tokenizer.eos_id)

        pred = tokenizer.decode(out[0])

        gold = row["answers"]["text"][0]

        pred = normalize(pred)
        gold = normalize(gold)

        overlap = len(set(pred.split()) & set(gold.split()))
        precision = overlap / len(pred.split()) if pred else 0
        recall = overlap / len(gold.split()) if gold else 0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        f1_scores.append(f1)

    return np.mean(f1_scores)



def main():
    results = []

    tokenizer = Tokenizer(TOKENIZER_DIR)

    for model_name, (ckpt, config_name) in MODELS.items():

        print(f"\nEvaluating {model_name}")

        config = Config.from_name(config_name)
        model = GPT(config).to(DEVICE, dtype=torch.bfloat16)

        weights = torch.load(ckpt, map_location=DEVICE, weights_only=True)

        if "model_state_dict" in weights:
            weights = weights["model_state_dict"]

        model.load_state_dict(weights, strict=False)
        model.eval()

        mgsm = evaluate_mgsm(model, tokenizer)
        # tydiqa = evaluate_tydiqa(model, tokenizer)

        results.append({
            "Model": model_name,
            "MGSM": mgsm,
            # "TyDiQA_F1": tydiqa
        })

        del model
        torch.cuda.empty_cache()


    df = pd.DataFrame(results)

    df.to_csv(f"{OUTPUT_DIR}/benchmark_results.csv", index=False)

    print(df)


    sns.set(style="whitegrid")

    plt.figure(figsize=(10,6))

    df_melt = df.melt(id_vars="Model")

    sns.barplot(data=df_melt, x="variable", y="value", hue="Model")

    plt.ylabel("Score")
    plt.xlabel("Benchmark")

    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/benchmark_plot.png")



    labels = df.columns[1:]

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)

    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)

    for i, row in df.iterrows():

        values = row[1:].values.astype(float)
        values = np.concatenate([values, [values[0]]])
        angles_closed = np.concatenate([angles, [angles[0]]])

        ax.plot(angles_closed, values, label=row["Model"])

    ax.set_thetagrids(angles * 180/np.pi, labels)

    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/radar_plot.png")

    with open(f"{OUTPUT_DIR}/benchmark_table.tex","w") as f:
        f.write(df.to_latex(index=False,float_format="%.3f"))


if __name__== "__main__":
    main()