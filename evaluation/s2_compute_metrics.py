import os
import time
import pandas as pd
import evaluate
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from comet import download_model, load_from_checkpoint


INPUT_FILE = "results/evaluation_outputs/mix_test_2/predictions.csv"
OUTPUT_DIR = Path(INPUT_FILE).parent

# Load metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

# Load COMET
comet_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_path)


# ---------------------------------------------------------
# Token F1
# ---------------------------------------------------------

def normalize_text(text):
    return " ".join(text.lower().split())


def token_f1(pred, ref):

    p = set(normalize_text(pred).split())
    r = set(normalize_text(ref).split())

    if len(p) == 0 or len(r) == 0:
        return 0

    inter = len(p & r)

    precision = inter / len(p)
    recall = inter / len(r)

    if precision + recall == 0:
        return 0

    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------
# COMET
# ---------------------------------------------------------

def compute_comet(preds, refs):

    data = [{"src": "", "mt": p, "ref": r} for p, r in zip(preds, refs)]

    scores = comet_model.predict(
        data,
        batch_size=32,
        gpus=1
    )

    return sum(scores["scores"]) / len(scores["scores"])


# ---------------------------------------------------------
# MAIN METRICS
# ---------------------------------------------------------

def compute_metrics():

    start_total = time.time()

    df = pd.read_csv(INPUT_FILE)

    results = []

    for model in df["model"].unique():

        print(f"\nComputing metrics for {model}")

        start_model = time.time()

        subset = df[df["model"] == model]

        preds = subset["prediction"].tolist()
        refs = subset["reference"].tolist()

        # BLEU
        bleu_refs = [[r] for r in refs]
        b = bleu.compute(predictions=preds, references=bleu_refs)

        # ROUGE
        r = rouge.compute(predictions=preds, references=refs)

        # BERTScore
        bert = bertscore.compute(
            predictions=preds,
            references=refs,
            model_type="xlm-roberta-large"
        )

        bert_f1 = sum(bert["f1"]) / len(bert["f1"])

        # Token F1
        token_f1_scores = [token_f1(p, r) for p, r in zip(preds, refs)]
        token_f1_avg = sum(token_f1_scores) / len(token_f1_scores)

        # COMET
        comet_score = compute_comet(preds, refs)

        # Router entropy (from generation step)
        router_entropy = subset["router_entropy"].mean()

        # Generation time stats
        avg_gen_time = subset["generation_time_sec"].mean()
        total_gen_time = subset["generation_time_sec"].sum()

        # Metric computation time
        metric_time = time.time() - start_model

        print(f"{model} metric computation time: {metric_time:.2f} sec")

        results.append({
            "Model": model,
            "BLEU": b["bleu"],
            "ROUGE-L": r["rougeL"],
            "BERTScore": bert_f1,
            "TokenF1": token_f1_avg,
            "COMET": comet_score,
            "RouterEntropy": router_entropy,
            "Avg_Generation_Time_sec": avg_gen_time,
            "Total_Generation_Time_sec": total_gen_time,
            "Metric_Compute_Time_sec": metric_time
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    total_time = time.time() - start_total

    print("\nEvaluation Results:\n")
    print(results_df)

    print(f"\nTotal evaluation runtime: {total_time:.2f} sec")


    sns.set(style="whitegrid")

    metric_columns = [
        "BLEU",
        "ROUGE-L",
        "BERTScore",
        "TokenF1",
        "COMET",
        "RouterEntropy"
    ]

    melted = results_df.melt(
        id_vars="Model",
        value_vars=metric_columns
    )

    plt.figure(figsize=(12,6))

    sns.barplot(data=melted, x="variable", y="value", hue="Model")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR/ "evaluation_plot.png")
    )

    print(f"saved at {OUTPUT_DIR}")


if __name__ == "__main__":
    compute_metrics()