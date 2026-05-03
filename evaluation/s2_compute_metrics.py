import os
import time
import pandas as pd
import evaluate
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from comet import download_model, load_from_checkpoint


INPUT_FILE = "results/evaluation_outputs/01_04_run_test_80k_A45F72/predictions.csv"
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
        # avg_perplexity = subset["perplexity"].mean()
        avg_log = np.log(subset["perplexity"]).mean()
        avg_perplexity = np.exp(avg_log)

        preds = subset["prediction"].tolist()
        refs = subset["reference"].tolist()

        # BLEU
        # bleu_refs = [[r] for r in refs]
        # b = bleu.compute(predictions=preds, references=bleu_refs)

        # ROUGE
        # r = rouge.compute(predictions=preds, references=refs)

        # BERTScore
        # BERTScore (Precision / Recall / F1)
        bert = bertscore.compute(
            predictions=preds,
            references=refs,
            model_type="xlm-roberta-large",
            batch_size=32,
            lang="hi",
        )

        bert_precision = sum(bert["precision"]) / len(bert["precision"])
        bert_recall = sum(bert["recall"]) / len(bert["recall"])
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
            "BERTScore_P": bert_precision,
            "BERTScore_R": bert_recall,
            "BERTScore_F1": bert_f1,
            "TokenF1": token_f1_avg,
            "COMET": comet_score,
            "RouterEntropy": router_entropy,
            "Perplexity": avg_perplexity,
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
        "BERTScore_P",
        "BERTScore_R",
        "BERTScore_F1",
        # "Perplexity",
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


    corr_metrics = [
        "BERTScore_P",
        "BERTScore_R",
        "BERTScore_F1",
        "Perplexity",
        "TokenF1",
        "COMET"
    ]

    corr = results_df[corr_metrics].corr()

    plt.figure(figsize=(8,6))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Metric Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "metric_correlation_heatmap.png"
    )


    # ---------------------------------------------------------
    # GENERATION METRICS RADAR CHART
    # ---------------------------------------------------------

    results_df["Perplexity_inv"] = 1 / results_df["Perplexity"]
    radar_metrics = [
        "BERTScore_F1",
        "Perplexity_inv",
        "TokenF1",
        "COMET"
    ]

    labels = radar_metrics
    num_metrics = len(labels)

    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False)

    fig = plt.figure(figsize=(7,7))
    ax = plt.subplot(111, polar=True)

    for i, row in results_df.iterrows():

        values = row[radar_metrics].values.astype(float)

        values = np.concatenate([values, [values[0]]])
        angles_closed = np.concatenate([angles, [angles[0]]])

        ax.plot(
            angles_closed,
            values,
            linewidth=2,
            label=row["Model"]
        )

        ax.fill(
            angles_closed,
            values,
            alpha=0.1
        )

    ax.set_thetagrids(angles * 180/np.pi, labels)

    plt.title("Model Performance Radar Chart")

    plt.legend(loc="upper right")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "generation_radar_plot.png"
    )

    #all metrics
    quality_metrics = [
        "BERTScore_P",
        "BERTScore_R",
        "BERTScore_F1",
        "TokenF1",
        "COMET",
        "RouterEntropy"
    ]

    melted_quality = results_df.melt(
        id_vars="Model",
        value_vars=quality_metrics
    )

    plt.figure(figsize=(12,6))

    sns.barplot(
        data=melted_quality,
        x="variable",
        y="value",
        hue="Model"
    )

    plt.ylabel("Score")
    plt.xlabel("Metric")

    plt.xticks(rotation=40)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "quality_metrics_plot.png",
        dpi=300
    )

    # Perplexity plot
    plt.figure(figsize=(6,5))

    sns.barplot(
        data=results_df,
        x="Model",
        y="Perplexity"
    )

    plt.title("Model Perplexity Comparison")
    plt.ylabel("Perplexity (Lower is Better)")
    plt.xlabel("Model")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "perplexity_plot.png",
        dpi=300
    )


    # Perplexity vs Quality scatter plot - tells Does lower perplexity → better advisory quality?
    plt.figure(figsize=(6,5))

    sns.scatterplot(
        data=results_df,
        x="Perplexity",
        y="BERTScore_F1",
        hue="Model",
        s=150
    )

    plt.title("Perplexity vs Semantic Quality")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "perplexity_vs_quality.png",
        dpi=300
    )
    plt.close()

    #table
    latex_table = results_df[
        [
            "Model",
            "BERTScore_F1",
            "TokenF1",
            "COMET",
            "Perplexity",
            "RouterEntropy"
        ]
    ].round(3)

    with open(OUTPUT_DIR / "metrics_table.tex", "w") as f:
        f.write(
            latex_table.to_latex(
                index=False,
                float_format="%.3f",
                column_format="lccccc",
                escape=False,
                bold_rows=False
            )
        )
    
    
    plt.figure(figsize=(11,2.5))
    plt.axis("off")

    table_df = results_df[
        [
            "Model",
            "BERTScore_P",
            "BERTScore_R",
            "BERTScore_F1",
            "Perplexity",
            "TokenF1",
            "COMET"
        ]
    ].round(3)

    table = plt.table(
        cellText=table_df.round(3).values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1,1.5)

    n_rows = len(table_df) + 1
    n_cols = len(table_df.columns)

    # Remove all borders first
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0)
        cell.visible_edges = ""

    # Top rule
    for col in range(n_cols):
        cell = table[(0, col)]
        cell.visible_edges = "T"
        cell.set_linewidth(1.5)

    # Mid rule (under header)
    for col in range(n_cols):
        cell = table[(0, col)]
        cell.visible_edges += "B"
        cell.set_linewidth(1.0)

    # Bottom rule
    for col in range(n_cols):
        cell = table[(n_rows-1, col)]
        cell.visible_edges = "B"
        cell.set_linewidth(1.5)

    # Bold header
    for col in range(n_cols):
        table[(0, col)].get_text().set_weight("bold")

    plt.savefig(
        OUTPUT_DIR / "metrics_table.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

if __name__ == "__main__":
    compute_metrics()