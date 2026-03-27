#!/usr/bin/env python3
"""
Production-grade evaluation script for model predictions.

Features
--------
• Robust schema validation
• Supports multi-model evaluation
• Computes modern semantic metrics
• Handles optional reasoning fields
• Aggregates efficiency metrics
• Saves results + plots
• Deterministic execution

Required columns in predictions.csv
-----------------------------------
model
prediction
reference

Optional columns
----------------
thought
generation_time_sec
router_entropy
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List

import torch
import pandas as pd
import numpy as np

import evaluate
import matplotlib.pyplot as plt
import seaborn as sns

from comet import download_model, load_from_checkpoint


torch.manual_seed(42)
np.random.seed(42)

# -------------------------------------------------------
# Logging
# -------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------
# Metrics Utilities
# -------------------------------------------------------

INPUT_FILE = "results/evaluation_outputs/mix_test/predictions.csv"

def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def token_f1(pred: str, ref: str) -> float:

    p = set(normalize_text(pred).split())
    r = set(normalize_text(ref).split())

    if len(p) == 0 or len(r) == 0:
        return 0.0

    inter = len(p & r)

    precision = inter / len(p)
    recall = inter / len(r)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


# -------------------------------------------------------
# COMET Metric
# -------------------------------------------------------

class CometEvaluator:

    def __init__(self, batch_size: int = 64):

        logger.info("Loading COMET model...")

        model_path = download_model("Unbabel/wmt22-comet-da")
        self.model = load_from_checkpoint(model_path)

        self.batch_size = batch_size
 
    def compute(self, preds: List[str], refs: List[str]) -> float:

        data = [
            {"src": "", "mt": p, "ref": r}
            for p, r in zip(preds, refs)
        ]

        scores = self.model.predict(
            data,
            batch_size=self.batch_size,
            gpus=1 if torch.cuda.is_available() else 0
        )

        return float(np.mean(scores["scores"]))


# -------------------------------------------------------
# Metric Evaluator
# -------------------------------------------------------

class MetricEvaluator:

    def __init__(self):

        logger.info("Loading evaluation metrics...")

        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")

        self.comet = CometEvaluator()

    def compute_metrics(
        self,
        preds: List[str],
        refs: List[str]
    ) -> Dict[str, float]:

        results = {}

        # BLEU
        bleu_refs = [[r] for r in refs]

        bleu_res = self.bleu.compute(
            predictions=preds,
            references=bleu_refs
        )

        results["BLEU"] = bleu_res["bleu"]

        # ROUGE
        rouge_res = self.rouge.compute(
            predictions=preds,
            references=refs
        )

        results["ROUGE-L"] = rouge_res["rougeL"]

        # BERTScore
        bert = self.bertscore.compute(
            predictions=preds,
            references=refs,
            model_type="xlm-roberta-large"
        )

        results["BERTScore"] = float(np.mean(bert["f1"]))

        # Token F1
        token_scores = [
            token_f1(p, r)
            for p, r in zip(preds, refs)
        ]

        results["TokenF1"] = float(np.mean(token_scores))

        # COMET
        results["COMET"] = self.comet.compute(preds, refs)

        return results


# -------------------------------------------------------
# Evaluation Pipeline
# -------------------------------------------------------

class EvaluationPipeline:

    REQUIRED_COLUMNS = [
        "model",
        "prediction",
        "reference"
    ]

    OPTIONAL_COLUMNS = [
        "generation_time_sec",
        "router_entropy",
        "thought"
    ]

    def __init__(
        self,
        input_file: Path,
        output_dir: Path
    ):

        self.input_file = input_file
        self.output_dir = output_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = MetricEvaluator()

    def validate_schema(self, df: pd.DataFrame):

        for col in self.REQUIRED_COLUMNS:

            if col not in df.columns:

                raise ValueError(
                    f"Missing required column: {col}"
                )

    def evaluate(self):

        logger.info("Loading predictions file")

        df = pd.read_csv(self.input_file)

        df["prediction"] = df["prediction"].astype(str).str.strip()
        df["reference"] = df["reference"].astype(str).str.strip()

        self.validate_schema(df)

        results = []

        total_start = time.time()

        for model_name in sorted(df["model"].unique()):

            logger.info(f"Evaluating model: {model_name}")

            model_start = time.time()

            subset = df[df["model"] == model_name]

            preds = subset["prediction"].astype(str).tolist()
            refs = subset["reference"].astype(str).tolist()

            filtered = [(p, r) for p, r in zip(preds, refs) if p.strip() != ""]
            preds = [p for p, _ in filtered]
            refs = [r for _, r in filtered]

            metric_values = self.metrics.compute_metrics(preds, refs)

            # Optional metrics
            if "generation_time_sec" in subset.columns:

                metric_values["Avg_Generation_Time"] = float(
                    subset["generation_time_sec"].mean()
                )

                metric_values["Total_Generation_Time"] = float(
                    subset["generation_time_sec"].sum()
                )

            if "router_entropy" in subset.columns:

                metric_values["RouterEntropy"] = float(
                    subset["router_entropy"].mean()
                )

            metric_values["Model"] = model_name
            metric_values["Metric_Compute_Time"] = time.time() - model_start

            results.append(metric_values)

        total_runtime = time.time() - total_start

        results_df = pd.DataFrame(results)

        logger.info("Saving results")

        results_path = self.output_dir / "metrics.csv"

        results_df.to_csv(results_path, index=False)

        logger.info(f"Metrics saved → {results_path}")

        logger.info(f"Total evaluation runtime: {total_runtime:.2f} sec")

        self.plot_metrics(results_df)

    def plot_metrics(self, results_df: pd.DataFrame):

        logger.info("Generating evaluation plots")

        excluded_cols = {
            "Model",
            "Avg_Generation_Time",
            "Total_Generation_Time",
            "Metric_Compute_Time",
            "RouterEntropy"
        }

        metric_cols = [
            "BLEU",
            "ROUGE-L",
            "BERTScore",
            "TokenF1",
            "COMET"
        ]

        plot_df = results_df.melt(
            id_vars="Model",
            value_vars=metric_cols,
            var_name="Metric",
            value_name="Score"
        )

        sns.set(style="whitegrid")

        plt.figure(figsize=(12, 6))

        sns.barplot(
            data=plot_df,
            x="Metric",
            y="Score",
            hue="Model",
            order=["BLEU", "ROUGE-L", "BERTScore", "TokenF1", "COMET"]
        )

        plt.xticks(rotation=45)
        plt.tight_layout()

        plot_path = self.output_dir / "evaluation_plot.png"

        plt.savefig(plot_path)

        logger.info(f"Plot saved → {plot_path}")



# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    input_file = Path(INPUT_FILE)

    if not input_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {input_file}")

    output_dir = input_file.parent

    pipeline = EvaluationPipeline(
        input_file=input_file,
        output_dir=output_dir
    )

    pipeline.evaluate()


if __name__ == "__main__":
    main()