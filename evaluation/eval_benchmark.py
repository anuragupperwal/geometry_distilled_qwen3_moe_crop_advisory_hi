import os
import torch
import pandas as pd
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

from tqdm import tqdm
from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer

# =========================================================
# CONFIG
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_DATA_PATH = "data/test_data/test_part_003.parquet"

NUM_SAMPLES = 1   # 🔥 IMPORTANT: increase for real evaluation
MAX_NEW_TOKENS = 1024

MODELS = {
    # "Teacher(Base)": (
    #     "checkpoints/Qwen/Qwen3-8B/lit_model.pth",
    #     "Qwen3-8B",
    #     "two_phase"
    # ),
    # "Student (Base)": (
    #     "checkpoints/Qwen/Qwen3-0.6B-moe-init/lit_model.pth",
    #     "Qwen3-0.6B-MoE",
    #     "two_phase"
    # ),
    "Student (Distilled Final)": (
        "checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/20_03_run_test_80k_1FC9E6/lit_model.pth",
        "Qwen3-0.6B-MoE",
        "two_phase"
    )
}

# =========================================================
# LOAD METRICS
# =========================================================

print("⏳ Loading Metrics...")
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

# =========================================================
# HELPERS
# =========================================================

def normalize_text(text):
    return " ".join(text.strip().lower().split())


def exact_match(preds, refs):
    return sum(int(normalize_text(p) == normalize_text(r)) for p, r in zip(preds, refs)) / len(preds)


def token_f1(preds, refs):
    scores = []
    for p, r in zip(preds, refs):
        p_tokens = set(normalize_text(p).split())
        r_tokens = set(normalize_text(r).split())

        if len(p_tokens) == 0 or len(r_tokens) == 0:
            scores.append(0)
            continue

        common = len(p_tokens & r_tokens)
        precision = common / len(p_tokens)
        recall = common / len(r_tokens)

        if precision + recall == 0:
            scores.append(0)
        else:
            scores.append(2 * precision * recall / (precision + recall))

    return sum(scores) / len(scores)


def get_config(config_name):
    config = Config.from_name(config_name)
    if "8B" in config_name:
        config.n_embd = 4096
        config.n_head = 32
        config.intermediate_size = 12288
    return config


# =========================================================
# GENERATION (DETERMINISTIC FOR EVAL)
# =========================================================

def generate(model, idx, max_new_tokens, stop_tokens=None):
    B, T = idx.shape
    model.set_kv_cache(batch_size=B, max_seq_length=T + max_new_tokens, device=DEVICE)

    input_pos = torch.arange(0, T, device=DEVICE)
    logits = model(idx, input_pos=input_pos)[:, -1, :]

    generated = []

    for i in range(max_new_tokens):

        # 🔥 deterministic (no randomness)
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if stop_tokens and idx_next.item() in stop_tokens:
            break

        idx = torch.cat((idx, idx_next), dim=1)
        generated.append(idx_next)

        input_pos = torch.tensor([T + i], device=DEVICE)
        logits = model(idx_next, input_pos=input_pos)[:, -1, :]

    if len(generated) == 0:
        return torch.tensor([[]], device=DEVICE)

    return torch.cat(generated, dim=1)


# =========================================================
# PLOTTING
# =========================================================

def plot_metrics(df, save_path):
    sns.set_theme(style="whitegrid")

    df_melted = df.melt(
        id_vars="Model",
        value_vars=["BLEU", "ROUGE-L", "BERTScore", "ExactMatch", "TokenF1"],
        var_name="Metric",
        value_name="Score"
    )

    plt.figure(figsize=(14, 6))

    chart = sns.barplot(
        data=df_melted,
        x="Metric",
        y="Score",
        hue="Model",
        palette="viridis",
        edgecolor="black"
    )

    for container in chart.containers:
        chart.bar_label(container, fmt='%.3f', padding=3)

    plt.title("Model Evaluation Metrics", fontsize=16)
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================================================
# MAIN EVAL
# =========================================================

def run_evaluation():

    tokenizer = Tokenizer("checkpoints/Qwen/Qwen3-0.6B-moe-init")

    eos_id = tokenizer.eos_id

    # ChatML tokens
    assistant_tag = "<|im_start|>assistant\n"
    assistant_ids = tokenizer.encode(assistant_tag, bos=False, eos=False).to(DEVICE)

    df = pd.read_parquet(TEST_DATA_PATH)
    df = df.head(1)

    if len(df) > NUM_SAMPLES:
        df = df.sample(NUM_SAMPLES, random_state=42)

    results = []
    qualitative_log = []

    for model_name, (ckpt_path, config_name, strategy) in MODELS.items():

        print(f"\n🧠 Evaluating: {model_name}")

        if not os.path.exists(ckpt_path):
            print("❌ Checkpoint missing")
            continue

        config = get_config(config_name)

        model = GPT(config).to(DEVICE, dtype=torch.float16)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True), strict=False)
        model.eval()

        predictions = []
        references = []

        for row in tqdm(df.itertuples(index=False), total=len(df)):
            crop = getattr(row, "crop", "")
            region = getattr(row, "region", "")
            stage = getattr(row, "stage", "")

            query = getattr(row, "prompt", "")
            target = getattr(row, "advisory", "")
            
            scenario = f"फसल: {getattr(row, 'crop', '')} | क्षेत्र: {getattr(row, 'region', '')} | चरण: {getattr(row, 'stage', '')}"
            row.prompt
            row.advisory

            # =====================================================
            # CHATML PROMPT
            # =====================================================

            base_prompt = (
                "<|im_start|>system\n"
                "You are an agricultural expert.\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"स्थिति:\n{scenario}\nसमस्या: {query}\n"
                "<|im_end|>\n"
            )

            if strategy == "two_phase":

                # Thought phase
                thought_prompt = base_prompt + "<|im_start|>assistant\n"
                input_ids = tokenizer.encode(thought_prompt, bos=True, eos=False).to(DEVICE).unsqueeze(0)
                
                with torch.no_grad():
                    model.clear_kv_cache()
                    thought_ids = generate(model, input_ids, 256, stop_tokens={eos_id})
                    thought_text = tokenizer.decode(thought_ids[0])

                    # Final answer
                    final_prompt = thought_prompt + thought_text + "\n<|im_end|>\n<|im_start|>assistant\n"
                    input_ids = tokenizer.encode(final_prompt, bos=True, eos=False).to(DEVICE).unsqueeze(0)
                    model.clear_kv_cache()
                    output_ids = generate(model, input_ids, MAX_NEW_TOKENS, stop_tokens={eos_id})

            else:

                prompt = base_prompt + "<|im_start|>assistant\n"
                input_ids = tokenizer.encode(prompt, bos=True, eos=False).to(DEVICE).unsqueeze(0)

                output_ids = generate(model, input_ids, MAX_NEW_TOKENS, stop_tokens={eos_id})

            prediction = tokenizer.decode(output_ids[0]).strip()

            if not prediction:
                prediction = "<empty>"

            predictions.append(prediction)
            references.append(target)

            qualitative_log.append({
                "Model": model_name,
                "Query": query,
                "GroundTruth": target,
                "Prediction": prediction
            })

        # =====================================================
        # METRICS
        # =====================================================

        print("📉 Computing metrics...")

        bleu_refs = [[r] for r in references]

        b = bleu.compute(predictions=predictions, references=bleu_refs)
        r = rouge.compute(predictions=predictions, references=references, use_stemmer=True)

        bert = bertscore.compute(
            predictions=predictions,
            references=references,
            model_type="xlm-roberta-large",
            device=DEVICE
        )

        bert_f1 = sum(bert["f1"]) / len(bert["f1"])

        em = exact_match(predictions, references)
        f1 = token_f1(predictions, references)

        scores = {
            "Model": model_name,
            "BLEU": b["bleu"],
            "ROUGE-L": r["rougeL"],
            "BERTScore": bert_f1,
            "ExactMatch": em,
            "TokenF1": f1
        }

        print(scores)

        results.append(scores)

        del model
        torch.cuda.empty_cache()

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    os.makedirs("outputs", exist_ok=True)

    if results:
        df_res = pd.DataFrame(results)
        print("\n🏆 Final Results:")
        print(df_res.to_markdown(index=False))

        df_res.to_csv("outputs/eval_metrics.csv", index=False)
        plot_metrics(df_res, "outputs/eval_plot.png")

    if qualitative_log:
        pd.DataFrame(qualitative_log).to_csv("outputs/qualitative.csv", index=False)


if __name__ == "__main__":
    run_evaluation()