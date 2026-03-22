import os
import re
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

import google.generativeai as genai
# from bleurt import score as bleurt_score
from moverscore import word_mover_score
from comet import download_model, load_from_checkpoint



comet_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_path)

# BLEURT_MODEL = "bleurt/BLEURT-20"
# bleurt_scorer = bleurt_score.BleurtScorer(BLEURT_MODEL)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

judge_model = genai.GenerativeModel("gemini-2.5-flash")

# =========================================================
# CONFIGURATION
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_DATA = "data/test/extracted_100_rows.parquet"
TOKENIZER_DIR = "checkpoints/Qwen/Qwen3-0.6B"

MAX_NEW_TOKENS = 3072

OUTPUT_DIR = "results/evaluation_outputs/20_03_run_test_80k_1FC9E6"

MODELS = {
    # "Teacher": (
    #     "checkpoints/Qwen/Qwen3-8B/lit_model.pth",
    #     "Qwen3-8B"
    # ),
    # "Base": (
    #     "checkpoints/Qwen/Qwen3-0.6B/lit_model.pth",
    #     "Qwen3-0.6B"
    # ),
    "Distilled": (
        "checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/20_03_run_test_80k_1FC9E6/lit_model.pth",
        "Qwen3-0.6B-MoE"
    )
}


# =========================================================
# METRIC INITIALIZATION
# =========================================================

print("Loading metrics...")

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def get_config(name):
    config = Config.from_name(name)

    if "8B" in name:
        config.n_embd = 4096
        config.n_head = 32
        config.intermediate_size = 12288

    return config


# ---------------------------------------------------------
# Split reasoning and advisory
# ---------------------------------------------------------

def split_thought_advisory(text):

    text_lower = text.lower()

    if "</think>" in text_lower:
        split_idx = text_lower.find("</think>") + len("</think>")

        thought = text[:split_idx]
        advisory = text[split_idx:]

        thought = thought.replace("<think>", "").replace("</think>", "")
        advisory = advisory.replace("<advisory>", "").replace("</advisory>", "")
        thought = re.sub(r"\s+", " ", thought).strip()

        advisory = re.sub(r"\s+", " ", advisory).strip()

        return thought, advisory

    return "", text.strip()


### improved splitting - test after above - as this covers all failure cases.
# def split_thought_advisory(text):

#     text = text.strip()

#     # Normalize casing
#     lower = text.lower()

#     think_start = lower.find("<think>")
#     think_end = lower.find("</think>")

#     thought = ""
#     advisory = ""

#     if think_start != -1 and think_end != -1 and think_end > think_start:

#         thought = text[think_start + len("<think>"):think_end]

#         advisory = text[think_end + len("</think>"):]

#     elif think_end != -1:
#         # thought exists but start tag missing
#         thought = text[:think_end]
#         advisory = text[think_end + len("</think>"):]

#     else:
#         # model failed to produce think block
#         advisory = text

#     # Clean leftover tags
#     advisory = advisory.replace("<advisory>", "").replace("</advisory>", "")

#     # Normalize whitespace
#     thought = re.sub(r"\s+", " ", thought).strip()
#     advisory = re.sub(r"\s+", " ", advisory).strip()

#     return thought, advisory

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


# =========================================================
# GENERATION FUNCTION
# =========================================================

def generate(model, idx, max_new_tokens, eos_id, tokenizer):
    temperature = 0.65
    top_k = 40
    top_p = 0.9
    repetition_penalty = 1.1

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

        logits = logits / temperature
        
        # repetition penalty
        unique_tokens = torch.unique(idx[0])
        logits[0, unique_tokens] /= repetition_penalty

        # penalty for n-gram repetition blocking
        if len(idx[0]) > 3:
            last_trigram = tuple(idx[0][-3:].tolist())

            for j in range(len(idx[0]) - 3):
                if tuple(idx[0][j:j+3].tolist()) == last_trigram:
                    logits *= 0.9

        # Top-K
        values, _ = torch.topk(logits, top_k)
        min_val = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_val, torch.full_like(logits, -1e10), logits)

        # Top-P
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        cutoff = cumulative_probs > top_p
        cutoff[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, -1e10)

        logits = torch.zeros_like(sorted_logits).scatter(-1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)

        idx_next = torch.multinomial(probs, num_samples=1)

        token = idx_next.item()

        # stop if EOS
        if token == eos_id:
            break

        # decode partial output occasionally to check stop tokens
        if i % 10 == 0:
            if len(generated) > 0:
                partial = tokenizer.decode(torch.cat(generated)[0])
            else:
                partial = ""
            if "</think>" in partial and "धन्यवाद" in partial:
                break


        idx = torch.cat((idx, idx_next), dim=1)
        generated.append(idx_next)

        input_pos = torch.tensor([T + i], device=DEVICE)
        logits = model(idx_next, input_pos=input_pos)[:, -1, :]

    model.clear_kv_cache()

    if len(generated) == 0:
        return torch.tensor([[]], device=DEVICE)

    return torch.cat(generated, dim=1)


# =========================================================
# EVALUATION FUNCTIONS
# =========================================================

# ---------------------------------------------------------
# Advisory Quality Metrics
# ---------------------------------------------------------

def compute_advisory_metrics(preds, refs):

    bleu_refs = [[r] for r in refs]

    b = bleu.compute(predictions=preds, references=bleu_refs)
    r = rouge.compute(predictions=preds, references=refs)

    bert = bertscore.compute(
        predictions=preds,
        references=refs,
        model_type="xlm-roberta-large",
        device=DEVICE
    )

    bert_f1 = sum(bert["f1"]) / len(bert["f1"])

    f1_scores = [token_f1(p, r) for p, r in zip(preds, refs)]

    # bleurt_score = compute_bleurt(preds, refs)
    mover_score = compute_moverscore(preds, refs)
    comet_score = compute_comet(preds, refs)
    

    return {
        "BLEU": b["bleu"],
        "ROUGE-L": r["rougeL"],
        "BERTScore": bert_f1,
        "TokenF1": sum(f1_scores) / len(f1_scores),
        "BLEURT_MODEL": bleurt_score,
        "Mover Score": mover_score,
        "Comet Score": comet_score,
    }


# ---------------------------------------------------------
# Reasoning Quality Evaluation
# ---------------------------------------------------------

def compute_reasoning_metrics(pred_thoughts, ref_thoughts):

    filtered_preds = []
    filtered_refs = []

    for p, r in zip(pred_thoughts, ref_thoughts):
        if p.strip() == "" or r.strip() == "":
            continue
        filtered_preds.append(p)
        filtered_refs.append(r)

    if len(filtered_preds) == 0:
        return {
            "Thought_BERTScore": 0,
            "Thought_Length_Ratio": 0
        }

    bert = bertscore.compute(
        predictions=filtered_preds,
        references=filtered_refs,
        model_type="xlm-roberta-large",
        device=DEVICE
    )

    bert_f1 = sum(bert["f1"]) / len(bert["f1"])

    length_ratios = [len(p)/len(r) for p, r in zip(filtered_preds, filtered_refs)]

    return {
        "Thought_BERTScore": bert_f1,
        "Thought_Length_Ratio": sum(length_ratios)/len(length_ratios)
    }


# ---------------------------------------------------------
# Expert Specialization (MoE)
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
# Quantitative Dataset Analysis
# ---------------------------------------------------------

def dataset_statistics(df):

    advisory_lengths = df["advisory"].apply(len)

    stats = {
        "avg_advisory_length": advisory_lengths.mean(),
        "min_advisory_length": advisory_lengths.min(),
        "max_advisory_length": advisory_lengths.max()
    }

    return stats


# ---------------------------------------------------------
# LLM-as-a-Judge
# ---------------------------------------------------------

def llm_judge_score(question, prediction, reference):

    prompt = f"""
You are evaluating an agricultural advisory system.

Question:
{question}

Reference Answer:
{reference}

Model Answer:
{prediction}

Score the model answer from 1 to 5 based on:

1. Correctness of information
2. Relevance to the question
3. Practical usefulness for farmers

Return ONLY the score.
"""

    response = judge_model.generate_content(prompt)

    try:
        score = float(re.findall(r"\d", response.text)[0])
    except:
        score = 3.0

    return score


def clean_text(text):
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



def compute_bleurt(preds, refs):
    scores = bleurt_scorer.score(
        references=refs,
        candidates=preds
    )
    return sum(scores) / len(scores)


def compute_moverscore(preds, refs):
    scores = word_mover_score(
        refs,
        preds,
        idf_dict_ref=None,
        idf_dict_hyp=None,
        stop_words=[]
    )
    return sum(scores) / len(scores)


def compute_comet(preds, refs):
    data = []
    for p, r in zip(preds, refs):

        data.append({
            "src": "",
            "mt": p,
            "ref": r
        })
    scores = comet_model.predict(
        data,
        batch_size=8,
        gpus=1 if torch.cuda.is_available() else 0
    )
    return sum(scores["scores"]) / len(scores["scores"])


def correctness_score(question, prediction, reference):

    prompt = f"""
Evaluate the correctness of this agricultural advisory.

Question:
{question}

Reference Answer:
{reference}

Model Answer:
{prediction}

Score:

1 = incorrect
0.5 = partially correct
0 = incorrect

Return only the score.
"""

    response = judge_model.generate_content(prompt)

    try:
        score = float(re.findall(r"0\\.5|1|0", response.text)[0])
    except:
        score = 0.5

    return score

# =========================================================
# MAIN EVALUATION PIPELINE
# =========================================================

def run_evaluation():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = Tokenizer(TOKENIZER_DIR)
    eos_id = tokenizer.eos_id

    df = pd.read_parquet(TEST_DATA)
    df = df.head(1)

    predictions_log = []
    metrics_log = []

    dataset_stats = dataset_statistics(df)

    print("Dataset statistics:", dataset_stats)

    for model_name, (ckpt, config_name) in MODELS.items():

        print("\nEvaluating:", model_name)

        config = get_config(config_name)

        model = GPT(config).to(DEVICE, dtype=torch.bfloat16)

        weights = torch.load(ckpt, map_location=DEVICE)

        if "model_state_dict" in weights:
            weights = weights["model_state_dict"]

        model.load_state_dict(weights, strict=False)

        model.eval()

        preds = []
        refs = []
        judge_scores = []
        correctness_scores = []

        pred_thoughts = []
        ref_thoughts = []

        for _, row in tqdm(df.iterrows(), total=len(df)):

            prompt = (
                "<|im_start|>system\n"
                + row["system_instruction"]
                + "\n\nOutput format strictly:\n"
                "<think>\nReasoning here\n</think>\n"
                "<advisory>\nFinal advisory here\n</advisory>\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                + row["prompt"]
                + "\n<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

            input_ids = tokenizer.encode(prompt, bos=True, eos=False).to(DEVICE).unsqueeze(0)

            with torch.no_grad():
                output_ids = generate(model, input_ids, MAX_NEW_TOKENS, eos_id, tokenizer)


            output = tokenizer.decode(output_ids[0])
            output = output.replace("<|im_end|>", "")
            output = output.replace("<|im_start|>", "")
            output = output.strip()
            
            print(output)

            thought, advisory = split_thought_advisory(output)

            reference = row["advisory"]

            preds.append(advisory)
            refs.append(reference)

            pred_thoughts.append(thought)
            ref_thoughts.append(row["thoughts"])

            thought_len = len(thought.split())
            advisory_len = len(advisory.split())


            judge = llm_judge_score(
                row["prompt"],
                advisory,
                reference
            )

            correctness = correctness_score(
                row["prompt"],
                advisory,
                reference
            )

            judge_scores.append(judge)
            correctness_scores.append(correctness)
        
            predictions_log.append({
                "id": row["custom_id"],
                "model": model_name,
                "thought": thought,
                "advisory": advisory,
                "raw_output": output,
                "thought_len": thought_len,
                "advisory_len": advisory_len,
                "judge_scores": judge_scores,
                "correctness_scores": correctness_scores,
            })

            print(f"thought length: {thought_len}, advisory length: {advisory_len}")
        preds = [clean_text(p) for p in preds]
        refs = [clean_text(r) for r in refs]

        advisory_metrics = compute_advisory_metrics(preds, refs)

        # reasoning_metrics = compute_reasoning_metrics(pred_thoughts, ref_thoughts)

        router_entropy = compute_router_entropy(model)
        

        metrics_log.append({
            "Model": model_name,
            **advisory_metrics,
            # **reasoning_metrics,
            "RouterEntropy": router_entropy,
            "LLM_Judge": sum(judge_scores)/len(judge_scores),
            "Correctness": sum(correctness_scores)/len(correctness_scores),
        })

        del model
        torch.cuda.empty_cache()


    # =====================================================
    # SAVE OUTPUTS
    # =====================================================

    pd.DataFrame(predictions_log).to_csv(
        os.path.join(OUTPUT_DIR, "predictions.csv"), index=False
    )

    results_df = pd.DataFrame(metrics_log)

    results_df.to_csv(
        os.path.join(OUTPUT_DIR, "final_metrics.csv"), index=False
    )

    sns.set(style="whitegrid")

    melted = results_df.melt(id_vars="Model")

    plt.figure(figsize=(12,6))

    sns.barplot(data=melted, x="variable", y="value", hue="Model")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "evaluation_plot.png")
    )

    print(f"Generation completed! Saved results at {OUTPUT_DIR}")
# =========================================================

if __name__ == "__main__":
    run_evaluation()