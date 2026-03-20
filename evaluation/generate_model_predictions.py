import torch
import pandas as pd
import os
import gc
import torch.nn.functional as F
from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SUBSET_PARQUET = "data/test/extracted_100_rows.parquet"
OUTPUT_FINAL_CSV = "results/14_03_run_test_80k_FA4FAC/my_test_data_100.csv"

TOKENIZER_DIR = "checkpoints/Qwen/Qwen3-0.6B"

MAX_CONTEXT = 4096

MODELS = {

    # "Base": {
    #     "config_name": "Qwen3-0.6B-MoE",
    #     "path": "checkpoints/Qwen/Qwen3-0.6B/lit_model.pth"
    # },

    "Distilled": {
        "config_name": "Qwen3-0.6B-MoE",
        "path": "checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/16_03_run_test_80k_5DEB07/lit_model.pth"
    },

    # "Teacher": {
    #     "config_name": "Qwen3-8B",
    #     "path": "checkpoints/Qwen/Qwen3-8B/lit_model.pth"
    # }
}

# ==========================================
# GENERATION FUNCTION
# ==========================================

def generate_continuous(
    model,
    idx,
    max_new_tokens,
    stop_tokens,
    temperature=0.6,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.05
):

    B, T = idx.shape

    model.set_kv_cache(
        batch_size=B,
        max_seq_length=T + max_new_tokens,
        device=DEVICE
    )

    input_pos = torch.arange(0, T, device=DEVICE)
    logits = model(idx, input_pos=input_pos)
    logits = logits[:, -1, :]

    generated = []

    for i in range(max_new_tokens):

        logits = logits / temperature

        # ---------- TOP K ----------
        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            min_val = values[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_val,
                torch.full_like(logits, -1e10),
                logits
            )

        # ---------- TOP P ----------
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)

        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        cutoff = cumulative_probs > top_p

        # keep at least 1 token
        cutoff[..., 0] = False

        sorted_logits = sorted_logits.masked_fill(cutoff, -1e10)

        logits = torch.zeros_like(sorted_logits).scatter(
            -1, sorted_indices, sorted_logits
        )

        probs = F.softmax(logits, dim=-1)

        # ---------- NAN SAFETY ----------
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        idx_next = torch.multinomial(probs, num_samples=1)

        if idx_next.item() in stop_tokens:
            break

        idx = torch.cat((idx, idx_next), dim=1)
        generated.append(idx_next)

        input_pos = torch.full((B,), T + i, device=DEVICE)

        logits = model(idx_next, input_pos=input_pos)
        logits = logits[:, -1, :]

    model.clear_kv_cache()

    if len(generated) == 0:
        return torch.tensor([[]], device=DEVICE)

    return torch.cat(generated, dim=1)


# ==========================================
# MAIN SCRIPT
# ==========================================

def main():

    print("🚀 Initializing Model Prediction Generator...")

    tokenizer = Tokenizer(TOKENIZER_DIR)
    eos_id = tokenizer.eos_id

    print(f"📂 Loading extracted rows from {INPUT_SUBSET_PARQUET}...")

    if not os.path.exists(INPUT_SUBSET_PARQUET):
        print("❌ File not found.")
        return

    df_eval = pd.read_parquet(INPUT_SUBSET_PARQUET)

    df_eval = df_eval.head(10)

    df_eval['Reference_Thought'] = df_eval['thoughts']
    df_eval['Reference_Advisory'] = df_eval['advisory']

    # store prompt
    df_eval['Full_Prompt'] = (
        "<|im_start|>system\n"
        + df_eval['system_instruction']
        + "<|im_end|>\n"
        + "<|im_start|>user\n"
        + df_eval['prompt']
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )

    # ==========================================
    # MODEL LOOP
    # ==========================================

    for model_key, config_data in MODELS.items():

        print("\n========================================")
        print(f"🧠 Loading Model: {model_key}")
        print("========================================")

        config = Config.from_name(config_data["config_name"])

        model = GPT(config).to(DEVICE, dtype=torch.bfloat16)

        try:

            # weights = torch.load(
            #     config_data["path"],
            #     map_location=DEVICE,
            #     weights_only=True
            # )
            
            weights = torch.load(config_data["path"], map_location=DEVICE)

            if "model_state_dict" in weights:
                weights = weights["model_state_dict"]

            missing, unexpected = model.load_state_dict(weights, strict=False)

            print("Missing:", len(missing))
            print("Unexpected:", len(unexpected))

        except Exception as e:

            print("⚠️ Model load failed:", e)

            df_eval[f"{model_key}_Thought"] = "LOAD_FAILED"
            df_eval[f"{model_key}_Advisory"] = "LOAD_FAILED"

            continue

        model.eval()

        model_thoughts = []
        model_advisories = []

        stop_tokens = [eos_id]

        try:
            stop_tokens.append(
                tokenizer.encode("<|im_end|>", bos=False, eos=False)[0]
            )
        except:
            pass

        print(f"Generating {len(df_eval)} predictions...")

        for _, row in tqdm(df_eval.iterrows(), total=len(df_eval)):

            # ==========================================
            # PROMPT
            # ==========================================

            clean_system = str(row['system_instruction']).split("Internal Processing")[0].strip()

            prompt_text = (
                f"<|im_start|>system\n{clean_system}\n"
                "Respond strictly in language specified.\n"
                "<|im_end|>\n"
                f"<|im_start|>user\n{row['prompt']}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

            input_ids = tokenizer.encode(prompt_text, bos=False, eos=False)

            # context protection
            if len(input_ids) > MAX_CONTEXT - 200:
                input_ids = input_ids[-(MAX_CONTEXT - 200):]

            input_ids = input_ids.to(DEVICE)

            prompt_len = input_ids.size(0)

            max_new_tokens = min(
                4072,
                MAX_CONTEXT - prompt_len - 5
            )

            with torch.no_grad():

                output_ids = generate_continuous(
                    model,
                    input_ids.unsqueeze(0),
                    max_new_tokens=max_new_tokens,
                    stop_tokens=stop_tokens
                )

            raw_output = tokenizer.decode(output_ids[0])

            # clean tokens
            raw_output = raw_output.replace("<|im_end|>", "")
            raw_output = raw_output.replace("<|im_start|>", "")
            raw_output = raw_output.replace("</think>", "")
            raw_output = raw_output.replace("<think>", "")

            advisory_text = raw_output.strip()

            thought_text = "NO_THOUGHT_GENERATED"

            model_thoughts.append(thought_text)
            model_advisories.append(advisory_text)

        df_eval[f"{model_key}_Thought"] = model_thoughts
        df_eval[f"{model_key}_Advisory"] = model_advisories

        print("🧹 Clearing VRAM...")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ==========================================
    # SAVE CSV
    # ==========================================

    columns = [
        'custom_id',
        'Token_Count',
        'Full_Prompt',
        'Reference_Thought',
        'Reference_Advisory',
        'Base_Thought',
        'Base_Advisory',
        'Distilled_Thought',
        'Distilled_Advisory',
        'Teacher_Thought',
        'Teacher_Advisory'
    ]

    columns = [c for c in columns if c in df_eval.columns]

    final_df = df_eval[columns]

    os.makedirs(os.path.dirname(OUTPUT_FINAL_CSV), exist_ok=True)

    final_df.to_csv(OUTPUT_FINAL_CSV, index=False)

    print(f"\n✅ Saved predictions → {OUTPUT_FINAL_CSV}")


if __name__ == "__main__":
    main()