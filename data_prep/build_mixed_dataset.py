import os
import pandas as pd
from datasets import load_dataset
from sklearn.utils import shuffle
from tqdm import tqdm

# ==========================================================
# CONFIG
# ==========================================================

LOCAL_AGRI_HINDI_PATH = "data/train_agri_data_65k.parquet"

OUTPUT_PATH = "data/train_bilingual_mixed_83k_agri65k.parquet"

NUM_EN_AGRI = 8000
NUM_HI_CHAT = 5000
NUM_EN_CHAT = 5000

SEED = 42


# ==========================================================
# CHATML FORMATTER
# ==========================================================

def format_chatml(system, user, thought, answer):
    """
    Convert components into Qwen compatible ChatML format
    """

    return (
        f"<|im_start|>system\n"
        f"{system.strip()}\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{user.strip()}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<think>\n"
        f"{thought.strip()}\n"
        f"</think>\n\n"
        f"<advisory>\n\n"
        f"{answer.strip()}\n"
        f"</advisory>\n\n"
        f"<|im_end|>"
    )


# ==========================================================
# SIMPLE CLEANING
# ==========================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

    # collapse multiple spaces
    text = " ".join(text.split())

    return text.strip()


# ==========================================================
# VALIDATION
# ==========================================================

def validate_sample(sample):

    required = [
        "<|im_start|>system",
        "<|im_start|>user",
        "<|im_start|>assistant",
        "<|im_end|>",
        "<think>",
        "</think>"
    ]

    for tag in required:
        if tag not in sample:
            return False

    return True


# ==========================================================
# 1. CUSTOM AGRI DATA
# ==========================================================

def load_custom_agri():

    print("\nLoading custom agricultural Hindi dataset...")

    df = pd.read_parquet(LOCAL_AGRI_HINDI_PATH)

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        system = clean_text(row.get("system_instruction", "You are an agricultural advisor."))
        user = clean_text(row.get("prompt", ""))
        thought = clean_text(row.get("thoughts", ""))
        answer = clean_text(row.get("advisory", ""))

        sample = format_chatml(system, user, thought, answer)

        if validate_sample(sample):
            rows.append(sample)

    print(f"Added {len(rows)} agri Hindi samples")

    return rows


# ==========================================================
# 2. ENGLISH AGRI QA
# ==========================================================

def load_en_agri():

    print("\nLoading English agriculture QA dataset...")

    ds = load_dataset(
        "KisanVaani/agriculture-qa-english-only",
        split="train"
    )

    ds = ds.shuffle(seed=SEED).select(range(NUM_EN_AGRI))

    rows = []

    for row in tqdm(ds):

        question = clean_text(row["question"])
        answer = clean_text(row["answers"])

        thought = (
            "The farmer is asking an agricultural question. "
            "I should provide scientifically correct farming guidance."
        )

        system = "You are an agricultural advisor. Provide accurate farming advice."

        sample = format_chatml(system, question, thought, answer)

        if validate_sample(sample):
            rows.append(sample)

    print(f"Added {len(rows)} English agri samples")

    return rows


# ==========================================================
# 3. HINDI CHAT
# ==========================================================

def load_hi_chat():

    print("\nLoading Hindi conversational dataset...")

    ds = load_dataset(
        "FreedomIntelligence/alpaca-gpt4-hindi",
        split="train"
    )

    ds = ds.shuffle(seed=SEED).select(range(NUM_HI_CHAT))

    rows = []

    for row in tqdm(ds):

        if "instruction" in row:

            user = clean_text(row["instruction"])

            if row.get("input"):
                user += "\n" + clean_text(row["input"])

            answer = clean_text(row["output"])

        elif "conversations" in row:

            user = clean_text(row["conversations"][0]["value"])
            answer = clean_text(row["conversations"][1]["value"])

        else:
            continue

        thought = (
            "The user asked a question in Hindi. "
            "I should respond clearly and helpfully."
        )

        system = "You are a helpful assistant."

        sample = format_chatml(system, user, thought, answer)

        if validate_sample(sample):
            rows.append(sample)

    print(f"Added {len(rows)} Hindi chat samples")

    return rows


# ==========================================================
# 4. ENGLISH CHAT
# ==========================================================

def load_en_chat():

    print("\nLoading English Alpaca dataset...")

    ds = load_dataset(
        "tatsu-lab/alpaca",
        split="train"
    )

    ds = ds.shuffle(seed=SEED).select(range(NUM_EN_CHAT))

    rows = []

    for row in tqdm(ds):

        user = clean_text(row["instruction"])

        if row.get("input"):
            user += "\n" + clean_text(row["input"])

        answer = clean_text(row["output"])

        thought = (
            "The user asked a question. "
            "Provide a helpful and clear response."
        )

        system = "You are a helpful assistant."

        sample = format_chatml(system, user, thought, answer)

        if validate_sample(sample):
            rows.append(sample)

    print(f"Added {len(rows)} English chat samples")

    return rows


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("\nBuilding training dataset...\n")

    agri_hi = load_custom_agri()
    agri_en = load_en_agri()
    hi_chat = load_hi_chat()
    en_chat = load_en_chat()

    all_samples = agri_hi + agri_en + hi_chat + en_chat

    print(f"\nTotal samples before shuffle: {len(all_samples)}")

    df = pd.DataFrame({"text": all_samples})

    df = shuffle(df, random_state=SEED).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nSaved dataset to {OUTPUT_PATH}")
    print("Dataset creation complete.")


# ==========================================================

if __name__ == "__main__":
    main()