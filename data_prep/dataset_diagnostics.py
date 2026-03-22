import pandas as pd
import random
import numpy as np
from tqdm import tqdm

from litgpt.tokenizer import Tokenizer


# ============================================
# CONFIG
# ============================================

DATA_PATH = "data/train_bilingual_mixed_83k_agri65k.parquet"
TOKENIZER_PATH = "checkpoints/Qwen/Qwen3-0.6B"

NUM_SAMPLES_TO_PRINT = 5


# ============================================
# LOAD DATASET
# ============================================

def load_dataset():

    print("\nLoading dataset...")

    df = pd.read_parquet(DATA_PATH)

    print(f"Total rows: {len(df)}")

    return df


# ============================================
# FORMAT VALIDATION
# ============================================

def check_chatml_format(df):

    print("\nChecking ChatML format...")

    required_tags = [
        "<|im_start|>system",
        "<|im_start|>user",
        "<|im_start|>assistant",
        "<|im_end|>",
        "<think>",
        "</think>"
    ]

    broken = 0

    for text in df["text"]:

        for tag in required_tags:

            if tag not in text:
                broken += 1
                break

    print(f"Broken rows: {broken}")
    print(f"Valid rows: {len(df) - broken}")


# ============================================
# TOKEN LENGTH ANALYSIS
# ============================================

def token_length_stats(df, tokenizer):

    print("\nComputing token statistics...")

    lengths = []

    for text in tqdm(df["text"]):

        ids = tokenizer.encode(text, bos=False, eos=False)

        lengths.append(len(ids))

    lengths = np.array(lengths)

    print("\nToken length statistics:")

    print("mean:", lengths.mean())
    print("median:", np.median(lengths))
    print("max:", lengths.max())
    print("min:", lengths.min())

    print("\nPercentiles:")

    for p in [50, 75, 90, 95, 99]:
        print(f"{p}% :", np.percentile(lengths, p))


# ============================================
# REASONING CHECK
# ============================================

def reasoning_stats(df):

    print("\nChecking reasoning tags...")

    with_think = 0

    for text in df["text"]:

        if "<think>" in text and "</think>" in text:
            with_think += 1

    print("Rows with reasoning:", with_think)
    print("Rows without reasoning:", len(df) - with_think)


# ============================================
# LANGUAGE ESTIMATION
# ============================================

def language_estimate(df):

    print("\nEstimating language distribution...")

    hindi = 0
    english = 0

    for text in df["text"]:

        if any("\u0900" <= c <= "\u097F" for c in text):
            hindi += 1
        else:
            english += 1

    print("Hindi samples:", hindi)
    print("English samples:", english)


# ============================================
# VISUAL SAMPLE INSPECTION
# ============================================

def inspect_samples(df):

    print("\nRandom dataset samples:\n")

    samples = random.sample(list(df["text"]), NUM_SAMPLES_TO_PRINT)

    for i, s in enumerate(samples):

        print("="*60)
        print(f"SAMPLE {i+1}")
        print("="*60)
        print(s)
        print("\n")


# ============================================
# MAIN
# ============================================

def main():

    tokenizer = Tokenizer(TOKENIZER_PATH)

    df = load_dataset()

    check_chatml_format(df)

    reasoning_stats(df)

    language_estimate(df)

    token_length_stats(df, tokenizer)

    inspect_samples(df)


if __name__ == "__main__":
    main()