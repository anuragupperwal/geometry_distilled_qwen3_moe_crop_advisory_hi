import pandas as pd
from litgpt.tokenizer import Tokenizer
from tqdm import tqdm


INPUT_PATH = "data/raw_train_final_merged_220221.parquet"
OUTPUT_PATH = "data/train_agri_filtered_141k.parquet"
TOKENIZER_DIR = "checkpoints/Qwen/Qwen3-0.6B"

MAX_SEQ_LENGTH = 7000
TARGET_ROW_COUNT = 141000  # Exactly how many rows to keep in the final file

def build_full_prompt(row):
    """
    Simulates the final token structure to get an accurate length count.
    """
    sys_msg = str(row.get('system_instruction', '')).strip()
    user_msg = str(row.get('prompt', '')).strip()
    thoughts = str(row.get('thoughts', '')).strip()
    advisory = str(row.get('advisory', '')).strip()
    
    return (
        f"<|system|>\n{sys_msg}\n"
        f"<|user|>\n{user_msg}\n"
        f"<|thought|>\n{thoughts}\n"
        f"<|assistant|>\n{advisory}\n"
        f"<|end|>"
    )

def count_and_extract_safe_rows(input_path, tokenizer_dir, max_len, min_len):
    """
    Function 1: Scans the dataset, formats columns into a text string, 
    tokenizes it, and returns a DataFrame of ONLY the rows under the length limit.
    """
    print(f"🧹 Loading raw dataset: {input_path}")
    df = pd.read_parquet(input_path)
    tokenizer = Tokenizer(tokenizer_dir)
    
    safe_rows = []
    
    print(f"Scanning for rows strictly under {max_len} tokens...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Filtering"):
        
        # 1. Build the text dynamically from your distinct columns
        full_text = build_full_prompt(row)
        
        # 2. Tokenize to check length
        tokens = tokenizer.encode(full_text, bos=False, eos=False)
        
        # 3. Keep the original row if it fits
        if len(tokens) >= min_len and len(tokens)<max_len:
            safe_rows.append(row)
            
    df_safe = pd.DataFrame(safe_rows)
    
    print("\n" + "="*50)
    print("SCAN COMPLETE")
    print("="*50)
    print(f"Original Rows:       {len(df):,}")
    print(f"Safe Rows Available: {len(df_safe):,}")
    
    return df_safe

def save_target_amount(df_safe, target_count, output_path):
    """
    Function 2: Takes the safe DataFrame and exports a new Parquet file 
    containing exactly the target amount of rows.
    """
    total_safe = len(df_safe)
    
    if total_safe < target_count:
        print(f"\nWARNING: You requested {target_count:,} rows, but only {total_safe:,} are safe.")
        print("Saving all available safe rows instead.")
        df_final = df_safe
    else:
        # .sample() picks random rows so you don't just grab the first X rows.
        # random_state=42 ensures you get the exact same random rows if you run it twice.
        df_final = df_safe.sample(n=target_count, random_state=42).reset_index(drop=True)
        
    print(f"\nSaving {len(df_final):,} rows to: {output_path}")
    df_final.to_parquet(output_path, index=False)
    print("Done!")

def main():
    # 1. Extract all rows that are under 4000 tokens
    df_safe = count_and_extract_safe_rows(INPUT_PATH, TOKENIZER_DIR, MAX_SEQ_LENGTH, 4000)
    
    # 2. Save exactly 12,000 of them (or whatever TARGET_ROW_COUNT is set to)
    # save_target_amount(df_safe, TARGET_ROW_COUNT, OUTPUT_PATH)
    OUTPUT_PATH_65k= "data/test/extracted_100_rows.parquet"
    save_target_amount(df_safe, 100, OUTPUT_PATH_65k)

if __name__ == "__main__":
    main()