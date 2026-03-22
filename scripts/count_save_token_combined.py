import pandas as pd
import numpy as np
from litgpt.tokenizer import Tokenizer
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_PATH = "data/train_bilingual_mixed_83k_agri65k.parquet"
OUTPUT_PATH = "data/train_bilingual_mixed_filtered_exact.parquet"
TOKENIZER_DIR = "checkpoints/Qwen/Qwen3-0.6B"

MAX_SEQ_LENGTH = 4096
# TARGET_ROW_COUNT = 70000  

def count_and_extract_safe_rows(input_path, tokenizer_dir, max_len):
    """
    Function 1: Scans the 'text' column, tokenizes it, prints statistics, 
    and returns a DataFrame of ONLY the rows under the length limit.
    """
    print(f"🧹 Loading dataset: {input_path}")
    df = pd.read_parquet(input_path)
    tokenizer = Tokenizer(tokenizer_dir)
    
    safe_indices = []
    lengths = []
    
    print(f"📏 Scanning '{max_len}' token limit on the 'text' column...")
    
    # Iterate over the text column specifically, keeping track of the original index
    for idx, text in enumerate(tqdm(df['text'], desc="Tokenizing")):
        tokens = tokenizer.encode(text, bos=False, eos=False)
        tok_len = len(tokens)
        lengths.append(tok_len)
        
        # If it fits in the context window, record its index
        if tok_len <= max_len:
            safe_indices.append(idx)
            
    # Calculate stats for visibility
    lengths = np.array(lengths)
    
    # Extract only the safe rows using the indices we collected
    df_safe = df.iloc[safe_indices].copy()
    
    print("\n" + "="*50)
    print("📊 TOKEN LENGTH STATISTICS")
    print("="*50)
    print(f"Total Original Rows:  {len(lengths):,}")
    print(f"Average Length:       {lengths.mean():.0f} tokens")
    print(f"Max Length:           {lengths.max():.0f} tokens")
    print("-" * 50)
    print(f"Safe Rows Available:  {len(df_safe):,} (under {max_len} tokens)")
    print(f"Rows Dropped:         {len(df) - len(df_safe):,} (Too long)")
    print("="*50)
    
    return df_safe

def save_target_amount(df_safe, target_count, output_path):
    """
    Function 2: Takes the safe DataFrame and exports a new Parquet file 
    containing exactly the target amount of rows.
    """
    total_safe = len(df_safe)
    
    if total_safe < target_count:
        print(f"\n⚠️ WARNING: Requested {target_count:,} rows, but only {total_safe:,} are safe.")
        print("Saving all available safe rows instead.")
        df_final = df_safe
    else:
        # Randomly sample the exact amount needed so we don't skew the dataset
        df_final = df_safe.sample(n=target_count, random_state=42).reset_index(drop=True)
        
    print(f"\n💾 Saving {len(df_final):,} rows to: {output_path}")
    df_final.to_parquet(output_path, index=False)
    print("🎉 Done!")

def main():
    # 1. Analyze token lengths and extract the safe rows
    df_safe = count_and_extract_safe_rows(INPUT_PATH, TOKENIZER_DIR, MAX_SEQ_LENGTH)
    
    # 2. Extract exactly the target number and save
    save_target_amount(df_safe, len(df_safe), OUTPUT_PATH)

if __name__ == "__main__":
    main()