import pandas as pd
from litgpt.tokenizer import Tokenizer
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_PATH = "data/train_bilingual_mixed_80k_agri62k.parquet"
OUTPUT_PATH = "data/train_bilingual_mixed_filtered_4k.parquet"
TOKENIZER_DIR = "checkpoints/Qwen/Qwen3-0.6B"
NEW_MAX_SEQ_LENGTH = 4096

def main():
    print(f"🧹 Loading dataset for filtering: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    tokenizer = Tokenizer(TOKENIZER_DIR)
    
    safe_rows = []
    dropped_count = 0
    
    print(f"📏 Filtering rows strictly under {NEW_MAX_SEQ_LENGTH} tokens...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scanning"):
        tokens = tokenizer.encode(row['text'], bos=False, eos=False)
        
        # Only keep rows that fit entirely inside the new context window
        if len(tokens) <= NEW_MAX_SEQ_LENGTH:
            safe_rows.append(row)
        else:
            dropped_count += 1
            
    # Rebuild the dataframe
    df_safe = pd.DataFrame(safe_rows)
    
    print("\n" + "="*50)
    print("✅ FILTERING COMPLETE")
    print("="*50)
    print(f"Original Rows:    {len(df):,}")
    print(f"Rows Dropped:     {dropped_count:,} (Too long)")
    print(f"Safe Rows Kept:   {len(df_safe):,}")
    print(f"Retention Rate:   {(len(df_safe)/len(df))*100:.2f}%")
    
    print(f"\n💾 Saving safe dataset to: {OUTPUT_PATH}")
    df_safe.to_parquet(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()