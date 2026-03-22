import pandas as pd
import random

# --- CONFIGURATION ---
# Ensure this matches the exact name of the file you just generated
PARQUET_FILE = "data/train_bilingual_mixed_83k_agri65k.parquet" 


def inspect_dataset():
    print(f"🔍 Loading dataset: {PARQUET_FILE}...\n")
    
    try:
        df = pd.read_parquet(PARQUET_FILE)
    except Exception as e:
        print(f"❌ Failed to load file: {e}")
        return

    # 1. Basic Stats
    total_rows = len(df)
    print("="*50)
    print("📊 DATASET STATISTICS")
    print("="*50)
    print(f"Total Rows:       {total_rows:,}")
    print(f"Columns:          {df.columns.tolist()}")
    
    # 2. Structure Health Check
    # Count how many rows have the <|thought|> tag (Your Custom Agri Data)
    thought_count = df['text'].str.contains("<\|thought\|>").sum()
    general_count = total_rows - thought_count
    
    print(f"Agri MoE Rows:    {thought_count:,} (Has <|thought|> tag)")
    print(f"General Rows:     {general_count:,} (Direct <|assistant|> response)")
    
    if thought_count == 0:
        print("\n⚠️ WARNING: No <|thought|> tags found! Your MoE routing logic might be broken.")
    
    # 3. Visual Inspection (Print 3 random samples)
    print("\n" + "="*50)
    print("👀 RANDOM SAMPLES INSPECTION")
    print("="*50)
    
    # Pick 3 random indices
    sample_indices = random.sample(range(total_rows), 6)
    
    for i, idx in enumerate(sample_indices, 1):
        print(f"\n--- SAMPLE {i} (Row {idx}) ---")
        text = df.iloc[idx]['text']
        
        # Truncate if it's absurdly long just for terminal readability
        if len(text) > 1500:
            print(text[:1500] + "\n... [TRUNCATED] ...\n<|end|>")
        else:
            print(text)
            
    print("\n✅ Inspection Complete.")

if __name__ == "__main__":
    inspect_dataset()