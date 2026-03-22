import pandas as pd
import glob
import os
from sklearn.utils import shuffle

# --- CONFIGURATION ---
INPUT_FOLDER = "data/merge_all_these"
# Only merge files that end with '.parquet' to avoid leaking test data
FILE_PATTERN = "*.parquet" 
OUTPUT_FILE = "data/merge_all_these/train_final_merged_"

def merge_datasets():
    # 1. Find all training files
    search_path = os.path.join(INPUT_FOLDER, FILE_PATTERN)
    files = sorted(glob.glob(search_path))
    
    print(f"🔍 Looking for files in: {search_path}")
    print(f"   Found {len(files)} files to merge.")
    
    if not files:
        print("No files found! Check that your training files start with 'train_'.")
        return

    # 2. Load and Concatenate
    df_list = []
    total_rows = 0
    
    print("\n   Processing files:")
    for f in files:
        try:
            df = pd.read_parquet(f)
            df_list.append(df)
            total_rows += len(df)
            print(f"   -> Included: {os.path.basename(f)} ({len(df)} rows)")
        except Exception as e:
            print(f"   ⚠️ Error loading {f}: {e}")

    # 3. Combine
    if not df_list:
        print("No data loaded.")
        return
        
    full_df = pd.concat(df_list, ignore_index=True)
    
    # 4. GLOBAL SHUFFLE
    # This ensures the model doesn't memorize the order of the files
    full_df = shuffle(full_df, random_state=42).reset_index(drop=True)

    # 5. Save
    final_output_path = f"{OUTPUT_FILE}{len(full_df)}.parquet"
    full_df.to_parquet(final_output_path)
    
    print("-" * 40)
    print(f"MERGE SUCCESSFUL")
    print(f"Output File:   {final_output_path}")
    print(f"Total Rows:    {len(full_df)}")
    print(f"File Size:     {os.path.getsize(final_output_path) / (1024*1024):.2f} MB")
    print("-" * 40)

if __name__ == "__main__":
    merge_datasets()