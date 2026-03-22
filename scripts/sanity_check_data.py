import torch
import sys
from pathlib import Path

# Standardize pathing for the -m execution
from litgpt.tokenizer import Tokenizer
from litgpt.config import Config
from agri_data import AgriDataset
from torch.utils.data import DataLoader

def run_sanity_check():
    # 1. Setup - Use the directory, not just the file
    checkpoint_dir = Path("checkpoints/Qwen/Qwen3-0.6B-moe-init")
    
    # Load config to get the correct tokenizer settings
    config = Config.from_name("Qwen3-0.6B-MoE")
    
    print(f"Initializing Tokenizer from: {checkpoint_dir}")
    # Using the directory allows the tokenizer to find both tokenizer.json and any config files
    try:
        tokenizer = Tokenizer(checkpoint_dir)
    except NotImplementedError:
        # Fallback: manually point to the json if the dir-load fails in your version
        print("Manual fallback for Tokenizer...")
        tokenizer = Tokenizer(checkpoint_dir / "tokenizer.json")
    
    data_path = "data/agri_hi_train.parquet"
    
    # 2. Load Dataset
    print(f"Verifying data at: {data_path}")
    dataset = AgriDataset(data_path, tokenizer, max_seq_length=4096)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 3. Get and Decode
    batch = next(iter(loader))
    tokens = batch[0] 
    
    # Decode only the non-padding tokens for clarity
    eos_id = tokenizer.eos_id
    clean_tokens = tokens[tokens != eos_id]
    decoded_text = tokenizer.decode(clean_tokens)
    
    print("\n" + "="*50)
    print("DECODED SAMPLE (THINKING MODEL FORMAT):")
    print("="*50)
    print(decoded_text)
    print("="*50)
    print(f"Original Length: {len(tokens)} | Actual Content: {len(clean_tokens)}")
    print("="*50)

if __name__ == "__main__":
    run_sanity_check()