import torch
from litgpt.tokenizer import Tokenizer
from agri_data import AgriDataset

# --- CONFIG ---
DATA_PATH = "data/agri_hi_train.parquet"
TOKENIZER_DIR = "checkpoints/Qwen/Qwen3-0.6B-moe-initial"
MAX_SEQ_LENGTH = 1024  # Keep small for inspection

def color_print(text, color):
    """
    Helper to print colored text to terminal.
    Red = Ignored (Masked), Green = Learned (Unmasked)
    """
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "reset": "\033[0m"
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"

def inspect_dataset():
    print(f"🔎 Loading Tokenizer from {TOKENIZER_DIR}...")
    try:
        tokenizer = Tokenizer(TOKENIZER_DIR)
    except:
        print("⚠️  Could not load specific tokenizer. Falling back to default Qwen config if possible.")
        return

    print(f"📂 Loading Dataset from {DATA_PATH}...")
    dataset = AgriDataset(data_path=DATA_PATH, tokenizer=tokenizer, max_seq_length=MAX_SEQ_LENGTH)
    
    # Pick a random sample (e.g., index 5)
    sample_idx = 5
    print(f"\n--- Inspecting Sample Index: {sample_idx} ---\n")
    
    input_ids, mask = dataset[sample_idx]
    
    # Convert to list for iteration
    input_ids = input_ids.tolist()
    mask = mask.tolist()
    
    # Visual Reconstruction
    reconstructed_text = ""
    
    print("--- 🚦 MASK VISUALIZATION (Red=Ignored, Green=Learned) ---")
    
    for token_id, is_learned in zip(input_ids, mask):
        # Decode single token
        word = tokenizer.decode(torch.tensor([token_id]))
        
        # Stop if we hit padding
        if token_id == dataset.pad_id:
            break
            
        if is_learned == 1:
            # This is part of the loss calculation (Thought/Advisory)
            print(color_print(word, "green"), end="")
        else:
            # This is masked out (System/User)
            print(color_print(word, "red"), end="")
            
    print("\n\n" + "="*50)
    print("✅ CHECKLIST:")
    print("1. Is the System Prompt RED? (Should be YES)")
    print("2. Is the User Scenario RED? (Should be YES)")
    print("3. Does GREEN start EXACTLY at <|thought|>?")
    print("4. Is the final <|assistant|> answer GREEN?")
    print("="*50)

if __name__ == "__main__":
    inspect_dataset()