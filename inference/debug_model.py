import torch
from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer

# --- CONFIGURATION ---
CHECKPOINT_PATH = "checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/run_test_E103C2/step-500.pth"
# Use the folder where you copied the files
TOKENIZER_DIR = "checkpoints/Qwen/Qwen3-0.6B-moe-init" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def debug_loading():
    print(f"🕵️‍♂️ Debugging Checkpoint: {CHECKPOINT_PATH}")
    
    # 1. Load Config & Tokenizer
    config = Config.from_name("Qwen3-0.6B-MoE")
    tokenizer = Tokenizer(TOKENIZER_DIR)
    
    print(f"✅ Config Vocab Size: {config.vocab_size}")
    print(f"✅ Tokenizer Vocab Size: {tokenizer.vocab_size}")
    
    if config.vocab_size != tokenizer.vocab_size:
        print("⚠️ WARNING: Mismatch! This causes gibberish output.")
        # Fix for Qwen usually:
        # config.vocab_size = 151936 
        # config.padded_vocab_size = 151936
    
    # 2. Load Model STRICTLY
    print("\n🏋️‍♂️ Attempting STRICT Load...")
    model = GPT(config).to(DEVICE, dtype=torch.bfloat16)
    
    try:
        # Try to load with strict=True to catch missing keys
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        print("✅ Success! Weights match perfectly.")
    except RuntimeError as e:
        print("❌ FATAL ERROR: Weight Key Mismatch!")
        print("The model is not loading correctly. First 3 missing keys:")
        # Parse error message to show user what's missing
        str_e = str(e)
        if 'Missing key(s)' in str_e:
            print(str_e.split('Missing key(s) in state_dict:')[1].split('\n')[:3])
        return

    model.eval()
    
    # 3. Simple Forward Pass (No Generation Loop bugs)
    # Testing a simple English sentence to see if the brain works.
    test_prompt = "The capital of India is"
    input_ids = tokenizer.encode(test_prompt).to(DEVICE)
    
    print(f"\n🧠 Probing Model Brain...")
    print(f"Input: '{test_prompt}' (IDs: {input_ids.tolist()})")
    
    with torch.no_grad():
        logits = model(input_ids.unsqueeze(0))
        # Get the predicted next token
        next_token_logits = logits[0, -1, :]
        next_token_id = torch.argmax(next_token_logits).item()
        decoded_word = tokenizer.decode(torch.tensor([next_token_id]))
        
    print(f"Prediction ID: {next_token_id}")
    print(f"Prediction Word: '{decoded_word}'")
    
    if "New Delhi" in decoded_word or "Delhi" in decoded_word or "New" in decoded_word:
        print("✅ DIAGNOSIS: Model is HEALTHY. The issue was the inference loop/KV cache.")
    else:
        print("❌ DIAGNOSIS: Model is BROKEN (Output is random/wrong).")

if __name__ == "__main__":
    debug_loading()