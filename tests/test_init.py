import torch
import os
from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer

CHECKPOINT_PATH = "checkpoints/Qwen/Qwen3-0.6B/lit_model.pth" 
TOKENIZER_DIR = "checkpoints/Qwen/Qwen3-0.6B-moe-init" 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_sanity_check():
    print(f"🕵️‍♂️ Loading Upcycled Model from: {CHECKPOINT_PATH}")
    
    if not os.path.exists(CHECKPOINT_PATH):
        print("Error: Checkpoint not found.")
        return

    # 1. Load Tokenizer
    try:
        tokenizer = Tokenizer(TOKENIZER_DIR)
        print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"Tokenizer Error: {e}")
        return

    # 2. Load Model
    config = Config.from_name("Qwen3-0.6B-MoE")
    model = GPT(config).to(DEVICE, dtype=torch.bfloat16)
    
    # ⚠️ CRITICAL: Use strict=False for MoE because keys might be slightly different 
    # (e.g., _extra_state), but we check the important ones manually.
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    keys = model.load_state_dict(state_dict, strict=False)
    
    # Verify that we actually loaded weights, not just empty air
    if len(keys.missing_keys) > 0:
        print(f"Warning: {len(keys.missing_keys)} missing keys.")
        # If 'mlp.experts' are missing, that's a FAILURE.
        if any("experts" in k for k in keys.missing_keys):
            print("CRITICAL FAIL: Expert weights did not load! Check your key mapping.")
            return
    
    print("Weights loaded successfully.")
    model.eval()

    # 3. The "Brain Probe" Tests
    # We test 3 things: Basic English, Basic Hindi, and Domain Knowledge
    prompts = [
        "The capital of India is",
        "नमस्ते, आप कैसे",
        "Sugarcane requires a soil type of", 
        "What is the natianl animal of India?"
    ]
    
    print("\nBRAIN PROBE (Step 0) ")
    print("Goal: The model should complete these sentences coherently.\n")
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, bos=False, eos=False).to(DEVICE)
        
        with torch.no_grad():
            # Simple greedy generation (no fancy sampling) to test raw weights
            output_ids = model.generate(
                input_ids.unsqueeze(0), 
                max_new_tokens=100, 
                temperature=0.3, 
                top_k=1
            )
            
        generated_text = tokenizer.decode(output_ids[0])
        print(f"Input:  {prompt}")
        print(f"Output: {generated_text}")
        print("-" * 30)

if __name__ == "__main__":
    run_sanity_check()