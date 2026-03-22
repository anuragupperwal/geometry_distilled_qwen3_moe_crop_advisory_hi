import torch
import os
import sys
import torch.nn.functional as F
from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer

# --- CONFIGURATION ---
CHECKPOINT_PATH = "checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/16_02_run_test_39k_full_run/lit_model.pth"
TOKENIZER_DIR = "checkpoints/Qwen/Qwen3-0.6B-moe-init" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. SOTA GENERATION LOGIC (Now with Top-P) ---
def generate_stepwise(model, idx, max_new_tokens, temperature=1.0, top_k=50, top_p=0.9, stop_tokens=None, repetition_penalty=1.2):
    """
    Generates tokens using Nucleus Sampling (Top-P) and Repetition Penalty.
    """
    B, T = idx.shape
    model.set_kv_cache(batch_size=B, max_seq_length=T + max_new_tokens, device=DEVICE)
    
    input_pos = torch.arange(0, T, device=DEVICE)
    logits = model(idx, input_pos=input_pos)
    logits = logits[:, -1, :]
    
    generated = []
    print(f"   [Debug] Starting generation... Input shape: {idx.shape}")

    for i in range(max_new_tokens):
        # A. Repetition Penalty
        if repetition_penalty > 1.0 and idx.size(1) > 1:
            context_len = 200 # Look back window
            start_idx = max(0, idx.size(1) - context_len)
            current_context = idx[:, start_idx:].long()
            score = torch.gather(logits, 1, current_context)
            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
            logits.scatter_(1, current_context, score)

        # B. Temperature Scale
        logits = logits / temperature
        
        # C. Top-K Filter
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # D. Top-P (Nucleus) Filter [NEW SOTA LOGIC]
        if top_p is not None and top_p < 1.0:
            # Sort logits in descending order
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')

        # E. Sampling
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # F. Stop Check
        if stop_tokens is not None and idx_next.item() in stop_tokens:
            print(f"   [Debug] Stop token detected: {idx_next.item()}")
            break
        
        # Append
        idx = torch.cat((idx, idx_next), dim=1)
        generated.append(idx_next)

        input_pos = torch.tensor([T + i], device=DEVICE, dtype=torch.long)
        logits = model(idx_next, input_pos=input_pos)
        logits = logits[:, -1, :]

    if len(generated) > 0:
        return torch.cat(generated, dim=1)
    else:
        return torch.tensor([], device=DEVICE, dtype=torch.long).reshape(B, 0)

def build_agri_prompt(data):
    context_str = []
    if "Crop" in data: context_str.append(f"फसल: {data['Crop']}")
    if "Growth Stage" in data: context_str.append(f"चरण: {data['Growth Stage']}")
    if "Region" in data: context_str.append(f"क्षेत्र: {data['Region']}")
    if "Weather" in data: context_str.append(f"मौसम: {data['Weather']}")
    if "Soil Type" in data: context_str.append(f"मिट्टी: {data['Soil Type']}")
    if "Farming Practice" in data: context_str.append(f"खेती: {data['Farming Practice']}")
    
    scenario_text = " | ".join(context_str)
    stress = data.get("Stress", "Not specified")
    
    prompt = f"""<|system|>
You are an intelligent agricultural advisor for Indian farmers. Answer strictly in Hindi.
First, analyze the crop, weather, and symptoms in the <|thought|> section.
Then, provide a clear, actionable advisory in the <|assistant|> section.
<|user|>
स्थिति (Scenario):
{scenario_text}

समस्या (Problem):
{stress}
<|thought|>
"""
    return prompt

def run_inference():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"CRITICAL ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    print(f"Loading model from {CHECKPOINT_PATH}...")
    config = Config.from_name("Qwen3-0.6B-MoE")
    model = GPT(config).to(DEVICE, dtype=torch.bfloat16)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True), strict=False)
    model.eval()

    tokenizer = Tokenizer(TOKENIZER_DIR)

    # --- DEFINE INPUT ---
    input_data = {
        "Growth Stage": "फूल आना / पुष्पन",
        "Weather": "सामान्य / अनुकूल मौसम",
        "Soil Type": "हल्की मिट्टी / बलुई दोमट (Light/Sandy Loam Soil)",
        "Farming Practice": "जैविक / प्राकृतिक खेती (Organic/Natural Farming)",
        "Region": "गुजरात",
        "Language": "Hindi",
        "Crop": "मक्का",
        "Stress": "तुलासिता रोग (Downy Mildew)"
    }
    
    full_prompt = build_agri_prompt(input_data)
    input_ids = tokenizer.encode(full_prompt, bos=True, eos=False).to(DEVICE)
    if input_ids.ndim == 1: input_ids = input_ids.unsqueeze(0)

    # Token IDs
    assistant_tag_ids = tokenizer.encode("<|assistant|>", bos=False, eos=False).to(DEVICE)
    if assistant_tag_ids.ndim == 1: assistant_tag_ids = assistant_tag_ids.unsqueeze(0)
    
    assistant_stop_id = assistant_tag_ids[0, 0].item()
    eos_id = tokenizer.eos_id

    # ==========================================
    # PHASE 1: THINKING (Balanced Limit)
    # ==========================================
    print("\nPHASE 1: Generating Thought Analysis...")
    
    with torch.no_grad():
        thought_ids = generate_stepwise(
            model, 
            input_ids, 
            max_new_tokens=500, # Optimized: 350 allows reasoning but kills infinity loops
            temperature=0.4,    # Slight bump for creativity in analysis
            top_k=40,
            top_p=0.9,          # Nucleus Sampling enabled
            stop_tokens={assistant_stop_id, eos_id}, 
            repetition_penalty=1.2
        )
    
    thought_text = tokenizer.decode(thought_ids[0])
    thought_text = thought_text.replace("<|assistant|>", "").strip()
    
    print(f"\033[90m{thought_text}\033[0m") 

    # ==========================================
    # PHASE 2: ADVISORY
    # ==========================================
    print("\nPHASE 2: Generating Advisory...")

    clean_thought_ids = tokenizer.encode(thought_text, bos=False, eos=False).to(DEVICE)
    if clean_thought_ids.ndim == 1: clean_thought_ids = clean_thought_ids.unsqueeze(0)

    # Hard Reset: Inject <|assistant|> tag
    phase2_input = torch.cat([input_ids, clean_thought_ids, assistant_tag_ids], dim=1)

    print(f"   [Debug] Phase 2 Input Shape: {phase2_input.shape}")

    with torch.no_grad():
        advisory_ids = generate_stepwise(
            model, 
            phase2_input, 
            max_new_tokens=1024, 
            temperature=0.5,     # Stable advising
            top_k=50,
            top_p=0.9,           # Nucleus Sampling enabled
            stop_tokens={eos_id}, 
            repetition_penalty=1.1
        )

    advisory_text = tokenizer.decode(advisory_ids[0])
    
    print(f"\033[92m{advisory_text.strip()}\033[0m")
    print("="*40 + "\n")

if __name__ == "__main__":
    run_inference()