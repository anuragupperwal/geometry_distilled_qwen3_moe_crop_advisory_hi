import torch
import os
import torch.nn.functional as F
from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer

# --- CONFIGURATION ---
CHECKPOINT_PATH = "checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/12_03_run_test_80k_5C0943/step-3200.pth" 
TOKENIZER_DIR = "checkpoints/Qwen/Qwen3-0.6B-moe-init" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_continuous(model, idx, max_new_tokens, eos_id, temperature=0.4, top_k=50, repetition_penalty=1.15):
    """
    Generates tokens in a single continuous pass, maintaining KV-cache memory perfectly.
    """
    B, T = idx.shape
    model.set_kv_cache(batch_size=B, max_seq_length=T + max_new_tokens, device=DEVICE)
    
    input_pos = torch.arange(0, T, device=DEVICE)
    logits = model(idx, input_pos=input_pos)
    logits = logits[:, -1, :]
    
    generated = []
    
    for i in range(max_new_tokens):
        # A. Repetition Penalty
        if repetition_penalty > 1.0:
            context_len = 150
            start_idx = max(0, idx.size(1) - context_len)
            current_context = idx[:, start_idx:].long()
            score = torch.gather(logits, 1, current_context)
            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
            logits.scatter_(1, current_context, score)

        # B. Sampling
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # C. Stop Check (Only stop on exact EOS token)
        if idx_next.item() == eos_id:
            break
        
        idx = torch.cat((idx, idx_next), dim=1)
        generated.append(idx_next)

        input_pos = torch.tensor([T + i], device=DEVICE, dtype=torch.long)
        logits = model(idx_next, input_pos=input_pos)
        logits = logits[:, -1, :]

    if len(generated) > 0:
        return torch.cat(generated, dim=1)
    else:
        return torch.tensor([[]], device=DEVICE)

def build_agri_prompt(data):
    """
    Exactly mirrors the training data format to trigger optimal latent space alignment.
    """
    # 1. Provide the exact same system instruction from your training data
    sys_msg = (
        "You are a professional Agri-Business Consultant working with the government. \n"
        "Your objective is to generate detailed weekly advisories that focus on maximizing yield, "
        "reducing input costs, and ensuring the economic well-being of the farmer.\n\n"
        "Immersion Protocol:\n"
        "You are speaking to a farmer in their mother tongue, matching the language found in the input. "
        "Ensure deep linguistic alignment. \n"
        "Use the respectful and persuasive tone common in local dialects. \n"
        "Your response should feel 'Desi' (indigenous/local) and authentic, avoiding stiff, formal, or Anglicized sentence constructions.\n"
        "Ensure the tone is welcoming to all farmers, avoiding gender-specific address.\n\n"
        "Safety Guardrail: Strictly adhere to the 'Farming Practice'. If Organic/Natural, DO NOT mention synthetic chemicals."
    )

    # 2. Format the user input exactly as it was formatted in the parquet file
    user_msg = (
        f"Growth Stage: {data.get('Growth Stage', '')}\n"
        f"Weather: {data.get('Weather', '')}\n"
        f"Soil Type: {data.get('Soil Type', '')}\n"
        f"Farming Practice: {data.get('Farming Practice', '')}\n"
        f"Region: {data.get('Region', '')}\n"
        f"Language: {data.get('Language', 'Hindi')}\n"
        f"Crop: {data.get('Crop', '')}\n"
        f"Stress: {data.get('Stress', '')}"
    ).strip()
    
    # 3. Assemble with strict tagging
    prompt = (
        f"<|system|>\n{sys_msg}\n"
        f"<|user|>\n{user_msg}\n"
        f"<|thought|>\n"
    )
    return prompt

def run_inference():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Error: Model not found at {CHECKPOINT_PATH}")
        return

    print(f"⚙️ Loading distilled MoE model...")
    config = Config.from_name("Qwen3-0.6B-MoE")
    model = GPT(config).to(DEVICE, dtype=torch.bfloat16)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True), strict=False)
    model.eval()

    print("🧠 Loading Tokenizer...")
    tokenizer = Tokenizer(TOKENIZER_DIR)
    eos_id = tokenizer.eos_id

    # Test Data
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
    input_ids = tokenizer.encode(full_prompt, bos=False, eos=False).to(DEVICE) # Ensure bos=False to match training

    print("\n" + "="*50)
    print("🚀 GENERATING ADVISORY (Single Pass)")
    print("="*50 + "\n")

    with torch.no_grad():
        # Notice temperature is 0.4. It's a good middle ground for logic + natural language.
        output_ids = generate_continuous(
            model, 
            input_ids.unsqueeze(0), 
            max_new_tokens=2500, 
            eos_id=eos_id,
            temperature=0.4,     
            repetition_penalty=1.15
        )

    # Decode the newly generated tokens
    raw_output = tokenizer.decode(output_ids[0])
    
    # Parse the output into Thought and Advisory for display
    if "<|assistant|>" in raw_output:
        parts = raw_output.split("<|assistant|>")
        thought_text = parts[0].strip()
        advisory_text = parts[1].strip() if len(parts) > 1 else ""
    else:
        # Fallback if the model failed to generate the transition tag
        thought_text = raw_output.strip()
        advisory_text = "⚠️ [Model did not transition to assistant phase]"

    # Print nicely formatted to the terminal
    print("\033[90m--- MODEL THOUGHTS ---\033[0m")
    print(f"\033[90m{thought_text}\033[0m\n")
    
    print("\033[92m--- FINAL ADVISORY ---\033[0m")
    print(f"\033[92m{advisory_text}\033[0m")
    print("\n" + "="*50)

if __name__ == "__main__":
    run_inference()