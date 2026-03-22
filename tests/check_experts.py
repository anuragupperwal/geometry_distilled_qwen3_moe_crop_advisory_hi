import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer

# --- CONFIGURATION ---
device = torch.device("cuda")
checkpoint_path = "checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/lit_model.pth"
tokenizer_path = "checkpoints/Qwen/Qwen3-0.6B-moe-initial"
LAYER_TO_PROBE = 12  # We analyze the middle layer (where reasoning often happens)

print("Loading Model...")
model = GPT(Config.from_name("Qwen3-0.6B-MoE")).to(device, dtype=torch.bfloat16)
model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True), strict=False)
model.eval()
tokenizer = Tokenizer(tokenizer_path)

def get_expert_usage(text, label):
    # 1. Correct Input Shaping (Add Batch Dim)
    input_ids = tokenizer.encode(text, bos=False, eos=False).to(device)
    input_ids = input_ids.unsqueeze(0)  # Shape: [1, T]
    
    # 2. Variable to store captured data
    captured_routing = {}

    # 3. Define the Hook Function
    def hook_fn(module, input, output):
        # 'output' of the gate layer is the raw logits [Batch*Seq, Num_Experts]
        # We find which experts were selected (Top-K)
        logits = output.detach()
        probs, indices = torch.topk(logits, k=2, dim=-1) # Assuming Top-2 routing
        captured_routing['indices'] = indices.flatten().cpu().numpy()

    # 4. Attach Hook to the Router (Gate)
    # Target: transformer.h[LAYER].mlp.gate
    target_mlp = model.transformer.h[LAYER_TO_PROBE].mlp
    
    if not hasattr(target_mlp, 'gate'):
        print(f"Layer {LAYER_TO_PROBE} is not an MoE layer (no gate found).")
        return pd.Series()

    handle = target_mlp.gate.register_forward_hook(hook_fn)

    # 5. Run Standard Forward Pass
    with torch.no_grad():
        model(input_ids)
    
    # 6. Cleanup
    handle.remove()
    
    # 7. Analyze
    if 'indices' in captured_routing:
        return pd.Series(captured_routing['indices']).value_counts().sort_index()
    return pd.Series()

# --- RUN EXPERIMENT ---

# 1. Test Logic/Thinking (English/Hinglish)
print("Probing Thinking Logic...")
thought_text = "The humidity is high which promotes fungal growth like Red Rot. Immediate drainage is required to lower soil moisture."
thought_usage = get_expert_usage(thought_text, "Thinking")

# 2. Test Advisory (Hindi)
print("Probing Hindi Advisory...")
advisory_text = "किसान भाइयों, अपने खेत से पानी तुरंत निकाल दें क्योंकि इससे लाल सड़न का खतरा बढ़ जाता है।"
advisory_usage = get_expert_usage(advisory_text, "Advisory")

# --- PLOTTING ---
print("Generating Plot...")
df = pd.DataFrame({"Thinking (Logic)": thought_usage, "Advisory (Hindi)": advisory_usage}).fillna(0)

# Normalize to see relative preference (Percentages)
df = df.div(df.sum(axis=0), axis=1) * 100

plt.figure(figsize=(12, 6))
df.plot(kind='bar', width=0.8, color=['#1f77b4', '#2ca02c'])
plt.title(f"Expert Specialization in Layer {LAYER_TO_PROBE}\n(Thinking vs Advisory)", fontsize=14)
plt.xlabel("Expert ID", fontsize=12)
plt.ylabel("Usage Frequency (%)", fontsize=12)
plt.legend(title="Context Type")
plt.grid(axis='y', linestyle='--', alpha=0.5)

output_file = "outputs/expert_specialization_chart.png"
plt.savefig(output_file, bbox_inches='tight', dpi=300)
print(f"Analysis saved to: {output_file}")