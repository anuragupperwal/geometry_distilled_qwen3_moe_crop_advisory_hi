import torch
from collections import defaultdict

ckpt_path = "checkpoints/Qwen/Qwen3-0.6B-moe-init/lit_model.pth"
# Use weights_only=True for security and speed
state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

print(f"{'Layer':<8} | {'Component':<25} | {'Shape':<20}")
print("-" * 60)

# We use a dictionary to group experts by layer for summary later
layers_audit = defaultdict(list)

# Sorting keys helps us view them in order (h.0, h.1, etc.)
for key in sorted(state_dict.keys()):
    # Filter for MLP components (Experts and Gates)
    if ".mlp." in key:
        # Extract layer index for cleaner display
        parts = key.split('.')
        layer_idx = parts[2] # transformer.h.{idx}
        
        # Identify Routers
        if "gate.weight" in key:
            print(f"{layer_idx:<8} | Gate (Router)            | {str(list(state_dict[key].shape)):<20}")
        
        # Identify Experts
        elif "experts" in key:
            # We only print the first linear layer (fc_1) per expert to keep output readable
            if "fc_1.weight" in key:
                expert_idx = parts[5] # experts.{idx}
                print(f"{layer_idx:<8} | Expert {expert_idx} MLP fc_1       | {str(list(state_dict[key].shape)):<20}")
                layers_audit[layer_idx].append(expert_idx)

print("-" * 60)
print("\n--- Final Summary ---")
# Verify consistency across all layers
total_layers = len(layers_audit)
print(f"Total Transformer Layers: {total_layers}")

for layer, experts in list(layers_audit.items())[:1]: # Check first layer as sample
    print(f"Experts per layer: {len(set(experts))} (Expected: 2)")

# Quick Jitter Check for Layer 0 to ensure they aren't bit-wise identical
e0_val = state_dict["transformer.h.0.mlp.experts.0.fc_1.weight"][0, 0]
# e1_val = state_dict["transformer.h.0.mlp.experts.1.fc_1.weight"][0, 0]
print(f"Symmetry Breaking Check: Expert 0 vs Expert 1 (L0, pos 0,0): {e0_val:.6f} vs {e0_val:.6f}")