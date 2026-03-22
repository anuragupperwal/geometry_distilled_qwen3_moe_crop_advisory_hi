import torch
from litgpt.config import Config

def audit_moe_checkpoint(checkpoint_path, config_name):
    # Load state dict
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = Config.from_name(config_name)
    
    total_params = 0
    active_params = 0
    non_mlp_params = 0
    
    print(f"\n--- MoE Architecture Audit: {config_name} ---")
    print(f"Target Experts: {config.n_expert} | Top-K: {config.n_expert_per_token}\n")

    # Group keys by layer to see distribution
    layer_info = {}

    for name, param in sd.items():
        param_count = param.numel()
        total_params += param_count
        
        # Identify if it's an MLP/Expert layer
        if ".mlp." in name:
            # Format: transformer.h.{idx}.mlp...
            layer_idx = name.split(".")[2]
            if layer_idx not in layer_info:
                layer_info[layer_idx] = {"experts": set(), "params_per_expert": 0, "gate_params": 0}
            
            if "experts" in name:
                # Get expert index: transformer.h.0.mlp.experts.0.fc_1.weight -> 0
                expert_idx = name.split(".")[5]
                layer_info[layer_idx]["experts"].add(expert_idx)
                
                # Parameters in a single expert (we count fc_1, fc_2, proj)
                # We only count Expert 0 for the "Active" calculation later
                if expert_idx == "0":
                    layer_info[layer_idx]["params_per_expert"] += param_count
            
            if "gate" in name:
                layer_info[layer_idx]["gate_params"] += param_count
                active_params += param_count # Gates are always active
        else:
            # Non-MLP (Attention, Embeddings, Norms)
            non_mlp_params += param_count
            active_params += param_count # Non-MLP layers are always active

    # Finalize Active Param calculation
    # Active = Non-MLP + (Params of 1 Expert * Top-K) + Gates
    for idx, info in sorted(layer_info.items(), key=lambda x: int(x[0])):
        expert_count = len(info["experts"])
        layer_active_mlp = (info["params_per_expert"] * config.n_expert_per_token)
        
        print(f"Layer {idx:2}: Experts: {expert_count} | Params/Expert: {info['params_per_expert']:,}")
        
    print("\n" + "="*40)
    print(f"Total Parameters (on disk):   {total_params/1e6:.2f}M")
    
    # Calculation for Active Parameters
    # We sum Non-MLP + (Per-layer MLP active experts)
    total_active = non_mlp_params + sum([(info["params_per_expert"] * config.n_expert_per_token) + info["gate_params"] for info in layer_info.values()])
    
    print(f"Active Parameters (per token): {total_active/1e6:.2f}M")
    print(f"Non-MLP Shared Parameters:    {non_mlp_params/1e6:.2f}M")
    print("="*40)

if __name__ == "__main__":
    audit_moe_checkpoint(
        "checkpoints/Qwen/Qwen3-0.6B-moe-initial/lit_model.pth", 
        "Qwen3-0.6B-MoE"
    )