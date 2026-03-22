import os
import torch
import logging
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from litgpt.model import GPT
from litgpt.config import Config

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def upcycle_checkpoint(
    dense_ckpt_path: Path,
    output_dir: Path,
    student_config_name: str = "Qwen3-0.6B-MoE"
):
    """
    Implements SOTA 'Sparse Upcycling' (Komatsuzaki et al., Mistral AI).
    Strategy: Identical Clones for Experts + Random Initialization for Router.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu") # on C PU to avoid OOM

    # Load Dense Weights
    logger.info(f"Loading dense weights from {dense_ckpt_path}")
    dense_state_dict = torch.load(dense_ckpt_path, map_location=device, weights_only=True)

    # Initialize Student Model with MoE Config
    logger.info(f"Initializing student model architecture: {student_config_name}")
    config = Config.from_name(student_config_name)
    student_model = GPT(config)
    student_state_dict = student_model.state_dict()

    new_state_dict = {}
    # Track which keys we've mapped for verification
    mapped_keys = set()

    # COPY WEIGHTS 
    logger.info("Applying Sparse Upcycling: Cloning Dense Weights to All Experts...")
    for key, weight in dense_state_dict.items():
        # Handle MLP to MoE mapping
        # Dense: transformer.h.0.mlp.fc_1.weight
        # Student: transformer.h.0.mlp.experts.0.fc_1.weight, ... .experts.7.fc_1.weight
        if ".mlp." in key:
            for i in range(config.n_expert):
                # Construct the new expert key
                expert_key = key.replace(".mlp.", f".mlp.experts.{i}.")
                
                # Copy weight 
                # jitter = torch.randn_like(weight) * 0.02
                new_state_dict[expert_key] = weight.clone() #+ jitter
                mapped_keys.add(expert_key)
        else:
            # Handle non-MLP layers (Attention, Norm, Embedding, Head)
            if key in student_state_dict:
                new_state_dict[key] = weight.clone()
                mapped_keys.add(key)

    # Initialize the Router (Gate) weights
    # Routers are new, so we initialize them randomly using model's default init
    for key in student_state_dict.keys():
        if ".gate." in key and key not in new_state_dict:
            logger.info(f"Initializing new router weights: {key}")
            # Router weights are usually kept small
            new_state_dict[key] = torch.randn_like(student_state_dict[key]) * 0.02
            mapped_keys.add(key)

    # Final Safety Check and Save
    missing_keys = set(student_state_dict.keys()) - mapped_keys
    if missing_keys:
        logger.warning(f"Weights missing for keys: {missing_keys}")
    
    save_path = output_dir / "lit_model.pth"
    torch.save(new_state_dict, save_path)
    logger.info(f"Successfully saved upcycled MoE model to {save_path}")



def verify_experts_and_plot(folder_path, config_name="Qwen3-0.6B-MoE"):
    device = torch.device("cpu")    
    ckpt_path = folder_path / "lit_model.pth"
    
    print(f"\nVerifying Upcycled Model: {ckpt_path}")
    if not ckpt_path.exists():
        print(f"Error: File not found at {ckpt_path}")
        return
    
    # 1. Load Config Dynamically
    config = Config.from_name(config_name)
    n_experts = config.n_expert
    n_layers = config.n_layer
    print(f"Model Config: {n_experts} Experts, {n_layers} Layers")

    # 2. Define Target Layers (Indices are 0-based)
    # Layer 2, Mid and last layer
    target_layers = [1, n_layers // 2, n_layers - 1]
    
    # Load State Dict once
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    
    for layer_idx in target_layers:
        print(f"\nAnalyzing Layer {layer_idx} (Index {layer_idx})...")
        
        expert_weights = []
        for i in range(n_experts):
            key = f"transformer.h.{layer_idx}.mlp.experts.{i}.fc_1.weight"
            if key not in state_dict:
                print(f"Key not found: {key}")
                continue
            expert_weights.append(state_dict[key].float())

        if len(expert_weights) != n_experts:
            print("Skipping plot due to missing weights.")
            continue

        # Compute Similarity Matrix
        sim_matrix = torch.zeros((n_experts, n_experts))
        for i in range(n_experts):
            for j in range(n_experts):
                sim = torch.nn.functional.cosine_similarity(
                    expert_weights[i].flatten().unsqueeze(0), 
                    expert_weights[j].flatten().unsqueeze(0)
                ).item()
                sim_matrix[i, j] = sim

        # Text Report (Row 0)
        ref_sims = sim_matrix[0]
        print(f"Layer {layer_idx} Similarity (Exp 0 vs Others):")
        all_clones = True
        for i in range(n_experts):
            sim = ref_sims[i].item()
            status = "PASS" if sim > 0.99999 else "FAIL"
            if sim < 0.99999: all_clones = False
            # Only print first few to keep logs clean
            if i < 3 or i > n_experts - 2: 
                print(f"  Exp 0 vs {i}: {sim:.6f} [{status}]")

        if all_clones:
            print(f"Layer {layer_idx}: All experts are identical clones.")
        else:
            print(f"Layer {layer_idx}: Experts differ (Check upcycling logic).")

        # Plot Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            sim_matrix.numpy(), 
            annot=True, 
            fmt=".4f", 
            cmap="YlGnBu",
            vmin=0.0,  # Force bottom of scale
            vmax=1.0,  # Force top of scale to strictly 1.0 
            xticklabels=[f"E{i}" for i in range(n_experts)],
            yticklabels=[f"E{i}" for i in range(n_experts)]
        )
        plt.title(f"Expert Similarity - Layer {layer_idx}\n(1.000 = Success)")
        
        plot_filename = f"heatmap_layer_{layer_idx}.png"
        plot_path = folder_path / plot_filename
        plt.savefig(plot_path)
        print(f"Saved heatmap to: {plot_path}")
        plt.close()
        

if __name__ == "__main__":
    # Example usage - Update paths as per your local setup
    DENSE_CKPT = Path("checkpoints/Qwen/Qwen3-0.6B/lit_model.pth")
    OUTPUT_DIR = Path("checkpoints/Qwen/Qwen3-0.6B-moe-init/")
    
    upcycle_checkpoint(DENSE_CKPT, OUTPUT_DIR)
    verify_experts_and_plot(OUTPUT_DIR, config_name="Qwen3-0.6B-MoE")








