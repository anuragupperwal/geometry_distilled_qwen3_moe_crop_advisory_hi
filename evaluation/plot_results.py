    # Caption: "Figure 1(a): Training loss trajectory. The blue line (Total Loss) and orange dashed line (KL Loss) show the student successfully minimizing the divergence from the teacher's output distribution. A steep drop followed by stabilization indicates effective knowledge transfer."
    # Caption: "Figure 1(b): Feature-level alignment using CKA Loss. This metric measures the similarity between the Teacher's hidden states and the Student's hidden states. The downward trend confirms the student is learning to replicate the teacher's internal reasoning process, not just the final answer."
    # Caption: "Figure 1(c): Expert utilization over time. The red line tracks the load on the most active expert. Values hovering near the ideal 12.5% (dotted line) indicate that the router is effectively distributing tokens across all 8 experts, avoiding mode collapse where a single expert dominates."

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# --- CONFIGURATION ---
log_file = "outputs/20_03_run_test_80k_1FC9E6/training_log.csv"
output_dir = Path(log_file).parent
# output_dir = "outputs/plots/" # Folder to save individual plots

def smooth(scalars, weight=0.85):
    """Exponential Moving Average for cleaner lines."""
    if len(scalars) == 0: return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_training_metrics():
    if not Path(log_file).exists():
        print(f"❌ Error: Log file not found at {log_file}")
        return

    # 1. Load Data
    df = pd.read_csv(log_file)
    print(f"✅ Loaded {len(df)} steps.")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 2. Apply Smoothing
    cols_to_smooth = ['total_loss', 'kl_loss', 'ce_loss', 'cka_loss', 'div_loss', 'max_load']
    for col in cols_to_smooth:
        if col in df.columns:
            df[f'{col}_smooth'] = smooth(df[col])

    # 3. Setup Research Style
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)

    # PLOT 1: Optimization (Loss Convergence)
    plt.figure(figsize=(12, 8))
    
    if 'total_loss' in df.columns:
        sns.lineplot(data=df, x="step", y="total_loss_smooth", 
                     label="Total Loss", color="#1f77b4", linewidth=1)
    if 'kl_loss' in df.columns:
        sns.lineplot(data=df, x="step", y="kl_loss_smooth", 
                     label="Distillation (KL) Loss", color="#ff7f0e", linestyle="-", linewidth=1)
    if 'ce_loss' in df.columns:
        sns.lineplot(data=df, x="step", y="ce_loss_smooth", 
                     label="CE Loss", color="#127f0e", linestyle="-", linewidth=1)
    
    plt.title("Optimization Convergence", fontweight='bold', pad=15)
    plt.ylabel("Loss Value")
    plt.xlabel("Training Steps")
    plt.legend(frameon=True, loc='upper right')
    plt.grid(True, which='major', linestyle='--', alpha=0.6)
    
    save_path_1 = os.path.join(output_dir, "1_optimization_convergence.png")
    plt.savefig(save_path_1, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path_1}")

    # PLOT 2: Representation Alignment (CKA)
    plt.figure(figsize=(12, 8))
    
    if 'cka_loss' in df.columns:
        sns.lineplot(data=df, x="step", y="cka_loss_smooth", 
                     color="#2ca02c", linewidth=1, label="Feature Alignment (CKA)")
        
        plt.title("Internal Representation Alignment", fontweight='bold', pad=15)
        plt.ylabel("Feature Distance")
        plt.xlabel("Training Steps")
        plt.legend(frameon=True, loc='upper right')
        plt.grid(True, which='major', linestyle='--', alpha=0.6)
        
        save_path_2 = os.path.join(output_dir, "2_feature_alignment.png")
        plt.savefig(save_path_2, dpi=400, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {save_path_2}")
    else:
        print("⚠️ CKA Loss not found, skipping Plot 2.")

    # PLOT 3: Router Health (Expert Load)
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    if 'max_load' in df.columns:
        sns.lineplot(data=df, x="step", y="max_load_smooth", ax=ax1,
                     color="#d62728", linewidth=1, label="Max Expert Load")
        
        # Ideal Baseline
        ax1.axhline(y=0.125, color='black', linestyle=':', linewidth=1, label="Ideal Balance (12.5%)")
        
        ax1.set_ylabel("Load Factor (0.0 - 1.0)", color="#d62728", fontweight='bold')
        ax1.tick_params(axis='y', labelcolor="#d62728")
        ax1.set_xlabel("Training Steps")
        ax1.set_title("MoE Routing Dynamics", fontweight='bold', pad=15)
        
        # Secondary Axis for Diversity Loss
        if 'div_loss' in df.columns:
            ax2 = ax1.twinx()
            sns.lineplot(data=df, x="step", y="div_loss_smooth", ax=ax2,
                         color="gray", alpha=0.5, linestyle="-.", linewidth=1, label="Diversity Loss")
            ax2.set_ylabel("Auxiliary Loss", color="gray")
            ax2.tick_params(axis='y', labelcolor="gray")
            
            # Combine legends
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='center right', frameon=True)
        else:
            ax1.legend(loc='upper right', frameon=True)

        save_path_3 = os.path.join(output_dir, "3_routing_dynamics.png")
        plt.savefig(save_path_3, dpi=400, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {save_path_3}")

if __name__ == "__main__":
    plot_training_metrics()