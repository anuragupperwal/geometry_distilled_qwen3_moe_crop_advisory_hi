import pandas as pd
import matplotlib.pyplot as plt

def plot_training_results(csv_path="outputs/training_log.csv"):
    df = pd.read_csv(csv_path)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plotting primary losses
    ax1.plot(df['step'], df['kl_loss'], label='KL (Logit) Loss', color='blue', alpha=0.6)
    ax1.plot(df['step'], df['cka_feat_loss'], label='CKA (Feature) Loss', color='green', alpha=0.8)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss Magnitude')
    ax1.legend(loc='upper left')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # Secondary axis for Expert Load
    ax2 = ax1.twinx()
    ax2.plot(df['step'], df['max_expert_load'], label='Max Expert Load', color='red', linestyle=':')
    ax2.set_ylabel('Expert Utilization (%)')
    ax2.legend(loc='upper right')

    plt.title('Distillation Progress: Reasoning Alignment & Expert Balancing')
    plt.savefig("outputs/12_03_run_test_80k_5C0943/distillation_curves.png")
    print("Graph saved to outputs/12_03_run_test_80k_5C0943/distillation_curves.png")

if __name__ == "__main__":
    plot_training_results()