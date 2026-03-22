import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer
from agri_data import AgriDataset


CHECKPOINT_PATH = "checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/16_02_run_test_39k_A5206A/step-500.pth"
TOKENIZER_DIR = "checkpoints/Qwen/Qwen3-0.6B-moe-init"
DATA_PATH = "data/agri_hi_train.parquet" 
OUTPUT_DIR = Path("outputs/plots")
MAX_SEQ_LENGTH = 3072
BATCH_SIZE = 1
name="expert_similarity"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. CKA Similarity Metric ---
class LinearCKALoss(nn.Module):
    def forward(self, x, y):
        # x, y: [Batch * Seq, Hidden_Dim]
        x = x.view(-1, x.size(-1)).to(torch.float32)
        y = y.view(-1, y.size(-1)).to(torch.float32)
        
        # Center the features (Critical for CKA)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        
        # Calculate CKA
        dot_product = torch.norm(torch.matmul(x.t(), y))**2
        norm_x = torch.norm(torch.matmul(x.t(), x))
        norm_y = torch.norm(torch.matmul(y.t(), y))
        
        cka_score = dot_product / (norm_x * norm_y + 1e-6)
        return cka_score # Returns SIMILARITY (1.0 = Identical)

# --- 2. The Hook & Plot Logic ---
def generate_plot(student, cka_fn, input_ids):
    print(f"\n--- Generatig Heatmap ---")
    student.eval()
    
    # A. Hook Layer 14 (Middle Layer)
    target_layer_idx = student.config.n_layer // 2 
    target_block = student.transformer.h[target_layer_idx]
    
    captured_input = {}
    def hook_fn(module, args, output):
        captured_input['hidden_state'] = args[0].detach()

    handle = target_block.mlp.register_forward_hook(hook_fn)
    
    # B. Run Forward Pass
    with torch.no_grad():
        student(input_ids)
    handle.remove()
    
    real_hidden_state = captured_input['hidden_state'] # [B, T, Emb]

    # C. Calculate Similarity Matrix
    layer = target_block.mlp
    if hasattr(layer, 'experts'):
        expert_outputs = [expert(real_hidden_state) for expert in layer.experts]
        n_exp = len(expert_outputs)
        cka_matrix = torch.zeros((n_exp, n_exp))
        latent_dim = real_hidden_state.size(-1)
        
        for i in range(n_exp):
            for j in range(n_exp):
                sim = cka_fn(expert_outputs[i].view(-1, latent_dim), 
                             expert_outputs[j].view(-1, latent_dim))
                cka_matrix[i, j] = sim
        
        # D. Plotting (Publication Style)
        plt.figure(figsize=(10, 8), dpi=300) # High DPI for papers
        sns.set_theme(style="white")
        
        # Plot Heatmap
        # cmap="viridis" or "mako" looks professional. "coolwarm" is good for divergence.
        ax = sns.heatmap(
            cka_matrix.detach().cpu().numpy(),
            annot=True, 
            fmt=".2f", 
            cmap="viridis_r", # Reversed Viridis (Dark = Similar, Light = Different) - or just "viridis"
            vmin=0.5, vmax=1.0, # Focus the range to show subtle diffs
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .8, "label": "CKA Similarity"}
        )
        
        plt.title(f"Expert Specialization (Layer {target_layer_idx})\nLower = More Specialized", fontsize=14, pad=20)
        plt.xlabel("Expert ID", fontsize=12)
        plt.ylabel("Expert ID", fontsize=12)
        
        # Save as PNG and PDF (Vector format for LaTeX)
        save_path_png = OUTPUT_DIR / f"{name}.png"
        save_path_pdf = OUTPUT_DIR / f"{name}.pdf"
        
        plt.savefig(save_path_png, bbox_inches='tight')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Plots saved to:\n  - {save_path_png}\n  - {save_path_pdf}")
        print(f"📊 Avg Similarity: {cka_matrix.mean().item():.4f}")

# --- 3. Main ---
def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    tokenizer = Tokenizer(TOKENIZER_DIR)
    dataset = AgriDataset(DATA_PATH, tokenizer, max_seq_length=MAX_SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Get batch
    input_ids, _ = next(iter(dataloader))
    input_ids = input_ids.to(device)
    
    # Slice for speed if needed
    if input_ids.size(1) > 256:
        input_ids = input_ids[:, :256]

    # Load Model
    config = Config.from_name("Qwen3-0.6B-MoE")
    model = GPT(config).to(device, dtype=torch.bfloat16)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True), strict=False)
    
    cka = LinearCKALoss()
    generate_plot(model, cka, input_ids)

if __name__ == "__main__":
    run()