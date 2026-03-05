import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
import csv
import json
import uuid
import os

from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from evaluation.plot_specialization import plot_cka_heatmap
from agri_data import AgriDataset
from litgpt.tokenizer import Tokenizer 
from litgpt.model import GPT
from litgpt.config import Config
from litgpt.utils import chunked_cross_entropy

#config

RUN_TAG = "02_03_run_test_80k"  # <--- EDIT THIS PER RUN

DISTILL_CONFIG = {
    "alpha": 0.5,
    "lambda": 0.5,  
    "beta": 1.0, 
    "gamma": 1.5, 
    "T": 2.0 
}
MAX_SEQ_LENGTH = 4096
BATCH_SIZE = 2 
ACCUMULATE_GRAD_STEPS = 8  # 1 * 4 = Effective Batch Size 4 (Better stability)
NUM_EPOCHS = 1

# Generate a short 6-char unique ID (e.g., 'A9D2') to prevent overwrites
short_id = uuid.uuid4().hex[:6].upper()
run_name = f"{RUN_TAG}_{short_id}"

OUTPUT_ROOT = Path(f"outputs/{run_name}")
CHECKPOINT_ROOT = Path(f"checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/{run_name}")

# Create directories
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "audits").mkdir(exist_ok=True)

DATA_PATH = "data/train_bilingual_mixed_83k_agri65k.parquet"
TEACHER_CKPT = "checkpoints/Qwen/Qwen3-8B/lit_model.pth"
STUDENT_INIT = "checkpoints/Qwen/Qwen3-0.6B-moe-init/lit_model.pth" 

print(f"Starting Run: {run_name}")
print(f"Output Dir:   {OUTPUT_ROOT}")

# --- 3. SAVE FULL CONFIG (The "Black Box" Recorder) ---
# We save the numbers here, so we don't need them in the filename.
config_path = OUTPUT_ROOT / "experiment_config.json"
experiment_settings = {
    "run_name": run_name,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    **DISTILL_CONFIG,
    "max_seq_length": MAX_SEQ_LENGTH,
    "batch_size": BATCH_SIZE,
    "accum_steps": ACCUMULATE_GRAD_STEPS,
    "dataset": DATA_PATH
}
with open(config_path, 'w') as f:
    json.dump(experiment_settings, f, indent=4)
print(f"Parameters saved to: {config_path}")



class LossLogger:
    def __init__(self, log_dir):
        self.file_path = log_dir / "training_log.csv"
        self.headers = ["step", "total_loss", "kl_loss", "ce_loss", "cka_loss", "div_loss", "max_load"]
        with open(self.file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def log(self, step, total, kl, ce, cka, div, load):
        with open(self.file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, f"{total:.6f}", f"{kl:.6f}", f"{ce:.6f}", f"{cka:.6f}", f"{div:.6f}", f"{load:.4f}"])


#CKA Loss
class LinearCKALoss(nn.Module):
    """
    Minibatch Linear CKA for feature manifold alignment.
    Handles different hidden dimensions (1024 vs 5120) naturally.
    """
    def forward(self, x, y):
        # x (Student): [B*T, 1024], y (Teacher): [B*T, 5120]
        x = x.view(-1, x.size(-1)).to(torch.float32)
        y = y.view(-1, y.size(-1)).to(torch.float32)
        
        # Center the features
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        
        dot_product = torch.norm(torch.matmul(x.t(), y))**2
        norm_x = torch.norm(torch.matmul(x.t(), x))
        norm_y = torch.norm(torch.matmul(y.t(), y))
        
        cka_score = dot_product / (norm_x * norm_y + 1e-6)
        return 1 - cka_score



@torch.no_grad()
def audit_expert_specialization(student, cka_fn, batch, step):
    """
    Live Training Audit: Uses a forward hook to capture real contextual 
    hidden states from the current batch and calculates expert similarity.
    """
    student.eval()
    device = next(student.parameters()).device
    
    # 1. Prepare data from the current batch
    input_ids = batch[0] # batch is (input_ids, loss_mask)
    # Slice to a small context for speed/VRAM during audit
    if input_ids.size(1) > 256:
        input_ids = input_ids[:, :256]
    
    save_dir = OUTPUT_ROOT / "audits"
    matrix_path = save_dir / f"cka_matrix_step_{step}.pth"
    
    # 2. Setup the Hook to capture Layer 14 (Middle) input
    target_layer_idx = student.config.n_layer // 2
    target_block = student.transformer.h[target_layer_idx]
    
    captured_input = {}
    def hook_fn(module, args, output):
        # args[0] is the hidden state entering the MLP/Expert block
        captured_input['hidden_state'] = args[0].detach()

    handle = target_block.mlp.register_forward_hook(hook_fn)
    
    # 3. Run a mini-forward pass to trigger the hook
    try:
        student(input_ids)
    finally:
        handle.remove() # Always remove hooks to prevent memory leaks
    
    real_hidden_state = captured_input['hidden_state']
    layer = target_block.mlp
    
    if hasattr(layer, 'experts'):
        print(f"\nStep {step}: Expert Specialization Audit (Real-Data Hook)")
        
        # 4. Pass captured real states through each expert
        expert_outputs = [expert(real_hidden_state) for expert in layer.experts]
        
        n_exp = len(expert_outputs)
        cka_matrix = torch.zeros((n_exp, n_exp))
        latent_dim = real_hidden_state.size(-1)
        
        for i in range(n_exp):
            for j in range(n_exp):
                # We calculate similarity (1.0 = identical)
                # Note: If your cka_fn in train_distill returns (1 - score), 
                # make sure to adjust this to store (1 - result) for a similarity heatmap.
                sim = 1 - cka_fn(expert_outputs[i].view(-1, latent_dim), 
                                 expert_outputs[j].view(-1, latent_dim))
                cka_matrix[i, j] = sim
        
        # 5. Save and Plot
        torch.save(cka_matrix, matrix_path)
        
        # Call your existing plotting function (Ensure it handles Similarity)
        plot_cka_heatmap(matrix_path, save_path=save_dir / f"heatmap_step_{step}.png")
        
        avg_sim = cka_matrix.mean().item()
        print(f"Layer {target_layer_idx} Avg Expert Similarity: {avg_sim:.4f}")
    
    student.train()


def train_distill():
    start_time = datetime.now()
    device = torch.device("cuda") 
    logger = LossLogger(OUTPUT_ROOT)

    tokenizer = Tokenizer("checkpoints/Qwen/Qwen3-0.6B")
    dataset = AgriDataset(
            data_path=DATA_PATH, 
            tokenizer=tokenizer, 
            max_seq_length=MAX_SEQ_LENGTH 
        )

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    print("Loading 8B Teacher in BF16...")
    teacher = GPT(Config.from_name("Qwen3-8B")).to(device, dtype=torch.bfloat16) 
    teacher.load_state_dict(torch.load(TEACHER_CKPT, mmap=True, weights_only=True), strict=False)
    teacher.eval()
    
    for param in teacher.parameters():
        param.requires_grad = False

    print("Loading 0.6B-MoE Student in BF16...")
    student = GPT(Config.from_name("Qwen3-0.6B-MoE")).to(device, dtype=torch.bfloat16)
    student.load_state_dict(torch.load(STUDENT_INIT, map_location=device, weights_only=True))

    #adding guassian noise
    print("Applying (Gaussian noise) Symmetry Breaking to Experts...")
    with torch.no_grad():
        for name, param in student.named_parameters():
            if "mlp.experts" in name and "weight" in name:
                param.add_(torch.randn_like(param) * 1e-4)

    # Enable Gradient Checkpointing on Student to save VRAM, critical for the H100 80GB when running two models
    student.gradient_checkpointing_enable()

    # Slightly higher LR because we are using a Scheduler now
    # optimizer = bnb.optim.AdamW8bit(student.parameters(), lr=5e-5) # 8-bit AdamW saves ~2GB
    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5, eps=1e-8)


    # Calculate total steps for scheduler
    total_steps = len(data_loader) * NUM_EPOCHS
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    cka_fn = LinearCKALoss()

    # Layer Mapping (teacher to student)
    # mapping = {s_i: int(s_i * (32-1) / (28-1)) for s_i in range(28)}
    n_student_layers = student.config.n_layer
    n_teacher_layers = teacher.config.n_layer
    print(f"Mapping {n_student_layers} Student layers -> {n_teacher_layers} Teacher layers")
    mapping = {
        s_i: int(s_i * (n_teacher_layers - 1) / (n_student_layers - 1)) 
        for s_i in range(n_student_layers)
    }


    n_experts = student.config.n_expert    
    print(f"Training Started. Steps per Epoch: {len(data_loader)}")
    optimizer.zero_grad() 
    
    global_step = 0
    for epoch in range(NUM_EPOCHS):
        print(f"\n=== EPOCH {epoch+1}/{NUM_EPOCHS} ")

        for batch_idx, (input_ids, loss_mask) in enumerate(tqdm(data_loader, desc="Training")):
            input_ids = input_ids.to(device).long()
            loss_mask = loss_mask.to(device)

            # Teacher Forward (No Grads)
            with torch.no_grad():
                t_logits, t_features = teacher(input_ids, return_features=True)

            # Student Forward
            s_logits, s_features = student(input_ids, return_features=True)

            # LOSS A: Soft Logit Distillation (With Masking)
            # We calculate KL Divergence only on the unmasked tokens (Thoughts + Advisory)
            T = DISTILL_CONFIG["T"] # Distillation Temperature

            # Shapes: [B, T, Vocab]
            student_log_probs = F.log_softmax(s_logits / T, dim=-1)
            teacher_probs = F.softmax(t_logits / T, dim=-1)
            # Standard KLDiv reduces to a scalar, so we need to implement manual reduction for masking
            # kl_loss_pointwise shape: [B, T, Vocab] -> Sum over vocab -> [B, T]
            kl_loss_pointwise = F.kl_div(student_log_probs, teacher_probs, reduction="none", log_target=False).sum(dim=-1) 
            # Apply Mask
            # loss_mask is [B, T], 1 for learnable tokens, 0 for prompt
            active_loss = (kl_loss_pointwise * loss_mask).sum() / (loss_mask.sum() + 1e-6)
            loss_kl = active_loss * (T**2)


            #LOSS B: Cross-Entropy 
            # Shift targets: logits[0] predicts input_ids[1]
            shift_logits = s_logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = loss_mask[..., 1:].contiguous()

            # Calculate standard CE
            ce_loss_pointwise = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none").view(shift_labels.size())
            # Apply mask
            loss_ce = (ce_loss_pointwise * shift_mask).sum() / (shift_mask.sum() + 1e-6)


            # LOSS C: Feature CKA Distillation 
            loss_cka = sum(cka_fn(s_features[s], t_features[t]) for s, t in mapping.items()) / len(mapping)

            # LOSS D: Expert Diversity (The Research Key) 
            loss_diversity = 0
            moe_layers = [h.mlp for h in student.transformer.h if hasattr(h.mlp, 'experts')]
            moe_blocks = [(idx, h) for idx, h in enumerate(student.transformer.h) if hasattr(h.mlp, 'experts')]

            for i, (layer_idx, block) in enumerate(moe_blocks):
                if layer_idx == 0: continue 
                # We need the input to the MLP for this layer. 
                # Since s_features contains block outputs, let's use the 
                # features from the PREVIOUS layer as the input to this MLP.
                
                # s_features[i-1] is the input to layer i
                # We only need a small slice of tokens to save VRAM during diversity check
                latent_input = s_features[layer_idx - 1].detach()[:, :128, :]

                expert_outputs = [block.mlp.experts[exp_idx](latent_input) for exp_idx in range(n_experts)]

                layer_div = 0
                for i_exp in range(n_experts):
                    for j_exp in range(i_exp + 1, n_experts):
                        # Functional Diversity: Experts must produce different outputs
                        layer_div += (1 - cka_fn(expert_outputs[i_exp], expert_outputs[j_exp]))
                
                loss_diversity += (layer_div / (n_experts * (n_experts - 1) / 2))
                
                del expert_outputs #to del the tensors accumulated after slicing

            loss_diversity /= len(moe_layers)


            # Total Loss 
            total_loss = (DISTILL_CONFIG["alpha"] * loss_kl) + \
                        (DISTILL_CONFIG["lambda"] * loss_ce) + \
                        (DISTILL_CONFIG["beta"] * loss_cka) + \
                        (DISTILL_CONFIG["gamma"] * loss_diversity)        


            loss_scaled = total_loss / ACCUMULATE_GRAD_STEPS
            loss_scaled.backward()

            # Step Optimizer ONLY every 'N' steps

            is_accumulation_step = (batch_idx + 1) % ACCUMULATE_GRAD_STEPS == 0
            is_last_batch = (batch_idx + 1) == len(data_loader)
            # if (batch_idx + 1) % ACCUMULATE_GRAD_STEPS == 0:
            if is_accumulation_step or is_last_batch:
                # Gradient Clipping for MoE Stability
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                scheduler.step() # Update LR
                optimizer.zero_grad()

                global_step += 1

                #logging
                all_indices = torch.cat([layer.mlp.last_indices for layer in student.transformer.h if hasattr(layer.mlp, 'last_indices')], dim=0)
                max_load = (torch.bincount(all_indices.view(-1), minlength=n_experts).float() / all_indices.numel()).max().item()

                logger.log(global_step, total_loss.item(), loss_kl.item(), loss_ce.item(), loss_cka.item(), loss_diversity.item(), max_load)

                # Print status
                if global_step % 20 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Step {global_step} | Total_Loss: {total_loss:.4f} | Load: {max_load:.2%} | LR: {current_lr:.2e} ")

                #  THE AUDIT (Every 500 Steps) 
                if global_step % 200 == 0:
                    audit_expert_specialization(student, cka_fn, [input_ids], global_step)
                    # torch.save(student.state_dict(), CHECKPOINT_ROOT / f"step-{global_step}.pth")
                    checkpoint = {
                        'step': global_step,
                        'model_state_dict': student.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }
                    torch.save(checkpoint, CHECKPOINT_ROOT / f"step-{global_step}.pth")


    # Save the final student state separately
    final_path = CHECKPOINT_ROOT / "lit_model.pth"
    # torch.save(student.state_dict(), final_path)
    checkpoint = {
        'step': global_step,
        'model_state_dict': student.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, CHECKPOINT_ROOT / "lit_model.pth")
    audit_expert_specialization(student, cka_fn, [input_ids], global_step)


    # Copy configuration files to make the model portable
    from litgpt.utils import copy_config_files
    copy_config_files(Path("checkpoints/Qwen/Qwen3-0.6B"), CHECKPOINT_ROOT)
    print(f"Distilled model saved to: {CHECKPOINT_ROOT}")

    # --- 2. CAPTURE END TIME & LOG SUMMARY ---
    end_time = datetime.now()
    duration = end_time - start_time
    
    summary = {
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": duration.total_seconds(),
        "duration_formatted": str(duration)
    }
    
    with open(OUTPUT_ROOT / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"✅ Finished! Total Duration: {duration}")



if __name__ == "__main__":
    train_distill()






