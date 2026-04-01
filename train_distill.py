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


# GPU optimisation
torch.set_float32_matmul_precision("high")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
#flash attention
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


#config

RUN_TAG = "01_04_run_test_80k"  # <--- EDIT THIS PER RUN

DISTILL_CONFIG = {
    "alpha": 0.8,   # KL
    "lambda": 0.5,  # CE
    "beta": 0.1,    # CKA
    "gamma": 0.1,   # load balance
    "delta": 0.08, #div 
    "T": 3,
}


MAX_SEQ_LENGTH = 4096
BATCH_SIZE = 2 
ACCUMULATE_GRAD_STEPS = 8  # 1 * 4 = Effective Batch Size 4 (Better stability)
NUM_EPOCHS = 3
WARMUP_STEPS = 1000

# Generate a short 6-char unique ID (e.g., 'A9D2') to prevent overwrites
short_id = uuid.uuid4().hex[:6].upper()
run_name = f"{RUN_TAG}_{short_id}"

OUTPUT_ROOT = Path(f"outputs/{run_name}")
CHECKPOINT_ROOT = Path(f"checkpoints/Qwen/Qwen3-1.7B-Agri-Distilled/{run_name}")

# Create directories
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "audits").mkdir(exist_ok=True)

DATA_PATH = "data/train_bilingual_mixed_83k_agri65k.parquet"
TEACHER_CKPT = "checkpoints/Qwen/Qwen3-8B/lit_model.pth"
STUDENT_INIT = "checkpoints/Qwen/Qwen3-1.7B-moe-init/lit_model.pth" 

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
        self.headers = ["step", "total_loss", "kl_loss", "ce_loss", "cka_loss", "div_loss", "router_loss", "max_load"]
        with open(self.file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def log(self, step, total, kl, ce, cka, div, router, load):
        with open(self.file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, f"{total:.6f}", f"{kl:.6f}", f"{ce:.6f}", f"{cka:.6f}", f"{div:.6f}", f"{router:.6f}", f"{load:.4f}"])


#CKA Loss
class LinearCKALoss(nn.Module):
    """
    Stable minibatch Linear CKA for representation geometry alignment.
    Works with different hidden sizes. 
    """
    def forward(self,x,y):

        B,T,D1 = x.shape
        _,_,D2 = y.shape

        x = x.reshape(-1,D1).float()
        y = y.reshape(-1,D2).float()

        x = x - x.mean(0,keepdim=True)
        y = y - y.mean(0,keepdim=True)

        hsic = torch.norm(x.T @ y,p="fro")**2
        norm_x = torch.norm(x.T @ x,p="fro")
        norm_y = torch.norm(y.T @ y,p="fro")

        cka = hsic/(norm_x*norm_y + 1e-8)

        return 1-cka


# ------------------------------------------------------------
# TOKEN SAMPLING FOR MINIBATCH CKA
# ------------------------------------------------------------
def sample_tokens(x,max_tokens=3072):

    B,T,D = x.shape
    total = B*T

    if total <= max_tokens:
        return x

    idx = torch.randperm(total,device=x.device)[:max_tokens]

    x = x.reshape(total,D)[idx]

    return x.view(1,-1,D)



def paired_token_sample(x, y, max_tokens=1024):

    B,T,Dx = x.shape
    _,_,Dy = y.shape

    x = x.reshape(-1, Dx)
    y = y.reshape(-1, Dy)

    total = x.shape[0]

    if total <= max_tokens:
        return x.view(1,-1,Dx), y.view(1,-1,Dy)

    idx = torch.randperm(total, device=x.device)[:max_tokens]

    x = x[idx]
    y = y[idx]

    return x.unsqueeze(0), y.unsqueeze(0)

# ------------------------------------------------------------
# SWITCH TRANSFORMER LOAD BALANCE LOSS
# ------------------------------------------------------------
def compute_load_balance_loss(student,n_experts):

    losses=[]

    for layer in student.transformer.h:

        if not hasattr(layer.mlp,"last_router_probs"):
            continue

        probs=layer.mlp.last_router_probs
        indices=layer.mlp.last_indices

        if probs is None:
            continue

        tokens=probs.shape[0]

        importance = probs.mean(0) + 1e-6
        importance = importance / importance.sum()

        load=torch.bincount(
            indices.view(-1),
            minlength=n_experts
        ).float()/tokens
        load = load / (load.sum() + 1e-6)
        
        balance = ((importance - load) ** 2).sum()
        losses.append(balance)

    if len(losses)==0:
        return torch.tensor(0.0,device=probs.device)

    return torch.stack(losses).mean()


# ------------------------------------------------------------
# INTERPRETABILITY: TEACHER vs EXPERT CKA
# ------------------------------------------------------------

@torch.no_grad()
def audit_teacher_expert_alignment(student,teacher,cka_fn,batch,step, epoch):
    '''
    How similar each expert’s representation is to the teacher layer representation.
    0 → completely different
    1 → identical representation 

    '''
    student.eval()
    teacher.eval()

    device = next(teacher.parameters()).device
    input_ids = batch[0].to(device)
    
    t_logits,t_features=teacher(input_ids,return_features=True)
    s_logits,s_features=student(input_ids,return_features=True)

    target_layer=len(s_features)//2
    teacher_feat=t_features[target_layer]

    block=student.transformer.h[target_layer]

    if not hasattr(block.mlp,"experts"):
        return

    experts=block.mlp.experts
    hidden = block.mlp.last_hidden_input
    hidden = hidden.contiguous().view(1, -1, hidden.size(-1))

    matrix=torch.zeros(len(experts), device=hidden.device)

    teacher_feat = teacher_feat.contiguous().view(1,-1,teacher_feat.size(-1))

    for i,expert in enumerate(experts):

        out = expert(hidden.view(-1, hidden.size(-1)))
        out = out.view(1,-1,hidden.size(-1))

        out_sample, teacher_sample = paired_token_sample(out, teacher_feat)

        sim = 1 - cka_fn(out_sample, teacher_sample)

        matrix[i]=sim

    save_dir = OUTPUT_ROOT / "audits"
    save_path=save_dir/f"teacher_expert_alignment_epoch_{epoch}_step_{step}.pt"
    torch.save(matrix,save_path)
    
    plot_cka_heatmap(
        save_path,
        save_path.with_suffix(".png"),
        title=f"Teacher-Expert Alignment Epoch {epoch} Step {step}"
    )


    print(f"Teacher alignment avg: {matrix.mean().item():.4f}")



@torch.no_grad()
def audit_expert_specialization(student, cka_fn, batch, step, epoch):
    """
    Live Training Audit: Uses a forward hook to capture real contextual 
    hidden states from the current batch and calculates expert similarity.
    """
    student.eval()
    device = next(student.parameters()).device
    
    # 1. Prepare data from the current batch
    input_ids = batch[0].to(device) # batch is (input_ids, loss_mask)
    # Slice to a small context for speed/VRAM during audit
    if input_ids.size(1) > 512:
        input_ids = input_ids[:, :512]
    
    save_dir = OUTPUT_ROOT / "audits"
    matrix_path = save_dir / f"cka_matrix_epoch_{epoch}_step_{step}.pth"
    
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
        print(f"\nEpoch {epoch} Step {step}: Expert Specialization Audit (Real-Data Hook)")
        
        # 4. Pass captured real states through each expert
        expert_outputs = []
        for expert in layer.experts:
            expert_outputs.append(expert(real_hidden_state).detach())
        
        n_exp = len(expert_outputs)
        cka_matrix = torch.zeros((n_exp, n_exp), device=real_hidden_state.device)
        latent_dim = real_hidden_state.size(-1)
        
        for i in range(n_exp):
            for j in range(n_exp):
                # We calculate similarity (1.0 = identical)
                # Note: If your cka_fn in train_distill returns (1 - score), 
                # make sure to adjust this to store (1 - result) for a similarity heatmap.
                xi = expert_outputs[i].view(1, -1, latent_dim)
                xj = expert_outputs[j].view(1, -1, latent_dim)
                sim = 1 - cka_fn(xi, xj)
                cka_matrix[i, j] = sim
        
        # 5. Save and Plot
        torch.save(cka_matrix, matrix_path)

        valid_vals = cka_matrix[~torch.isnan(cka_matrix)]
        avg = valid_vals.mean().item()
        std = valid_vals.std().item()
        print(f"Expert similarity avg: {avg:.4f} ± {std:.4f}")

        plot_cka_heatmap(
            matrix_path,
            save_path=save_dir / f"heatmap_epoch_{epoch}_step_{step}.png",
            title=f"Expert CKA Similarity (Epoch {epoch}, Step {step})"
        )        
        avg_sim = cka_matrix.mean().item()
        print(f"Layer {target_layer_idx} Avg Expert Similarity: {avg_sim:.4f}")
    
    student.train()


@torch.no_grad()
def audit_expert_specialization_routerwise(student, cka_fn, batch, step, epoch):
    """
    Correct MoE Audit:
    - Uses REAL routed tokens
    - Compares ONLY active experts
    - Computes distribution-level CKA - experts see different tokens
    0.6 – 0.9 - experts redundant
    0.15 – 0.35 - healthy specialization
    <0.05 - over-diversification

    """

    student.eval()
    device = next(student.parameters()).device

    # --------------------------------------------------
    # 1. Prepare batch (small slice for speed)
    # --------------------------------------------------
    input_ids = batch[0].to(device)

    if input_ids.size(1) > 512:
        input_ids = input_ids[:, :512]

    save_dir = OUTPUT_ROOT / "audits"
    matrix_path = save_dir / f"cka_matrix_epoch_{epoch}_step_{step}.pth"

    # --------------------------------------------------
    # 2. Forward pass (to populate router states)
    # --------------------------------------------------
    student(input_ids)

    # --------------------------------------------------
    # 3. Pick middle layer
    # --------------------------------------------------
    target_layer_idx = student.config.n_layer // 2
    layer = student.transformer.h[target_layer_idx].mlp

    if not hasattr(layer, "experts"):
        student.train()
        return

    if not hasattr(layer, "last_indices") or layer.last_indices is None:
        student.train()
        return

    if not hasattr(layer, "last_hidden_input") or layer.last_hidden_input is None:
        student.train()
        return

    print(f"\nEpoch {epoch} Step {step}: Expert Specialization Audit (ROUTING-AWARE)")

    # --------------------------------------------------
    # 4. Get routing + hidden states
    # --------------------------------------------------
    indices = layer.last_indices            # [tokens]
    hidden = layer.last_hidden_input.view(-1, layer.last_hidden_input.size(-1))

    experts = layer.experts
    n_exp = len(experts)
    latent_dim = hidden.size(-1)

    # --------------------------------------------------
    # 5. Compute outputs ONLY on routed tokens
    # --------------------------------------------------
    expert_outputs = [None] * n_exp

    for i in range(n_exp):

        mask = (indices == i).any(dim=-1)

        if mask.sum() < 2:   # skip tiny samples (important for CKA stability)
            continue

        tokens_i = hidden[mask]  # only tokens routed to expert i

        out = experts[i](tokens_i)  # [Ni, D]

        expert_outputs[i] = out

    # --------------------------------------------------
    # 6. Compute CKA matrix (only valid pairs)
    # --------------------------------------------------
    cka_matrix = torch.full((n_exp, n_exp), float('nan'), device=device)

    for i in range(n_exp):
        for j in range(n_exp):

            if i == j:
                cka_matrix[i, j] = 1.0
                continue

            if expert_outputs[i] is None or expert_outputs[j] is None:
                continue

            Xi = expert_outputs[i]
            Xj = expert_outputs[j]

            # 🔥 MATCH TOKEN COUNT
            n = min(Xi.size(0), Xj.size(0))

            if n < 2:
                continue

            idx_i = torch.randperm(Xi.size(0), device=Xi.device)[:n]
            idx_j = torch.randperm(Xj.size(0), device=Xj.device)[:n]

            Xi = Xi[idx_i].unsqueeze(0)
            Xj = Xj[idx_j].unsqueeze(0)

            try:
                sim = 1 -cka_fn(Xi, Xj)
                cka_matrix[i, j] = sim
            except:
                # safety against numerical instability
                continue

    # --------------------------------------------------
    # 7. Save matrix
    # --------------------------------------------------
    torch.save(cka_matrix, matrix_path)

    # --------------------------------------------------
    # 8. Plot (handle NaNs)
    # --------------------------------------------------
    plot_cka_heatmap(
        matrix_path,
        save_path=save_dir / f"heatmap_routing_aware_epoch_{epoch}_step_{step}.png",
        title=f"Expert CKA Similarity (Routing-Aware) Epoch {epoch} Step {step}"
    )

    # --------------------------------------------------
    # 9. Stats
    # --------------------------------------------------
    valid_vals = cka_matrix[~torch.isnan(cka_matrix)]

    if valid_vals.numel() > 0:
        avg_sim = valid_vals.mean().item()
        print(f"Layer {target_layer_idx} Avg Expert Similarity: {avg_sim:.4f}")
    else:
        print("No valid expert pairs for similarity.")

    student.train()



def expert_diversity_cka(student):

    losses = []
    linear_cka = LinearCKALoss()
    for layer in student.transformer.h:

        if not hasattr(layer.mlp, "last_expert_outputs"):
            continue

        expert_outputs = layer.mlp.last_expert_outputs

        if expert_outputs is None:
            continue

        E = expert_outputs.shape[0]

        pairs = torch.randperm(E)[:4]

        for i in pairs:
            for j in pairs:
                if i >= j:
                    continue

                Xi = expert_outputs[i].unsqueeze(0)
                Xj = expert_outputs[j].unsqueeze(0)

                cka_val = linear_cka(Xi, Xj)

                losses.append(cka_val)

    if len(losses) == 0:
        return torch.tensor(0.0, device=student.lm_head.weight.device)

    return torch.stack(losses).mean()


def expert_diversity_cka_new(student):

    losses = []
    linear_cka = LinearCKALoss()

    for layer in student.transformer.h:

        if not hasattr(layer.mlp, "last_hidden_input"):
            continue
        if not hasattr(layer.mlp, "last_indices"):
            continue

        hidden = layer.mlp.last_hidden_input.view(-1, layer.mlp.last_hidden_input.size(-1))
        indices = layer.mlp.last_indices

        experts = layer.mlp.experts
        active_experts = torch.unique(indices)

        if len(active_experts) < 2:
            continue

        active_experts = active_experts.tolist()

        # 🔥 SAMPLE SMALL SUBSET OF PAIRS
        pairs = torch.randperm(len(active_experts))[:4]

        for i_idx in range(len(pairs)):
            for j_idx in range(i_idx + 1, len(pairs)):

                i = active_experts[pairs[i_idx]]
                j = active_experts[pairs[j_idx]]

                mask_i = (indices == i).any(dim=-1)
                mask_j = (indices == j).any(dim=-1)

                if mask_i.sum() < 2 or mask_j.sum() < 2:
                    continue

                Xi = experts[i](hidden[mask_i]).detach()
                Xj = experts[j](hidden[mask_j]).detach()

                # match tokens
                n = min(Xi.size(0), Xj.size(0), 256)

                idx_i = torch.randperm(Xi.size(0), device=Xi.device)[:n]
                idx_j = torch.randperm(Xj.size(0), device=Xj.device)[:n]

                Xi = Xi[idx_i].unsqueeze(0)
                Xj = Xj[idx_j].unsqueeze(0)

                sim = 1 - linear_cka(Xi, Xj)

                losses.append(sim)

    if len(losses) == 0:
        return torch.tensor(0.0, device=student.lm_head.weight.device)

    return torch.stack(losses).mean()

# def expert_diversity_loss(student):

#     losses = []

#     for layer in student.transformer.h:

#         if not hasattr(layer.mlp, "experts"):
#             continue

#         if not hasattr(layer.mlp, "last_hidden_input"):
#             continue

#         hidden = layer.mlp.last_hidden_input.detach()

#         experts = layer.mlp.experts
#         outputs = []

#         for expert in experts:
#             out = expert(hidden.view(-1, hidden.size(-1)))
#             outputs.append(out)

#         for i in range(len(outputs)):
#             for j in range(i + 1, len(outputs)):

#                 sim = F.cosine_similarity(
#                     outputs[i],
#                     outputs[j],
#                     dim=-1
#                 ).mean()

#                 losses.append(sim)

#     if len(losses) == 0:
#         return torch.tensor(0.0, device=hidden.device)

#     return torch.stack(losses).mean()


def train_distill():
    start_time = datetime.now()
    device = torch.device("cuda") 

    tokenizer = Tokenizer("checkpoints/Qwen/Qwen3-1.7B")
    dataset = AgriDataset(
            data_path=DATA_PATH, 
            tokenizer=tokenizer, 
            max_seq_length=MAX_SEQ_LENGTH 
        )
    # cka_fns = {
    #     "linear": LinearCKALoss(),
    #     "unbiased": UnbiasedLinearCKA()
    # }

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    print("Loading 8B Teacher in BF16...")
    teacher = GPT(Config.from_name("Qwen3-8B")).to(device, dtype=torch.bfloat16) 
    teacher.load_state_dict(torch.load(TEACHER_CKPT, mmap=True, weights_only=True), strict=False)
    teacher.eval()
    
    for param in teacher.parameters():
        param.requires_grad = False

    print("Loading 0.6B-MoE Student in BF16...")
    student = GPT(Config.from_name("Qwen3-1.7B-MoE")).to(device, dtype=torch.bfloat16)
    student.load_state_dict(torch.load(STUDENT_INIT, map_location=device, weights_only=True))

    for p in teacher.parameters():
        p.requires_grad=False

    #adding guassian noise (5% of weight std)
    print("Applying symmetry-breaking noise to experts...")
    with torch.no_grad():
        for name, param in student.named_parameters():

            if "mlp.experts" in name and "weight" in name:
                weight_std = param.std()
                noise = torch.randn_like(param) * weight_std * 0.02
                param.add_(noise)

    # Enable Gradient Checkpointing on Student to save VRAM, critical for the H100 80GB when running two models
    student.gradient_checkpointing_enable()

    optimizer=torch.optim.AdamW(
            student.parameters(),
            lr=3e-5,
            weight_decay=0.01
        )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    
    # Slightly higher LR because we are using a Scheduler now
    # optimizer = bnb.optim.AdamW8bit(student.parameters(), lr=5e-5) # 8-bit AdamW saves ~2GB
    # optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5, eps=1e-8, weight_decay=0.01)


    # Calculate total steps for scheduler
    total_steps = (len(data_loader) * NUM_EPOCHS) // ACCUMULATE_GRAD_STEPS
    scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=1e-6
        )
    # scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    cka_fn = LinearCKALoss()

    # Layer Mapping (teacher to student)
    # mapping = {s_i: int(s_i * (32-1) / (28-1)) for s_i in range(28)}
    n_student_layers = student.config.n_layer
    n_teacher_layers = teacher.config.n_layer
    
    num_pairs = 10
    student_layers = torch.linspace(
            0,
            n_student_layers - 1,
            steps=num_pairs
        ).long()
    mapping = {
            int(s): int(s * (n_teacher_layers - 1) / (n_student_layers - 1))
            for s in student_layers
        }
    print(f"Mapping {n_student_layers} Student layers -> {n_teacher_layers} Teacher layers")


    n_experts = student.config.n_expert   
    logger=LossLogger(OUTPUT_ROOT) 
    print(f"Training Started. Steps per Epoch: {len(data_loader)}")
    # optimizer.zero_grad() 

    # torch.backends.cuda.matmul.allow_tf32 = True
    
    update_step = 0
    step = 0
    last_cka = torch.tensor(0.0, device=device)
    

    print("\nDataset sanity check:\n")

    # data sanity check
    for i in range(3):

        tokens, mask = dataset[i]

        print("\n============================")
        print("Sample", i)
        print("============================")

        decoded = tokenizer.processor.decode(tokens.tolist())

        print(decoded[:1000])

        print("\nMASK CHECK")

        for t, m in zip(tokens[:120], mask[:120]):

            word = tokenizer.processor.decode([t.item()])


            if m == 1:
                print(f"{word} → LOSS")
            else:
                print(f"{word} → MASK")




    for epoch in range(NUM_EPOCHS):
        print(f"\n=== EPOCH {epoch+1}/{NUM_EPOCHS} ")

        # for batch_idx, (input_ids, loss_mask) in enumerate(tqdm(data_loader, desc="Training")):
        for batch in tqdm(data_loader):
            input_ids, loss_mask = batch
            input_ids = input_ids.to(device)
            loss_mask = loss_mask.to(device)

            if step < WARMUP_STEPS:
                lr_scale = (step + 1) / WARMUP_STEPS
                for g in optimizer.param_groups:
                    g["lr"] = 3e-5 * lr_scale

            # Teacher 
            with torch.no_grad():
                t_logits, t_features = teacher(input_ids, return_features=True)

            # Student Forward
            s_logits, s_features = student(input_ids, return_features=True)

            T = DISTILL_CONFIG["T"] # Distillation Temperature

            # KL
            student_log=F.log_softmax(s_logits/T,dim=-1)
            teacher_prob=F.softmax(t_logits/T,dim=-1)

            kl_pointwise = F.kl_div(student_log, teacher_prob, reduction="none").sum(-1)

            kl = (kl_pointwise * loss_mask).sum() / (loss_mask.sum() + 1e-6)
            kl = kl * (T**2)

            # CE
            shift_logits=s_logits[:,:-1]
            shift_labels=input_ids[:,1:]

            ce_pointwise = F.cross_entropy(
                shift_logits.reshape(-1,shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction="none"
            ).view(shift_labels.shape)

            shift_mask = loss_mask[:,1:]

            ce = (ce_pointwise * shift_mask).sum() / (shift_mask.sum() + 1e-6)


            # CKA

            if step % 4 == 0:
                cka = 0
                for s,t in mapping.items():
                    sf, tf = paired_token_sample(
                        s_features[s].detach(),
                        t_features[t].detach()
                    )
                    cka += cka_fn(sf, tf)

                cka = cka / max(len(mapping),1)
                last_cka = cka
            else:
                cka = last_cka

            router_loss = compute_load_balance_loss(student,n_experts)
            div_loss = expert_diversity_cka_new(student)
            
            # entropy
            # student_prob = torch.exp(student_log)
            # entropy = -(student_prob * student_log).sum(-1).mean()
            prob = F.softmax(s_logits, dim=-1) 
            logprob = F.log_softmax(s_logits, dim=-1)
            entropy = -(prob * logprob).sum(-1).mean()


            loss=(
                DISTILL_CONFIG["alpha"]*kl
                +DISTILL_CONFIG["lambda"]*ce
                +DISTILL_CONFIG["beta"]*cka
                +DISTILL_CONFIG["delta"]*div_loss
                +DISTILL_CONFIG["gamma"]*router_loss
            )

            #training step
            (loss / ACCUMULATE_GRAD_STEPS).backward()

            if (step + 1) % ACCUMULATE_GRAD_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(),0.8)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                update_step+=1

            step+=1

            max_load = 0.0
            all_indices = []

            for layer in student.transformer.h:
                if hasattr(layer.mlp, "last_indices") and layer.mlp.last_indices is not None:
                    all_indices.append(layer.mlp.last_indices)

            if len(all_indices) > 0:
                all_indices = torch.cat(all_indices, dim=0).detach()
                max_load = (
                    torch.bincount(all_indices.view(-1), minlength=n_experts).float()
                    / all_indices.numel()
                ).max().item()

            logger.log(
                step,
                loss.item(),
                kl.item(),
                ce.item(),
                cka.item(),
                div_loss.item(),
                router_loss.item(),
                max_load
            )

            # Print status
            if step % 20 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"Step {step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"KL: {kl.item():.4f} | "
                    f"CE: {ce.item():.4f} | "
                    f"CKA: {cka.item():.4f} | "
                    f"DIV: {div_loss.item():.6f} | "
                    f"Router: {router_loss.item():.4f} | "
                    f"MaxLoad: {max_load:.2%} | "
                    f"LR: {current_lr:.2e}"
                )

            #  THE AUDIT 
            if (step + 1) % ACCUMULATE_GRAD_STEPS == 0 and (update_step+1)%1500 == 0:
                audit_teacher_expert_alignment(
                    student,teacher,cka_fn,batch,step, epoch
                )
                audit_expert_specialization(student, cka_fn, [input_ids], step, epoch)
                audit_expert_specialization_routerwise(student, cka_fn, [input_ids], step, epoch)
                checkpoint = {
                    'step': step,
                    'model_state_dict': student.state_dict(),
                }
                torch.save(checkpoint, CHECKPOINT_ROOT / f"epoch-{epoch}_step-{step}.pth", _use_new_zipfile_serialization=False)

        checkpoint = {
            'step': step,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(checkpoint, CHECKPOINT_ROOT / f"epoch-{epoch}.pth", _use_new_zipfile_serialization=False)
 

    # Save the final student state separately
    final_path = CHECKPOINT_ROOT / "lit_model.pth"
    # torch.save(student.state_dict(), final_path)
    checkpoint = {
        'step': step,
        'model_state_dict': student.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, CHECKPOINT_ROOT / "lit_model.pth", _use_new_zipfile_serialization=False)
    audit_expert_specialization(student, cka_fn, [input_ids], step)
    audit_expert_specialization_routerwise(student, cka_fn, [input_ids], step)


    # Copy configuration files to make the model portable
    from litgpt.utils import copy_config_files
    copy_config_files(Path("checkpoints/Qwen/Qwen3-1.7B"), CHECKPOINT_ROOT)
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
        
    print(f"Finished! Total Duration: {duration}")



if __name__ == "__main__":
    train_distill()







