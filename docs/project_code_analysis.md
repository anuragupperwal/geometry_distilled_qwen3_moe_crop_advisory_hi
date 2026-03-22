Below is a **structured technical report** of the scripts you provided.
You can directly paste this into a **`PROJECT_CODE_ANALYSIS.md`** file.

I will explain for each file:

1. **Purpose of the file**
2. **What pipeline stage it belongs to**
3. **Step-by-step execution**
4. **Core ML / systems concepts used**
5. **Important implementation details**

This will help you maintain a **mental map of the whole system**.

---

# Project Script Analysis Report

This document explains the functionality, concepts, and logic used in the scripts of the **MoE Distillation Project** built using **LitGPT + Qwen3 architecture**.

The system converts dense models to MoE models, trains them via distillation, and provides tools for debugging and verification.

---

# 1. `scripts/check_weights.py`

## Purpose

This script **inspects a checkpoint to verify the MoE structure**.

Specifically it confirms:

* routers exist
* experts exist
* the expected number of experts per layer
* expert weights are not identical

This is a **sanity check tool for the MoE architecture**.

---

## Pipeline Stage

Used **after model creation / upcycling**.

Pipeline position:

```
Dense Model
     ↓
Upcycling to MoE
     ↓
check_weights.py  ← validation step
     ↓
training
```

---

## Step-by-Step Logic

### 1️⃣ Load checkpoint

```python
state_dict = torch.load(...)
```

Concept:

**State Dict**

A PyTorch checkpoint storing model parameters.

Example:

```
transformer.h.0.mlp.experts.0.fc_1.weight
```

---

### 2️⃣ Filter only MLP components

```
if ".mlp." in key
```

Reason:

MoE routing happens only inside the **MLP block**.

Transformer layer structure:

```
TransformerBlock
 ├ attention
 ├ layernorm
 └ MLP / MoE
```

---

### 3️⃣ Identify router weights

Router key:

```
mlp.gate.weight
```

Concept:

**MoE Router**

A small neural network that decides:

```
token → which expert
```

Mathematically:

[
g(x) = softmax(Wx)
]

---

### 4️⃣ Identify expert weights

Experts are independent MLPs.

Structure:

```
mlp.experts.0.fc_1.weight
mlp.experts.1.fc_1.weight
...
```

Each expert contains:

```
fc1 → activation → fc2 → projection
```

---

### 5️⃣ Count experts per layer

```
layers_audit[layer_idx].append(expert_idx)
```

Concept:

This verifies that every layer has the correct number of experts.

Expected:

```
n_expert = 2
```

---

### 6️⃣ Symmetry check

```
Expert0 weight vs Expert1 weight
```

Purpose:

Ensure experts are **not identical**.

If identical:

```
experts collapse
```

Which is a common MoE problem.

---

# Concepts Used

| Concept                  | Explanation                        |
| ------------------------ | ---------------------------------- |
| State dict inspection    | checking architecture via weights  |
| MoE router               | gating network selecting experts   |
| expert cloning detection | prevents degenerate initialization |
| model auditing           | debugging model structure          |

---

# 2. `scripts/convert_hf_checkpoint.py`

## Purpose

Converts **HuggingFace model checkpoints → LitGPT format**.

This is required because:

```
HF architecture != LitGPT architecture
```

So weight names must be remapped.

---

## Pipeline Stage

```
HF Model
   ↓
convert_hf_checkpoint.py
   ↓
LitGPT compatible model
```

---

## Core Idea

Different frameworks store weights with different names.

Example:

HF:

```
model.layers.0.self_attn.q_proj.weight
```

LitGPT:

```
transformer.h.0.attn.qkv.weight
```

So we must **map the names**.

---

# Core Technique Used

## Weight Mapping

Example mapping dictionary:

```
"model.embed_tokens.weight"
→
"transformer.wte.weight"
```

This converts embedding weights.

---

# Step-by-Step Pipeline

## 1️⃣ Determine architecture

```
config = Config.from_name(model_name)
```

Concept:

**Model configuration**

Defines:

```
n_layer
n_head
hidden_size
```

---

## 2️⃣ Select conversion function

Based on architecture:

```
Qwen
Gemma
Phi
Falcon
Llama
```

Example:

```
copy_weights_qwen_3
```

---

## 3️⃣ Load HF checkpoint files

Checkpoints may be stored in:

```
.bin
.safetensors
```

So the script loads both.

---

## 4️⃣ Lazy Loading

```
lazy_load()
```

Concept:

**Memory-efficient loading**

Large models (8B+) cannot be loaded fully into RAM.

Lazy loading loads weights **on demand**.

---

## 5️⃣ QKV reassembly

Attention weights sometimes stored differently.

Example:

HF layout:

```
[q,k,v,q,k,v]
```

LitGPT layout:

```
[q,q,q,k,k,k,v,v,v]
```

So the script rearranges them.

Mathematically this is just **tensor slicing and concatenation**.

---

# Key Concepts

| Concept               | Explanation                    |
| --------------------- | ------------------------------ |
| checkpoint conversion | translating model formats      |
| tensor remapping      | renaming weight tensors        |
| lazy loading          | memory-efficient loading       |
| QKV reassembly        | attention weight restructuring |

---

# 3. `scripts/convert_lit_checkpoint.py`

## Purpose

Reverse of the previous script.

Converts:

```
LitGPT checkpoint → HuggingFace format
```

Useful for:

* model sharing
* inference with HF tools
* publishing models

---

## Pipeline

```
LitGPT training checkpoint
       ↓
convert_lit_checkpoint.py
       ↓
HuggingFace model
```

---

# Core Logic

Similar to previous script but **mapping reversed**.

Example mapping:

```
transformer.wte.weight
→
model.embed_tokens.weight
```

---

## QKV splitting

LitGPT stores QKV together:

```
attn.qkv.weight
```

HF expects separate tensors:

```
q_proj
k_proj
v_proj
```

So this script **splits the tensor**.

---

# Concepts

| Concept                    | Explanation                       |
| -------------------------- | --------------------------------- |
| framework interoperability | convert models between frameworks |
| tensor slicing             | splitting QKV                     |
| model export               | preparing models for external use |

---

# 4. `scripts/upcycle_moe.py`

This is one of the **most important files**.

---

# Purpose

Implements **Sparse Upcycling**.

Paper:

```
"Sparse Upcycling"
Komatsuzaki et al
(Mistral / DeepMind technique)
```

---

# Concept: Sparse Upcycling

Convert dense model → MoE model **without retraining from scratch**.

Strategy:

```
Dense MLP
↓
Clone it into multiple experts
```

Example:

Dense model layer:

```
MLP
```

MoE version:

```
Expert0 = clone
Expert1 = clone
Expert2 = clone
Expert3 = clone
```

---

# Why This Works

Dense model already learned good representations.

Cloning it into experts allows **faster MoE training**.

---

# Step-by-Step

## 1️⃣ Load dense checkpoint

```
dense_state_dict
```

---

## 2️⃣ Initialize MoE model

```
student_model = GPT(config)
```

This creates the MoE architecture.

---

## 3️⃣ Copy dense weights to experts

For each dense MLP weight:

```
mlp.fc1.weight
```

Create:

```
experts.0.fc1.weight
experts.1.fc1.weight
...
experts.N.fc1.weight
```

This clones the weights.

---

## 4️⃣ Initialize router

Router is new so random init:

```
torch.randn() * 0.02
```

Concept:

**Gating network initialization**

Small weights prevent routing instability.

---

# Verification Function

## `verify_experts_and_plot()`

Purpose:

Confirm experts were cloned correctly.

---

### Method

Compute **cosine similarity** between expert weights.

Formula:

[
\cos(x,y)=\frac{x\cdot y}{||x|| ||y||}
]

Expected result:

```
similarity ≈ 1
```

Meaning identical clones.

---

### Visualization

Creates heatmap:

```
expert similarity matrix
```

Example:

```
E0 E1 E2
E1 1 1 1
E2 1 1 1
```

---

# Concepts

| Concept               | Explanation                    |
| --------------------- | ------------------------------ |
| Sparse upcycling      | converting dense models to MoE |
| weight cloning        | copying learned parameters     |
| router initialization | gating network creation        |
| cosine similarity     | expert comparison              |

---

# 5. `scripts/sanity_check_data.py`

## Purpose

Verifies that **dataset + tokenizer + formatting** work correctly.

This prevents silent data errors.

---

# Pipeline Position

```
Dataset
↓
sanity_check_data.py
↓
Training
```

---

# Step-by-Step

### 1️⃣ Load tokenizer

```
Tokenizer(checkpoint_dir)
```

The tokenizer defines:

```
text → tokens
tokens → text
```

---

### 2️⃣ Load dataset

```
AgriDataset(...)
```

Dataset returns:

```
tokenized sequences
```

---

### 3️⃣ Decode sample

The script decodes tokens to verify formatting.

This ensures training data looks correct.

Example output:

```
<think>
reasoning here
</think>
final answer
```

---

# Important Concept

### Masked Training

Prompt tokens are masked:

```
labels = -100
```

So loss is computed only on:

```
CoT + answer
```

---

# Concepts Used

| Concept                 | Explanation                    |
| ----------------------- | ------------------------------ |
| tokenization            | converting text to tokens      |
| dataset validation      | ensuring data correctness      |
| sequence masking        | ignoring prompt tokens in loss |
| training data debugging | verifying format               |

---

# Overall Project Architecture

The scripts together implement the following pipeline.

---

# Full System Pipeline

```
HuggingFace Model
        │
        │
        ▼
convert_hf_checkpoint.py
        │
        ▼
LitGPT Dense Model
        │
        │
        ▼
upcycle_moe.py
(Dense → MoE)
        │
        ▼
check_weights.py
(verify experts)
        │
        ▼
sanity_check_data.py
(verify dataset)
        │
        ▼
Training Script
(distillation + MoE)
        │
        ▼
Evaluation scripts
        │
        ▼
convert_lit_checkpoint.py
(export model)
```

---

# Key ML Concepts Used Across the Project

## Mixture of Experts (MoE)

Instead of one MLP:

```
MLP(x)
```

We use:

[
y=\sum g_i(x)E_i(x)
]

Where

```
g_i(x) = router probability
E_i = expert network
```

---

## Distillation

Teacher → student knowledge transfer.

Loss used:

```
KL divergence
cross entropy
CKA feature alignment
diversity loss
```

---

## Sparse Upcycling

Technique to convert dense model to MoE quickly.

---

## Router Gating

Router chooses which expert processes each token.

---

## Weight Conversion

Necessary because different frameworks use different tensor names.

---

# Summary

Your project contains **four major system components**:

### Model conversion

```
convert_hf_checkpoint
convert_lit_checkpoint
```

### Model architecture transformation

```
upcycle_moe
```

### Debugging tools

```
check_weights
sanity_check_data
```

### Training + evaluation

Handled by the other scripts you shared earlier.

---

# My Assessment

Your project architecture is **well structured and follows modern MoE research practices**.

It includes:

* Sparse upcycling
* Distillation
* MoE specialization analysis
* Conversion pipelines
* Data verification tools

This is **very similar to research pipelines used in large MoE training systems**.
