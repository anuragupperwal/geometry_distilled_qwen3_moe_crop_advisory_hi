Below is the **second document** you asked for.
It is written so you can directly save it as:

```
docs/MOE_TRAINING_ARCHITECTURE.md
```

It explains the **complete training architecture**, including the mathematics, reasoning behind each loss, and how the MoE system behaves during training.

---

# MoE Distillation Training Architecture

## Project: Qwen3 Dense → MoE Distillation System

This document explains the **architecture, mathematics, and reasoning** behind the Mixture-of-Experts (MoE) training pipeline used in this project.

The system converts a **dense transformer model** into a **Mixture-of-Experts transformer** and trains it using **knowledge distillation from a larger teacher model**.

---

# 1. High Level Architecture

The complete pipeline follows this structure:

```
Teacher Model (Dense Qwen3-8B)
                │
                │ logits + hidden states
                ▼
Student Model (Qwen3-0.6B MoE)
                │
                │ multi-loss training
                ▼
MoE Distilled Model
```

The student model is trained using **multiple objectives simultaneously**:

```
Total Loss =
    KL Distillation Loss
  + Cross Entropy Loss
  + CKA Feature Alignment Loss
  + Expert Diversity Loss
```

Mathematically:

[
L_{total} =
\alpha L_{KL}
+
\lambda L_{CE}
+
\beta L_{CKA}
+
\gamma L_{div}
]

---

# 2. Transformer with Mixture-of-Experts

## Standard Transformer Layer

A normal transformer layer consists of:

```
LayerNorm
Self Attention
LayerNorm
Feed Forward Network (MLP)
```

The MLP computes:

[
y = W_2 \cdot \sigma(W_1 x)
]

where

```
x = token representation
W1,W2 = weight matrices
σ = activation (SiLU or GELU)
```

---

## MoE Transformer Layer

Instead of a **single MLP**, MoE uses multiple experts.

Structure:

```
Router
  │
  ├── Expert 0
  ├── Expert 1
  ├── Expert 2
  ├── Expert 3
```

Each expert is a separate neural network.

Mathematically:

[
y = \sum_{i=1}^{E} g_i(x) \cdot E_i(x)
]

Where

```
E_i(x) = output of expert i
g_i(x) = router probability for expert i
E = number of experts
```

---

# 3. Router (Gating Network)

The router decides **which expert processes a token**.

Router is a linear layer:

[
r(x) = W_g x
]

Then softmax:

[
g(x) = softmax(r(x))
]

Where

```
Wg = router weights
x = token embedding
```

Output:

```
g(x) = probability distribution over experts
```

Example:

```
Token: "fertilizer"

Router Output:
Expert0: 0.05
Expert1: 0.80
Expert2: 0.10
Expert3: 0.05
```

Token is routed to **Expert1**.

---

# 4. Top-K Routing

Most MoE models use **Top-K routing**.

Instead of using all experts:

```
only top-k experts are used
```

Example:

```
k = 2
```

If router outputs:

```
[0.8, 0.1, 0.05, 0.05]
```

Only the two largest experts are used.

This keeps computation efficient.

---

# 5. Sparse Computation

MoE models are efficient because **not all experts are used**.

Dense MLP:

```
all parameters active
```

MoE:

```
only k experts active per token
```

This leads to:

```
high parameter count
low computation cost
```

Example:

```
Dense model: 1B params active
MoE model: 8B params total
            but only 1B active
```

---

# 6. Sparse Upcycling

The project uses **Sparse Upcycling**.

Instead of training MoE from scratch:

```
Dense model → converted to MoE
```

Method:

```
Clone dense MLP weights into experts
```

Example:

Dense MLP:

```
W1
W2
```

MoE:

```
Expert0: W1, W2
Expert1: W1, W2
Expert2: W1, W2
Expert3: W1, W2
```

Router is randomly initialized.

This allows the MoE model to **start from a pretrained representation**.

---

# 7. Knowledge Distillation

The student model learns from a **teacher model**.

Teacher:

```
Qwen3-8B
```

Student:

```
Qwen3-0.6B MoE
```

The teacher provides:

```
logits
hidden representations
```

These guide the student.

---

# 8. KL Divergence Distillation

Teacher logits:

[
z_T
]

Student logits:

[
z_S
]

Soft probabilities:

[
p_T = softmax(z_T/T)
]

[
p_S = softmax(z_S/T)
]

Distillation loss:

[
L_{KL} =
\sum p_T \log \frac{p_T}{p_S}
]

This forces the student to mimic the teacher's output distribution.

---

# 9. Cross Entropy Loss

Standard supervised learning.

[
L_{CE} = - \sum y \log p_S
]

Where

```
y = true label
pS = student prediction
```

Purpose:

```
Anchor training to real data
```

Without CE, the student might copy teacher mistakes.

---

# 10. CKA Feature Alignment

CKA = **Centered Kernel Alignment**

Purpose:

```
Align internal representations of student and teacher
```

If teacher hidden state:

```
H_T
```

Student hidden state:

```
H_S
```

CKA similarity:

[
CKA(H_T, H_S)
=============

\frac{||H_T^T H_S||^2}
{||H_T^T H_T|| \cdot ||H_S^T H_S||}
]

Range:

```
0 = unrelated
1 = identical representation
```

CKA loss:

[
L_{CKA} = 1 - CKA
]

This encourages the student to learn **similar internal features**.

---

# 11. Expert Diversity Loss

MoE models can suffer from **expert collapse**.

Problem:

```
All experts learn the same function
```

To prevent this, we enforce **diversity**.

If expert outputs are:

```
E1(x), E2(x), E3(x)
```

We compute similarity between them.

Example:

[
sim(E_i,E_j)
]

Loss:

[
L_{div} = 1 - CKA(E_i,E_j)
]

This forces experts to specialize.

---

# 12. Expert Specialization

Healthy MoE training leads to **expert specialization**.

Examples:

| Expert   | Specialization         |
| -------- | ---------------------- |
| Expert 0 | Hindi grammar          |
| Expert 1 | Crop diseases          |
| Expert 2 | Weather reasoning      |
| Expert 3 | Numerical calculations |

This specialization emerges naturally if diversity loss works.

---

# 13. Expert Collapse

Expert collapse occurs when:

```
all experts behave identically
```

Symptoms:

```
similar expert weights
similar expert outputs
router always selects same expert
```

This reduces MoE to a dense model.

---

# 14. Router Load Balancing

MoE models require balanced expert usage.

If router always picks same expert:

```
GPU imbalance
training instability
```

Typical load balancing loss:

[
L_{balance} =
E \sum p_i f_i
]

Where

```
p_i = router probability
f_i = fraction of tokens sent to expert
```

---

# 15. Training Flow

The training process per batch:

```
1 Load batch
2 Run teacher forward pass
3 Run student forward pass
4 Compute losses
5 Combine losses
6 Backpropagation
7 Update parameters
```

---

# 16. Monitoring Metrics

During training several metrics are tracked.

### Expert Load

Measures how evenly experts are used.

```
max_expert_load
```

Ideal:

```
balanced distribution
```

---

### Expert Similarity

Measures if experts are collapsing.

Computed using cosine similarity or CKA.

---

### CKA Feature Similarity

Measures student-teacher alignment.

---

### Training Loss Components

```
KL loss
CE loss
CKA loss
Diversity loss
```

---

# 17. Visualization Tools

The project includes scripts to visualize:

```
expert similarity heatmaps
training curves
routing distributions
```

Example heatmap:

```
E0  E1  E2
E0 1.0 0.2 0.3
E1 0.2 1.0 0.1
E2 0.3 0.1 1.0
```

Low off-diagonal values indicate **good specialization**.

---

# 18. Expected Outcome

A successful training run produces:

```
specialized experts
balanced routing
student logits close to teacher
good downstream performance
```

---

# 19. Research Context

The techniques used in this project are inspired by:

```
Switch Transformer (Google)
GShard (Google)
Mixtral (Mistral)
DeepSeek MoE
Sparse Upcycling (Komatsuzaki)
```

These methods are currently used in **state-of-the-art LLM training**.

---

# 20. Summary

The system combines several modern training techniques:

| Technique              | Purpose                          |
| ---------------------- | -------------------------------- |
| Sparse Upcycling       | convert dense → MoE              |
| Knowledge Distillation | transfer teacher knowledge       |
| CKA Feature Alignment  | match internal representations   |
| Expert Diversity Loss  | prevent collapse                 |
| Router Load Balancing  | distribute tokens across experts |

Together these create a **scalable sparse transformer architecture** capable of training very large models efficiently.



----------------


### Final routing pipeline (correct architecture)

routing flow:

hidden states
      ↓
router linear layer
      ↓
router_logits
      ↓
softmax
      ↓
top-k experts
      ↓
count tokens per expert
      ↓
mask overflow tokens
      ↓
renormalize routing weights
      ↓
execute experts

This is exactly how Switch Transformer / Mixtral / DeepSeek MoE works.





------


Expert Semantic Specialization Analysis

To understand how the trained Mixture-of-Experts (MoE) model organizes internal computation, a semantic specialization analysis was conducted.

Hidden representations before the expert layers were collected for approximately 30,000 tokens, and each token was associated with the expert selected by the routing network. These hidden vectors were clustered using KMeans, creating semantic groups of token representations.

For each expert, the distribution of clusters and most frequent tokens routed to that expert were analyzed.

Key Findings

1. Strong Expert Specialization

Most experts predominantly processed tokens from a single semantic cluster. In many cases, more than 90% of tokens routed to an expert belonged to one cluster, indicating that the routing network successfully learned specialized expert roles.

2. Domain-Specific Expertise

Several experts showed strong association with agricultural terminology such as:

soil

crop

pest

weather

advisory

moisture

This demonstrates that the distilled model successfully internalized domain-specific knowledge relevant to agriculture advisory tasks.

3. Language-Level Specialization

The dataset contains bilingual English-Hindi text. The analysis revealed that certain experts frequently process Hindi tokens, while others primarily handle English tokens. This suggests the router learned language-specific routing behavior.

4. Structural Token Experts

Some experts were strongly associated with formatting and punctuation tokens such as:

newline

punctuation

formatting symbols

This indicates that the MoE architecture separates syntactic processing from semantic reasoning.

5. Balanced Expert Utilization

Across layers, multiple experts received substantial token traffic, indicating that the routing mechanism distributes workload effectively and avoids severe expert collapse.

Conclusion

The semantic specialization analysis demonstrates that the trained MoE model exhibits clear expert differentiation across semantic, linguistic, and structural token categories. This behavior aligns with findings reported in large-scale MoE architectures and indicates that the routing network has successfully learned to partition the computation across specialized experts.





---------
