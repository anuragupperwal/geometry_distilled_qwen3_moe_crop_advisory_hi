I’ll explain exactly what your geometry-based distillation pipeline is doing, step by step, and specifically how hidden representations and CKA are used. I’ll stay technical but structured so the whole mechanism becomes clear.

⸻

1. What geometry distillation means in your training setup

Your training objective is not only to make the student produce the same output tokens as the teacher. It also tries to make the internal representations of the student resemble those of the teacher.

The internal representation of a transformer at a given layer is the hidden state tensor produced after the self-attention and feedforward block. If the input sequence has length T and the hidden dimension is D, then the hidden state of a layer has shape:

[B, T, D]

where
	•	B = batch size
	•	T = sequence length
	•	D = embedding dimension

These hidden states encode the contextual meaning of tokens after processing through the transformer layers.

In your training procedure, geometry distillation attempts to ensure that:

The structure of the representation space produced by the student resembles the structure produced by the teacher.

This is why the method is called geometry distillation. It is not trying to match the raw vectors exactly, but rather the relationships between vectors in the representation space.

⸻

2. What components are used in your distillation objective

Your training loss is a weighted combination of several components:

1. KL divergence loss

This aligns the probability distributions over tokens between teacher and student.

The teacher produces logits:

t_logits

The student produces logits:

s_logits

The KL divergence loss encourages:

P_{student}(token | context) \approx P_{teacher}(token | context)

This ensures the student produces similar next-token probabilities.

⸻

2. Cross-entropy loss (teacher forcing)

This aligns the student output with the ground truth tokens.

The loss is computed using the shifted sequence:

shift_logits = s_logits[:, :-1]
shift_labels = input_ids[:, 1:]

This is standard autoregressive training where the student learns to predict the correct next token given the previous tokens.

⸻

3. Geometry distillation using CKA

This is the main component responsible for aligning internal representations.

The goal is to make the student’s hidden representations resemble the teacher’s hidden representations in terms of structure.

Instead of forcing the vectors to be identical, the algorithm matches the similarity relationships between tokens.

This is done using Centered Kernel Alignment (CKA).

⸻

4. Router load balancing (MoE)

Because the student is a mixture-of-experts model, the training also includes a term that encourages tokens to be distributed across experts rather than collapsing into a few experts.

This is not related to geometry distillation but helps maintain a functional MoE routing mechanism.

⸻

5. Expert diversity loss

This term encourages different experts to produce different outputs, preventing the MoE from degenerating into identical experts.

⸻

3. How hidden representations are used

When you run a forward pass with the teacher and student:

t_logits, t_features = teacher(input_ids, return_features=True)
s_logits, s_features = student(input_ids, return_features=True)

The model returns a list of hidden states:

t_features[layer]
s_features[layer]

Each element corresponds to the hidden representation after a transformer layer.

Each tensor has shape:

[B, T, D]

where
	•	B is batch size
	•	T is sequence length
	•	D is hidden dimension

Your teacher has more layers than the student. Therefore you create a mapping between student layers and teacher layers.

For example:

mapping = {
    student_layer_i → teacher_layer_j
}

This mapping determines which teacher layer representation should be compared to which student layer representation.

⸻

4. Why hidden representation sizes do not need to match

The teacher hidden dimension may be larger than the student hidden dimension.

For example:

teacher: D = 4096
student: D = 1536

CKA can compare these representations because it does not require equal dimensionality. It compares the relationships between vectors, not the vectors themselves.

This is a key advantage of CKA for representation alignment.

⸻

5. Why token sampling is used

The hidden states have shape:

[B, T, D]

If the sequence length is large (for example 4096), computing CKA on the full matrix is expensive.

Therefore your code samples tokens:

paired_token_sample(x, y)

This function randomly selects a subset of token representations from both teacher and student.

This produces smaller matrices:

[1, N, D_student]
[1, N, D_teacher]

where N is a sampled subset of tokens.

This reduces computational cost while still capturing the representation geometry.

⸻

6. What CKA measures

CKA measures the similarity between two sets of representations.

Given two matrices:

X \in R^{n \times d_1}

Y \in R^{n \times d_2}

each row corresponds to a token representation.

The algorithm first centers the matrices by subtracting the mean:

X = X - mean(X)

Y = Y - mean(Y)

Then it computes the cross-covariance matrix:

X^T Y

The squared Frobenius norm of this matrix measures how strongly the two representation spaces align.

The final CKA score is:

CKA = \frac{||X^T Y||^2}{||X^T X|| \cdot ||Y^T Y||}

The value ranges from:

0 → completely unrelated representations
1 → identical geometry


⸻

7. How CKA is used as a loss

Your code computes:

cka_loss = 1 - CKA

So the objective is:

minimize (1 − CKA)

which means maximizing the alignment between teacher and student representation geometries.

⸻

8. What the student learns from this

By minimizing this loss, the student learns to produce hidden states that preserve the relative relationships between token representations.

This means that if two tokens are close in the teacher’s representation space, they should also be close in the student’s space.

Similarly, if two tokens are far apart in the teacher representation space, the student should also represent them as distant.

This enforces structural similarity between representation manifolds.

⸻

9. How the geometry loss is aggregated

For each training step, your code loops through mapped layers:

for s_layer, t_layer in mapping:
    cka += CKA(student_feature, teacher_feature)

Then it averages the value across layers.

This ensures that multiple levels of representation hierarchy are aligned.

Lower layers capture lexical and syntactic structure, while deeper layers capture semantic and task-specific information.

Aligning multiple layers allows the student to approximate the teacher’s entire representational hierarchy.

⸻

10. How this interacts with other losses

Your final loss combines:

KL divergence
cross entropy
CKA geometry loss
router balancing
expert diversity

KL divergence aligns output distributions.
Cross entropy aligns target tokens.
CKA aligns internal representation geometry.

This combination ensures the student learns both:

what the teacher predicts
how the teacher internally represents information


⸻

11. Why geometry distillation is useful

Standard distillation using only KL divergence can produce a student that mimics token probabilities but has a very different internal representation structure.

Geometry distillation prevents this by enforcing representation similarity.

This often improves:

generalization
reasoning consistency
multi-step generation quality

because the student learns not only the outputs but also the internal feature organization of the teacher.

⸻

Final summary

Your geometry distillation works as follows:
	1.	Both teacher and student process the same input sequence.
	2.	Hidden states from selected layers are extracted.
	3.	A subset of token representations is sampled.
	4.	CKA is computed between teacher and student representations.
	5.	The training loss penalizes differences in representation geometry.
	6.	This loss is combined with KL divergence and cross-entropy.

As a result, the student learns both:

teacher output distributions
teacher representation geometry

which is the core idea of geometry-based distillation.

⸻

If you’d like, I can also explain why CKA was chosen instead of cosine similarity or MSE, which will clarify an important design choice in your training setup.