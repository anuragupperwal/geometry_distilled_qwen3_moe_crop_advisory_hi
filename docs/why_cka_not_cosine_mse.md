This is an important question because it explains why geometry-based distillation uses CKA instead of simpler measures like cosine similarity or MSE. The choice is not arbitrary. It comes directly from the properties of neural representations inside deep networks. To understand the decision clearly, we need to examine how hidden representations behave and what we actually want to preserve during distillation.

⸻

1. What we want to preserve during geometry distillation

When a teacher model processes an input sequence, each transformer layer produces hidden representations for every token. These hidden vectors encode contextual meaning and relationships between tokens.

If we denote the hidden representation matrix of a layer as:

H \in \mathbb{R}^{T \times D}

then
	•	T = number of tokens in the sequence
	•	D = hidden dimension of the layer

Each row corresponds to the representation of a token after that layer.

When performing geometry distillation, we do not want to force the student to reproduce the exact numeric vectors of the teacher. Instead, we want to preserve the structure of the representation space. This means that the relationships between token representations should be similar.

For example, if two tokens have very similar contextual meaning in the teacher representation space, their vectors will be close together. If two tokens represent unrelated concepts, their vectors will be far apart. The student should preserve this relative structure.

This is why the distillation objective focuses on representation geometry rather than raw vector values.

⸻

2. Why MSE is not suitable

A straightforward way to compare teacher and student hidden states would be Mean Squared Error:

MSE = \frac{1}{n} \sum (X - Y)^2

where X and Y are hidden representations from teacher and student.

However, this approach has a major problem.

Transformer hidden representations are not uniquely determined. The same semantic information can be represented using different coordinate systems. In other words, the representation space can undergo transformations such as:
	•	rotation
	•	scaling
	•	linear transformation

and still encode the same information.

If the student representation is simply a rotated version of the teacher representation, MSE would report a large error even though the representations are functionally equivalent.

Therefore MSE incorrectly penalizes differences that do not matter for the model’s behavior.

⸻

3. Why cosine similarity is also insufficient

Cosine similarity compares two vectors by measuring the angle between them:

\cos(\theta) = \frac{x \cdot y}{||x|| \, ||y||}

This metric solves one problem that MSE has. It is insensitive to vector magnitude and focuses only on direction.

However cosine similarity still compares individual vectors directly. This means it assumes that the teacher vector and student vector for the same token should point in the same direction.

This assumption is too strict. If the student representation is produced by a different linear transformation of the teacher representation space, cosine similarity will still report low similarity even if the internal relationships between tokens are preserved.

Therefore cosine similarity still forces the student to imitate the exact representation coordinates rather than the structure of the representation space.

⸻

4. What CKA measures instead

Centered Kernel Alignment (CKA) compares two representation matrices by measuring how similar their internal pairwise relationships are.

Suppose we have:

X \in \mathbb{R}^{n \times d_1}

Y \in \mathbb{R}^{n \times d_2}

where each row corresponds to a token representation.

Instead of comparing vectors directly, CKA measures how similar the similarity matrices of the two representations are.

First, it computes similarity between all pairs of tokens:

S_X = X X^T

S_Y = Y Y^T

Each entry in these matrices measures the similarity between two tokens.

If the teacher representation considers tokens i and j to be similar, the student representation should also consider them similar.

CKA evaluates how closely these two similarity matrices match.

⸻

5. The mathematical form of CKA

The linear CKA used in your code is computed as:

CKA(X,Y) =
\frac{||X^T Y||_F^2}
{||X^T X||_F \cdot ||Y^T Y||_F}

where || \cdot ||_F is the Frobenius norm.

The numerator measures the strength of alignment between the two representations.

The denominator normalizes the value so that the score lies between 0 and 1.

Interpretation:

CKA = 1   → identical geometry
CKA ≈ 0   → unrelated representations

Your training objective minimizes:

1 − CKA

so that the student maximizes alignment with the teacher representation geometry.

⸻

6. Important properties of CKA

CKA was chosen because it has several mathematical properties that make it ideal for comparing neural representations.

Invariance to orthogonal transformations

If the student representation is a rotated version of the teacher representation, CKA still reports high similarity.

This is important because neural networks often learn equivalent representations in different coordinate systems.

⸻

Invariance to isotropic scaling

If the magnitude of vectors changes but their relationships remain the same, CKA is unaffected.

This avoids problems where representation norms differ between teacher and student.

⸻

Ability to compare different dimensionalities

Teacher and student hidden dimensions may differ:

Teacher hidden size = 4096
Student hidden size = 1536

CKA can still compare them because it operates on the relationships between samples rather than on individual coordinates.

Neither MSE nor cosine similarity can handle this properly.

⸻

7. Why this matters for distillation

If you used MSE or cosine similarity, the student would be forced to reproduce the teacher representation vectors exactly. This would constrain the student unnecessarily and could make training unstable or inefficient.

By using CKA, the student is allowed to learn its own representation coordinates as long as the relative relationships between tokens remain consistent with the teacher.

This means the student can compress the teacher representation space into a lower-dimensional space while preserving the semantic structure.

This is exactly what we want in model compression and distillation.

⸻

8. Why CKA is widely used in representation analysis

CKA was introduced in research on neural representation similarity and has become a standard tool for comparing internal representations across models.

It is widely used for:
	•	analyzing layer similarity between models
	•	studying representation collapse
	•	comparing models trained on different tasks
	•	representation distillation

Because it captures structural similarity rather than coordinate similarity, it is particularly suitable for comparing deep neural networks.

⸻

9. Why your training uses CKA with token sampling

Computing CKA on full hidden states would be expensive because the token count can be large. Your code therefore samples a subset of tokens from both teacher and student representations.

This reduces the computational cost while preserving enough information to estimate the geometry of the representation space.

The sampled token representations are then used to compute the CKA score for that layer.

⸻

Final understanding

The reason CKA is used instead of MSE or cosine similarity is that it compares representation geometry rather than raw vector coordinates. This allows the student model to learn a compressed representation space that preserves the structural relationships present in the teacher’s representations.

Because transformer hidden states can differ in scale, orientation, and dimensionality between teacher and student models, a geometry-based metric like CKA provides a robust way to align internal representations during distillation.

⸻

If you want, I can also explain one subtle but extremely important limitation of CKA in distillation, which is relevant to your training setup and could affect how well the student actually learns from the teacher.