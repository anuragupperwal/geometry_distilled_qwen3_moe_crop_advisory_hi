# Distilled-Qwen3-MoE: Upcycled Sparse Mixture-of-Experts for Hindi Crop Advisory

This project focuses on the development of a high-efficiency, reasoning-capable Language Model specifically optimized for the Indian agricultural sector. By "upcycling" the **Qwen3-0.6B** dense model into a **Sparse Mixture-of-Experts (MoE)** architecture, we achieve a model that maintains a small deployment footprint while significantly increasing its specialized capacity for Hindi-language crop diagnostics and advisory.

---

## Technical Architecture: From Dense to Sparse

The core of this project is the transformation of a standard dense transformer into a sparse MoE via **Sparse Upcycling**.

### 1. The Architecture Shift

Instead of training a MoE from scratch, we perform "weight surgery" on the Qwen3-0.6B model:

* **Frozen Attention:** We retain the pre-trained Attention and LayerNorm weights, as these already capture the linguistic structure of Hindi and English.
* **MLP Proliferation:** We duplicate the original Feed-Forward Network (FFN) layers into  identical "experts."
* **Router Initialization:** We introduce a Top- gating mechanism (the Router) that learns to dispatch tokens to the most relevant experts.
* **CKA Distillation:** To ensure the upcycled model retains the base model's knowledge, we use **Centered Kernel Alignment (CKA)** loss during initial training to minimize the representational shift between the dense teacher and the sparse student.

### 2. The "Thinking Token" Extension

To improve the model's agricultural reasoning (e.g., "If  pest is present and the weather is , then apply "), we experiment with **Reasoning Distillation**:

* **Vocab Expansion:** Adding `<thought>` tokens to the vocabulary.
* **Latent Reasoning:** Training the model to generate internal rationales before providing a final answer in Hindi.
* **Masked Distillation:** Using a teacher model to guide the final output while allowing the student MoE to utilize its "thinking" tokens to bridge the gap.

---

## Application: Dynamic Crop Advisory (Hindi)

While the architecture is cutting-edge, the objective is practical: providing real-time, context-aware agricultural support to farmers in their native language.

### Key Capabilities:

* **Localized Diagnostics:** Farmers can describe crop symptoms in Hindi (e.g., "गेहूं के पत्तों पर पीले धब्बे," *Yellow spots on wheat leaves*), and the model identifies potential diseases.
* **Dynamic Advisory:** Unlike static lookup tables, the model synthesizes advice based on variables like soil health, crop age, and current weather conditions.
* **Bilingual Reasoning:** While the input and output are in Hindi, the model leverages its English pre-training for technical botanical knowledge, translating complex agricultural science into actionable Hindi instructions.


---

## 🚀 Research Overview: CKA-Guided MoE Distillation

This project implements a "Thesis-Grade" knowledge distillation pipeline to compress a **Qwen3-14B (Teacher)** into an upcycled **Qwen3-0.6B-MoE (Student)** using a specialized Hindi Agricultural dataset.

### 🧪 Key Technical Innovations

* **Linear CKA Distillation:** Replaces rigid MSE loss with Centered Kernel Alignment (CKA) for feature mapping. This allows the Student () to align its internal manifolds with the Teacher () without requiring noisy projection layers.
* **Expert Diversity via CKA:** Utilizes an internal diversity penalty (minimizing CKA similarity between MoE experts) to mitigate "Expert Collapse" and force specialization across different agricultural domains.
* **On-the-Fly Simultaneous Distillation:** Implements real-time feature extraction from both models during the training loop, eliminating the need for massive disk I/O and intermediate storage of teacher features.

### 🛠️ Architectural Enhancements (`model.py`)

To support this pipeline, the core architecture has been modified to:

1. **Expose Hidden States:** The `forward` pass now optionally returns a list of intermediate hidden states from all transformer blocks for layer-wise CKA alignment.
2. **Expert Contribution Hooks:** The `LLaMAMoE` class now captures and stores individual expert outputs during the forward pass to enable diversity loss calculations.
3. **Dynamic Layer Mapping:** Implements a skip-based mapping strategy to align the Student's 28 layers with the Teacher's 40 layers effectively.

### 📊 Distillation Strategy Summary

| Loss Component | Objective | Metric |
| --- | --- | --- |
| **Logit Alignment** | Capture "Dark Knowledge" from final predictions. | KL-Divergence |
| **Feature Alignment** | Align internal reasoning manifolds across architectures. | **Linear CKA** |
| **Expert Diversity** | Prevent expert collapse and force specialization. | **1 - CKA** |

---



## Project Structure

```text
qwen3-moe-project/
├── litgpt/                 <-- The Core (Keep these)
│   ├── model.py            # MODIFIED: Add LLaMAMoE class here
│   ├── config.py           # MODIFIED: Add Qwen-MoE hyperparams here
│   ├── tokenizer.py        # Keep (Standard)
│   └── utils.py            # ADD: CKA calculation function
├── scripts/                <-- Tools & Surgery
│   ├── download.py         # Keep (To get 0.6B weights)
│   ├── upcycle_moe.py      # YOUR SCRIPT: Dense -> MoE conversion
│   └── convert_hf.py       # Keep (To convert HuggingFace weights)
├── configs/                <-- Configuration Hub
│   ├── upsample.yaml       # Hyperparams for the MoE shift
│   └── distill.yaml        # Distillation settings
├── train_distill.py        # YOUR MAIN WORK: The training loop
├── eval_cka.py             # YOUR ANALYSIS: Run this to get similarity scores
├── requirements.txt        # Add 'torch-cka' if using the library
└── data/                   # Dataset for distillation
```

scripts/upcycle_moe.py
scripts/convert_hf_checkpoint.py
scripts/download.py
litgpt/utils.py
litgpt/tokenizer.py
litgpt/model.py
litgpt/config.py


#### 1. `src/modeling/` (The Architecture)

* **`modular_qwen3_moe.py`**: This is where you write your code. Define your `Qwen3MoeSparseMoeBlock` here.
* **`modeling_qwen3_moe.py`**: You will use a script (or manual copy-paste) to flatten the modular code into this file so it can be used with `AutoModel`.

#### 2. `src/surgery/` (The Weight Mapping)

This is the most critical part for your specific project.

* **`upcycle.py`**: This script loads the **Qwen3-0.6B** (Dense) state dict, creates a new **Qwen3-MoE** (Sparse) model, and maps the weights.
* *Logic:* It takes `layer.i.mlp.gate_proj` and maps it to `layer.i.mlp.experts.gate_up_proj` across all  experts.



#### 3. `src/utils/` (The Math)

* **`cka_loss.py`**: Put your CKA implementation here. You will import this into your training script to ensure the "Upsampled" model still behaves like the "Base" model.
* **`distil_utils.py`**: Since you want to use "Thinking Tokens," you'll need a custom loss that masks out the thought tokens when comparing the Student (MoE) to the Teacher (Dense).

#### 4. `scripts/train_distill.py` (The Execution)

This is your entry point for training. It should:

1. Load the Teacher (Qwen3-14B).
2. Load the Student (Your Upcycled MoE).
3. Load the data.
4. Run the forward passes and calculate `Loss = Alpha * Task_Loss + Beta * CKA_Loss`.


---

### Contact & Contribution

This project is currently in the **Architecture Validation** phase. If you are interested in the intersection of MoE upcycling and Indic-language LLMs for social good, please reach out.



---

### files:
File,Brief Description,Key Role in Thesis
train_distill.py
Main execution engine for simultaneous distillation of Teacher (14B) and Student (0.6B)
Orchestrates CKA manifold alignment and Expert Diversity loss.

agri_data.py
Custom Data Loader for Parquet files handling Scenario → Thinking → Advisory formatting.
"Forces the model to learn reasoning by training on specific ""Thinking"" tokens."

litgpt/model.py
Modified LitGPT core with hooks for hidden states and MoE routing decisions.
"Provides the structural ""hooks"" needed for CKA calculation and expert monitoring."

litgpt/config.py
Architectural registry for the upcycled 8-expert Qwen3-MoE configuration.
Ensures model weights map correctly to the new sparse 0.6B-MoE structure.

litgpt/tokenizer.py
Tiktoken-based tokenization logic optimized for Qwen3 and Devanagari script.
Correctly processes complex Hindi agricultural terminology.

litgpt/utils.py
Utility suite including chunked cross-entropy and parameter counting.
Manages memory spikes on H100 and verifies model size metrics.

scripts/upcycle_moe.py
Weight surgery script mapping 1 dense MLP to 8 sparse experts with jitter.
"Performs the initialization ""surgery"" for the sparse Student model."

scripts/check_weights.py
Audit utility for verifying layer shapes and router symmetry post-surgery.
Confirms the Student model is structurally ready for distillation.

scripts/convert_hf_checkpoint.py
Bidirectional weight mapper for Hugging Face to LitGPT formats.
Enables loading of pre-trained Qwen3 weights into the research engine.

scripts/convert_lit_checkpoint.py
Utility to convert distilled Student models back to Hugging Face format.
Prepares the final model for standard NLP evaluation benchmarks.

scripts/sanity_check_data.py
Pre-flight script to decode and verify Hindi CoT data formatting.
Ensures reasoning and advice boundaries are correctly tokenized.

launch_train.sh
H100 shell launcher setting PYTHONPATH and TensorFloat-32 optimizations.
Ensures the server processes distillation at peak hardware efficiency.

plot_specialization.py
Visualization tool converting CKA matrices into heatmaps.
Provides the visual proof of Expert Specialization for the thesis defense.


---

### Commands:

* conda activate proj_agri_iitr

* to run as package:
pip install -e .


* Surgery: Run 
    python -m scripts.upcycle_moe


* Verification: Run 
    python -m scripts.check_weights 
    and 
    python -m scripts.sanity_check_data

* interactive GPU access
srun --partition=debug --gres=gpu:1 --time=01:00:00 --pty bash

* To check GPU load
nvidia-smi
* update VRAM info every 1s
watch -n 1 nvidia-smi

* Start Training: Run 
    ./runners/launch_train.sh 
    within a tmux session 

        Screen & Tmux: Why and How
        When you run a long training job (like distillation) over SSH, if your internet flickers or your laptop goes to sleep, the SSH session dies—and so does your training.

        Tmux (Terminal Multiplexer) creates a session on the server side. Even if you disconnect, the code keeps running.

        How to use it:

        The Workflow:
        Start a new session: Type tmux and press Enter. You will see a green bar at the bottom of your terminal.
        Run your code: Inside this session, 
            run ./launch_train.sh.
        Foolproof Way to Monitor : 
            Press Ctrl + B, release, then press " (Double Quote key). This splits your screen horizontally.
            watch -n 1 nvidia-smi
        Detach (Leave it running): 
            Press Ctrl + B, release both keys, and then press D. You can now safely close your terminal or laptop.
        Re-attach (Check progress): 
            When you log back in later, type tmux attach to return to your running session.

* to submit slurm
sbatch runners/run_distill.slurm

*cancel job
scancel <JOB_ID>

* to check progress
squeue -u anurag

* check last log
cat logs/train_22408.err

* Watch the Live Progress Bar To see the tqdm bar and loss values updating in real-time (just like in tmux), "tail" the output file:

    Bash
    # Replace 'train_22365.out' with your actual filename from the logs folder
    tail -f logs/train_22365.out
    You will see the training lines appearing as they happen.

    Press Ctrl + C to stop watching (this does not stop the training; it just closes the view).


* Resume if Time Runs Out
    If the 4-hour limit hits and the job dies.

    Check the logs: tail -n 20 logs/train_22365.out to see the last step saved.

    Submit again: sbatch run_distill.slurm.

    Automatic Resume: Because your train_distill.py loads the weights from student_path, ensure you update the student_path in your python script to point to the latest checkpoint (e.g., checkpoints/.../step-2000.pth) before submitting the new job.

    Pro Tip: A smarter script would automatically look for the latest checkpoint, but manually updating the path takes 10 seconds.


* 
python -m inference.test_inference

* 
python plot_results.py

* check folder storage
du -sh checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/*

---


### Chat format structure
<|system|>
{system_instruction}
<|user|>
स्थिति (Scenario):
{Formatted Metadata from Prompt}
<|thought|>
{thoughts}
<|assistant|>
{advisory}
<|endoftext|>

---


### Files:

config.py - Defines the 40-layer Teacher and 28-layer MoE Student specs.
model.py - The logic for Attention and MoE blocks; now modified to return hidden states and expert data.
download.py - Downloads raw weights from Hugging Face.
convert_hf.py - Converts Hugging Face naming conventions to your local litgpt format.
upcycle_moe.py - Clones dense MLP weights into 8 (or 2) experts with symmetry-breaking jitter.
tokenizer.py - Converts your Hindi/English agricultural text into numerical IDs.
train_distill.py - The Main Command Center. Runs the White-Box distillation loop.



---
### Check file size

* individually break down each file size
du -sh *




### References
* your current $1e-5$ approach is the standard baseline used in the original Sparse Upcycling paper (Komatsuzaki et al.).

* for eval: 
    Reference Paper: Feng et al. (2020), "Language-agnostic BERT Sentence Embedding" (Published by Google Research).
        LaBSE: Tests semantic meaning (Did it say the same thing, even if it used different words?).
    
    Reference Paper: Popović (2015), "chrF: character n-gram F-score for automatic MT evaluation" (Standard in WMT - Workshop on Machine Translation).
        chrF++: Tests exact grammar and character-level spelling.
    
    Reference Paper: Zheng et al. (2023), "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (Published by LMSYS / UC Berkeley).
        LLM-as-a-Judge: Tests human vibes (Tone, Safety, Logic).



    three advanced extensions of this paper that researchers now use:
        1️⃣ Kernel CKA vs Linear CKA differences
        2️⃣ CKA for Transformers and attention heads
        3️⃣ How OpenAI / DeepMind analyze LLM layers using CKA


    Minibatch CKA
    https://arxiv.org/html/2405.08657v1


### Tip for your Defense
During your presentation, you can show a comparison of the Student's attention maps.

Before Distillation: The attention is scattered.

After CKA Distillation: The attention in the advisory section will show strong "activation peaks" back to the relevant parts of the thoughts section. This is visual proof that your model is actually "using" its internal thoughts.


H100 Health: Keep a separate tmux pane open with watch -n 1 nvidia-smi. Ensure your VRAM usage is stable around 40-50GB.