#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=rl
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# =========================================================
# 1. Load Conda Environment
# =========================================================

eval "$(/cm/shared/apps/miniconda3/bin/conda shell.bash hook)"
conda activate proj_agri_iitr

echo "Using Python: $(which python)"

# =========================================================
# 2. Move to Project Root
# =========================================================

cd ~/Geometry_Distilled_qwen3_moe_crop_advisory_hi/

# =========================================================
# 3. Runtime Optimizations
# =========================================================

export PYTHONPATH=$(pwd)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Prevent tokenizer multi-thread warning
export TOKENIZERS_PARALLELISM=false

# =========================================================
# 4. Debug Info (printed in logs)
# =========================================================

echo "========== JOB INFO =========="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi
echo "================================"

# =========================================================
# 5. Run Evaluation
# =========================================================

python -u evaluation/evaluation.py

echo "Evaluation finished."