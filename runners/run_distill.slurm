#!/bin/bash
#SBATCH --job-name=qwen
#SBATCH --partition=translation
#SBATCH --gres=gpu:1
#SBATCH --time=45:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# 1. Load Environment
# We use the direct hook to ensure Conda works in non-interactive batch mode
# Ensure this path matches your system's conda location:
eval "$(/cm/shared/apps/miniconda3/bin/conda shell.bash hook)"

conda activate proj_agri_iitr

# Debugging: Print python path to logs to confirm it worked
echo "Using Python: $(which python)"

# 2. Navigate to Project Root
cd ~/Geometry_Distilled_qwen3_moe_crop_advisory_hi/

# 3. Optimizations
export PYTHONPATH=$(pwd)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 4. Run the Training
python -u train_distill.py