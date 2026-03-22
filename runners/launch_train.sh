# Project Root and Python Path
# ensures that modified litgpt/model.py is used instead of any installed version
export PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# H100 Specific Optimizations
# Force use of TensorFloat-32 for faster matmuls on H100
export TORCH_CUDNN_V8_API_ENABLED=1
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO

# --- NEW: Fragmentation Monitoring & Prevention ---
# This tells PyTorch to use a 'round_up' strategy for small allocations (prevents gaps)
# and set the max_split_size_mb to 512 to reduce fragmentation.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# 3. Model and Data Paths
STUDENT_PATH="checkpoints/Qwen/Qwen3-0.6B-moe-init/lit_model.pth"
TEACHER_PATH="checkpoints/Qwen/Qwen3-8B/lit_model.pth"
DATA_PATH="data/train_bilingual_mixed_83k_agri65k.parquet"

# 4. Launch Training
# Using 'torchrun' is best practice for H100 to handle process monitoring
echo "Starting CKA-Guided MoE Distillation on H100..."

python train_distill.py \
    --student_path $STUDENT_PATH \
    --teacher_path $TEACHER_PATH \
    --data_path $DATA_PATH \
    --batch_size 2 \
    --max_seq_length 4096 \
    

# 5. Instructions
# To run this script:
# chmod +x launch_train.sh
# ./launch_train.sh