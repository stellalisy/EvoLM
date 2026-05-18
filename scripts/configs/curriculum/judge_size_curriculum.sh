# Judge Size Curriculum Configuration
# Start with larger (stronger) judge models and progressively use smaller ones
# This implements "easy to hard" curriculum for rubric training

# Curriculum models from largest to smallest
JUDGE_SIZE_CURRICULUM=${JUDGE_SIZE_CURRICULUM:-"Qwen/Qwen3-32B,Qwen/Qwen3-14B,Qwen/Qwen3-8B,Qwen/Qwen3-4B"}

# Schedule: 10 cycles total (K=50, 1000 steps), split 3-3-2-2 across models
# Cycles 0-2 (steps 1-300): 32B, Cycles 3-5 (steps 301-600): 14B,
# Cycles 6-7 (steps 601-800): 8B, Cycles 8-9 (steps 801-1000): 4B
JUDGE_CURRICULUM_SCHEDULE=${JUDGE_CURRICULUM_SCHEDULE:-"0,3,6,8"}

# Initial model = first (largest) in curriculum
RUBRIC_JUDGE_MODEL="Qwen/Qwen3-32B"

# TP=2 so all models fit (32B needs ~32GB/GPU at TP=2, fits H100 80GB).
# 4 engines x TP=2 = 8 GPUs for judge (reduced from 8 engines to avoid OOM).
# Shared policy/rubric engines reduced to 32 to fit within 64 GPU budget:
# 32 shared (TP=1) + 8 judge + 8 training = 48 active (16 idle/buffer).
RUBRIC_JUDGE_TENSOR_PARALLEL_SIZE=2
RUBRIC_JUDGE_NUM_ENGINES=4
RUBRIC_JUDGE_GPU_MEMORY_UTILIZATION=0.5
RUBRIC_JUDGE_MAX_MODEL_LEN=16384
VLLM_GPU_MEMORY_UTILIZATION=0.45
