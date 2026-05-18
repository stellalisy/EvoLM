#!/bin/bash
# Launch sharded evaluation for a single (experiment, step, task) combination.
#
# Usage:
#   ./launch_sharded_eval.sh <exp_name> <checkpoint_dir> <output_dir> <step> <task> <num_shards>
#
# Example:
#   ./launch_sharded_eval.sh v3_mj5_maj \
#       /path/to/checkpoints/policy \
#       /path/to/olmes_eval \
#       950 "popqa::olmo3:adapt" 16
#
# This will:
#   1. Seed each shard directory with existing checkpoint progress
#   2. Submit num_shards SLURM jobs, each processing ~1/N of the data
#   3. Print a merge command to run after all shards complete

set -e

# Project root: defaults to two levels above this script's directory.
PROJ_ROOT="${PROJ_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"

EXP_NAME=${1:?Error: exp_name required}
CHECKPOINT_DIR=${2:?Error: checkpoint_dir required}
OUTPUT_DIR=${3:?Error: output_dir required}
STEP=${4:?Error: step required}
TASK=${5:?Error: task required}
NUM_SHARDS=${6:?Error: num_shards required}

EVAL_QOS=${EVAL_QOS:-h100_sage_high}
GPUS_PER_SHARD=${GPUS_PER_SHARD:-1}
OLD_NUM_SHARDS=${OLD_NUM_SHARDS:-0}

TASK_STEM=$(echo "$TASK" | cut -d: -f1)
# Allow overriding the checkpoint file name stem when it differs from the task stem
CKPT_TASK_STEM=${CKPT_TASK_STEM:-$TASK_STEM}

MERGE_SCRIPT="${PROJ_ROOT}/scripts/eval/merge_eval_shards.py"
RUNNER_SCRIPT="${PROJ_ROOT}/scripts/eval/run_olmes_on_checkpoints.py"
LOG_DIR="${PROJ_ROOT}/logs/eval/sharded/${EXP_NAME}"
mkdir -p "$LOG_DIR"

echo "=== Sharded eval: ${EXP_NAME} / step ${STEP} / ${TASK_STEM} x${NUM_SHARDS} ==="

# Step 0 (optional): Gather checkpoint progress from old shards into central file
if [ "$OLD_NUM_SHARDS" -gt 0 ]; then
    echo "Gathering checkpoints from ${OLD_NUM_SHARDS} old shards..."
    python3 "$MERGE_SCRIPT" \
        --output-root "$OUTPUT_DIR" \
        --step "$STEP" \
        --task-name "$CKPT_TASK_STEM" \
        --num-shards "$NUM_SHARDS" \
        --old-num-shards "$OLD_NUM_SHARDS" \
        --gather-checkpoints
fi

# Step 1: Seed shard directories with existing checkpoint progress
echo "Seeding shard checkpoints from existing progress..."
python3 "$MERGE_SCRIPT" \
    --output-root "$OUTPUT_DIR" \
    --step "$STEP" \
    --task-name "$CKPT_TASK_STEM" \
    --num-shards "$NUM_SHARDS" \
    --seed-checkpoints

# Step 2: Submit one SLURM job per shard
SUBMITTED=0
for ((SID=0; SID<NUM_SHARDS; SID++)); do
    SHARD_OUTPUT="${OUTPUT_DIR}/eval_shards/${CKPT_TASK_STEM}/shard_${SID}"
    JOB_NAME="sh_${EXP_NAME}_${CKPT_TASK_STEM}_${SID}of${NUM_SHARDS}"
    JOB_NAME="${JOB_NAME:0:50}"

    # Skip if this shard already has a metrics file
    METRICS_GLOB="${SHARD_OUTPUT}/step_${STEP}/*-${CKPT_TASK_STEM}-metrics.json"
    if ls $METRICS_GLOB 1>/dev/null 2>&1; then
        echo "  Shard ${SID}: already complete, skipping"
        continue
    fi

    JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=sage
#SBATCH --qos=${EVAL_QOS}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --chdir=${PROJ_ROOT}
#SBATCH --export=all
#SBATCH --gres=gpu:${GPUS_PER_SHARD}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}-%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}-%j.err

source ${PROJ_ROOT}/olmes/.venv/bin/activate

export HF_HOME=\${HF_HOME:-\$HOME/.cache/huggingface}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_\${SLURM_JOB_ID}
export VLLM_CACHE_ROOT=/tmp/vllm_cache_\${SLURM_JOB_ID}

cd ${PROJ_ROOT}/olmes

python "${RUNNER_SCRIPT}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --output-dir "${SHARD_OUTPUT}" \
    --task "${TASK}" \
    --steps ${STEP} \
    --model-type vllm \
    --trust-remote-code \
    --max-length 16384 \
    --max-gen-toks 16384 \
    --gpus ${GPUS_PER_SHARD} \
    --shard-id ${SID} \
    --num-shards ${NUM_SHARDS}

echo "Done: shard ${SID}/${NUM_SHARDS} of ${TASK_STEM}"
EOF
)
    echo "  Shard ${SID}: Job ${JOB_ID}"
    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "Submitted ${SUBMITTED}/${NUM_SHARDS} shard jobs."
echo ""
echo "After all complete, merge with:"
echo "  python3 ${MERGE_SCRIPT} --output-root ${OUTPUT_DIR} --step ${STEP} --task-name ${TASK_STEM} --num-shards ${NUM_SHARDS}"
echo ""
echo "Monitor: squeue -u \$USER | grep sh_${EXP_NAME}_${TASK_STEM}"
