#!/bin/bash
# Submit a RewardBench 2 evaluation job via SLURM
#
# Usage:
#   ./eval_rewardbench2.sh <exp_name> <rubric_generator> <judge_model> [num_gpus] [vllm_gpu_util] [score_sample_batch_size] [subset_filter] [disable_per_criteria]
#
# Examples:
#   bash stella_run_scripts/paper_experiments/eval/eval_rewardbench2.sh \
#       "main_s950" \
#       "/path/to/checkpoints/step_950" \
#       "Qwen/Qwen3-1.7B"

set -e

# Project root: defaults to two levels above this script's directory.
PROJ_ROOT="${PROJ_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"

# Prefix used for RB2 SLURM job names and log directory names.
# Example: EVAL_PREFIX=v3 -> job "v3_rb2_<exp>_<subset>" and logs under log_v3/.
EVAL_PREFIX=${EVAL_PREFIX:-v3}

EXP_NAME=${1:?Error: exp_name required}
RUBRIC_GENERATOR=${2:?Error: rubric_generator path or model name required}
JUDGE_MODEL=${3:?Error: judge model path or model name required}
NUM_GPUS=${4:-2}
VLLM_GPU_UTIL=${5:-0.45}
SCORE_SAMPLE_BATCH_SIZE=${6:-16}
SINGLE_SUBSET_FILTER=${7:-}
DISABLE_PER_CRITERIA=${8:-1}
QUESTION_ONLY_RUBRIC=${9:-0}
RUBRIC_PROMPT_KEY=${10:-}  # e.g. "rubric_generation_v3" to match V3 training prompt

# Sharding: set RB2_NUM_SHARDS>1 to parallelize rubric generation across N jobs per subset.
RB2_NUM_SHARDS=${RB2_NUM_SHARDS:-1}

LOG_DIR="${PROJ_ROOT}/logs/eval/rb2_batch"
mkdir -p "${LOG_DIR}"

echo "Submitting RB2 eval(s): ${EXP_NAME}"
echo "  Rubric generator: ${RUBRIC_GENERATOR}"
echo "  Judge model: ${JUDGE_MODEL}"
echo "  GPUs: ${NUM_GPUS}"
echo "  vLLM GPU util: ${VLLM_GPU_UTIL}"
echo "  score micro-batch size: ${SCORE_SAMPLE_BATCH_SIZE}"
echo "  disable per-criteria mode: ${DISABLE_PER_CRITERIA}"
echo "  question-only rubric: ${QUESTION_ONLY_RUBRIC}"
echo "  rubric prompt key: ${RUBRIC_PROMPT_KEY:-default}"
echo "  eval prefix: ${EVAL_PREFIX}"

DISABLE_PER_CRITERIA_FLAG=""
RB2_SUBDIR="rb2_results"
if [ "${DISABLE_PER_CRITERIA}" = "1" ]; then
    DISABLE_PER_CRITERIA_FLAG="--disable_per_criteria"
    RB2_SUBDIR="rb2_results_v2"
fi

QUESTION_ONLY_FLAG=""
if [ "${QUESTION_ONLY_RUBRIC}" = "1" ]; then
    QUESTION_ONLY_FLAG="--question_only_rubric"
    RB2_SUBDIR="${RB2_SUBDIR}_qonly"
fi

RUBRIC_PROMPT_KEY_FLAG=""
if [ -n "${RUBRIC_PROMPT_KEY}" ]; then
    RUBRIC_PROMPT_KEY_FLAG="--rubric_prompt_key ${RUBRIC_PROMPT_KEY}"
    RB2_SUBDIR="${RB2_SUBDIR}_${RUBRIC_PROMPT_KEY}"
fi

if [ -n "${SINGLE_SUBSET_FILTER}" ]; then
    SUBSETS=("${SINGLE_SUBSET_FILTER}")
else
    SUBSETS=("Factuality" "Focus" "Math" "Precise IF" "Safety" "Ties")
fi

for SUBSET_FILTER in "${SUBSETS[@]}"; do
SUBSET_SLUG=$(echo "${SUBSET_FILTER}" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | tr -cd 'a-z0-9_')

for SHARD_ID in $(seq 0 $((RB2_NUM_SHARDS - 1))); do

if [ "${RB2_NUM_SHARDS}" -gt 1 ]; then
    JOB_NAME="${EVAL_PREFIX}_rb2_${EXP_NAME}_${SUBSET_SLUG}_s${SHARD_ID}"
    SHARD_SUFFIX="subset_${SUBSET_SLUG}_shard${SHARD_ID}of${RB2_NUM_SHARDS}"
    SHARD_FLAGS="--shard_id ${SHARD_ID} --num_shards ${RB2_NUM_SHARDS}"
else
    JOB_NAME="${EVAL_PREFIX}_rb2_${EXP_NAME}_${SUBSET_SLUG}"
    SHARD_SUFFIX="subset_${SUBSET_SLUG}"
    SHARD_FLAGS=""
fi
JOB_NAME="${JOB_NAME:0:50}"

echo "  Subset: ${SUBSET_FILTER}  Shard: ${SHARD_ID}/${RB2_NUM_SHARDS}"

SBATCH_SCRIPT=$(mktemp)
cat > "${SBATCH_SCRIPT}" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=sage
#SBATCH --qos=h100_sage_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --chdir=${PROJ_ROOT}
#SBATCH --export=all
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}-%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}-%j.err

set -euo pipefail

# Unset Claude Code proxy vars that --export=all would forward to compute nodes
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

export HF_HOME=\${HF_HOME:-\$HOME/.cache/huggingface}

# Configure API credentials (needed for GPT-4.1 rubric generator)
source ${PROJ_ROOT}/scripts/configure_api.sh gpt-4.1 2>/dev/null || true

echo "========================================"
echo "RewardBench 2 Evaluation"
echo "========================================"
echo "Experiment: ${EXP_NAME}"
echo "Subset filter: ${SUBSET_FILTER}"
echo "Shard: ${SHARD_ID}/${RB2_NUM_SHARDS}"
echo "Rubric generator: ${RUBRIC_GENERATOR}"
echo "Judge model: ${JUDGE_MODEL}"
echo "GPUs: ${NUM_GPUS}"
echo "vLLM GPU util: ${VLLM_GPU_UTIL}"
echo "score micro-batch size: ${SCORE_SAMPLE_BATCH_SIZE}"
echo "========================================"

# Save local JSON outputs into the rubric checkpoint directory when available.
# Use rb2_results_v2 when disable_per_criteria is active to avoid overwriting old per-item results.
if [ -d "${RUBRIC_GENERATOR}" ]; then
    export REWARDBENCH_LOCAL_RESULTS_DIR="${RUBRIC_GENERATOR}/${RB2_SUBDIR}"
else
    export REWARDBENCH_LOCAL_RESULTS_DIR="${PROJ_ROOT}/reward-bench/results_${EXP_NAME}"
fi
echo "Local RB2 JSON output dir: ${REWARDBENCH_LOCAL_RESULTS_DIR}"

cd ${PROJ_ROOT}/reward-bench
source .venv/bin/activate

# Ensure local rewardbench package + deps are importable in this venv.
# This prevents silent failures from partial/old environments.
if ! python - <<'PY'
import rewardbench  # noqa: F401
import transformers  # noqa: F401
PY
then
    python -m pip install -e ".[api,vllm]"
fi

python scripts/run_generative_v2_rubric.py \\
    --rubric_generator="${RUBRIC_GENERATOR}" \\
    --model="${JUDGE_MODEL}" \\
    --subset_filter="${SUBSET_FILTER}" \\
    --result_suffix="${SHARD_SUFFIX}" \\
    --trust_remote_code \\
    --vllm_gpu_util "${VLLM_GPU_UTIL}" \\
    --score_sample_batch_size ${SCORE_SAMPLE_BATCH_SIZE} \\
    --do_not_save \\
    --disable_beaker_save \\
    --num_gpus ${NUM_GPUS} \\
    ${DISABLE_PER_CRITERIA_FLAG} \\
    ${QUESTION_ONLY_FLAG} \\
    ${RUBRIC_PROMPT_KEY_FLAG} \\
    ${SHARD_FLAGS}

echo "Done: ${EXP_NAME} (${SUBSET_FILTER}) shard ${SHARD_ID}/${RB2_NUM_SHARDS}"
EOF

JOB_ID=$(sbatch ${SBATCH_DEPENDENCY:+--dependency=${SBATCH_DEPENDENCY}} "${SBATCH_SCRIPT}" | awk '{print $4}')
echo "    Job ${JOB_ID}"

rm "${SBATCH_SCRIPT}"
done
done
