#!/bin/bash
# Submit a JudgeBench evaluation job via SLURM.
#
# Launches two vLLM servers (rubric generator + judge) when both are local models,
# then runs JudgeBench's run_judge.py with the rubric judge.
#
# Usage:
#   ./eval_judgebench.sh <exp_name> <rubric_generator> <judge_model> [num_gpus] [vllm_gpu_util] [concurrency_limit]
#
# Examples:
#   # Trained rubric generator + Qwen3-1.7B judge
#   bash stella_run_scripts/paper_experiments/eval/eval_judgebench.sh \
#       "main_s1000" \
#       "/checkpoint/.../step_1000" \
#       "Qwen/Qwen3-1.7B"
#
#   # Prompted GPT-4.1 rubric generator (API) + Qwen3-1.7B judge (local)
#   bash stella_run_scripts/paper_experiments/eval/eval_judgebench.sh \
#       "prompted_gpt41" \
#       "gpt-4.1-2025-04-14" \
#       "Qwen/Qwen3-1.7B"

set -e

# Project root: defaults to two levels above this script's directory.
PROJ_ROOT="${PROJ_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"

# Prefix used for job names and log directory names.
# Example: EVAL_PREFIX=v3 -> job "v3_jb_<exp>" and logs under log_v3/.
EVAL_PREFIX=${EVAL_PREFIX:-v3}

EXP_NAME=${1:?Error: exp_name required}
RUBRIC_GENERATOR=${2:?Error: rubric_generator path or model name required}
JUDGE_MODEL=${3:?Error: judge model path or model name required}
NUM_GPUS=${4:-2}
VLLM_GPU_UTIL=${5:-0.9}
CONCURRENCY_LIMIT=${6:-10}
RUBRIC_PROMPT_KEY=${7:-}  # e.g. "rubric_generation_v3" to match V3 training prompt

# Health-check defaults for run_judge.py (override via environment if needed).
HEALTH_ENFORCE=${HEALTH_ENFORCE:-1}
HEALTH_MAX_FAILED_SCORE_RATE=${HEALTH_MAX_FAILED_SCORE_RATE:-0.02}
HEALTH_MAX_PAIR_FAILURE_RATE=${HEALTH_MAX_PAIR_FAILURE_RATE:-0.10}
HEALTH_MAX_NULL_DECISION_RATE=${HEALTH_MAX_NULL_DECISION_RATE:-0.05}

RUBRIC_PORT=8000
JUDGE_PORT=8001

# Ports are overridden inside the SLURM job using SLURM_JOB_ID to avoid
# conflicts when multiple eval jobs land on the same node.

LOG_DIR="${PROJ_ROOT}/logs/eval/jb"
JB_DIR="${PROJ_ROOT}/JudgeBench"
PAIRS_FILE="data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl"

mkdir -p "${LOG_DIR}"

JOB_NAME="${EVAL_PREFIX}_jb_${EXP_NAME}"
JOB_NAME="${JOB_NAME:0:50}"

echo "Submitting JudgeBench eval: ${EXP_NAME}"
echo "  Rubric generator: ${RUBRIC_GENERATOR}"
echo "  Judge model: ${JUDGE_MODEL}"
echo "  GPUs: ${NUM_GPUS}"
echo "  vLLM GPU util: ${VLLM_GPU_UTIL}"
echo "  Concurrency limit: ${CONCURRENCY_LIMIT}"
echo "  Rubric prompt key: ${RUBRIC_PROMPT_KEY:-default}"
echo "  Eval prefix: ${EVAL_PREFIX}"
echo "  Health enforce: ${HEALTH_ENFORCE}"
echo "  Health max failed score rate: ${HEALTH_MAX_FAILED_SCORE_RATE}"
echo "  Health max pair failure rate: ${HEALTH_MAX_PAIR_FAILURE_RATE}"
echo "  Health max null decision rate: ${HEALTH_MAX_NULL_DECISION_RATE}"

# Detect whether rubric generator is an API model (not a local path / HF model).
is_api_model() {
    local m="$1"
    [[ "$m" == gpt-* ]] || [[ "$m" == o1-* ]] || [[ "$m" == claude-* ]] || [[ "$m" == gemini-* ]]
}

SBATCH_SCRIPT=$(mktemp)
cat > "${SBATCH_SCRIPT}" << 'OUTER_EOF'
#!/bin/bash
#SBATCH --job-name=PLACEHOLDER_JOB_NAME
#SBATCH --account=sage
#SBATCH --qos=h100_sage_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --chdir=PLACEHOLDER_PROJ_ROOT
#SBATCH --export=all
#SBATCH --gres=gpu:PLACEHOLDER_NUM_GPUS
#SBATCH --output=PLACEHOLDER_LOG_DIR/PLACEHOLDER_JOB_NAME-%j.out
#SBATCH --error=PLACEHOLDER_LOG_DIR/PLACEHOLDER_JOB_NAME-%j.err

set -euo pipefail

# Unset Claude Code proxy vars that --export=all would forward to compute nodes
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

# Configure API credentials (needed for GPT-4.1 rubric generator)
source PLACEHOLDER_PROJ_ROOT/scripts/configure_api.sh gpt-4.1 2>/dev/null || true

echo "========================================"
echo "JudgeBench Evaluation"
echo "========================================"
echo "Experiment: PLACEHOLDER_EXP_NAME"
echo "Rubric generator: PLACEHOLDER_RUBRIC_GENERATOR"
echo "Judge model: PLACEHOLDER_JUDGE_MODEL"
echo "GPUs: PLACEHOLDER_NUM_GPUS"
echo "vLLM GPU util: PLACEHOLDER_VLLM_GPU_UTIL"
echo "Concurrency: PLACEHOLDER_CONCURRENCY_LIMIT"
echo "========================================"

PROJ_ROOT="PLACEHOLDER_PROJ_ROOT"
JB_DIR="${PROJ_ROOT}/JudgeBench"
RUBRIC_GENERATOR="PLACEHOLDER_RUBRIC_GENERATOR"
JUDGE_MODEL="PLACEHOLDER_JUDGE_MODEL"
NUM_GPUS=PLACEHOLDER_NUM_GPUS
VLLM_GPU_UTIL="PLACEHOLDER_VLLM_GPU_UTIL"
CONCURRENCY_LIMIT=PLACEHOLDER_CONCURRENCY_LIMIT
PAIRS_FILE="PLACEHOLDER_PAIRS_FILE"
RUBRIC_PROMPT_KEY="PLACEHOLDER_RUBRIC_PROMPT_KEY"
HEALTH_ENFORCE="${HEALTH_ENFORCE:-1}"
HEALTH_MAX_FAILED_SCORE_RATE="${HEALTH_MAX_FAILED_SCORE_RATE:-0.02}"
HEALTH_MAX_PAIR_FAILURE_RATE="${HEALTH_MAX_PAIR_FAILURE_RATE:-0.10}"
HEALTH_MAX_NULL_DECISION_RATE="${HEALTH_MAX_NULL_DECISION_RATE:-0.05}"

# Derive unique ports from SLURM_JOB_ID so multiple eval jobs on the same node
# don't collide on ports 8000/8001.
PORT_OFFSET=$(( (SLURM_JOB_ID % 500) * 2 ))
RUBRIC_PORT=$(( 10000 + PORT_OFFSET ))
JUDGE_PORT=$(( RUBRIC_PORT + 1 ))
echo "Derived ports from SLURM_JOB_ID=${SLURM_JOB_ID}: RUBRIC_PORT=${RUBRIC_PORT}, JUDGE_PORT=${JUDGE_PORT}"

# Make rewardbench importable (for RubricJudge's rubric logic).
export PYTHONPATH="${PROJ_ROOT}/reward-bench:${PYTHONPATH:-}"

# Activate reward-bench venv and ensure dependencies.
cd "${PROJ_ROOT}/reward-bench"
source .venv/bin/activate

# Ensure JudgeBench dependencies are installed via uv (venv has no pip).
python -c "import backoff" 2>/dev/null || uv pip install backoff

# Determine available GPUs and split them between two vLLM servers.
ALL_GPUS=$(python -c "import os; print(os.environ.get('CUDA_VISIBLE_DEVICES', ','.join(str(i) for i in range(${NUM_GPUS}))))")
IFS=',' read -ra GPU_LIST <<< "${ALL_GPUS}"
N_TOTAL=${#GPU_LIST[@]}
N_HALF=$(( N_TOTAL / 2 ))
if [ "${N_HALF}" -lt 1 ]; then N_HALF=1; fi

RUBRIC_GPUS=$(IFS=,; echo "${GPU_LIST[*]:0:${N_HALF}}")
JUDGE_GPUS=$(IFS=,; echo "${GPU_LIST[*]:${N_HALF}}")
N_RUBRIC_GPUS=${N_HALF}
N_JUDGE_GPUS=$(( N_TOTAL - N_HALF ))

cleanup() {
    echo "Cleaning up vLLM servers..."
    [ -n "${RUBRIC_PID:-}" ] && kill "${RUBRIC_PID}" 2>/dev/null || true
    [ -n "${JUDGE_PID:-}" ] && kill "${JUDGE_PID}" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

is_api_model() {
    local m="$1"
    [[ "$m" == gpt-* ]] || [[ "$m" == o1-* ]] || [[ "$m" == claude-* ]] || [[ "$m" == gemini-* ]]
}

wait_for_server() {
    local port=$1 name=$2 max_wait=1800
    echo "Waiting for ${name} vLLM server on port ${port} (timeout ${max_wait}s)..."
    for i in $(seq 1 ${max_wait}); do
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "${name} server ready after ${i}s."
            return 0
        fi
        sleep 1
    done
    echo "ERROR: ${name} server on port ${port} did not start within ${max_wait}s."
    exit 1
}

RUBRIC_PID=""
JUDGE_PID=""

# Launch rubric generator vLLM server (skip for API models).
if ! is_api_model "${RUBRIC_GENERATOR}"; then
    echo "Starting rubric generator vLLM server on port ${RUBRIC_PORT} with GPUs: ${RUBRIC_GPUS}"
    CUDA_VISIBLE_DEVICES="${RUBRIC_GPUS}" python -m vllm.entrypoints.openai.api_server \
        --model "${RUBRIC_GENERATOR}" \
        --port "${RUBRIC_PORT}" \
        --tensor-parallel-size "${N_RUBRIC_GPUS}" \
        --gpu-memory-utilization "${VLLM_GPU_UTIL}" \
        --trust-remote-code \
        --max-model-len 32768 \
        --generation-config vllm &
    RUBRIC_PID=$!
    wait_for_server "${RUBRIC_PORT}" "rubric generator"
fi

# Launch judge vLLM server.
if ! is_api_model "${JUDGE_MODEL}"; then
    # If rubric gen is API, give all GPUs to judge.
    if is_api_model "${RUBRIC_GENERATOR}"; then
        JUDGE_GPUS="${ALL_GPUS}"
        N_JUDGE_GPUS="${N_TOTAL}"
    fi
    echo "Starting judge vLLM server on port ${JUDGE_PORT} with GPUs: ${JUDGE_GPUS}"
    CUDA_VISIBLE_DEVICES="${JUDGE_GPUS}" python -m vllm.entrypoints.openai.api_server \
        --model "${JUDGE_MODEL}" \
        --port "${JUDGE_PORT}" \
        --tensor-parallel-size "${N_JUDGE_GPUS}" \
        --gpu-memory-utilization "${VLLM_GPU_UTIL}" \
        --trust-remote-code \
        --max-model-len 32768 &
    JUDGE_PID=$!
    wait_for_server "${JUDGE_PORT}" "judge"
fi

# Output results next to rubric checkpoint when available.
# Append prompt key to output dir to avoid clobbering results from different prompts.
JB_SUBDIR="jb_results"
if [ -n "${RUBRIC_PROMPT_KEY}" ]; then
    JB_SUBDIR="jb_results_${RUBRIC_PROMPT_KEY}"
fi
if [ -d "${RUBRIC_GENERATOR}" ]; then
    OUTPUT_DIR="${RUBRIC_GENERATOR}/${JB_SUBDIR}"
else
    OUTPUT_DIR="${JB_DIR}/outputs"
fi

echo "Output dir: ${OUTPUT_DIR}"

cd "${JB_DIR}"

RUBRIC_PROMPT_KEY_ARG=""
if [ -n "${RUBRIC_PROMPT_KEY}" ]; then
    RUBRIC_PROMPT_KEY_ARG="--rubric_prompt_key ${RUBRIC_PROMPT_KEY}"
    echo "Using rubric prompt key: ${RUBRIC_PROMPT_KEY}"
fi

HEALTH_ENFORCE_ARG=""
if [ "${HEALTH_ENFORCE}" = "1" ]; then
    HEALTH_ENFORCE_ARG="--health_enforce"
fi

python run_judge.py \
    --judge_name rubric \
    --judge_model "${JUDGE_MODEL}" \
    --rubric_model "${RUBRIC_GENERATOR}" \
    --rubric_port "${RUBRIC_PORT}" \
    --judge_port "${JUDGE_PORT}" \
    --pairs "${PAIRS_FILE}" \
    --concurrency_limit "${CONCURRENCY_LIMIT}" \
    --output_dir "${OUTPUT_DIR}" \
    --health_max_failed_score_rate "${HEALTH_MAX_FAILED_SCORE_RATE}" \
    --health_max_pair_failure_rate "${HEALTH_MAX_PAIR_FAILURE_RATE}" \
    --health_max_null_decision_rate "${HEALTH_MAX_NULL_DECISION_RATE}" \
    ${HEALTH_ENFORCE_ARG} \
    ${RUBRIC_PROMPT_KEY_ARG}

echo "Done: PLACEHOLDER_EXP_NAME"
OUTER_EOF

# Replace placeholders with actual values.
sed -i "s|PLACEHOLDER_JOB_NAME|${JOB_NAME}|g" "${SBATCH_SCRIPT}"
sed -i "s|PLACEHOLDER_PROJ_ROOT|${PROJ_ROOT}|g" "${SBATCH_SCRIPT}"
sed -i "s|PLACEHOLDER_LOG_DIR|${LOG_DIR}|g" "${SBATCH_SCRIPT}"
sed -i "s|PLACEHOLDER_EXP_NAME|${EXP_NAME}|g" "${SBATCH_SCRIPT}"
sed -i "s|PLACEHOLDER_RUBRIC_GENERATOR|${RUBRIC_GENERATOR}|g" "${SBATCH_SCRIPT}"
sed -i "s|PLACEHOLDER_JUDGE_MODEL|${JUDGE_MODEL}|g" "${SBATCH_SCRIPT}"
sed -i "s|PLACEHOLDER_NUM_GPUS|${NUM_GPUS}|g" "${SBATCH_SCRIPT}"
sed -i "s|PLACEHOLDER_VLLM_GPU_UTIL|${VLLM_GPU_UTIL}|g" "${SBATCH_SCRIPT}"
sed -i "s|PLACEHOLDER_CONCURRENCY_LIMIT|${CONCURRENCY_LIMIT}|g" "${SBATCH_SCRIPT}"
sed -i "s|PLACEHOLDER_PAIRS_FILE|${PAIRS_FILE}|g" "${SBATCH_SCRIPT}"
sed -i "s|PLACEHOLDER_RUBRIC_PROMPT_KEY|${RUBRIC_PROMPT_KEY}|g" "${SBATCH_SCRIPT}"

JOB_ID=$(sbatch ${SBATCH_DEPENDENCY:+--dependency=${SBATCH_DEPENDENCY}} "${SBATCH_SCRIPT}" | awk '{print $4}')
echo "  Job ${JOB_ID}"

rm "${SBATCH_SCRIPT}"
