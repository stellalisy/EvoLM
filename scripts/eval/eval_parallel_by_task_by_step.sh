#!/bin/bash
# Submit parallel evaluation jobs - one per (checkpoint, task) combination
# Maximum parallelism: checkpoints × tasks jobs (or 1 checkpoint × tasks if step specified)
#
# Usage:
#   ./eval_parallel_by_task_by_step.sh <exp_name> <checkpoint_dir> <output_dir> [step_number] [tasks...] [--reuse-alpaca-generations]
#
# checkpoint_dir can be:
#   - A directory containing step_N subdirectories (training checkpoints)
#   - A HuggingFace model name (e.g., Qwen/Qwen3-8B) for base model eval
#
# Example:
#   # Evaluate all steps from a training run
#   ./eval_parallel_by_task_by_step.sh main_alt50 /path/to/checkpoints /path/to/olmes_eval
#   
#   # Evaluate only step 1000
#   ./eval_parallel_by_task_by_step.sh main_alt50 /path/to/checkpoints /path/to/olmes_eval 1000
#   
#   # Evaluate a base HuggingFace model (use step 0 as placeholder)
#   ./eval_parallel_by_task_by_step.sh qwen3_8b_base Qwen/Qwen3-8B /path/to/olmes_eval 0

set -e

# Project root: defaults to two levels above this script's directory.
PROJ_ROOT="${PROJ_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"

EXP_NAME=${1:?Error: exp_name required}
CHECKPOINT_DIR=${2:?Error: checkpoint_dir required}
OUTPUT_DIR=${3:?Error: output_dir required}
shift 3

# Compute defaults; override through env vars if needed.
DEFAULT_GPUS=${DEFAULT_GPUS:-1}
DEFAULT_NUM_WORKERS=${DEFAULT_NUM_WORKERS:-1}
DEFAULT_CPUS_PER_TASK=${DEFAULT_CPUS_PER_TASK:-8}
EVAL_QOS=${EVAL_QOS:-h100_sage_high}

# Optional Omega subtask grouping:
#   0 => disabled (one job per expanded Omega subtask)
#   N >= 2 => group omega_* tasks by first N underscore tokens in task stem.
#   Example with N=3: omega_explorative_numbertheory_* go into one grouped job.
#   Default is 3 (enabled): launch Omega as subgroup jobs by prefix.
OMEGA_GROUP_PREFIX_TOKENS=${OMEGA_GROUP_PREFIX_TOKENS:-3}

# Heavy-codegen profile for sharded LiveCodeBench:
# default is 8 independent 1-GPU shards (more efficient than one 8-GPU job).
LIVECODEBENCH_SHARDS=${LIVECODEBENCH_SHARDS:-8}
LIVECODEBENCH_GPUS=${LIVECODEBENCH_GPUS:-1}
# Keep one worker for LiveCodeBench to avoid multiprocessing daemon-child errors.
LIVECODEBENCH_NUM_WORKERS=${LIVECODEBENCH_NUM_WORKERS:-1}
LIVECODEBENCH_CPUS_PER_TASK=${LIVECODEBENCH_CPUS_PER_TASK:-16}
LIVECODEBENCH_METRICS_FILE="task-000-livecodebench_codegeneration-metrics.json"
MERGE_LIVECODEBENCH_SCRIPT="${PROJ_ROOT}/scripts/eval/merge_livecodebench_shards.py"

# Check if first argument is a step number (numeric)
SPECIFIC_STEP=""
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
    SPECIFIC_STEP="$1"
    shift
fi

# Optional mode: reuse cached AlpacaEval generations (annotation-only API test).
REUSE_ALPACA_GENERATIONS=0
FILTERED_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--reuse-alpaca-generations" ]]; then
        REUSE_ALPACA_GENERATIONS=1
    else
        FILTERED_ARGS+=("$arg")
    fi
done
set -- "${FILTERED_ARGS[@]}"

# Default tasks if none specified
if [ $# -eq 0 ]; then
    TASKS=(
        # === OLMo3 Adapt Benchmarks (reasoning model evals) ===
        "mmlu:cot::olmo3:adapt"
        "popqa::olmo3:adapt"
        "bbh:cot::olmo3:adapt"
        "gpqa::olmo3:adapt"
        "zebralogic::olmo3:adapt"
        "agi_eval_english::olmo3:adapt"
        "minerva_math::olmo3:adapt"
        "gsm8k::olmo3:adapt"
        "omega::olmo3:adapt"
        "aime:2024::olmo3:adapt"
        "aime:2025::olmo3:adapt"
        "codex_humanevalplus::olmo3:adapt"
        "mbppplus::olmo3:adapt"
        "livecodebench_codegeneration::olmo3:adapt"
        "alpaca_eval_v3::olmo3:adapt"
        "ifeval::olmo3:adapt"

        # === Old: Tulu-3 Dev Benchmarks ===
        # "gsm8k::tulu"
        # "drop::llama3"
        # "minerva_math::tulu"
        # "codex_humaneval::tulu"
        # "codex_humanevalplus::tulu"
        # "ifeval::tulu"
        # "popqa::tulu"
        # "mmlu:mc::tulu"
        # "alpaca_eval_v2::tulu"
        # "bbh:cot-v1::tulu"
        # "truthfulqa::tulu"
        # === Old: OLMES Core Benchmarks (9 tasks, mc format) ===
        # "arc_easy:mc::olmes"
        # "arc_challenge:mc::olmes"
        # "boolq:mc::olmes"
        # "csqa:mc::olmes"
        # "hellaswag:mc::olmes"
        # "openbookqa:mc::olmes"
        # "piqa:mc::olmes"
        # "socialiqa:mc::olmes"
        # "winogrande:mc::olmes"
    )
else
    TASKS=("$@")
fi

# Expand composite task suites into per-subtask evals so each subtask runs independently.
# This enables parallel SLURM jobs for suites that would otherwise run sequentially:
#   minerva_math (7 subtasks), bbh (27 subtasks), mmlu (57 subtasks), omega (~96 subtasks)
# Disable with EXPAND_SUBTASKS=0.
EXPAND_SUBTASKS=${EXPAND_SUBTASKS:-1}

EXPANDED_TASKS=()
for TASK in "${TASKS[@]}"; do
    if [[ "${EXPAND_SUBTASKS}" == "1" ]]; then
        # Extract the task core (before ::) and the config suffix (after first ::)
        TASK_CORE="${TASK%%::*}"
        TASK_SUFFIX="${TASK#*::}"

        case "${TASK_CORE}" in
            minerva_math)
                while IFS= read -r SUB; do
                    [[ -n "${SUB}" ]] && EXPANDED_TASKS+=("${SUB}")
                done < <(python3 -c "
import sys; sys.path.insert(0, '${PROJ_ROOT}/olmes')
from oe_eval.data.math_task_types import MATH_TASK_TYPES
for t in MATH_TASK_TYPES: print(f'minerva_math_{t}::${TASK_SUFFIX}')
")
                continue
                ;;
            bbh:cot|bbh:cot-v1)
                VARIANT="${TASK_CORE#bbh:}"  # cot or cot-v1
                while IFS= read -r SUB; do
                    [[ -n "${SUB}" ]] && EXPANDED_TASKS+=("${SUB}")
                done < <(python3 -c "
import sys; sys.path.insert(0, '${PROJ_ROOT}/olmes')
from oe_eval.data.bbh_tasks import BBH_TASKS
for t in BBH_TASKS: print(f'bbh_{t}:${VARIANT}::${TASK_SUFFIX}')
")
                continue
                ;;
            mmlu:cot|mmlu:mc)
                VARIANT="${TASK_CORE#mmlu:}"  # cot or mc
                while IFS= read -r SUB; do
                    [[ -n "${SUB}" ]] && EXPANDED_TASKS+=("${SUB}")
                done < <(python3 -c "
import sys; sys.path.insert(0, '${PROJ_ROOT}/olmes')
from oe_eval.data.mmlu_tasks import MMLU_SUBJECTS
for t in MMLU_SUBJECTS: print(f'mmlu_{t}:${VARIANT}::${TASK_SUFFIX}')
")
                continue
                ;;
            agi_eval_english)
                while IFS= read -r SUB; do
                    [[ -n "${SUB}" ]] && EXPANDED_TASKS+=("${SUB}")
                done < <(python3 -c "
import sys; sys.path.insert(0, '${PROJ_ROOT}/olmes')
from oe_eval.configs.task_suites import TASK_SUITE_CONFIGS
suite = TASK_SUITE_CONFIGS.get('agi_eval_english::${TASK_SUFFIX}', {})
for t in suite.get('tasks', []): print(t)
")
                continue
                ;;
            omega)
                while IFS= read -r OMEGA_TASK; do
                    [[ -n "${OMEGA_TASK}" ]] && EXPANDED_TASKS+=("${OMEGA_TASK}")
                done < <(PROJ_ROOT="${PROJ_ROOT}" python3 - <<'PY'
import sys, os
sys.path.insert(0, os.path.join(os.environ["PROJ_ROOT"], "olmes"))
from oe_eval.data.omega_categories import OMEGA_SUB_CATEGORIES

for broad_cate, sub_categories in OMEGA_SUB_CATEGORIES.items():
    splits = ["test_in", "test_out"] if broad_cate == "explorative" else ["test"]
    for sub_cate in sub_categories:
        for split in splits:
            print(f"omega_{broad_cate}_{sub_cate}_{split}:0-shot-chat")
PY
)
                continue
                ;;
        esac
    fi
    EXPANDED_TASKS+=("${TASK}")
done
TASKS=("${EXPANDED_TASKS[@]}")

build_omega_group_key() {
    local TASK="$1"
    local PREFIX_TOKENS="$2"
    local TASK_STEM="${TASK%%:*}"  # e.g., omega_explorative_numbertheory_qr_sum_test_in
    local TOKENS=()
    IFS='_' read -r -a TOKENS <<< "${TASK_STEM}"

    if [ "${#TOKENS[@]}" -eq 0 ]; then
        echo "${TASK_STEM}"
        return
    fi

    local USE_TOKENS="${PREFIX_TOKENS}"
    if [ "${USE_TOKENS}" -gt "${#TOKENS[@]}" ]; then
        USE_TOKENS="${#TOKENS[@]}"
    fi

    local KEY="${TOKENS[0]}"
    local i
    for ((i=1; i<USE_TOKENS; i++)); do
        KEY="${KEY}_${TOKENS[i]}"
    done
    echo "${KEY}"
}

count_matching_metrics_for_task() {
    local TASK="$1"
    local STEP_OUTPUT="$2"
    local TASK_BASE
    TASK_BASE=$(echo "$TASK" | cut -d: -f1)

    local EXPECTED_COUNT=1
    case "$TASK_BASE" in
        minerva_math) EXPECTED_COUNT=7 ;;       # 7 math subtasks
        mmlu) EXPECTED_COUNT=57 ;;              # 57 MMLU subjects
        bbh) EXPECTED_COUNT=27 ;;               # 27 BBH subtasks
        agi_eval_english) EXPECTED_COUNT=5 ;;   # 5 AGI eval subtasks
        omega) EXPECTED_COUNT=96 ;;             # omega::olmo3:adapt suite expansion
    esac

    local MATCHING_COUNT=0
    if [ -d "${STEP_OUTPUT}" ]; then
        local mfile
        for mfile in "${STEP_OUTPUT}"/*-"${TASK_BASE}"*-metrics.json; do
            [ -f "$mfile" ] || continue
            local ALIAS
            ALIAS=$(python3 -c "import json; d=json.load(open('$mfile')); print(d.get('task_config',{}).get('metadata',{}).get('alias',''))" 2>/dev/null)
            if [[ "$ALIAS" == *"${TASK##*::}"* ]]; then
                MATCHING_COUNT=$((MATCHING_COUNT + 1))
            fi
        done
    fi

    echo "${MATCHING_COUNT} ${EXPECTED_COUNT}"
}

livecodebench_shard_metrics_path() {
    local STEP="$1"
    local SHARD_ID="$2"
    echo "${OUTPUT_DIR}/livecodebench_shards/shard_${SHARD_ID}/step_${STEP}/${LIVECODEBENCH_METRICS_FILE}"
}

maybe_merge_livecodebench_step() {
    local STEP="$1"
    if [ "${LIVECODEBENCH_SHARDS}" -le 1 ]; then
        return
    fi
    if [ ! -f "${MERGE_LIVECODEBENCH_SCRIPT}" ]; then
        return
    fi
    local SID
    for ((SID=0; SID<${LIVECODEBENCH_SHARDS}; SID++)); do
        local SHARD_PATH
        SHARD_PATH=$(livecodebench_shard_metrics_path "${STEP}" "${SID}")
        if [ ! -f "${SHARD_PATH}" ]; then
            return
        fi
    done
    python3 "${MERGE_LIVECODEBENCH_SCRIPT}" \
        --output-root "${OUTPUT_DIR}" \
        --step "${STEP}" \
        --num-shards "${LIVECODEBENCH_SHARDS}" >/dev/null 2>&1 || true
}

# Build submission units:
# - default: one task per unit
# - optional grouping: Omega subtasks grouped by omega prefix tokens
declare -a UNIT_LABELS=()
declare -a UNIT_TASKS_JOINED=()
declare -a UNIT_SHARD_IDS=()
declare -a UNIT_NUM_SHARDS=()

append_unit() {
    local LABEL="$1"
    local TASKS_JOINED="$2"
    local SHARD_ID="${3:--1}"
    local NUM_SHARDS="${4:-1}"
    UNIT_LABELS+=("${LABEL}")
    UNIT_TASKS_JOINED+=("${TASKS_JOINED}")
    UNIT_SHARD_IDS+=("${SHARD_ID}")
    UNIT_NUM_SHARDS+=("${NUM_SHARDS}")
}

if [[ "${OMEGA_GROUP_PREFIX_TOKENS}" =~ ^[0-9]+$ ]] && [ "${OMEGA_GROUP_PREFIX_TOKENS}" -ge 2 ]; then
    declare -A OMEGA_GROUP_INDEX=()
    for TASK in "${TASKS[@]}"; do
        # LiveCodeBench: split into shard-specific singleton units.
        if [[ "${TASK}" == *"livecodebench_codegeneration"* ]]; then
            if [ "${LIVECODEBENCH_SHARDS}" -le 1 ]; then
                append_unit "${TASK}" "${TASK}" -1 1
            else
                for ((SID=0; SID<${LIVECODEBENCH_SHARDS}; SID++)); do
                    append_unit "${TASK}#shard_${SID}" "${TASK}" "${SID}" "${LIVECODEBENCH_SHARDS}"
                done
            fi
        elif [[ "${TASK}" == omega_* ]]; then
            GROUP_KEY=$(build_omega_group_key "${TASK}" "${OMEGA_GROUP_PREFIX_TOKENS}")
            if [ -z "${OMEGA_GROUP_INDEX[$GROUP_KEY]+x}" ]; then
                IDX=${#UNIT_LABELS[@]}
                append_unit "${GROUP_KEY}" "${TASK}" -1 1
                OMEGA_GROUP_INDEX["${GROUP_KEY}"]="${IDX}"
            else
                IDX=${OMEGA_GROUP_INDEX["${GROUP_KEY}"]}
                UNIT_TASKS_JOINED[${IDX}]="${UNIT_TASKS_JOINED[${IDX}]}"$'\n'"${TASK}"
            fi
        else
            append_unit "${TASK}" "${TASK}" -1 1
        fi
    done
    echo "Omega grouping enabled: prefix tokens=${OMEGA_GROUP_PREFIX_TOKENS} (submission units=${#UNIT_LABELS[@]})."
else
    if [ "${OMEGA_GROUP_PREFIX_TOKENS}" != "0" ]; then
        echo "Warning: OMEGA_GROUP_PREFIX_TOKENS=${OMEGA_GROUP_PREFIX_TOKENS} is invalid for grouping. Using default one-task-per-job mode."
    fi
    for TASK in "${TASKS[@]}"; do
        if [[ "${TASK}" == *"livecodebench_codegeneration"* ]] && [ "${LIVECODEBENCH_SHARDS}" -gt 1 ]; then
            for ((SID=0; SID<${LIVECODEBENCH_SHARDS}; SID++)); do
                append_unit "${TASK}#shard_${SID}" "${TASK}" "${SID}" "${LIVECODEBENCH_SHARDS}"
            done
        else
            append_unit "${TASK}" "${TASK}" -1 1
        fi
    done
fi

# Find step directories
if [ -n "$SPECIFIC_STEP" ]; then
    # User specified a step
    if [ -d "${CHECKPOINT_DIR}/step_${SPECIFIC_STEP}" ]; then
        echo "Evaluating checkpoint step ${SPECIFIC_STEP}"
    elif [ ! -d "${CHECKPOINT_DIR}" ] || [[ "${CHECKPOINT_DIR}" == *"/"*"/"* && ! -d "${CHECKPOINT_DIR}/step_${SPECIFIC_STEP}" ]]; then
        # checkpoint_dir might be a HuggingFace model name (e.g., Qwen/Qwen3-8B)
        echo "Evaluating model directly: ${CHECKPOINT_DIR} (step ${SPECIFIC_STEP} as label)"
    else
        echo "Error: Step ${SPECIFIC_STEP} not found in ${CHECKPOINT_DIR}"
        echo "Available steps:"
        ls -d "${CHECKPOINT_DIR}"/step_* 2>/dev/null | sed 's/.*step_/  /' || echo "  (none)"
        exit 1
    fi
    STEPS="$SPECIFIC_STEP"
else
    # Find all step_N directories
    STEPS=$(ls -d "${CHECKPOINT_DIR}"/step_* 2>/dev/null | sed 's/.*step_//' | sort -n)
    
    if [ -z "$STEPS" ]; then
        echo "Error: No step_* directories found in ${CHECKPOINT_DIR}"
        exit 1
    fi
    echo "Evaluating all steps"
fi

STEP_COUNT=$(echo "$STEPS" | wc -w)
UNIT_COUNT=${#UNIT_LABELS[@]}
TOTAL_JOBS=$((STEP_COUNT * UNIT_COUNT))

echo "Found ${STEP_COUNT} checkpoint(s), ${#TASKS[@]} expanded tasks, ${UNIT_COUNT} submission units"
echo "Steps: ${STEPS}"
echo "Tasks: ${TASKS[*]}"
echo "LiveCodeBench shards: ${LIVECODEBENCH_SHARDS} (GPUs/shard=${LIVECODEBENCH_GPUS})"
echo "Reuse AlpacaEval generations: ${REUSE_ALPACA_GENERATIONS}"
echo "This will submit up to ${TOTAL_JOBS} jobs"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted"
    exit 0
fi

SUBMITTED=0
SKIPPED=0

# Submit a job for each (checkpoint, submission-unit) combination
for STEP in $STEPS; do
    STEP_OUTPUT="${OUTPUT_DIR}/step_${STEP}"
    maybe_merge_livecodebench_step "${STEP}"

    for UNIT_IDX in "${!UNIT_LABELS[@]}"; do
        UNIT_LABEL="${UNIT_LABELS[$UNIT_IDX]}"
        UNIT_TASKS_RAW="${UNIT_TASKS_JOINED[$UNIT_IDX]}"
        UNIT_SHARD_ID="${UNIT_SHARD_IDS[$UNIT_IDX]}"
        UNIT_NUM_SHARDS_VALUE="${UNIT_NUM_SHARDS[$UNIT_IDX]}"
        UNIT_TASKS=()
        while IFS= read -r UNIT_TASK; do
            [[ -n "${UNIT_TASK}" ]] && UNIT_TASKS+=("${UNIT_TASK}")
        done <<< "${UNIT_TASKS_RAW}"

        PENDING_TASKS=()
        for TASK in "${UNIT_TASKS[@]}"; do
            if [[ "${TASK}" == *"livecodebench_codegeneration"* ]] && [ "${UNIT_NUM_SHARDS_VALUE}" -gt 1 ] && [ "${UNIT_SHARD_ID}" -ge 0 ]; then
                SHARD_METRICS_PATH=$(livecodebench_shard_metrics_path "${STEP}" "${UNIT_SHARD_ID}")
                if [ -f "${SHARD_METRICS_PATH}" ]; then
                    echo "  Task ${TASK} shard ${UNIT_SHARD_ID}/${UNIT_NUM_SHARDS_VALUE}: Already complete, skipping"
                else
                    PENDING_TASKS+=("${TASK}")
                fi
            else
                read -r MATCHING_COUNT EXPECTED_COUNT < <(count_matching_metrics_for_task "${TASK}" "${STEP_OUTPUT}")
                if [ "$MATCHING_COUNT" -ge "$EXPECTED_COUNT" ]; then
                    echo "  Task ${TASK}: Already complete ($MATCHING_COUNT/$EXPECTED_COUNT files), skipping"
                else
                    if [ "$MATCHING_COUNT" -gt 0 ]; then
                        echo "  Task ${TASK}: Incomplete ($MATCHING_COUNT/$EXPECTED_COUNT files), will re-run"
                    fi
                    PENDING_TASKS+=("${TASK}")
                fi
            fi
        done

        if [ "${#PENDING_TASKS[@]}" -eq 0 ]; then
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        # Safety rail: never run LiveCodeBench together with any other task.
        HAS_LIVECODEBENCH=0
        for TASK in "${PENDING_TASKS[@]}"; do
            if [[ "${TASK}" == *"livecodebench_codegeneration"* ]]; then
                HAS_LIVECODEBENCH=1
                break
            fi
        done
        if [[ "${HAS_LIVECODEBENCH}" == "1" && "${#PENDING_TASKS[@]}" -gt 1 ]]; then
            echo "Error: LiveCodeBench was grouped with other tasks in unit ${UNIT_LABEL}. This is not allowed."
            exit 1
        fi
        UNIT_LIVECODEBENCH_SHARD_ID=-1
        UNIT_LIVECODEBENCH_NUM_SHARDS=1
        if [[ "${HAS_LIVECODEBENCH}" == "1" ]] && [ "${UNIT_NUM_SHARDS_VALUE}" -gt 1 ] && [ "${UNIT_SHARD_ID}" -ge 0 ]; then
            UNIT_LIVECODEBENCH_SHARD_ID="${UNIT_SHARD_ID}"
            UNIT_LIVECODEBENCH_NUM_SHARDS="${UNIT_NUM_SHARDS_VALUE}"
        fi

        if [ "${#PENDING_TASKS[@]}" -eq 1 ]; then
            JOB_TASK_TAG=$(echo "${PENDING_TASKS[0]}" | sed 's/[^a-zA-Z0-9]/_/g' | sed 's/__*/_/g')
            if [ "${UNIT_LIVECODEBENCH_SHARD_ID}" -ge 0 ]; then
                JOB_TASK_TAG="${JOB_TASK_TAG}_sh${UNIT_LIVECODEBENCH_SHARD_ID}"
            fi
        else
            UNIT_SAFE=$(echo "${UNIT_LABEL}" | sed 's/[^a-zA-Z0-9]/_/g' | sed 's/__*/_/g')
            JOB_TASK_TAG="${UNIT_SAFE}_n${#PENDING_TASKS[@]}"
        fi
        JOB_NAME="v2_37c14bf_ev_${EXP_NAME}_s${STEP}_${JOB_TASK_TAG}"
        JOB_NAME="${JOB_NAME:0:50}"  # SLURM job name limit

        # Per-job compute profile.
        TASK_GPUS=${DEFAULT_GPUS}
        TASK_NUM_WORKERS=${DEFAULT_NUM_WORKERS}
        TASK_CPUS_PER_TASK=${DEFAULT_CPUS_PER_TASK}
        for TASK in "${PENDING_TASKS[@]}"; do
            if [[ "${TASK}" == *"livecodebench_codegeneration"* ]]; then
                TASK_GPUS=${LIVECODEBENCH_GPUS}
                TASK_NUM_WORKERS=${LIVECODEBENCH_NUM_WORKERS}
                TASK_CPUS_PER_TASK=${LIVECODEBENCH_CPUS_PER_TASK}
                break
            fi
        done

        TASK_ARGS_CLI=""
        for TASK in "${PENDING_TASKS[@]}"; do
            TASK_ARGS_CLI="${TASK_ARGS_CLI} \"${TASK}\""
        done
        TASK_LIST_PRINT="${PENDING_TASKS[*]}"
        UNIT_HAS_ALPACA=0
        for TASK in "${PENDING_TASKS[@]}"; do
            if [[ "${TASK}" == *"alpaca_eval_v3"* ]]; then
                UNIT_HAS_ALPACA=1
                break
            fi
        done

        UNIT_OUTPUT_ROOT="${OUTPUT_DIR}"
        UNIT_SHARD_ARGS=""
        if [[ "${UNIT_LIVECODEBENCH_SHARD_ID}" -ge 0 ]] && [[ "${UNIT_LIVECODEBENCH_NUM_SHARDS}" -gt 1 ]]; then
            UNIT_OUTPUT_ROOT="${OUTPUT_DIR}/livecodebench_shards/shard_${UNIT_LIVECODEBENCH_SHARD_ID}"
            UNIT_SHARD_ARGS="--livecodebench-shard-id ${UNIT_LIVECODEBENCH_SHARD_ID} --livecodebench-num-shards ${UNIT_LIVECODEBENCH_NUM_SHARDS}"
        fi
        UNIT_REUSE_ALPACA_ARG=""
        if [[ "${REUSE_ALPACA_GENERATIONS}" == "1" ]]; then
            UNIT_REUSE_ALPACA_ARG="--reuse-alpaca-generations"
        fi
        UNIT_TASK_ARGS_JSON=$(python3 - <<PY
import json
task_args = {
    "generation_kwargs": {
        "max_gen_toks": 16384,
        "truncate_context": False,
        "temperature": 0.6,
        "top_p": 0.95,
        "do_sample": True,
    }
}
shard_id = ${UNIT_LIVECODEBENCH_SHARD_ID}
num_shards = ${UNIT_LIVECODEBENCH_NUM_SHARDS}
if shard_id >= 0 and num_shards > 1:
    task_args["livecodebench_shard_id"] = shard_id
    task_args["livecodebench_num_shards"] = num_shards
print(json.dumps(task_args))
PY
)
        
        # Create SLURM script
        SBATCH_SCRIPT=$(mktemp)
        cat > "${SBATCH_SCRIPT}" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=sage
#SBATCH --qos=${EVAL_QOS}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${TASK_CPUS_PER_TASK}
#SBATCH --time=7-00:00:00
#SBATCH --chdir=${PROJ_ROOT}
#SBATCH --export=all
#SBATCH --gres=gpu:${TASK_GPUS}
#SBATCH --output=${PROJ_ROOT}/logs/eval/${EXP_NAME}/${JOB_NAME}-%j.out
#SBATCH --error=${PROJ_ROOT}/logs/eval/${EXP_NAME}/${JOB_NAME}-%j.err

source ${PROJ_ROOT}/olmes/.venv/bin/activate

# Unset Claude Code proxy vars that --export=all would forward to compute nodes
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

echo "========================================"
echo "OLMES Evaluation"
echo "========================================"
echo "Step: ${STEP}"
echo "Task count: ${#PENDING_TASKS[@]}"
echo "Tasks: ${TASK_LIST_PRINT}"
echo "LiveCodeBench shard: ${UNIT_LIVECODEBENCH_SHARD_ID}/${UNIT_LIVECODEBENCH_NUM_SHARDS}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${TASK_GPUS}, Workers: ${TASK_NUM_WORKERS}, CPUs: ${TASK_CPUS_PER_TASK}"
echo "========================================"

export HF_HOME=\${HF_HOME:-\$HOME/.cache/huggingface}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_\${SLURM_JOB_ID}
export VLLM_CACHE_ROOT=/tmp/vllm_cache_\${SLURM_JOB_ID}

cd ${PROJ_ROOT}/olmes

# AlpacaEval v3 requires judge API credentials; configure Azure GPT-4.1 client.
if [[ "${UNIT_HAS_ALPACA}" == "1" ]]; then
    source ${PROJ_ROOT}/scripts/configure_api.sh gpt-4.1 2>/dev/null || true
    if [[ "${REUSE_ALPACA_GENERATIONS}" == "1" ]]; then
        export OLMES_REUSE_ALPACA_GENERATIONS=1
        echo "Enabled OLMES_REUSE_ALPACA_GENERATIONS=1 for AlpacaEval task(s)"
    fi
    # alpaca_eval reads client YAML literally and does not expand ${ENV_VAR} placeholders.
    # Build a runtime config with fully expanded Azure values to avoid malformed endpoint URLs.
    AZURE_ENDPOINT="\${AZURE_API_BASE:-}"
    if [[ -n "\${AZURE_ENDPOINT}" && "\${AZURE_ENDPOINT}" != http://* && "\${AZURE_ENDPOINT}" != https://* ]]; then
        AZURE_ENDPOINT="https://\${AZURE_ENDPOINT}"
    fi
    if [[ -z "\${AZURE_API_KEY:-}" || -z "\${AZURE_API_VERSION:-}" || -z "\${AZURE_ENDPOINT:-}" ]]; then
        echo "ERROR: Missing Azure API env vars for AlpacaEval (AZURE_API_KEY / AZURE_API_VERSION / AZURE_API_BASE)" >&2
        exit 1
    fi
    ALPACA_CLIENT_CONFIG=$(mktemp /tmp/alpaca_eval_openai_config.XXXXXX.yaml)
    cat > "\${ALPACA_CLIENT_CONFIG}" << ALPACA_CFG_EOF
gpt-4.1-2025-04-14:
  - client_class: "openai.AzureOpenAI"
    azure_deployment: "gpt-4.1"
    azure_endpoint: "\${AZURE_ENDPOINT}"
    api_version: "\${AZURE_API_VERSION}"
    api_key: "\${AZURE_API_KEY}"
ALPACA_CFG_EOF
    export OPENAI_CLIENT_CONFIG_PATH="\${ALPACA_CLIENT_CONFIG}"
    # Keep compatibility with codepaths that still read OPENAI_API_KEY directly.
    if [ -n "${AZURE_API_KEY:-}" ] && [ -z "${AZURE_EVAL_API_KEY:-}" ]; then
        export AZURE_EVAL_API_KEY="${AZURE_API_KEY}"
    fi
    if [ -n "${AZURE_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
        export OPENAI_API_KEY="${AZURE_API_KEY}"
    fi
fi

mkdir -p "${UNIT_OUTPUT_ROOT}"

# If checkpoint dir contains step_N subdirectories, use the wrapper script.
# Otherwise (e.g., HuggingFace model name), call oe_eval.launch directly.

if [ -d "${CHECKPOINT_DIR}/step_${STEP}" ]; then
    python ${PROJ_ROOT}/scripts/eval/run_olmes_on_checkpoints.py \\
        --checkpoint-dir "${CHECKPOINT_DIR}" \\
        --output-dir "${UNIT_OUTPUT_ROOT}" \\
        --task ${TASK_ARGS_CLI} \\
        --steps ${STEP} \\
        --model-type vllm \\
        --trust-remote-code \\
        --max-length 16384 \\
        --max-gen-toks 16384 \\
        --gpus ${TASK_GPUS} \\
        --num-workers ${TASK_NUM_WORKERS} \\
        ${UNIT_SHARD_ARGS} \\
        ${UNIT_REUSE_ALPACA_ARG}
else
    # CHECKPOINT_DIR is a model name/path (e.g., Qwen/Qwen3-8B)
    mkdir -p "${UNIT_OUTPUT_ROOT}/step_${STEP}"
    python -m oe_eval.launch \\
        --model "${CHECKPOINT_DIR}" \\
        --model-type vllm \\
        --model-args '{"trust_remote_code": true, "max_length": 16384}' \\
        --task-args '${UNIT_TASK_ARGS_JSON}' \\
        --gpus ${TASK_GPUS} \\
        --num-workers ${TASK_NUM_WORKERS} \\
        --output-dir "${UNIT_OUTPUT_ROOT}/step_${STEP}" \\
        --task ${TASK_ARGS_CLI}
fi

if [[ "${UNIT_LIVECODEBENCH_SHARD_ID}" -ge 0 ]] && [[ "${UNIT_LIVECODEBENCH_NUM_SHARDS}" -gt 1 ]]; then
    ALL_SHARDS_DONE=1
    for ((SID=0; SID<${UNIT_LIVECODEBENCH_NUM_SHARDS}; SID++)); do
        SHARD_PATH="${OUTPUT_DIR}/livecodebench_shards/shard_${SID}/step_${STEP}/${LIVECODEBENCH_METRICS_FILE}"
        if [ ! -f "${SHARD_PATH}" ]; then
            ALL_SHARDS_DONE=0
            break
        fi
    done
    if [[ "${ALL_SHARDS_DONE}" == "1" ]]; then
        python3 "${MERGE_LIVECODEBENCH_SCRIPT}" \\
            --output-root "${OUTPUT_DIR}" \\
            --step "${STEP}" \\
            --num-shards "${UNIT_LIVECODEBENCH_NUM_SHARDS}" || true
    fi
fi

echo "Done: step ${STEP}, tasks: ${TASK_LIST_PRINT}"
EOF
        
        # Submit the job
        JOB_ID=$(sbatch "${SBATCH_SCRIPT}" | awk '{print $4}')
        echo "Step ${STEP}, unit ${UNIT_LABEL}, tasks ${#PENDING_TASKS[@]}: Job ${JOB_ID}"
        SUBMITTED=$((SUBMITTED + 1))
        
        rm "${SBATCH_SCRIPT}"
    done
done

echo ""
echo "Submitted ${SUBMITTED} jobs, skipped ${SKIPPED} (already evaluated)"
echo "Monitor with: squeue -u \$USER | grep ev_${EXP_NAME}"
