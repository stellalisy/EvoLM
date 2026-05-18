#!/bin/bash
# Accumulate nodes with single-node placeholder jobs, then swap for the real job.
# Automatically detaches so it survives terminal/SSH disconnects.
#
# Usage:
#   ./scripts/node_accumulator.sh <job_script> [-- extra sbatch flags]
#
# Examples:
#   ./scripts/node_accumulator.sh stella_run_scripts/paper_experiments/main_v3_margin_format.sh
#   POLL_INTERVAL=15 ./scripts/node_accumulator.sh my_job.sh -- --dependency=afterany:12345

set -euo pipefail

JOB_SCRIPT="${1:?Usage: $0 <job_script> [-- extra_sbatch_flags...]}"
shift
EXTRA_SBATCH_ARGS=()
if [[ "${1:-}" == "--" ]]; then shift; EXTRA_SBATCH_ARGS=("$@"); fi
[[ -f "$JOB_SCRIPT" ]] || { echo "ERROR: $JOB_SCRIPT not found"; exit 1; }

NUM_NODES=$(grep -m1 '#SBATCH.*--nodes=' "$JOB_SCRIPT" | sed 's/.*--nodes=\([0-9]*\).*/\1/')
ACCOUNT=$(grep -m1 '#SBATCH.*--account=' "$JOB_SCRIPT" | sed 's/.*--account=\([^ ]*\).*/\1/')
QOS=$(grep -m1 '#SBATCH.*--qos=' "$JOB_SCRIPT" | sed 's/.*--qos=\([^ ]*\).*/\1/')
GPUS_PER_NODE=$(grep -m1 '#SBATCH.*--gres=gpu:' "$JOB_SCRIPT" | sed 's/.*--gres=gpu:\([0-9]*\).*/\1/')
TIME_LIMIT=$(grep -m1 '#SBATCH.*--time=' "$JOB_SCRIPT" | sed 's/.*--time=\([^ ]*\).*/\1/')
REAL_JOB_NAME=$(grep -m1 '#SBATCH.*--job-name=' "$JOB_SCRIPT" | sed 's/.*--job-name=\([^ ]*\).*/\1/')

: "${NUM_NODES:?Could not parse --nodes from $JOB_SCRIPT}"
: "${ACCOUNT:?Could not parse --account from $JOB_SCRIPT}"
: "${QOS:?Could not parse --qos from $JOB_SCRIPT}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TIME_LIMIT="${TIME_LIMIT:-7-00:00:00}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
HOLDER_NAME="${REAL_JOB_NAME}_prep"

# --- Detach into background on first run ---
LOG_DIR="${LOG_DIR:-./log_accumulator}"
mkdir -p "$LOG_DIR"
if [[ "${_ACCUM_BG:-}" != "1" ]]; then
    LOG="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_${REAL_JOB_NAME}.log"
    echo "Node Accumulator — running detached"
    echo "  Target : $NUM_NODES nodes × $GPUS_PER_NODE GPUs"
    echo "  Holders: ${HOLDER_NAME}_1 … ${HOLDER_NAME}_${NUM_NODES}"
    echo "  Log    : $LOG"
    _ACCUM_BG=1 nohup bash "$0" "$JOB_SCRIPT" \
        ${EXTRA_SBATCH_ARGS[@]+"--" "${EXTRA_SBATCH_ARGS[@]}"} \
        > "$LOG" 2>&1 &
    echo "  PID    : $!"
    echo ""
    echo "  tail -f $LOG"
    exit 0
fi

# --- Background process starts here ---
echo "=== Node Accumulator (PID $$) ==="
echo "Job: $JOB_SCRIPT"
echo "Target: $NUM_NODES nodes, holder name: $HOLDER_NAME"
echo "Started: $(date)"
echo ""

HOLDER_IDS=()
cleanup() {
    echo ""
    echo "[cleanup] Cancelling ${#HOLDER_IDS[@]} holders..."
    [[ ${#HOLDER_IDS[@]} -gt 0 ]] && scancel "${HOLDER_IDS[@]}" 2>/dev/null || true
    echo "[cleanup] Done."
}
trap cleanup INT TERM EXIT

echo "==> Submitting $NUM_NODES node holders..."
for i in $(seq 1 "$NUM_NODES"); do
    JID=$(sbatch --parsable \
        --job-name="${HOLDER_NAME}_${i}" \
        --account="$ACCOUNT" --qos="$QOS" \
        --nodes=1 --exclusive --gres="gpu:${GPUS_PER_NODE}" \
        --time="$TIME_LIMIT" \
        --output=/dev/null --error=/dev/null \
        --wrap="sleep 604800")
    HOLDER_IDS+=("$JID")
    echo "    $i/$NUM_NODES → $JID"
done
echo ""

sleep 5
echo "==> Waiting for all $NUM_NODES holders to start..."
CONSECUTIVE_EMPTY=0
while true; do
    RUNNING=0
    PENDING=0
    UNKNOWN=0
    ID_LIST=$(IFS=,; echo "${HOLDER_IDS[*]}")
    while IFS='|' read -r state node; do
        state="${state// /}"
        case "$state" in
            RUNNING)                RUNNING=$((RUNNING + 1)) ;;
            PENDING|CONFIGURING)    PENDING=$((PENDING + 1)) ;;
            "")                     ;;
            CANCELLED|FAILED|TIMEOUT|NODE_FAIL)
                echo "ERROR: A holder entered state $state. Aborting."
                exit 1 ;;
            *)                      PENDING=$((PENDING + 1)) ;;
        esac
    done < <(squeue -j "$ID_LIST" --noheader --format="%T|%N" 2>/dev/null || true)

    VISIBLE=$((RUNNING + PENDING))
    UNKNOWN=$((${#HOLDER_IDS[@]} - VISIBLE))

    echo "    [$(date '+%H:%M:%S')]  running=$RUNNING  pending=$PENDING  unknown=$UNKNOWN"

    if [[ $VISIBLE -eq 0 ]]; then
        CONSECUTIVE_EMPTY=$((CONSECUTIVE_EMPTY + 1))
        [[ $CONSECUTIVE_EMPTY -ge 5 ]] && { echo "ERROR: All holders gone."; exit 1; }
    else
        CONSECUTIVE_EMPTY=0
    fi

    [[ $RUNNING -eq $NUM_NODES ]] && { echo ""; echo "==> All $NUM_NODES holders RUNNING!"; break; }
    sleep "$POLL_INTERVAL"
done

# Collect nodes
NODELIST=""
for JID in "${HOLDER_IDS[@]}"; do
    NODE=$(squeue -j "$JID" --noheader --format="%N" 2>/dev/null | tr -d ' ')
    NODELIST="${NODELIST:+$NODELIST,}$NODE"
done
echo "    Nodes: $NODELIST"
echo ""

# Submit real job FIRST so it's in the queue (building age priority) and
# pinned to these nodes via --nodelist.  It will be PENDING (QOSGrpGRES)
# while the holders still occupy quota — that's expected and transient.
echo "==> Submitting real job with --nodelist=$NODELIST"
REAL_JOB_ID=$(sbatch --parsable \
    --nodelist="$NODELIST" \
    ${EXTRA_SBATCH_ARGS[@]+"${EXTRA_SBATCH_ARGS[@]}"} \
    "$JOB_SCRIPT")
echo "    Real job: $REAL_JOB_ID (PENDING until holders release quota)"

# Let the scheduler register the job before we free the nodes
sleep 3

# Now cancel holders — frees both the QoS quota and the nodes.
# Our real job is already queued targeting these exact nodes.
echo "==> Cancelling holders to release nodes..."
scancel "${HOLDER_IDS[@]}"
HOLDER_IDS=()
echo "    Done."
echo ""

# Watch real job
echo "==> Waiting for real job to start..."
for tick in $(seq 1 60); do
    STATE=$(squeue -j "$REAL_JOB_ID" --noheader --format="%T" 2>/dev/null | tr -d ' ')
    if [[ "$STATE" == "RUNNING" ]]; then
        NODES=$(squeue -j "$REAL_JOB_ID" --noheader --format="%N" 2>/dev/null)
        echo "    RUNNING on: $NODES"
        echo ""
        echo "=== SUCCESS ==="
        exit 0
    elif [[ -z "$STATE" ]]; then
        echo "    WARNING: job not in queue"
        exit 1
    fi
    echo "    [${tick}/60] $STATE"
    sleep 5
done
echo "WARNING: didn't start in 5 min. Check: squeue -j $REAL_JOB_ID"
exit 2
