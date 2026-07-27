#!/usr/bin/env bash
#SBATCH --job-name=exp8-face-spectrogram-open-mouth
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=bootstrap-%j.out
#SBATCH --error=bootstrap-%j.err

set -Eeuo pipefail
umask 007

# =============================================================================
# USER CONFIGURATION
# Normally, users change only these three values.
# =============================================================================

# Folder name under /data1/workspace/students/
USER_NAME="${USER_NAME:-soongswang-kornkanok}"

# The same project name is used under /home/workspace/<linux-user>/
# and /data1/workspace/students/<user-name>/
PROJECT_NAME="${PROJECT_NAME:-project-01}"

# Selected dataset folder under /data1/workspace/students/<user-name>/
DATASET_NAME="${DATASET_NAME:-silver_dataset_open_mouth}"

# =============================================================================
# AUTOMATIC PATHS
# Do not normally edit these.
# =============================================================================

LINUX_USER="${SLURM_JOB_USER:-$(id -un)}"

CODE_ROOT="/home/workspace/students/$USER_NAME/$PROJECT_NAME"
USER_DATA_ROOT="/data1/workspace/students/$USER_NAME"
PROJECT_DATA_ROOT="$USER_DATA_ROOT/$PROJECT_NAME"
DATASET_ROOT="$PROJECT_DATA_ROOT/dataset/$DATASET_NAME"
RUN_ROOT="$PROJECT_DATA_ROOT/slurm-runs"

PYTHON="$CODE_ROOT/.venv/bin/python"
PROGRAM="$CODE_ROOT/code/experiment/experiment_8_face_spectrogram_open_mouth.py"

# =============================================================================
# SMALL VALIDATION HELPERS
# =============================================================================

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

validate_path_name() {
    local label="$1"
    local value="$2"

    [[ "$value" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
        fail "$label contains unsupported characters: $value"
}

validate_path_name "USER_NAME" "$USER_NAME"
validate_path_name "PROJECT_NAME" "$PROJECT_NAME"
validate_path_name "DATASET_NAME" "$DATASET_NAME"

# =============================================================================
# REDIS SYSTEM SERVICE
# =============================================================================

REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_DB="${REDIS_DB:-0}"
export REDIS_HOST REDIS_PORT REDIS_DB

# =============================================================================
# RUN DIRECTORY
# =============================================================================

mkdir -p "$RUN_ROOT"

if [[ -n "${RESUME_RUN_DIR:-}" ]]; then
    [[ -d "$RESUME_RUN_DIR" ]] || fail "Resume directory not found: $RESUME_RUN_DIR"
    RUN_DIR="$(realpath -e "$RESUME_RUN_DIR")"
    RESUME=1
else
    RUN_DIR="$RUN_ROOT/run-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
    RESUME=0
fi

mkdir -p \
    "$RUN_DIR/log" \
    "$RUN_DIR/timestamp" \
    "$RUN_DIR/output"

# Move Slurm's short bootstrap logs into the logical run directory.
BOOTSTRAP_OUT="${SLURM_SUBMIT_DIR:-$PWD}/bootstrap-${SLURM_JOB_ID}.out"
BOOTSTRAP_ERR="${SLURM_SUBMIT_DIR:-$PWD}/bootstrap-${SLURM_JOB_ID}.err"
mv "$BOOTSTRAP_OUT" "$RUN_DIR/log/bootstrap.out" 2>/dev/null || true
mv "$BOOTSTRAP_ERR" "$RUN_DIR/log/bootstrap.err" 2>/dev/null || true

exec >>"$RUN_DIR/log/stdout.log" 2>>"$RUN_DIR/log/stderr.log"

export RUN_DIR RESUME DATASET_ROOT
export USER_NAME PROJECT_NAME DATASET_NAME

printf '%s\n' \
    "Slurm job ID : $SLURM_JOB_ID" \
    "Linux user   : $LINUX_USER" \
    "User name    : $USER_NAME" \
    "Project name : $PROJECT_NAME" \
    "Dataset name : $DATASET_NAME" \
    "Code root    : $CODE_ROOT" \
    "Dataset root : $DATASET_ROOT" \
    "Run directory: $RUN_DIR"

# =============================================================================
# PATH VALIDATION
# =============================================================================

[[ -d "$CODE_ROOT" ]] || fail "Project code directory not found: $CODE_ROOT"
[[ -d "$DATASET_ROOT" ]] || fail "Dataset directory not found: $DATASET_ROOT"
[[ -x "$PYTHON" ]] || fail "Python environment not found: $PYTHON"
[[ -f "$PROGRAM" ]] || fail "Python program not found: $PROGRAM"

cd "$CODE_ROOT"

# =============================================================================
# REDIS CHECK
# =============================================================================

"$PYTHON" - <<'PY'
import os
import redis

client = redis.Redis(
    host=os.environ["REDIS_HOST"],
    port=int(os.environ["REDIS_PORT"]),
    db=int(os.environ["REDIS_DB"]),
    password=os.environ.get("REDIS_PASSWORD"),
    socket_connect_timeout=10,
)

if not client.ping():
    raise RuntimeError("Redis did not respond to PING")

print("PASS: Redis system service is ready")
PY

# =============================================================================
# READ MODEL SETTINGS FROM PYTHON
# The model and provider are configured only in experiment_8_face_spectrogram_open_mouth.py.
# =============================================================================

read_python_constant() {
    local constant_name="$1"

    "$PYTHON" - "$PROGRAM" "$constant_name" <<'PY'
import ast
import sys
from pathlib import Path

program = Path(sys.argv[1])
constant_name = sys.argv[2]
tree = ast.parse(program.read_text(encoding="utf-8"), filename=str(program))

for node in tree.body:
    names = []
    value_node = None

    if isinstance(node, ast.Assign):
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        value_node = node.value
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        names = [node.target.id]
        value_node = node.value

    if constant_name in names and value_node is not None:
        print(ast.literal_eval(value_node))
        break
else:
    raise SystemExit(f"Constant not found in {program}: {constant_name}")
PY
}

PROVIDER="$(read_python_constant PROVIDER)"
MODEL="$(read_python_constant MODEL)"

printf '%s\n' \
    "Model provider: $PROVIDER" \
    "Model name    : $MODEL"

# =============================================================================
# OLLAMA INSIDE THE SLURM ALLOCATION
# =============================================================================

OLLAMA_PID=""

cleanup() {
    status=$?
    trap - EXIT INT TERM
    set +e

    if [[ -n "$OLLAMA_PID" ]]; then
        echo "Stopping Ollama process $OLLAMA_PID"
        kill "$OLLAMA_PID" 2>/dev/null || true
        wait "$OLLAMA_PID" 2>/dev/null || true
    fi

    exit "$status"
}

trap cleanup EXIT INT TERM

if [[ "$PROVIDER" == "ollama" ]]; then
    OLLAMA_PORT="${OLLAMA_PORT:-$(
        "$PYTHON" - <<'PY'
import socket

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
    )}"

    export OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"
    export OLLAMA_MODELS="${OLLAMA_MODELS:-/usr/share/ollama/.ollama/models}"
    export MODEL_HOST="http://$OLLAMA_HOST"

    echo "Starting Ollama at $MODEL_HOST"
    echo "Ollama models : $OLLAMA_MODELS"

    ollama serve >"$RUN_DIR/log/ollama.log" 2>&1 &
    OLLAMA_PID=$!

    OLLAMA_READY=0
    for _ in {1..90}; do
        if curl -fsS "$MODEL_HOST/api/tags" >/dev/null 2>&1; then
            OLLAMA_READY=1
            break
        fi
        sleep 1
    done

    [[ "$OLLAMA_READY" == 1 ]] || fail "Ollama did not become ready"

    if ! ollama show "$MODEL" >/dev/null 2>&1; then
        fail "Ollama model '$MODEL' is unavailable in $OLLAMA_MODELS"
    fi

    echo "PASS: Ollama is ready"
else
    MODEL_HOST="${MODEL_HOST:?MODEL_HOST is required for provider $PROVIDER}"
    export MODEL_HOST
fi

# =============================================================================
# EXECUTE THE PYTHON EXPERIMENT
# =============================================================================

# Allow Python to find project-01/lab_pipeline
export PYTHONPATH="$CODE_ROOT/code${PYTHONPATH:+:$PYTHONPATH}"

srun --unbuffered "$PYTHON" "$PROGRAM"
