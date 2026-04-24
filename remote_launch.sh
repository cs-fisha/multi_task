#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 12 ]]; then
  echo "usage: remote_launch.sh WORKDIR GPU_INDEX JOB_DIR STDOUT_LOG STDERR_LOG PID_FILE STATUS_FILE EXIT_CODE_FILE CONDA_BIN ENV_MODE ENV_VALUE CMD_B64" >&2
  exit 1
fi

WORKDIR="$1"
GPU_INDEX="$2"
JOB_DIR="$3"
STDOUT_LOG="$4"
STDERR_LOG="$5"
PID_FILE="$6"
STATUS_FILE="$7"
EXIT_CODE_FILE="$8"
CONDA_BIN="$9"
ENV_MODE="${10}"
ENV_VALUE="${11}"
CMD_B64="${12}"

mkdir -p "$JOB_DIR"

decode_b64() {
  python3 -c 'import base64,sys; print(base64.b64decode(sys.argv[1]).decode())' "$1"
}

CMD="$(decode_b64 "$CMD_B64")"

cat > "$JOB_DIR/run.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "__WORKDIR__"
export CUDA_VISIBLE_DEVICES="__GPU_INDEX__"
python3 - <<'PY'
from pathlib import Path
import time
Path("__STARTED_AT_FILE__").write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z'), encoding='utf-8')
PY
echo running > "__STATUS_FILE__"
set +e
if [[ "__ENV_MODE__" == "prefix" ]]; then
  "__CONDA_BIN__" run -p "__ENV_VALUE__" bash -lc '__CMD__' >>"__STDOUT_LOG__" 2>>"__STDERR_LOG__"
elif [[ "__ENV_MODE__" == "env" ]]; then
  "__CONDA_BIN__" run -n "__ENV_VALUE__" bash -lc '__CMD__' >>"__STDOUT_LOG__" 2>>"__STDERR_LOG__"
else
  ENV_SNIPPET="__ENV_VALUE__"
  if [[ -n "$ENV_SNIPPET" ]]; then
    eval "$ENV_SNIPPET"
  fi
  bash -lc '__CMD__' >>"__STDOUT_LOG__" 2>>"__STDERR_LOG__"
fi
RC=$?
printf '%s\n' "$RC" > "__EXIT_CODE_FILE__"
if [[ "$RC" -eq 0 ]]; then
  echo finished > "__STATUS_FILE__"
else
  echo failed > "__STATUS_FILE__"
fi
exit "$RC"
EOF

python3 - <<'PY' "$JOB_DIR/run.sh" "$WORKDIR" "$GPU_INDEX" "$STATUS_FILE" "$CONDA_BIN" "$ENV_MODE" "$ENV_VALUE" "$CMD" "$STDOUT_LOG" "$STDERR_LOG" "$EXIT_CODE_FILE" "$JOB_DIR/started_at"
from pathlib import Path
import sys

path = Path(sys.argv[1])
repl = {
    "__WORKDIR__": sys.argv[2],
    "__GPU_INDEX__": sys.argv[3],
    "__STATUS_FILE__": sys.argv[4],
    "__CONDA_BIN__": sys.argv[5],
    "__ENV_MODE__": sys.argv[6],
    "__ENV_VALUE__": sys.argv[7],
    "__CMD__": sys.argv[8],
    "__STDOUT_LOG__": sys.argv[9],
    "__STDERR_LOG__": sys.argv[10],
    "__EXIT_CODE_FILE__": sys.argv[11],
    "__STARTED_AT_FILE__": sys.argv[12],
}
text = path.read_text(encoding="utf-8")
for key, value in repl.items():
    text = text.replace(key, value.replace("\\", "\\\\").replace("'", "'\"'\"'"))
path.write_text(text, encoding="utf-8")
PY

chmod +x "$JOB_DIR/run.sh"
: > "$STDOUT_LOG"
: > "$STDERR_LOG"
echo queued > "$STATUS_FILE"
nohup "$JOB_DIR/run.sh" < /dev/null > /dev/null 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
printf '%s\n' "$PID"
