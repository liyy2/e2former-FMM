#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: codex_monitor.sh [--interval <seconds>] [--work-dir <dir>] [--once]
                        [--repo-root <dir>] [--master-prompt <file>]

Continuously delegates experiment monitoring/triage to Codex.
Codex can inspect jobs, stop unhealthy runs, patch bugs, and relaunch.
Exits when max calls is exceeded (or after one call with --once).

Work directory contents (default: <repo_root>/outputs/codex_monitor):
  - memory.md: append-only monitor memory (used for prompting and audit)
  - last_report.json: last model JSON response
  - slack/: cached Slack IDs to avoid feedback loops

Environment:
  CODEX_CMD        Codex CLI binary (default: codex)
  CODEX_ARGS       Extra args for Codex CLI (default: --dangerously-bypass-approvals-and-sandbox)
  MAX_CALLS        Max Codex calls before exit (default: 400)
  JOB_NAME_REGEX   Regex for target job names (default: dwnt_e2former|md22|e2former_fmm|hybrid)
  SUBMIT_SCRIPT    Default relaunch script (default: scripts/submit_md22_dwnt_jobs.sh)
  MASTER_PROMPT    Path to master_prompt.md (default: <script_dir>/master_prompt.md; CLI --master-prompt overrides)
  SLACK_BOT_TOKEN  Slack bot token (recommended; enables DM steering + status updates)
  SLACK_CHANNEL    Slack channel ID to post/read (DM channel ID recommended)
  SLACK_USER_ID    If SLACK_CHANNEL is unset, DM this Slack user and cache the DM channel ID

Notes:
  --state-dir is a deprecated alias for --work-dir.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

INTERVAL="${INTERVAL:-1200}"
ONCE="${ONCE:-0}"
STATE_DIR="${STATE_DIR:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
MASTER_PROMPT="${MASTER_PROMPT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval) INTERVAL="${2:-}"; shift 2 ;;
    --work-dir) STATE_DIR="${2:-}"; shift 2 ;;
    --state-dir) STATE_DIR="${2:-}"; shift 2 ;;
    --repo-root) REPO_ROOT="${2:-}"; shift 2 ;;
    --master-prompt) MASTER_PROMPT="${2:-}"; shift 2 ;;
    --once)     ONCE="1"; shift ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$STATE_DIR" ]]; then
  # Default state directory lives inside the repo (and under outputs/ which is gitignored).
  STATE_DIR="${REPO_ROOT}/outputs/codex_monitor"
fi

mkdir -p "$STATE_DIR"

MEMORY_FILE="$STATE_DIR/memory.md"
LAST_REPORT="$STATE_DIR/last_report.json"

CODEX_CMD="${CODEX_CMD:-codex}"
CODEX_ARGS="${CODEX_ARGS:---dangerously-bypass-approvals-and-sandbox}"
MAX_CALLS="${MAX_CALLS:-400}"
JOB_NAME_REGEX="${JOB_NAME_REGEX:-dwnt_e2former|md22|e2former_fmm|hybrid}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-scripts/submit_md22_dwnt_jobs.sh}"
SLACK_BOT_TOKEN="${SLACK_BOT_TOKEN:-}"
SLACK_CHANNEL="${SLACK_CHANNEL:-}"
SLACK_USER_ID="${SLACK_USER_ID:-}"

# Master prompt file (experiment-specific goals)
if [[ -z "$MASTER_PROMPT" ]]; then
  MASTER_PROMPT="$SCRIPT_DIR/master_prompt.md"
fi

TMP_DIR="$(mktemp -d /tmp/codexmon.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

# Monitor memory: append-only log used for prompting and audit.
memory_cycle_count() {
  python3 - "$MEMORY_FILE" <<'PY'
import re, sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.exists():
    print(0)
    raise SystemExit(0)

text = p.read_text(encoding="utf-8", errors="ignore")
print(len(re.findall(r"^### Cycle\\s+\\d+\\b", text, flags=re.M)))
PY
}

memory_tail() {
  local max_chars="${1:-8000}"
  python3 - "$MEMORY_FILE" "$max_chars" <<'PY'
import sys
from pathlib import Path

p = Path(sys.argv[1])
max_chars = int(sys.argv[2])
if not p.exists():
    print("")
    raise SystemExit(0)

text = p.read_text(encoding="utf-8", errors="ignore")
if len(text) > max_chars:
    text = text[-max_chars:]
print(text)
PY
}

slack_cache_dir() {
  echo "$STATE_DIR/slack"
}

ensure_slack_channel() {
  [[ -z "$SLACK_BOT_TOKEN" ]] && return 0

  local cache_dir
  cache_dir="$(slack_cache_dir)"
  mkdir -p "$cache_dir"

  # Persist user id for convenience across restarts.
  if [[ -z "$SLACK_USER_ID" && -f "$cache_dir/user_id" ]]; then
    SLACK_USER_ID="$(cat "$cache_dir/user_id" 2>/dev/null || true)"
  fi
  if [[ -n "$SLACK_USER_ID" ]]; then
    echo "$SLACK_USER_ID" > "$cache_dir/user_id"
  fi

  # If already configured, persist it for restarts and return.
  if [[ -n "$SLACK_CHANNEL" ]]; then
    echo "$SLACK_CHANNEL" > "$cache_dir/channel_id"
    return 0
  fi

  # Reuse cached channel id if present.
  if [[ -f "$cache_dir/channel_id" ]]; then
    SLACK_CHANNEL="$(cat "$cache_dir/channel_id" 2>/dev/null || true)"
    [[ -n "$SLACK_CHANNEL" ]] && return 0
  fi

  # If user id is available, send a one-time handshake DM to open the DM channel
  # and cache its channel id for future reads/writes.
  [[ -z "$SLACK_USER_ID" ]] && return 0

  local resp channel_id
  resp="$(curl -s -X POST \
    -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
    -H "Content-type: application/x-www-form-urlencoded" \
    --data-urlencode "channel=$SLACK_USER_ID" \
    --data-urlencode "text=codex_monitor: online" \
    "https://slack.com/api/chat.postMessage" || true)"

  channel_id="$(python3 -c 'import json,sys
try:
    d = json.loads(sys.stdin.read() or "{}")
    print(d.get("channel", "") or "")
except Exception:
    print("")' <<<"$resp" 2>/dev/null || true)"

  if [[ -n "$channel_id" ]]; then
    SLACK_CHANNEL="$channel_id"
    echo "$SLACK_CHANNEL" > "$cache_dir/channel_id"
  fi
}

slack_bot_identity() {
  [[ -z "$SLACK_BOT_TOKEN" ]] && return 0
  local cache_dir
  cache_dir="$(slack_cache_dir)"
  mkdir -p "$cache_dir"

  # Cache bot user_id and bot_id to avoid feedback loops (ignore our own messages).
  if [[ -f "$cache_dir/bot_user_id" && -f "$cache_dir/bot_id" ]]; then
    return 0
  fi

  # If IDs are already provided (via env/defaults), persist them and skip the API call.
  if [[ -n "${SLACK_BOT_USER_ID:-}" && -n "${SLACK_BOT_ID:-}" ]]; then
    echo "$SLACK_BOT_USER_ID" > "$cache_dir/bot_user_id"
    echo "$SLACK_BOT_ID" > "$cache_dir/bot_id"
    return 0
  fi

  local resp
  resp="$(curl -s -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
    "https://slack.com/api/auth.test" || true)"

  python3 -c 'import json, os, sys
cache_dir = sys.argv[1]
try:
    d = json.loads(sys.stdin.read() or "{}")
except Exception:
    d = {}
user_id = d.get("user_id") or ""
bot_id = d.get("bot_id") or ""
if user_id:
    with open(os.path.join(cache_dir, "bot_user_id"), "w") as f:
        f.write(user_id)
if bot_id:
    with open(os.path.join(cache_dir, "bot_id"), "w") as f:
        f.write(bot_id)
' "$cache_dir" <<<"$resp" >/dev/null 2>&1 || true
}

default_master_prompt() {
  cat <<'EOF'
You are Codex acting as an autonomous MD22 experiment operator in this repository.
Primary goal: reach strong MD22 results while keeping experiments healthy. Never stop experiments unless they are unhealthy. 
A typical experiment takes 1-2 days to complete. Another rule is that you should tune the FMM variant (hybrid or long-range) to surpass the baseline (short-range).
Launch as many experiments as possible to reach the goal.
You should always check the status of the experiments before making any decisions.
also minimize resume from checkpoint. try to start from scratch if possible.
try to use distributed training to speed up the training if possible.
At each cycle:
1) Check active SLURM jobs and recent logs (outputs/slurm and wandb local logs).
2) Assess run health (healthy / stalled / diverging / crashed).
3) If a run is unhealthy, stop it (scancel), identify root cause, patch code/config, and relaunch.
4) If no target jobs are running, launch new runs using the default submit scripts.
5) Keep changes minimal and practical; prioritize concrete progress over long explanations.
6) Treat the "User Feedback" section as high-priority steering instructions from the user.
7) You must maintain monitor memory yourself by appending one entry per cycle to the provided memory file.

Return exactly one JSON object on one line at the end:
{"summary":"...","recommendation":"...","patch_report":"..."}

`patch_report` must describe the key code/config/script patches you actually applied in THIS cycle (not suggestions).
- Slack-ready bullets; include file paths and what changed.
- If you applied no patches this cycle, set "patch_report" to an empty string.
- Ensure the memory entry you wrote is consistent with this JSON.
EOF
}

load_master_prompt() {
  if [[ -f "$MASTER_PROMPT" ]]; then
    cat "$MASTER_PROMPT"
  else
    default_master_prompt
  fi
}

build_prompt() {
  local cycle_num="$1"

  local master_content
  master_content="$(load_master_prompt)"

  # Read feedback from Slack channel
  local feedback=""
  feedback="$(read_slack_feedback)"
  printf '%s' "$feedback" > "$TMP_DIR/feedback.txt"

  local status_snapshot=""
  status_snapshot="$(job_snapshot)"
  printf '%s' "$status_snapshot" > "$TMP_DIR/status.txt"

  local memory_context=""
  memory_context="$(memory_tail 12000)"

  cat > "$TMP_DIR/prompt.txt" <<__CODEXMON_EOF__
$master_content

Repository: $REPO_ROOT
Target job regex: $JOB_NAME_REGEX
Default submit script: $SUBMIT_SCRIPT
Monitor memory file: $MEMORY_FILE
Cycle number to write: $cycle_num

${status_snapshot:+## Current Status (local snapshot)
$status_snapshot

}

${feedback:+## User Feedback
$feedback

}

${memory_context:+## Memory (tail)
$memory_context

}---

Return exactly one line of JSON:
{"summary":"...","recommendation":"what to do next or try in next run","patch_report":"slack-ready key patches applied in this cycle (or empty string)"}

Rules for patch_report:
- Only include patches you actually applied this cycle (code/config/scripts).
- Each bullet must start with a tag like [bugfix], [config], [perf], [logging], [infra].
- Include file paths and what changed (not just why).
- Keep it short: max ~8 bullets / ~1200 chars.

Memory file rules (you must do this yourself in this cycle):
- Append exactly one new block to "$MEMORY_FILE".
- Start block with: "### Cycle $cycle_num (<ISO-8601 timestamp>)".
- Include: summary, recommendation/next, and patch_report.
- Do not rewrite or delete previous memory entries.

Cycle: $cycle_num / $MAX_CALLS
__CODEXMON_EOF__
}

job_snapshot() {
  local user="${USER:-}"
  local sq
  sq="$(squeue -u "$user" -o '%.18i %.25j %.2t %.10M %.6D %R' 2>/dev/null || true)"
  if [[ -z "$sq" ]]; then
    echo "(squeue unavailable or no jobs)"
    return 0
  fi

  local filtered
  filtered="$(echo "$sq" | awk 'NR==1{print; next} {print}' | { head -n1; tail -n +2 | rg -e "$JOB_NAME_REGEX" || true; })"
  if [[ -z "$filtered" ]]; then
    echo "(no matching jobs; regex: $JOB_NAME_REGEX)"
    return 0
  fi

  echo "$filtered"

  # Add a small per-job metric tail (best-effort; avoids spamming huge logs).
  local jobids
  jobids="$(echo "$filtered" | awk 'NR>1{print $1}' | head -n 8)"
  if [[ -z "$jobids" ]]; then
    return 0
  fi

  echo
  echo "Latest metrics (best-effort):"

  while read -r jid; do
    [[ -z "$jid" ]] && continue
    local out_file
    out_file="$(ls -1 outputs/slurm/*-"$jid".out 2>/dev/null | head -n 1 || true)"
    if [[ -z "$out_file" ]]; then
      echo "$jid | (no outputs/slurm/*-$jid.out)"
      continue
    fi
    python3 - "$jid" "$out_file" <<'PY'
import re, sys
jid, path = sys.argv[1], sys.argv[2]
ansi = re.compile(r"\x1b\\[[0-9;]*m")
last_valid = None
last_train = None
try:
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if "| valid_loss=" in line:
                last_valid = line.strip()
            if " | loss=" in line:
                last_train = line.strip()
except OSError:
    pass
line = last_valid or last_train
if not line:
    print(f\"{jid} | (no metric lines yet)\")
else:
    line = ansi.sub('', line)
    # Keep it short for Slack/prompt readability.
    if len(line) > 260:
        line = line[-260:]
    print(f\"{jid} | {line}\")
PY
  done <<< "$jobids"
}

repo_patch_fingerprint() {
  if ! command -v git >/dev/null 2>&1; then
    echo ""
    return 0
  fi
  (
    git -C "$REPO_ROOT" status --porcelain=v1 -uall 2>/dev/null || true
    git -C "$REPO_ROOT" diff --no-ext-diff HEAD 2>/dev/null || true
  ) | python3 -c 'import hashlib,sys; print(hashlib.sha1(sys.stdin.buffer.read()).hexdigest())' 2>/dev/null || true
}

repo_patch_overview() {
  if ! command -v git >/dev/null 2>&1; then
    echo "(git unavailable)"
    return 0
  fi

  local status_lines status_head status_count diff_shortstat
  status_lines="$(git -C "$REPO_ROOT" status --porcelain=v1 -uall 2>/dev/null || true)"
  status_count="$(echo "$status_lines" | wc -l | tr -d ' ' || true)"
  status_head="$(echo "$status_lines" | head -n 60 || true)"
  diff_shortstat="$(git -C "$REPO_ROOT" diff --shortstat HEAD 2>/dev/null | tr -s ' ' || true)"

  if [[ -z "$status_lines" ]]; then
    echo "(clean repo; no changes vs HEAD)"
    return 0
  fi

  echo "git status --porcelain (top 60):"
  echo "$status_head"
  if [[ -n "$status_count" && "$status_count" -gt 60 ]]; then
    echo "... (truncated; $status_count lines)"
  fi
  echo
  echo "git diff --shortstat (vs HEAD):"
  echo "${diff_shortstat:-<empty>}"
}

run_codex() {
  if ! command -v "$CODEX_CMD" >/dev/null 2>&1; then
    echo "Codex not found: $CODEX_CMD" >&2
    exit 1
  fi
  read -r -a args <<< "$CODEX_ARGS"
  # Stream to terminal AND capture to file (unbuffered)
  stdbuf -oL "$CODEX_CMD" exec \
    "${args[@]}" \
    -C "$REPO_ROOT" \
    --output-last-message "$TMP_DIR/last_message.txt" \
    - < "$TMP_DIR/prompt.txt" 2>&1 | tee "$TMP_DIR/response.txt" || true
}

parse_json() {
  python3 - "$1" <<'PY'
import json, sys
path = sys.argv[1]
try:
    text = open(path, encoding="utf-8", errors="ignore").read().strip()
except OSError:
    text = ""
candidates = []

if text:
    candidates.append(text)
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            candidates.append(line)

for candidate in reversed(candidates):
    try:
        obj = json.loads(candidate)
        if "summary" in obj:
            print(json.dumps(obj, ensure_ascii=False))
            sys.exit(0)
    except Exception:
        pass

print('{"summary":"(no valid JSON)","recommendation":"","patch_report":""}')
PY
}

get_json_field() {
  python3 -c "import json; print(json.loads(open('$LAST_REPORT').readline()).get('$1',''))"
}

send_slack() {
  local message="$1"
  # Use bot token to post to DM/channel.
  #
  # IMPORTANT: Do not interpolate raw multi-line strings into JSON in bash.
  # Use Python to JSON-encode the message so newlines/quotes are escaped.
  if [[ -n "$SLACK_BOT_TOKEN" && -n "$SLACK_CHANNEL" ]]; then
    python3 - "$STATE_DIR" "$SLACK_CHANNEL" "$message" <<'PY' >/dev/null 2>&1 || true
import datetime
import json
import os
import sys
import urllib.request

state_dir, channel, text = sys.argv[1], sys.argv[2], sys.argv[3]
token = os.environ.get("SLACK_BOT_TOKEN", "")

if not token or not channel:
    raise SystemExit(0)

payload = json.dumps({"channel": channel, "text": text}).encode("utf-8")
req = urllib.request.Request(
    "https://slack.com/api/chat.postMessage",
    data=payload,
    headers={
        "Authorization": f"Bearer {token}",
        "Content-type": "application/json",
    },
)

try:
    with urllib.request.urlopen(req, timeout=20) as r:
        resp = json.loads(r.read().decode("utf-8"))
except Exception as e:
    try:
        os.makedirs(state_dir, exist_ok=True)
        with open(os.path.join(state_dir, "slack_send.log"), "a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now().isoformat()}] exception: {e}\n")
    except Exception:
        pass
    raise SystemExit(0)

if not resp.get("ok"):
    try:
        os.makedirs(state_dir, exist_ok=True)
        with open(os.path.join(state_dir, "slack_send.log"), "a", encoding="utf-8") as f:
            f.write(
                f"[{datetime.datetime.now().isoformat()}] ok=false error={resp.get('error')}\n"
            )
    except Exception:
        pass
PY
  fi
}

# Read recent messages from Slack channel (feedback from user)
read_slack_feedback() {
  if [[ -z "$SLACK_BOT_TOKEN" || -z "$SLACK_CHANNEL" ]]; then
    echo ""
    return
  fi
  
  local last_ts_file="$STATE_DIR/slack_last_ts"
  local oldest="0"
  [[ -f "$last_ts_file" ]] && oldest="$(cat "$last_ts_file")"
  
  # Fetch messages newer than last check
  local response
  response=$(curl -s -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
    "https://slack.com/api/conversations.history?channel=$SLACK_CHANNEL&oldest=$oldest&limit=10" 2>/dev/null)
  
  # Extract messages and update timestamp
  local cache_dir bot_user_id bot_id
  cache_dir="$(slack_cache_dir)"
  bot_user_id=""
  bot_id=""
  [[ -f "$cache_dir/bot_user_id" ]] && bot_user_id="$(cat "$cache_dir/bot_user_id" 2>/dev/null || true)"
  [[ -f "$cache_dir/bot_id" ]] && bot_id="$(cat "$cache_dir/bot_id" 2>/dev/null || true)"

  python3 - "$response" "$last_ts_file" "$bot_user_id" "$bot_id" <<'PY'
import json, sys
try:
    data = json.loads(sys.argv[1])
    if data.get("ok") and data.get("messages"):
        bot_user_id = sys.argv[3] or ""
        bot_id = sys.argv[4] or ""
        msgs = []
        for m in data["messages"]:
            if m.get("subtype") is not None:
                continue
            if bot_user_id and m.get("user") == bot_user_id:
                continue
            if bot_id and m.get("bot_id") == bot_id:
                continue
            txt = m.get("text", "")
            if txt:
                msgs.append(txt)
        if msgs:
            print("\n".join(reversed(msgs)))
        # Save latest timestamp
        latest = max(float(m.get("ts", "0")) for m in data["messages"])
        if latest > 0:
            with open(sys.argv[2], "w") as f:
                f.write(str(latest))
except: pass
PY
}

# ─────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────
call_count="$(memory_cycle_count)"

while true; do
  ensure_slack_channel
  slack_bot_identity

  cycle_num="$((call_count + 1))"

  # Bail if max calls exceeded
  if (( call_count >= MAX_CALLS )); then
    echo "Max Codex calls ($MAX_CALLS) reached. Exiting."
    send_slack "🛑 Monitor stopped: max calls ($MAX_CALLS) reached"
    exit 0
  fi

  # Build prompt and call Codex
  build_prompt "$cycle_num"
  
  # Emit a concise status snapshot to Slack before running Codex.
  if [[ -n "$SLACK_BOT_TOKEN" && -n "$SLACK_CHANNEL" ]]; then
    status="$(cat "$TMP_DIR/status.txt" 2>/dev/null || true)"
    feedback_preview="$(cat "$TMP_DIR/feedback.txt" 2>/dev/null || true)"
    slack_msg=""
    printf -v slack_msg "📡 [Monitor #%s] Status snapshot\n%s\n\nNew steering (if any):\n%s" \
      "$cycle_num" \
      "${status:-<empty>}" \
      "${feedback_preview:-<none>}"
    send_slack "$slack_msg"
  fi
  
  pre_patch_fp="$(repo_patch_fingerprint)"
  run_codex
  post_patch_fp="$(repo_patch_fingerprint)"
  parse_json "$TMP_DIR/last_message.txt" > "$LAST_REPORT"
  call_count="$cycle_num"

  # Extract response fields
  summary="$(get_json_field summary)"
  recommendation="$(get_json_field recommendation)"
  patch_report="$(get_json_field patch_report)"

  patched="0"
  repo_overview=""
  if [[ -n "$pre_patch_fp" && -n "$post_patch_fp" && "$pre_patch_fp" != "$post_patch_fp" ]]; then
    patched="1"
    repo_overview="$(repo_patch_overview)"
  fi

  # Send Slack notification (patch delta)
  if [[ "$patched" == "1" && -n "$SLACK_BOT_TOKEN" && -n "$SLACK_CHANNEL" ]]; then
    patch_msg=""
    printf -v patch_msg "🧩 [Monitor #%s] Patch report\n%s\n\nRepo changes:\n%s" \
      "$cycle_num" \
      "${patch_report:-<no patch_report provided by model>}" \
      "${repo_overview:-<empty>}"
    send_slack "$patch_msg"
  fi

  # Send Slack notification (Codex decision)
  slack_msg=""
  printf -v slack_msg "[Monitor #%s] %s\nNext: %s" "$cycle_num" "$summary" "$recommendation"
  send_slack "$slack_msg"

  # Single iteration mode
  [[ "$ONCE" == "1" ]] && exit 0

  sleep "$INTERVAL"
done
