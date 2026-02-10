#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: codex_monitor.sh [--interval <seconds>] [--state-dir <dir>] [--once]
                        [--repo-root <dir>] [--master-prompt <file>]

Continuously delegates experiment monitoring/triage to Codex.
Codex can inspect jobs, stop unhealthy runs, patch bugs, and relaunch.
Exits when max calls is exceeded (or after one call with --once).

Environment:
  CODEX_CMD        Codex CLI binary (default: codex)
  CODEX_ARGS       Extra args for Codex CLI (default: --dangerously-bypass-approvals-and-sandbox)
  MAX_CALLS        Max Codex calls before exit (default: 400)
  JOB_NAME_REGEX   Regex for target job names (default: dwnt_e2former|md22|e2former_fmm|hybrid)
  SUBMIT_SCRIPT    Default relaunch script (default: scripts/submit_md22_dwnt_jobs.sh)
  MASTER_PROMPT    Path to master_prompt.md (default: <script_dir>/master_prompt.md)
  SLACK_BOT_TOKEN  Slack bot token (recommended; enables DM steering + status updates)
  SLACK_CHANNEL    Slack channel ID to post/read (DM channel ID recommended)
  SLACK_USER_ID    If SLACK_CHANNEL is unset, DM this Slack user and cache the DM channel ID
  WANDB_API_KEY    Weights & Biases API key for experiment tracking
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

INTERVAL="600"
ONCE="0"
STATE_DIR="${HOME}/.codexmon"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
MASTER_PROMPT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval) INTERVAL="${2:-}"; shift 2 ;;
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

mkdir -p "$STATE_DIR"

STATE="$STATE_DIR/state.json"
REPORT="$STATE_DIR/report.json"
ALERTS_LOG="$STATE_DIR/alerts.log"

CODEX_CMD="${CODEX_CMD:-codex}"
CODEX_ARGS="${CODEX_ARGS:---dangerously-bypass-approvals-and-sandbox}"
MAX_CALLS="${MAX_CALLS:-400}"
JOB_NAME_REGEX="${JOB_NAME_REGEX:-dwnt_e2former|md22|e2former_fmm|hybrid}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-scripts/submit_md22_dwnt_jobs.sh}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"
SLACK_BOT_TOKEN="${SLACK_BOT_TOKEN:-}"
SLACK_CHANNEL="${SLACK_CHANNEL:-}"
SLACK_USER_ID="${SLACK_USER_ID:-}"

# Weights & Biases API key for experiment tracking:
# set `WANDB_API_KEY` in your environment or run `wandb login` (do not hardcode keys here).

# Master prompt file (experiment-specific goals)
if [[ -z "$MASTER_PROMPT" ]]; then
  MASTER_PROMPT="$SCRIPT_DIR/master_prompt.md"
fi

TMP_DIR="$(mktemp -d /tmp/codexmon.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

# Initialize state file if missing
init_state() {
  [[ -f "$STATE" ]] && return
  cat > "$STATE" <<'JSON'
{"last_summary":"","last_alert_level":"none","last_prompt_ts":0,"call_count":0}
JSON
}

read_state() {
  python3 -c "import json; print(json.load(open('$STATE')).get('$1', ''))"
}

write_state() {
  python3 - "$STATE" "$1" "$2" "$3" "$4" <<'PY'
import json, sys
f, summary, alert, ts, calls = sys.argv[1:]
s = json.load(open(f))
s.update({"last_summary": summary, "last_alert_level": alert,
          "last_prompt_ts": int(ts), "call_count": int(calls)})
json.dump(s, open(f, "w"))
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

  channel_id="$(python3 - <<'PY' <<<"$resp"
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get("channel", "") or "")
except Exception:
    print("")
PY
)"

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

  local resp
  resp="$(curl -s -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
    "https://slack.com/api/auth.test" || true)"

  python3 - "$cache_dir" <<'PY' <<<"$resp"
import json, os, sys
cache_dir = sys.argv[1]
try:
    d = json.load(sys.stdin)
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
PY
}

default_master_prompt() {
  cat <<'EOF'
You are Codex acting as an autonomous MD22 experiment operator in this repository.
Primary goal: reach strong MD22 results while keeping experiments healthy.

At each cycle:
1) Check active SLURM jobs and recent logs (outputs/slurm and wandb local logs).
2) Assess run health (healthy / stalled / diverging / crashed).
3) If a run is unhealthy, stop it (scancel), identify root cause, patch code/config, and relaunch.
4) If no target jobs are running, launch new runs using the default submit scripts.
5) Keep changes minimal and practical; prioritize concrete progress over long explanations.
6) Treat the "User Feedback" section as high-priority steering instructions from the user.

Return exactly one JSON object on one line at the end:
{"alert_level":"none|info|warning|critical","summary":"...","recommendation":"..."}
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
  local prior_summary="$1" prior_alert="$2" call_count="$3"

  local master_content
  master_content="$(load_master_prompt)"

  # Read feedback from Slack channel
  local feedback=""
  feedback="$(read_slack_feedback)"
  printf '%s' "$feedback" > "$TMP_DIR/feedback.txt"

  local status_snapshot=""
  status_snapshot="$(job_snapshot)"
  printf '%s' "$status_snapshot" > "$TMP_DIR/status.txt"

  cat > "$TMP_DIR/prompt.txt" <<EOF
$master_content

Repository: $REPO_ROOT
Target job regex: $JOB_NAME_REGEX
Default submit script: $SUBMIT_SCRIPT

${status_snapshot:+## Current Status (local snapshot)
$status_snapshot

}

${feedback:+## User Feedback
$feedback

}---

Return exactly one line of JSON:
{"alert_level":"none|info|warning|critical","summary":"...","recommendation":"what to do next or try in next run"}

Prior summary: $prior_summary
Prior alert: $prior_alert
Call count: $call_count / $MAX_CALLS
EOF
}

job_snapshot() {
  local user="${USER:-}"
  local header="JOBID NAME ST TIME NODES NODELIST(REASON)"
  local sq
  sq="$(squeue -u "$user" -o '%.18i %.25j %.2t %.10M %.6D %R' 2>/dev/null || true)"
  if [[ -z "$sq" ]]; then
    echo "(squeue unavailable or no jobs)"
    return 0
  fi

  local filtered
  filtered="$(echo "$sq" | awk 'NR==1{print; next} {print}' | { head -n1; tail -n +2 | rg -E "$JOB_NAME_REGEX" || true; })"
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
        if "alert_level" in obj and "summary" in obj:
            print(json.dumps(obj, ensure_ascii=False))
            sys.exit(0)
    except Exception:
        pass

print('{"alert_level":"info","summary":"(no valid JSON)","recommendation":""}')
PY
}

get_json_field() {
  python3 -c "import json; print(json.loads(open('$REPORT').readline()).get('$1',''))"
}

send_slack() {
  local message="$1"
  # Use bot token to post to DM/channel
  if [[ -n "$SLACK_BOT_TOKEN" && -n "$SLACK_CHANNEL" ]]; then
    curl -s -X POST -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
      -H 'Content-type: application/json' \
      --data "{\"channel\":\"$SLACK_CHANNEL\",\"text\":\"$message\"}" \
      "https://slack.com/api/chat.postMessage" >/dev/null 2>&1 || true
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
init_state

while true; do
  ensure_slack_channel
  slack_bot_identity

  last_summary="$(read_state last_summary)"
  last_alert="$(read_state last_alert_level)"
  last_prompt_ts="$(read_state last_prompt_ts)"
  call_count="$(read_state call_count)"

  # Bail if max calls exceeded
  if (( call_count >= MAX_CALLS )); then
    echo "Max Codex calls ($MAX_CALLS) reached. Exiting."
    send_slack "🛑 Monitor stopped: max calls ($MAX_CALLS) reached"
    exit 0
  fi

  now_ts="$(date +%s)"

  # Build prompt and call Codex
  build_prompt "$last_summary" "$last_alert" "$call_count"
  
  # Emit a concise status snapshot to Slack before running Codex.
  if [[ -n "$SLACK_BOT_TOKEN" && -n "$SLACK_CHANNEL" ]]; then
    status="$(cat "$TMP_DIR/status.txt" 2>/dev/null || true)"
    feedback_preview="$(cat "$TMP_DIR/feedback.txt" 2>/dev/null || true)"
    send_slack "📡 [Monitor #$((call_count + 1))] Status snapshot\n${status}\n\nNew steering (if any):\n${feedback_preview:-<none>}"
  fi
  
  run_codex
  parse_json "$TMP_DIR/last_message.txt" > "$REPORT"
  call_count=$(( call_count + 1 ))

  # Extract response fields
  alert_level="$(get_json_field alert_level)"
  summary="$(get_json_field summary)"
  recommendation="$(get_json_field recommendation)"

  # Log alerts and recommendations
  if [[ "$alert_level" == "warning" || "$alert_level" == "critical" ]]; then
    echo "[$(date)] $alert_level: $summary" >> "$ALERTS_LOG"
    [[ -n "$recommendation" ]] && echo "  -> $recommendation" >> "$ALERTS_LOG"
  fi

  # Always log recommendation to a separate file
  if [[ -n "$recommendation" ]]; then
    echo "[$(date)] $recommendation" >> "$STATE_DIR/recommendations.log"
  fi

  # Send Slack notification (Codex decision)
  send_slack "[Monitor #$call_count] $alert_level: $summary\nNext: $recommendation"

  # Update state
  write_state "$summary" "$alert_level" "$now_ts" "$call_count"

  # Single iteration mode
  [[ "$ONCE" == "1" ]] && exit 0

  sleep "$INTERVAL"
done
