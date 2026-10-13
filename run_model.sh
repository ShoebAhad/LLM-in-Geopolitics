#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
PY=${PY:-python}
TAG=$1; MODEL=$2; EXTRA=${3:-}
URL=${BASE_URL:-http://192.168.100.145:4000/v1}
COMMON=(--model "$MODEL" --base-url "$URL" --langs en bn --workers "${WORKERS:-32}" --tpm 100000000
        --max-tokens 4000 --max-tokens-cot 16000 --schema codebook --split all --out results/v2)
[ -n "$EXTRA" ] && COMMON+=(--extra-body "$EXTRA")
[ "${PROVIDER:-}" = anthropic ] && COMMON+=(--provider anthropic --no-temp --no-seed)
mkdir -p results/v2/logs
LOG=results/v2/logs/run_$TAG.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
run() { log "=== $*"; "$PY" -u run_eval.py "${COMMON[@]}" "$@" 2>&1 | tr '\r' '\n' | grep -E "^done|WARN|items \(|Error" | tee -a "$LOG"; }
ready() {
  [ "${PROVIDER:-}" = anthropic ] && return 0
  for i in $(seq 1 45); do
    curl -s -m 90 "$URL/chat/completions" -H "Authorization: Bearer $(cat ~/.bfpn_api_key)" -H "Content-Type: application/json" \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":50}" | grep -q '"choices"' && return 0
    log "model not answering (try $i/45)"; sleep 120
  done; return 1
}
OK=0
for attempt in 1 2 3 4 5; do
  log "attempt $attempt: waiting for $MODEL"; ready || { log "model never came back"; continue; }
  run --tag "$TAG-codebook" --conditions zs --runs 0
  run --tag "$TAG-codebook" --conditions fs --runs 0
  run --tag "$TAG-balex" --exemplars balanced --conditions fs --runs 0
  run --tag "$TAG-codebook" --conditions zs --runs 1 2 3 4 --temp 0.7
  run --tag "$TAG-codebook" --conditions zs_cot --runs 0
  run --tag "$TAG-codebook" --conditions fs_cot --runs 0
  run --tag "$TAG-codebook" --conditions fs_cot --runs 1 2 --temp 0.7
  if "$PY" check_complete.py "$TAG" >> "$LOG" 2>&1; then OK=1; log "all planned runs complete"; break; fi
  log "incomplete after attempt $attempt; retrying in 2 min"; sleep 120
done
STATUS=$([ $OK = 1 ] && echo complete || echo INCOMPLETE)
log "running analysis over all models (status=$STATUS)"
"$PY" analyze.py --include-dev --exclude bfpn_1719f47ad4 --B 10000 > results/v2/logs/final_analysis.log 2>&1 || log "ANALYSIS FAILED, see results/v2/logs/final_analysis.log"
echo "$STATUS $(date '+%F %T')" > "results/v2/DONE_$TAG"
log "DONE ($STATUS)"
