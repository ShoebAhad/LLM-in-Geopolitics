#!/usr/bin/env bash
# ONE command = the whole experiment for one model (same 527 items, same prompts, same runs as every other model).
# Waits for the model, runs every stage, retries until nothing is missing, then re-runs the analysis over all models.
#   PY=~/.venvs/bfpn/bin/python ./run_model.sh <tag> <gateway-model-name> ['<extra-body json>']
# WORKERS=<n> lowers the number of parallel requests (default 32); it does not change any prompt or result.
#   e.g. ./run_model.sh gpt-oss-20b gpt-oss-20b '{"reasoning_effort":"low","allowed_openai_params":["reasoning_effort"]}'
#        ./run_model.sh qwen38-27b Qwen3.8-27B '{"chat_template_kwargs":{"enable_thinking":false}}'
# Closed Claude model (key in ~/.anthropic_api_key; no gateway, no fixed temperature or seed, so run 0 is not deterministic):
#        PROVIDER=anthropic WORKERS=8 ./run_model.sh haiku55 claude-haiku-5-5 '{"thinking":{"type":"disabled"}}'
# Output: results/v2/<tag>-codebook, results/v2/<tag>-balex, log results/v2/logs/run_<tag>.log, marker results/v2/DONE_<tag>.
# Resumable: finished items are skipped, so re-running after a crash or reboot only fills what is missing.
set -u
cd "$(dirname "$0")"
PY=${PY:-python}
TAG=$1; MODEL=$2; EXTRA=${3:-}
URL=${BASE_URL:-http://192.168.100.145:4000/v1}   # OpenAI-compatible endpoint (vLLM / LiteLLM gateway)
COMMON=(--model "$MODEL" --base-url "$URL" --langs en bn --workers "${WORKERS:-32}" --tpm 100000000
        --max-tokens 4000 --max-tokens-cot 16000 --schema codebook --split all --out results/v2)
[ -n "$EXTRA" ] && COMMON+=(--extra-body "$EXTRA")
[ "${PROVIDER:-}" = anthropic ] && COMMON+=(--provider anthropic --no-temp --no-seed)
mkdir -p results/v2/logs
LOG=results/v2/logs/run_$TAG.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
run() { log "=== $*"; "$PY" -u run_eval.py "${COMMON[@]}" "$@" 2>&1 | tr '\r' '\n' | grep -E "^done|WARN|items \(|Error" | tee -a "$LOG"; }
ready() {   # wait up to 90 min for the model to answer
  [ "${PROVIDER:-}" = anthropic ] && return 0   # hosted API: nothing to wait for
  for i in $(seq 1 45); do
    curl -s -m 90 "$URL/chat/completions" -H "Authorization: Bearer $(cat ~/.bfpn_api_key)" -H "Content-Type: application/json" \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":50}" | grep -q '"choices"' && return 0
    log "model not answering (try $i/45)"; sleep 120
  done; return 1
}
OK=0
for attempt in 1 2 3 4 5; do
  log "attempt $attempt: waiting for $MODEL"; ready || { log "model never came back"; continue; }
  run --tag "$TAG-codebook" --conditions zs --runs 0                      # zero-shot (primary)
  run --tag "$TAG-codebook" --conditions fs --runs 0                      # few-shot
  run --tag "$TAG-balex" --exemplars balanced --conditions fs --runs 0    # few-shot, second (balanced) exemplar set
  run --tag "$TAG-codebook" --conditions zs --runs 1 2 3 4 --temp 0.7     # zero-shot variance: 5 runs in total
  run --tag "$TAG-codebook" --conditions zs_cot --runs 0                  # zero-shot + chain-of-thought
  run --tag "$TAG-codebook" --conditions fs_cot --runs 0                  # few-shot + chain-of-thought
  run --tag "$TAG-codebook" --conditions fs_cot --runs 1 2 --temp 0.7     # few-shot + CoT variance: 3 runs in total
  if "$PY" check_complete.py "$TAG" >> "$LOG" 2>&1; then OK=1; log "all planned runs complete"; break; fi
  log "incomplete after attempt $attempt; retrying in 2 min"; sleep 120
done
STATUS=$([ $OK = 1 ] && echo complete || echo INCOMPLETE)
log "running analysis over all models (status=$STATUS)"
"$PY" analyze.py --include-dev --exclude bfpn_1719f47ad4 --B 10000 > results/v2/logs/final_analysis.log 2>&1 || log "ANALYSIS FAILED, see results/v2/logs/final_analysis.log"
echo "$STATUS $(date '+%F %T')" > "results/v2/DONE_$TAG"
log "DONE ($STATUS)"
