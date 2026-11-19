#!/usr/bin/env bash
# Experimental two-GPU Strata run; model files stay on the dedicated runner PVC.
set -euo pipefail
: "${BENCH_TOKENS:?}" "${BENCH_RUNS:?}" "${RUNNER_TEMP:?}"
if [[ ! "$BENCH_TOKENS" =~ ^[1-9][0-9]*$ || ! "$BENCH_RUNS" =~ ^[1-9][0-9]*$ ]] || (( BENCH_TOKENS < 2 || BENCH_TOKENS > 2048 )); then
  echo 'Expected 2 to 2048 generated tokens and a positive run count' >&2
  exit 1
fi
py=/opt/strata/terminal/python/bin/python3
strata=/opt/strata/terminal/bin/strata
cache=/cache/magnetar-flash-next-q2/data
results="$RUNNER_TEMP/benchmark-results"
runtime="$RUNNER_TEMP/flash-next-runtime"
mkdir -p "$results" "$runtime"/{home,state,config,install}
export HOME="$runtime/home" XDG_STATE_HOME="$runtime/state" XDG_CONFIG_HOME="$runtime/config"
export STRATA_INSTALL_DIR="$runtime/install" CUDA_VISIBLE_DEVICES=0,1 STRATA_DL_CONNECTIONS=4 PYTHONUNBUFFERED=1
export STRATA_API_KEY="$("$py" -c 'import secrets; print(secrets.token_hex(24))')"
printf '%s\n' "$BENCH_MODEL_PROFILE" > "$results/model-profile.txt"
printf '%s\n' \
  $'magnetar\tunavailable: Flash Next architecture is not integrated' \
  $'candle\tunavailable: Flash Next is not integrated in this adapter' \
  $'llama_server\tunavailable: Flash Next is not integrated in this campaign' \
  $'ollama\tunavailable: Flash Next is not integrated in this campaign' \
  $'airllm\tunavailable: Flash Next is not integrated in this campaign' \
  $'strata\tfailed' > "$results/status.tsv"
telemetry_pid=
engine_log="$runtime/state/strata/launch/q2_0-4k.engine.log"
cleanup() {
  local code=$?
  trap - EXIT
  if [[ -n "$telemetry_pid" ]]; then kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true; fi
  "$strata" launch --stop >/dev/null 2>&1 || true
  if [[ -f "$engine_log" ]]; then cp "$engine_log" "$results/strata-engine.log" || true; fi
  if ! "$py" benchmark/campaign_report.py "$results" > "$results/comparison.md"; then code=1; fi
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" && -f "$results/comparison.md" ]]; then cat "$results/comparison.md" >> "$GITHUB_STEP_SUMMARY"; fi
  exit "$code"
}
trap cleanup EXIT
"$py" -c 'import os,sys; sys.exit(0 if os.path.ismount("/cache") else "A mounted /cache volume is required")'
mkdir -p "$cache"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv | tee "$results/gpu-info.csv"
if (( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) < 2 )); then
  echo 'Flash Next requires both GPUs to be visible' >&2; exit 1
fi
df -h /cache | tee "$results/cache-space.txt"
"$strata" launch --family qwen --model q2_0 --context 4096 --parallel 1 --gpu 0,1 --vision off \
  --drafting off --tune skip --access local --port 18080 --api-key-env STRATA_API_KEY \
  --data-dir "$cache" --yes --no-tui
if ! grep -Eq 'layer split: CUDA0 runs layers' "$engine_log" || ! grep -Eq 'layer split: CUDA1 runs layers' "$engine_log"; then
  echo 'Strata did not confirm a layer split over both GPUs; inspect strata-engine.log' >&2; exit 1
fi
model="$("$py" -c 'import json,os,urllib.request; r=urllib.request.Request("http://127.0.0.1:18080/v1/models",headers={"Authorization":"Bearer "+os.environ["STRATA_API_KEY"]}); print(json.load(urllib.request.urlopen(r,timeout=30))["data"][0]["id"])')"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv --loop=1 > "$results/gpu-usage.csv" &
telemetry_pid=$!
"$py" benchmark/run.py strata --device gpu --url http://127.0.0.1:18080 --model "$model" \
  --tokens "$BENCH_TOKENS" --runs "$BENCH_RUNS" --revision strata-engine-0.1.62 \
  --notes 'Q2_0 GGUF; two-GPU layer split checked in strata-engine.log; vision and drafting disabled' --output-dir "$results"
sed -i '$d' "$results/status.tsv"
printf 'strata\tsuccess\n' >> "$results/status.tsv"
