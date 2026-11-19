#!/usr/bin/env bash
# One job, one machine, engines run sequentially; preserve all failures.
set -uo pipefail
: "${BENCH_DEVICE:?}" "${BENCH_TOKENS:?}" "${BENCH_RUNS:?}" "${RUNNER_TEMP:?}"
BENCH_MODEL_PROFILE=${BENCH_MODEL_PROFILE:-qwen2.5-0.5b}
export BENCH_MODEL_PROFILE
case "$BENCH_MODEL_PROFILE" in qwen2.5-0.5b|qwen3.6-35b-experimental) ;; *) echo 'Unknown model profile' >&2; exit 1 ;; esac
if [[ ! "$BENCH_DEVICE" =~ ^(cpu|gpu)$ || ! "$BENCH_TOKENS" =~ ^[1-9][0-9]*$ || ! "$BENCH_RUNS" =~ ^[1-9][0-9]*$ ]] || (( BENCH_TOKENS < 2 )); then
  echo 'Expected cpu or gpu, tokens >= 2 and a positive run count' >&2
  exit 1
fi
# One GPU is shared sequentially by all engines, even on multi-GPU runners.
if [[ "$BENCH_DEVICE" == gpu ]]; then export CUDA_VISIBLE_DEVICES=0; else export CUDA_VISIBLE_DEVICES=""; fi
results="$RUNNER_TEMP/benchmark-results"
mkdir -p "$results"
status="$results/status.tsv"
: > "$status"
printf '%s\n' "$BENCH_MODEL_PROFILE" > "$results/model-profile.txt"
failed=0
if [[ "$BENCH_MODEL_PROFILE" == qwen2.5-0.5b ]]; then
  engines=(magnetar candle llama_server ollama airllm)
else
  printf 'magnetar\tunavailable: Qwen3.6 is not supported by this qualification campaign\n' >> "$status"
  printf 'candle\tunavailable: Qwen3.6 is not integrated in this adapter\n' >> "$status"
  printf 'ollama\tunavailable: Qwen3.6 is not integrated in this campaign\n' >> "$status"
  printf 'airllm\tunavailable: Qwen3.6 is not integrated in this campaign\n' >> "$status"
  engines=(llama_server)
  if [[ "$BENCH_DEVICE" == gpu ]]; then engines+=(strata_native); else
    printf 'strata_native\tunavailable: the Qwen3.6 engine requires a supported GPU\n' >> "$status"
  fi
fi
for engine in "${engines[@]}"; do
  echo "::group::Benchmark $engine"
  if BENCH_ENGINE="$engine" bash benchmark/campaign_engine.sh 2>&1 | tee "$results/$engine.log"; then
    printf '%s\t%s\n' "$engine" success >> "$status"
  else
    printf '%s\t%s\n' "$engine" failed >> "$status"
    failed=1
  fi
  echo '::endgroup::'
done
if [[ "$BENCH_MODEL_PROFILE" == qwen2.5-0.5b ]]; then
  printf 'strata\tunavailable: Qwen2.5 is not a documented Strata model\n' >> "$status"
fi
if ! python3 benchmark/campaign_report.py "$results" | tee "$results/comparison.md"; then failed=1; fi
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then cat "$results/comparison.md" >> "$GITHUB_STEP_SUMMARY"; fi
exit "$failed"
