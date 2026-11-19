#!/usr/bin/env bash
# One job, one machine, engines run sequentially; preserve all failures.
set -uo pipefail
: "${BENCH_DEVICE:?}" "${BENCH_TOKENS:?}" "${BENCH_RUNS:?}" "${RUNNER_TEMP:?}"
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
failed=0
for engine in magnetar llama_server ollama airllm; do
  echo "::group::Benchmark $engine"
  if BENCH_ENGINE="$engine" bash benchmark/campaign_engine.sh 2>&1 | tee "$results/$engine.log"; then
    printf '%s\t%s\n' "$engine" success >> "$status"
  else
    printf '%s\t%s\n' "$engine" failed >> "$status"
    failed=1
  fi
  echo '::endgroup::'
done
printf 'strata\tunavailable: Qwen2.5 is not a documented Strata model\n' >> "$status"
if ! python3 benchmark/campaign_report.py "$results" | tee "$results/comparison.md"; then failed=1; fi
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then cat "$results/comparison.md" >> "$GITHUB_STEP_SUMMARY"; fi
exit "$failed"
