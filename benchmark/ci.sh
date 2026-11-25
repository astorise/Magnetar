#!/usr/bin/env bash
# Prepare the selected engine and run one CPU or GPU benchmark cell.
set -euo pipefail

: "${BENCH_ENGINE:?}" "${BENCH_DEVICE:?}" "${BENCH_TOKENS:?}" "${BENCH_RUNS:?}" "${RUNNER_TEMP:?}"
case "$BENCH_DEVICE" in cpu|gpu) ;; *) echo 'Invalid device' >&2; exit 1 ;; esac
case "$BENCH_ENGINE" in magnetar|candle|llama_cpp|llama_server|ollama|strata|airllm) ;; *) echo 'Invalid engine' >&2; exit 1 ;; esac
if [[ ! "$BENCH_TOKENS" =~ ^[1-9][0-9]*$ || ! "$BENCH_RUNS" =~ ^[1-9][0-9]*$ ]]; then
  echo 'Tokens and runs must be positive integers' >&2
  exit 1
fi
if [[ "$BENCH_ENGINE" == magnetar && "$BENCH_TOKENS" == 1 ]]; then
  echo 'Magnetar qualification requires tokens >= 2 to measure decode latency' >&2
  exit 1
fi

work="$RUNNER_TEMP/benchmark-work"
results="$RUNNER_TEMP/benchmark-results"
mkdir -p "$work" "$results"
args=("$BENCH_ENGINE" --device "$BENCH_DEVICE" --tokens "$BENCH_TOKENS" --runs "$BENCH_RUNS" --output-dir "$results")

if [[ "$BENCH_ENGINE" == magnetar || "$BENCH_ENGINE" == candle || "$BENCH_ENGINE" == airllm ]]; then
  revision=7ae557604adf67be50417f59c2c2f167def9a775
  bundle="$work/qwen2.5-0.5b-instruct"
  mkdir -p "$bundle"
  for file in config.json generation_config.json tokenizer.json tokenizer_config.json model.safetensors; do
    if [[ ! -f "$bundle/$file" ]]; then
      curl --retry 3 -fsSL "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/$revision/$file" -o "$bundle/$file.tmp"
      mv "$bundle/$file.tmp" "$bundle/$file"
    fi
  done
  (
    cd "$bundle"
    sha256sum -c <<'CHECKSUMS'
18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45  config.json
e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6  generation_config.json
c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539  tokenizer.json
5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583  tokenizer_config.json
fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe  model.safetensors
CHECKSUMS
  )
  args+=(--model-path "$bundle")
fi

case "$BENCH_ENGINE" in
  magnetar)
    args+=(--revision "${GITHUB_SHA:-unknown}")
    ;;
  candle)
    git clone --no-tags https://github.com/huggingface/candle.git "$work/candle"
    git -C "$work/candle" checkout --detach "${BENCH_ENGINE_REF:-cfcbd7186d13271843ce7eb41d25659bd0835850}"
    engine_revision=$(git -C "$work/candle" rev-parse HEAD)
    features=()
    if [[ "$BENCH_DEVICE" == gpu ]]; then features=(--features cuda); fi
    cargo build --release --manifest-path "$work/candle/Cargo.toml" -p candle-examples --example qwen "${features[@]}"
    args+=(--binary "$work/candle/target/release/examples/qwen" --revision "$engine_revision")
    ;;
  llama_cpp|llama_server)
    gguf="$work/qwen2.5-0.5b-instruct-fp16.gguf"
    if [[ ! -f "$gguf" ]]; then
      curl --retry 3 -fsSL 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-fp16.gguf' -o "$gguf.tmp"
      mv "$gguf.tmp" "$gguf"
    fi
    echo "8e0ae26000627ed62de0e78e41860af70094558b9d2913385c842a6aa06cf3fc  $gguf" | sha256sum -c
    git clone --no-tags https://github.com/ggml-org/llama.cpp.git "$work/llama.cpp"
    git -C "$work/llama.cpp" checkout --detach "${BENCH_ENGINE_REF:-bafa0420c1e57edf04d71587d32fe57d45df7b95}"
    engine_revision=$(git -C "$work/llama.cpp" rev-parse HEAD)
    # No web UI: the benchmark never serves it, and fetching its prebuilt
    # assets from a Hugging Face bucket at build time can fail (run
    # 44516315731: the pinned release's assets were missing).
    cmake_args=(-DCMAKE_BUILD_TYPE=Release -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF)
    if [[ "$BENCH_DEVICE" == gpu ]]; then cmake_args+=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native); fi
    cmake -S "$work/llama.cpp" -B "$work/llama.cpp/build" "${cmake_args[@]}"
    target=llama-completion
    if [[ "$BENCH_ENGINE" == llama_server ]]; then target=llama-server; fi
    cmake --build "$work/llama.cpp/build" --config Release --target "$target" --parallel 2
    args+=(--binary "$work/llama.cpp/build/bin/$target" --model-path "$gguf" --revision "$engine_revision")
    ;;
  airllm)
    args+=(--revision "${BENCH_ENGINE_REF:-unknown}")
    ;;
  ollama|strata)
    if [[ -z "${BENCH_URL:-}" || -z "${BENCH_MODEL:-}" ]]; then
      echo 'A server URL and model name are required for Ollama and Strata' >&2
      exit 1
    fi
    args+=(--url "$BENCH_URL" --model "$BENCH_MODEL")
    if [[ -n "${BENCH_SERVER_LOG:-}" ]]; then args+=(--server-log "$BENCH_SERVER_LOG"); fi
    if [[ -n "${BENCH_ENGINE_REF:-}" ]]; then args+=(--revision "$BENCH_ENGINE_REF"); fi
    ;;
esac

python3 benchmark/run.py "${args[@]}"
