#!/usr/bin/env bash
# Run one campaign member in a subprocess so servers and virtualenvs are isolated.
set -euo pipefail
: "${BENCH_ENGINE:?}" "${BENCH_DEVICE:?}" "${RUNNER_TEMP:?}"
work="$RUNNER_TEMP/benchmark-work"
mkdir -p "$work"
if [[ "$BENCH_DEVICE" == cpu ]]; then export CUDA_VISIBLE_DEVICES=""; fi

case "$BENCH_ENGINE" in
  magnetar|candle|llama_server)
    exec bash benchmark/ci.sh
    ;;
  strata_native)
    if [[ "$BENCH_DEVICE" != gpu || "${BENCH_MODEL_PROFILE:-}" != qwen3.6-35b-experimental ]]; then
      echo 'Strata native Qwen3.6 requires the experimental GPU profile' >&2; exit 1
    fi
    source benchmark/qwen36.sh
    prepare_qwen36 "$work"
    archive="$work/strata-engine.tar.gz"
    curl --retry 3 -fsSL 'https://github.com/Niko1221/Strata/releases/download/v0.2.16/strata-engine-0.1.62-linux-x86_64-cuda-cuda13.tar.gz' -o "$archive"
    echo "6723e1b6b6042bfe1d368723ebf873afb0fa65f4f1908d948dd865098f6cfb9e  $archive" | sha256sum -c
    mkdir -p "$work/strata-engine"
    tar -xzf "$archive" -C "$work/strata-engine"
    binary=$(find "$work/strata-engine" -type f -name strata -print -quit)
    if [[ -z "$binary" ]]; then echo 'Strata engine binary missing from release' >&2; exit 1; fi
    export BENCH_PROMPT_IDS_PATH="$RUNNER_TEMP/benchmark-results/qwen36-prompt-ids.json"
    python3 benchmark/run.py strata_native --device gpu --tokens "$BENCH_TOKENS" --runs "$BENCH_RUNS" \
      --binary "$binary" --model-path "$QWEN36_MODEL" --revision 'strata-engine-0.1.62' --output-dir "$RUNNER_TEMP/benchmark-results"
    ;;
  airllm)
    revision=56e297a30511289fc85473f9911d6fa6cf9e27cf
    python3 -m venv "$work/airllm-venv"
    py="$work/airllm-venv/bin/python"
    export PIP_NO_CACHE_DIR=1
    "$py" -m pip install --disable-pip-version-check --upgrade 'pip==25.3'
    if [[ "$BENCH_DEVICE" == cpu ]]; then torch_index=https://download.pytorch.org/whl/cpu; else torch_index=https://download.pytorch.org/whl/cu128; fi
    "$py" -m pip install --disable-pip-version-check 'torch==2.8.0' --index-url "$torch_index"
    "$py" -m pip install --disable-pip-version-check 'transformers==4.57.3' 'accelerate==1.12.0' "git+https://github.com/lyogavin/airllm.git@$revision#subdirectory=air_llm"
    export PATH="$work/airllm-venv/bin:$PATH" BENCH_ENGINE_REF="$revision"
    exec bash benchmark/ci.sh
    ;;
  ollama)
    version=v0.40.16
    archive="$work/ollama-linux-amd64.tar.zst"
    curl --retry 3 -fsSL "https://github.com/ollama/ollama/releases/download/$version/ollama-linux-amd64.tar.zst" -o "$archive"
    echo "87bfafb2acbc89b2e6a1be0a9043de6aa75fb450696f86bc426e65fc448f788d  $archive" | sha256sum -c
    mkdir -p "$work/ollama"
    tar --zstd -xf "$archive" -C "$work/ollama"
    gguf="$work/qwen2.5-0.5b-instruct-fp16.gguf"
    if [[ ! -f "$gguf" ]]; then
      curl --retry 3 -fsSL 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-fp16.gguf' -o "$gguf.tmp"
      mv "$gguf.tmp" "$gguf"
    fi
    echo "8e0ae26000627ed62de0e78e41860af70094558b9d2913385c842a6aa06cf3fc  $gguf" | sha256sum -c
    port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')
    export OLLAMA_HOST="127.0.0.1:$port" OLLAMA_MODELS="$work/ollama-models"
    "$work/ollama/bin/ollama" serve > "$RUNNER_TEMP/benchmark-results/ollama-server.log" 2>&1 &
    server_pid=$!
    cleanup_ollama() {
      kill "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
      rm -rf -- "$work/ollama" "$work/ollama-models" "$archive"
    }
    trap cleanup_ollama EXIT
    ready=false
    for _ in $(seq 1 120); do
      if ! kill -0 "$server_pid" 2>/dev/null; then echo 'Ollama exited during startup' >&2; exit 1; fi
      if curl -fsS "http://$OLLAMA_HOST/api/version" >/dev/null; then ready=true; break; fi
      sleep 1
    done
    if [[ "$ready" != true ]]; then echo 'Ollama did not start' >&2; exit 1; fi
    printf 'FROM %s\n' "$gguf" > "$work/Modelfile"
    "$work/ollama/bin/ollama" create magnetar-bench -f "$work/Modelfile"
    export BENCH_URL="http://$OLLAMA_HOST" BENCH_MODEL=magnetar-bench BENCH_ENGINE_REF="$version"
    bash benchmark/ci.sh
    ;;
  *) echo "Unexpected campaign engine: $BENCH_ENGINE" >&2; exit 1 ;;
esac
