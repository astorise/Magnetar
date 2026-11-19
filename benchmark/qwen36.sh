#!/usr/bin/env bash
# Pinned model also used by Strata's Qwen3.6 engine. Source this file.
prepare_qwen36() {
  local work="$1"
  QWEN36_MODEL="$work/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
  export QWEN36_MODEL
  python3 benchmark/qwen36_preflight.py "$work" "$QWEN36_MODEL"
  if [[ ! -f "$QWEN36_MODEL" ]]; then
    curl --retry 3 -fsSL 'https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf' -o "$QWEN36_MODEL.tmp"
    mv "$QWEN36_MODEL.tmp" "$QWEN36_MODEL"
  fi
  echo "55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1  $QWEN36_MODEL" | sha256sum -c
}
