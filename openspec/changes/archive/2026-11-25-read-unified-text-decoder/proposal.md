## Why

`google/gemma-4-12B-it` declares `model_type: gemma4_unified`. Its
`text_config` describes the same decoder as the dense Gemma 4 checkpoints:
five sliding-window layers to one full-attention layer, full layers with one
K/V head of width 512 that derive V from K, a quarter of each full head
rotated, no per-layer inputs and no experts. The Hugging Face loader already
derives that layout from a nested `text_config`, but the bundle also stores
an image-patch embedder under `model.vision_embedder.`, a prefix the loader
did not yet recognize as non-text. Those tensors would have entered the text
inventory, which the Component's graph does not read.

## What Changes

- The Hugging Face loader leaves `model.vision_embedder.*` out of the text
  inventory, beside the vision and audio towers, their embedders and the
  multimodal projector.
- A test parses the published `gemma-4-12B-it` `config.json` and checks
  every layer's geometry.

The Artifact family is the bundle's `model_type`, `gemma4_unified`. A Gemma
Component registration that serves this checkpoint declares it among its
`compatibility.architecture_families`, beside `gemma4`; the Component's
graph does not read the family.

The `ggml-org/gemma-4-12B-it-GGUF` files declare `general.architecture:
gemma4`, with per-layer `head_count_kv`, no `attn_v` on full layers, and
`rope_freqs` rotating 64 of 256 pairs. The GGUF loader already derives that
layout (the 26B-A4B test covers the same form); no GGUF change is needed.

## Impact

- `loaders/huggingface`: `naming.rs`, `config.rs` test,
  `testdata/gemma4/gemma-4-12B-it.config.json`.
