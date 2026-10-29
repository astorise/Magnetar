## Why

`qualify-real-checkpoint-performance` (A02) compared Magnetar's real Qwen2.5-0.5B-Instruct logits with an independent numpy forward pass (`tools/qualification/qwen2_reference_logits.py`) and found every Provider wrong from the first step: for "The capital of France is" the reference generates " Paris. It is the largest city in Europe…" while Magnetar generates " 1000000…", with a 3.04 logit difference at step 0. Switching only the reference's RoPE to rotate consecutive pairs reproduces Magnetar's step-0 logits to four decimals.

Magnetar's `rope` kernels (Reference CPU in `magnetar-runtime`, `providers/cpu`, `providers/cuda`, `providers/wgpu`) rotate element pairs `(2i, 2i+1)` -- the GPT-J/GGML "normal" convention. Hugging Face Qwen2 and Llama checkpoints (and llama.cpp's NeoX-mode GGUF Qwen2) rotate `(i, i + d/2)` (`rotate_half`). CPU and CUDA agreed with each other only because they share the defect, so the existing cross-Provider token test could not catch it.

## What Changes

- The portable `rope` Operator gains an optional `rotation_pairing` string attribute: `interleaved` (pairs `2i, 2i+1`) or `half_split` (pairs `i, i + dimension/2`). Absent keeps today's `interleaved` behavior so existing graphs and fixtures are unchanged; an unknown value is rejected.
- Reference CPU (`magnetar-runtime`), `providers/cpu`, `providers/cuda` and `providers/wgpu` implement both pairings, including partial RoPE (`dimension < head width`) and multi-head blocks, and reject unknown values.
- The Qwen and Llama Model Components emit `rotation_pairing: half_split` on every `rope` node, as do `magnetar-runtime`'s Rust-side Qwen graph builder and the Runtime's Component-graph dispatch, which forwards the node's attribute to the Kernel. The checked-in compiled Components (`qwen-real`, `llama-real`, `magnetar-cli`'s bundled copy) are rebuilt and their pinned digests and manifests updated.
- Real-checkpoint evidence: Reference CPU generation on the pinned Qwen2.5-0.5B-Instruct checkpoint matches the independent corpus under `f32-cross-provider-v1`.

## Capabilities

### Modified Capabilities

- `operator`: the RoPE Operator requirement now represents the rotation pairing explicitly.

## Impact

`magnetar-runtime`, `components/qwen`, `components/llama`, `providers/cpu`, `providers/cuda`, `providers/wgpu` and checked-in Component fixtures/digests. No WIT signature change: graph attributes are already open key/value data; the Components' WIT interface versions are unchanged. Every real Qwen/Llama generation output changes (it becomes correct); synthetic-fixture expectations that encoded the interleaved result are updated where they assert specific values.

## Non-goals

- GGUF Llama checkpoints, whose converter permutes Q/K weights for the interleaved convention, are not ingested today; a future GGUF Llama path must either un-permute in the loader or have the Component emit `interleaved`.
- No new RoPE scaling modes (`position_mode` stays `sequential`).
