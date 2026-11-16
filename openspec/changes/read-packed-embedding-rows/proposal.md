## Why

Issue #187 left one measurement open: the resident size of Gemma 4 E2B's
per-layer embedding tables. The loaders give each of the 35 layers its own
slice of `embed_tokens_per_layer`, a `[262144, 256]` table read by an
`embedding` node. These sizes come from the tensor headers and the Providers'
storage rules:

| Artifact and Provider | Per layer | 35 layers |
|---|---|---|
| Hugging Face bundle, `BF16` (storage) | 134.2 MB | 4.70 GB |
| Same, CUDA Provider (stored as bfloat16, admitted at 4 bytes a value) | 134.2 MB stored, 268.4 MB admitted | 4.70 GB stored, 9.40 GB admitted |
| Reference CPU (dequantized to f32) | 268.4 MB | 9.40 GB |
| `gemma-4-E2B-it-qat-q4_0-gguf`, `Q6_K` (storage, CPU native Provider) | 55.1 MB | 1.93 GB |
| Same, CUDA Provider (208 quant bytes and an f32 `d` per super-block) | 55.6 MB | 1.95 GB |

The token table `[262144, 1536]` adds 0.33 GB in `Q6_K` (1.61 GB in f32).
The admission row follows `execute-native-half-precision`: the Memory
Manager still counts 4 bytes per weight value, so on a 12 GB Device a
Hugging Face E2B bundle is admitted at about twice the memory its compact
weights use. This change does not alter that.

Both of the GGUF's tables are `Q6_K` (`per_layer_token_embd.weight`,
`token_embd.weight`, read from the file's header). Both loaders keep them
packed, and each per-layer slice is one `Q6_K` super-block wide, so it is cut
on block boundaries.

Measuring the size showed a time and transient-memory cost on CUDA. Its
lookup read only `Q8_0` packed tables directly. A table in any other packed
format was first dequantized whole into a new f32 buffer, at every call. For
the E2B GGUF, each step thus wrote 35 × 268 MB plus 1.61 GB, about 11 GB of
f32, of which it read a few rows. The CPU native Provider already
dequantizes only the looked-up rows.

## What Changes

- The CUDA dequantization kernels take optional row ids. A lookup from a
  packed table stored untransposed dequantizes only the looked-up rows, in
  every packed format. An invalid id fails the lookup, as before.
- A GPU diagnostic (`packed_embedding_lookup_times`, run by the GPU workflow's
  kernels suite) times both paths at the E2B tables' shapes and at a `Q4_K`
  `[32000, 4096]` token table.

## Impact

- `providers/cuda`: `kernels.cu` (`packed_element`, the seven dequantization
  kernels), `kernels.rs` (`dequantize_packed`, `embedding_lookup`), tests.
- `.github/workflows/gpu-runner-smoke.yml`: the diagnostic step.
- Resident sizes are unchanged. The transient f32 copy of the whole table is
  gone.
