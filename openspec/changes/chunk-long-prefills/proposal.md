## Why

A prefill holds activations for every prompt row at once. Mistral 7B Q4_K_M
on one RTX 3060 (12 GB) prefills 4,096 tokens, but at 16,384 tokens every KV
precision ran out of device memory in the prefill's MLP (`gate_proj`), before
the history's size mattered (`turboquant-kv-history`, run 44183609847): one
`[16,384, 14,336]` f32 intermediate alone is 940 MB. TurboQuant made the
history 7 to 10 times smaller (228,798 tokens of `tq4` history fit beside the
weights), but the prompt's activations still bound the context.

## What Changes

- **Chunked prefill.** A prompt longer than the chunk length runs as
  consecutive passes of one prefill graph of the chunk length. Each pass
  continues the KV history the passes before it wrote in place
  (`append-kv-history-in-place`), from its first row's position: its RoPE
  positions start there, its K and V rows are written there, and its
  attention reads every row up to its own (`key_count`), causally offset.
  A pass's activations are bounded by the chunk length.
- **One graph and one plan.** The last pass is right-padded to the chunk
  length, as a bucketed prefill is (`replay-bucketed-prefill`), and selects
  its last real row's logits through the logits row input. The history ends
  at the prompt's last row: the padding rows lie beyond it and decode writes
  over them. Every prompt longer than the chunk shares the chunk's graph,
  its Prepared Execution Plan and its fusion.
- **Eligibility.** Chunking needs KV history kept in place and a prefill
  graph whose every attention reads K and V from its KV-history edges, which
  feed nothing but attention. Otherwise every prompt runs in one pass, as
  today: a Provider without in-place history, tensor parallelism, and a
  prefill whose attention reads K or V computed from its own rows only (an
  expanded latent attention, whose history holds the latent rows).
- **Controls.** `MAGNETAR_PREFILL_CHUNK` (2,048 rows when unset, `0` turns
  chunking off) and `ProductionLoadedModel::set_prefill_chunk`.
- **Qualification.** On CUDA, a long prompt's chunked prefill against its
  one-pass prefill (first token, first-step logits, TTFT), and the
  TurboQuant gains of Mistral 7B at 16,384 and 32,768 tokens.

## Capabilities

### New Capabilities

- `chunked-prefill`: prefill passes of a fixed chunk length continuing the
  KV history in place, padding of the last pass, eligibility, controls.

### Modified Capabilities

None. Bucketed prefills keep their buckets; a prompt of exactly the chunk's
length runs the chunk's graph in one pass, unpooled and unrecorded.

## Impact

- `magnetar-runtime` (first-native engine): the chunk length in the graph
  cache, the chunked prefill, eligibility, controls.
- No Provider, Component or WIT change: the Providers' attention Kernels
  already take query rows at the end of a longer key history (`key_count`
  and the causal offset), and their in-place `kv-write` any row offset.
- `integration-tests/production-loading`: chunked tiny checkpoints against
  `transformers`, the CUDA test against one pass, the gains report.
- `.github/workflows/gpu-runner-smoke.yml`: the `kv` group's chunked
  prefill step.

## Non-goals

- Chunking under tensor parallelism and for latent attention (which would
  need an absorbed prefill reading the latent history).
- Capturing and replaying chunk passes, and interleaving passes of several
  requests with decode steps (`serve-continuous-batches-with-qos`).
