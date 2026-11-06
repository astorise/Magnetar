## Context

This draft is derived from roadmap proposal 1.0 (2026-10-29), whose reference commit is `09976e7792e610a04aa2b8af6c6991ba7bcbc225`. Repository inspection for this proposal used `acbb95cab732f01ace7dbc4203ebf698cd7a5fa6`. Canonical specs and implemented paths at the latter revision take precedence over historical roadmap wording. This is proposed work, not a stable new API or measured result.

## Goals / Non-Goals

Deliver the explicitly enumerated F01, F04 outcomes through Model Components and loaders; generic Runtime validation; Provider operators, preserving existing trust, resolution, lifetime and portable Component contracts. Reuse the baseline contracts linked from proposal.md. Broader roadmap work is split into independent changes; adopting an inspiration project or copying its implementation is not a goal.

## Decisions

### Implementation boundary and data flow

Qualify a second dense family and added Qwen versions before widening claims. Specialized MLA/GQA/SWA/sparse or KDA/recurrent families use explicit portable state schemas and operators; family semantics stay in Components, not Runtime Core.

### First additional dense checkpoints (F01, first increment)

Pinned real-checkpoint profiles: the real-checkpoint tests read `MAGNETAR_REAL_CHECKPOINT_PROFILE` from a table pinned by Hugging Face revision and `model.safetensors` sha256. The variable defaults to Qwen2.5-0.5B-Instruct, the logit corpus's checkpoint. The GPU workflow pins each checkpoint's JSON files by sha256.

| Profile | Revision | Shape |
|---|---|---|
| Qwen/Qwen2.5-1.5B-Instruct | `989aa79` | 28 layers, 1536 hidden, 12 query / 2 KV heads of 128 dimensions, tied LM head |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | `ad9f0ae` | the same Qwen2 shape, rope_theta 10000, untied LM head; served by the Qwen Component |

What qualifying them required:

- **128-dimension attention.** The Tensor Core prefill attention and the grouped split decode attention served only 64-dimension heads. Both are now templates on the head dimension:
  - The prefill attention uses 68 KB of dynamic shared memory for 128 dimensions, so the launcher raises the function's 48 KB limit.
  - The decode attention uses 32-key chunks for 128 dimensions.
  - Five 128-dimension cases joined the Reference CPU conformance (CUDA Provider `acb47db`).
- **Prefill activation memory.** Every graph edge's output stayed Provider-resident until the step ended. At 4096 prompt tokens, Qwen2.5-1.5B ran its 12 GB RTX 3060 out of memory at layer 24 (run 42127910391).
  - A prefill now releases each node output once its last reader has run. It keeps graph inputs, weights, KV stores and the edges it materializes.
  - Decode steps keep their resources, because a captured step replays against them.
- **Shared special tokens.** DeepSeek-R1-Distill declares its end-of-sequence token as its pad token. The tokenizer metadata now accepts one token in several roles. One role declared twice, or one id with two texts, is still a conflict.
- **Bounded verification memory.** The `model.safetensors` digest is computed in 8 MB reads. The 1.5B checkpoints in f32 (about 6 GB) do not fit the runner next to a Reference CPU run (exit 137, runs 42116467880 and 42116844824). Their check is therefore the same campaign with the Tensor Cores off, which must generate the same tokens.

Evidence (RTX 3060, GPU CI run 42137496729):

| Warm medians | 5 / 512 / 4096 prompt tokens: TTFT | TPOT |
|---|---|---|
| Qwen2.5-0.5B-Instruct | 14.6 / 53.6 / 478.3 ms | 4.4 / 4.4 / 5.0 ms |
| Qwen2.5-1.5B-Instruct | 33.4 / 171.7 / 1480.8 ms | 12.1 / 12.2 / 14.1 ms |
| Same, Tensor Cores off | 33.4 / 643.2 / 6086.4 ms | 12.2 / 12.2 / 14.2 ms |
| DeepSeek-R1-Distill-Qwen-1.5B | 33.2 / 172.2 / 1491.1 ms | 12.0 / 12.2 / 14.1 ms |
| Same, Tensor Cores off | 33.5 / 643.4 / 6083.6 ms | 12.1 / 12.2 / 14.1 ms |

- Both 1.5B checkpoints generate the same 128 greedy tokens with the Tensor Cores on and off, after 5-, 512- and 4096-token prompts.
- Qwen2.5-0.5B keeps its Reference CPU comparison, its logit corpus match and 13 of 16 replayed decode steps.
- Releasing prefill intermediates also lowered Qwen2.5-0.5B's warm TTFT at 512 prompt tokens from 62.7 to 53.6 ms, and its TPOT at 4096 from 5.8 to 5.0 ms (run 42106862830 before).
- A 1.5B decode step reads about 3.1 GB of bfloat16 weights. At 12.1 ms, that is about 70% of the RTX 3060's memory bandwidth.

Limits:

- The 1.5B checkpoints have no Reference CPU or logit-corpus comparison yet; a bfloat16 Reference CPU path would make one affordable.
- The Hugging Face loader ignores `add_bos_token`, which DeepSeek-R1-Distill sets. Tokens still match across Providers, but its prompts are not yet encoded exactly as upstream.

### First non-Qwen2 checkpoint: Llama 3.2 (F01, second increment)

`meta-llama/Llama-3.2-1B-Instruct` (revision `9213176`, gated) is the first checkpoint the Qwen Component cannot serve:

- Shape: 16 layers, 2048 hidden, 32 query / 8 KV heads of 64 dimensions, tied LM head.
- RoPE: rope_theta 500000 with Llama 3 per-frequency RoPE scaling (`rope_scaling.rope_type = "llama3"`, factor 32).

It is qualified through the Model Component boundary, so the Runtime gains no Llama-specific semantics:

- **Component selection.** The campaign loads each checkpoint with the registered Model Component whose manifest declares the ingested architecture family (`compatibility.architecture_families`). It never selects by model name. `llama` resolves to the Llama Component; `qwen2` (Qwen2.5 and DeepSeek-R1-Distill) resolves to the Qwen Component.
- **Contract.** `magnetar:model-component-graph@1.4.0` adds `model-config.rope-scaling`. It is a new function, not a new `architecture-config` field, so Components compiled against 1.2.0/1.3.0 still link. It returns the checkpoint's per-frequency scheme as normalized by the loader. The Hugging Face loader reads `rope_type` and refuses schemes it does not implement (`yarn`, `dynamic`, ...) instead of ingesting them as unscaled.
- **The Component declares, the Runtime forwards.** The Llama Component (`334d221`, which also implements the 1.3.0 segment exports) declares the scheme on its `rope` nodes as the portable `frequency_scaling` attributes. The Runtime forwards node attributes unchanged.
- **The Core names the scheme, not a family.** The operator, its Kernel feature and the WIT case call it `banded` (kept, interpolated and divided wavelength bands). Only the Hugging Face loader knows that `rope_type = "llama3"` maps onto it. The `quality / model-family isolation` guard enforces this.
  - It refuses a Component-built graph whose `rope` nodes do not declare exactly the bound configuration's scaling, for example the Qwen Component (1.3.0) given a Llama 3 checkpoint.
  - It refuses a Provider whose `rope` Kernels do not advertise `banded-frequency-scaling`.
- **Kernels.** Reference CPU, the CPU Provider and the CUDA Provider rescale each pair's inverse frequency with the same f32 operations:
  - Reference CPU matches Hugging Face's `_compute_llama3_parameters` within 1e-5 relative, across all three bands.
  - CUDA matches the CPU Provider in conformance.

Evidence (RTX 3060, GPU CI run 42186542821):

| Warm medians | 5 / 512 / 4096 prompt tokens: TTFT | TPOT |
|---|---|---|
| Llama-3.2-1B-Instruct | 27.2 / 127.9 / 1087.8 ms | 10.0 / 10.2 / 11.4 ms |
| Same, Tensor Cores off | 27.2 / 475.3 / 4502.5 ms | 10.0 / 10.2 / 11.5 ms |

- The same 128 greedy tokens with the Tensor Cores on and off, after all three prompts. The text is coherent at every length: " Paris. It is located on the Seine River and is known for its rich history, cultural landmarks, and iconic architecture..." after 4096 tokens.
- The Qwen2 checkpoints, now loaded through the same family-selected path, still pass every earlier check. DeepSeek-R1-Distill generates the same tokens as before the change (run 42144929355).
- Limits:
  - No Reference CPU or logit comparison yet. The checkpoint in f32 (about 5 GB) does not fit the runner next to the test.
  - The plain-text prompt is encoded without `<|begin_of_text|>`, the same `add_bos_token` gap as DeepSeek-R1-Distill.

### Identity, eligibility and recovery

Inputs retain validated artifact/model/Component identities, dtype/layout, phase, shape and version provenance where relevant. Runtime checks trust, ResourceAffinity, capability support and MemoryManager admission before policy ranks candidates. Plans and state reference immutable generations and opaque leases. Cancellation stops future submissions; in-flight work retains its leases until completion. On Device loss or incompatible state, invalidate affected work and replan or fail explicitly; never silently rebind resources or restore an untrusted generation.

### Compatibility and WIT evolution

Preserve existing F32 and single-Device baseline fixtures and public request semantics. Reuse current WIT and Rust contracts first. Before implementation requires new graph operators, state envelopes or result types, record their exact versioned portable schema, negotiation, unsupported behavior and migration tests; native API types are prohibited. A new precision or model profile is opt-in until its validation evidence supports promotion. Vendor implementations stay in their repositories, followed by an explicit parent submodule pin update.

### Authority and isolation

Tools and service/transport adapters use caller-authorized inputs and bounded operations. Inference and Components gain no ambient filesystem, network, process, source-control or secret authority. Local source streaming checks artifact ranges and identity. External campaigns/artifacts are untrusted until verified. Snapshot or distributed data remains tenant-scoped and requires authorization in addition to checksums; signing keys and credentials never enter reports.

### Qualification and rollout

Use the shared real-checkpoint harness and baseline corpus rather than inventing isolated pass criteria. A profile records checkpoint revision/digests, code and Provider SHAs, hardware/driver, flags, dtype, shape, budgets and warmup. Exercise 1/4/16/32 sessions, 512/4096/16384-token contexts and 128-512 generated tokens only inside model/backend/capacity limits; report every skipped combination. Repeat at least five warm runs when affordable and document exceptions, uncertainty and cold costs. Check logits and downstream quality with declared thresholds; token identity alone is insufficient. Measure TTFT, TPOT/ITL p50/p95/p99, aggregate/per-request throughput, launch/phase/copy/I/O costs, RSS/VRAM, fragmentation, useful cache savings and energy only when valid telemetry exists.

Promotion requires appropriate real hardware and a real checkpoint, error/cancellation/memory-pressure/isolation tests, before/after evidence, a disable control and a qualified fallback or structured Unsupported result. Experimental features remain disabled by default until qualified. Run existing Rust, WIT, OpenSpec, license/SBOM/signature and artifact-reproducibility gates for affected code. Compare tail latency and total memory as well as throughput, under the same budget. Publish limitations and do not infer a blanket speedup from heterogeneous runs.

## Risks / Trade-offs

- Do not infer support from a shared family name, shared tensor dimensions or an attention label alone.
- A performance win can be below measurement noise or transfer cost; stop promotion until evidence meets the declared objective and uncertainty threshold.
- Wider hardware coverage is constrained by actual machines; record missing hardware as skipped, not passed, and deliver at most a contract or POC for that target.

## Migration and rollback

Land compatibility scaffolding and baseline tests before optimized execution. Keep new paths selectable and disableable by profile. Drain or fence old generations at safe boundaries; in-flight references remain alive. Roll back only to a still-trusted, resource-compatible qualified generation. If no safe fallback exists, return a structured error. Document any public schema migration before implementation and preserve supported old readers/linkers during the declared compatibility window.

## Open questions

Select exact hardware/checkpoint, numerical and downstream-quality thresholds, resource budgets and SLOs per qualification profile before promoting it. The roadmap's reference CPU/CUDA and Qwen checkpoint are the initial comparison, not evidence that other targets are available. Any Tachyon-facing interface additionally needs its independent repository audit and owner agreement. These decisions gate implementation promotion, not creation of this proposal.
