## Context

This draft is derived from roadmap proposal 1.0 (2026-10-29), whose reference commit is `09976e7792e610a04aa2b8af6c6991ba7bcbc225`. Repository inspection for this proposal used `acbb95cab732f01ace7dbc4203ebf698cd7a5fa6`. Canonical specs and implemented paths at the latter revision take precedence over historical roadmap wording. This is proposed work, not a stable new API or measured result.

## Goals / Non-Goals

Deliver the explicitly enumerated F02, F05, B09 outcomes through Model Component routing semantics; Runtime residency; Provider expert kernels, preserving existing trust, resolution, lifetime and portable Component contracts. Reuse the baseline contracts linked from proposal.md. Broader roadmap work is split into independent changes; adopting an inspiration project or copying its implementation is not a goal.

## Decisions

### Implementation boundary and data flow

Implement router/top-k/shared expert graphs, then qualify expert kernels and trace-driven cache policies with bounded hot-layer pinning and top-k prefetch. Distinguish a portable MoE correctness path from later grouped/fused quantized optimization.

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

- Skewed or adversarial routing must not exceed expert buffer quotas or expose another sequence state.
- A performance win can be below measurement noise or transfer cost; stop promotion until evidence meets the declared objective and uncertainty threshold.
- Wider hardware coverage is constrained by actual machines; record missing hardware as skipped, not passed, and deliver at most a contract or POC for that target.

## Migration and rollback

Land compatibility scaffolding and baseline tests before optimized execution. Keep new paths selectable and disableable by profile. Drain or fence old generations at safe boundaries; in-flight references remain alive. Roll back only to a still-trusted, resource-compatible qualified generation. If no safe fallback exists, return a structured error. Document any public schema migration before implementation and preserve supported old readers/linkers during the declared compatibility window.

## Open questions

Select exact hardware/checkpoint, numerical and downstream-quality thresholds, resource budgets and SLOs per qualification profile before promoting it. The roadmap's reference CPU/CUDA and Qwen checkpoint are the initial comparison, not evidence that other targets are available. Any Tachyon-facing interface additionally needs its independent repository audit and owner agreement. These decisions gate implementation promotion, not creation of this proposal.

## Initial profile: Qwen1.5-MoE-A2.7B (F02 first, then F05)

Chosen 6 November 2026. Checkpoint: `Qwen/Qwen1.5-MoE-A2.7B-Chat` (Hugging
Face `model_type = qwen2_moe`; llama.cpp GGUF `general.architecture =
qwen2moe`), pinned as the `Q8_0` GGUF export
`RichardErkhov/Qwen_-_Qwen1.5-MoE-A2.7B-Chat-gguf@947580e7b3904cd4081810d0efec4520a75609c7`
(`Qwen1.5-MoE-A2.7B-Chat.Q8_0.gguf`, 15,226,010,336 bytes, sha256
`d20e59b6180c775c648703fd63b8df7d01fc107672441fc3746c41f1dcaf904c`). 24
decoder layers, hidden 2048, 16 heads (no GQA), QKV bias, 60 routed experts
of intermediate 1408, top-4 without renormalization (`norm_topk_prob =
false`), and one shared expert of intermediate 5632 scaled by
`sigmoid(x . w_shared_gate)`. At 15 GB packed it exceeds the 12 GB
qualification Device, so F05 (experts outside VRAM) is required, not
simulated.

### Layer semantics (the Component's graph)

```
h        = rmsnorm(x, post_attn_norm)
logits   = matmul(h, router)                      # [T, E], router [H, E]
routed   = moe-experts(h, logits, experts_gate, experts_up, experts_down)
shared   = matmul(silu-mul(matmul(h, shared_gate_proj), matmul(h, shared_up_proj)), shared_down_proj)
scale    = activation(matmul(h, shared_expert_gate), kind = sigmoid)   # [T, 1]
out      = x + routed + row-scale(shared, scale)
```

The existing fusion passes turn the shared expert into a `gated-matmul`.

### New portable Operators (catalog additions, no WIT signature change)

- `moe-experts` (5 inputs, 1 output; attributes `experts`, `top_k`,
  `normalize_top_k`): per token row, `p = softmax(logits)` in f32, the
  `top_k` largest `p` (ties to the lower expert index), optionally divided
  by their sum, and `out = sum_k p_k * down_e(silu(gate_e(h)) * up_e(h))`.
  The stacked expert weights are 2D: `experts_gate`/`experts_up` are
  `[H, E * I]` (expert `e` owns columns `[e*I, (e+1)*I)`) and
  `experts_down` is `[I, E * H]` (expert `e` owns columns `[e*H,
  (e+1)*H)`). This is exactly the logical view of llama.cpp's stacked
  `[E, out, in]` `ffn_*_exps` storage read as transposed storage (`[E*out,
  in]` rows), so packed `Q8_0` experts reuse `execute-direct-quantized-
  weights` unchanged, each expert a contiguous range of stored rows -- the
  unit F05 caches.
- `row-scale` (2 inputs): `out[t, :] = x[t, :] * s[t]`, `s` of shape `[T, 1]`.
- `activation` gains `kind = sigmoid`.

Routing is data-dependent but stays on the Device: a Provider must not
download indices to choose experts on the decode path (it would break
step capture); prefill may group tokens by expert.

### Model configuration (WIT `1.5.0`, additive)

`model-config.mixture-of-experts: func() -> option<moe-config>` with
`moe-config { experts, experts-per-token, expert-intermediate-size,
shared-expert-intermediate-size: option<u64>, normalize-top-k }` -- a new
function, as `rope-scaling` was in `1.4.0`, so Components compiled against
earlier versions keep linking. The Runtime's `ModelArchitectureConfig`
gains the matching optional `mixture_of_experts`. A Component that does not
understand a declared MoE configuration must refuse to build a graph.

### Ownership

The Qwen Model Component declares the `qwen2moe` (GGUF) and `qwen2_moe`
(Hugging Face) families and emits the graph above. `loaders/gguf` maps
`ffn_gate_inp` to `layers.N.moe.router_proj`, `ffn_{gate,up,down}_exps` to
`layers.N.moe.experts_{gate,up,down}_proj` (stacked, 2D), the `shexp`
tensors to `layers.N.moe.shared_{gate,up,down}_proj` and
`ffn_gate_inp_shexp` (`[H]`) to `layers.N.moe.shared_expert_gate` declared
`[H, 1]`. The Hugging Face per-expert layout (60 separate tensors per
projection) is out of this first profile: at 14.3B parameters its f32
dequantized form cannot run on the qualification hardware.

## Evidence: Qwen1.5-MoE-A2.7B-Chat Q8_0 on two RTX 3060 (12 GB each)

GPU Runner Smoke Test run 42282603030 (commit `29905a7`): the 15 GB packed
checkpoint, more than one GPU holds, loaded on `CudaMultiProvider` over
both GPUs with the Runtime's `Fill` layer placement
(`enforce-local-multigpu-and-expert-placement` G05), routed experts read
packed by `moe-experts` on the Device. Load 100.1 s; warm TTFT 33.3 ms;
TPOT 19.01 ms over 64 greedy tokens (about 53 tokens/s), without step
capture across the two GPUs. Text: " Paris. The country is famous for its
fashion, food, and art. It is also known for its beautiful architecture,
such as the Eiffel Tower and the Louvre Museum…".
