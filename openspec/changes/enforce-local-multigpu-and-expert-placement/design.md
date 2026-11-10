## Context

This draft is derived from roadmap proposal 1.0 (2026-10-29), whose reference commit is `09976e7792e610a04aa2b8af6c6991ba7bcbc225`. Repository inspection for this proposal used `acbb95cab732f01ace7dbc4203ebf698cd7a5fa6`. Canonical specs and implemented paths at the latter revision take precedence over historical roadmap wording. This is proposed work, not a stable new API or measured result.

## Goals / Non-Goals

Deliver the explicitly enumerated G05, G06 outcomes through Runtime local placement; Providers native transfers and collectives, preserving existing trust, resolution, lifetime and portable Component contracts. Reuse the baseline contracts linked from proposal.md. Broader roadmap work is split into independent changes; adopting an inspiration project or copying its implementation is not a goal.

## Decisions

### Implementation boundary and data flow

Turn production placement from a descriptive record into an enforced plan generation, supporting staged layer segments first and tensor/collective profiles only when qualified. Add expert placement/duplication after MoE correctness and measured interconnect capacity.

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

- Preserve local-node scope and distinguish two segment instances from any future logical multi-Device Model Instance API.
- A performance win can be below measurement noise or transfer cost; stop promotion until evidence meets the declared objective and uncertainty threshold.
- Wider hardware coverage is constrained by actual machines; record missing hardware as skipped, not passed, and deliver at most a contract or POC for that target.

## Migration and rollback

Land compatibility scaffolding and baseline tests before optimized execution. Keep new paths selectable and disableable by profile. Drain or fence old generations at safe boundaries; in-flight references remain alive. Roll back only to a still-trusted, resource-compatible qualified generation. If no safe fallback exists, return a structured error. Document any public schema migration before implementation and preserve supported old readers/linkers during the declared compatibility window.

## Open questions

Select exact hardware/checkpoint, numerical and downstream-quality thresholds, resource budgets and SLOs per qualification profile before promoting it. The roadmap's reference CPU/CUDA and Qwen checkpoint are the initial comparison, not evidence that other targets are available. Any Tachyon-facing interface additionally needs its independent repository audit and owner agreement. These decisions gate implementation promotion, not creation of this proposal.

## Evidence: staged layer segments on two RTX 3060 (step 1, uncaptured)

`CudaMultiProvider` over GPUs 0 and 1, the Runtime's `LayerPlacement`
handed over before any weight write. GPU Runner Smoke Test run 42284508413
(commit `4686a72`), 64 greedy tokens after a warm-up:

| Checkpoint | GPUs, placement | Load | Warm TTFT | TPOT |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct Q8_0 | 1, fill | 3.8 s | 14.2 ms | 4.56 ms |
| Qwen2.5-0.5B-Instruct Q8_0 | 2, balanced | 3.9 s | 14.8 ms | 11.34 ms |
| Qwen1.5-MoE-A2.7B-Chat Q8_0 (15 GB) | 2, fill | 99.2 s | 33.0 ms | 18.73 ms |

The two-GPU Qwen2.5-0.5B run generates exactly the single-GPU tokens (the
CI step fails otherwise). Its TPOT is 2.5x the single GPU's because a step
spanning two GPUs is not captured into CUDA graphs yet (the single GPU
replays a captured decode step) and synchronizes the producing stream at
the boundary: capture per GPU, chained by events, is the next step.
Qwen1.5-MoE-A2.7B does not fit one 12 GB GPU; across two it generates "
Paris. The country is famous for its fashion, food, and art…".

### Multi-GPU step capture (in progress)

Opt-in (`MAGNETAR_CUDA_MULTI_CAPTURE=1`) while it is qualified. Run
42300438672 (commit `853796c`) reproduced the uncaptured two-GPU figures
(Qwen2.5-0.5B balanced: TPOT 11.41 ms, same tokens as one GPU; Qwen1.5-MoE:
TPOT 18.96 ms) and the captured run's failure: a boundary copy refused
with `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` while neither stream was
capturing -- cudarc's deferred report of an earlier free of GPU 1's stale
copy made while GPU 0 was capturing. Stale copies are now kept while any
capture is open and freed afterwards.

Qualified and on by default (`MAGNETAR_CUDA_MULTI_CAPTURE=0` disables it).
Run 42311822286 (commit `8f159ad`), 64 greedy tokens:

| Checkpoint | GPUs | Step capture | TPOT |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct Q8_0 | 1 | yes | 4.48 ms |
| Qwen2.5-0.5B-Instruct Q8_0 | 2 (balanced) | yes | 4.67 ms |
| Qwen2.5-0.5B-Instruct Q8_0 | 2 (balanced) | no (control) | 11.61 ms |
| Qwen1.5-MoE-A2.7B-Chat Q8_0 | 2 (fill) | yes | 14.01 ms |

All Qwen2.5-0.5B runs generate identical tokens. Staged layer segments
cost 0.2 ms per token over one GPU once captured; Qwen1.5-MoE went from
18.5 ms to 14.0 ms, about 180 GB/s of the 2.5 GB of weights a token reads.

## Tensor parallelism (chosen 9 November 2026)

Layer segments add capacity, not speed: one GPU works at a time. On the
2 x RTX 3060 profile a decode token is bound by weight bytes (Qwen1.5-MoE:
2.5 GB per token, 14.0 ms captured), so the faster layout splits every
weight between the GPUs and lets both read their half at once, at the
cost of two reductions per layer (expected about 8 ms per token for
Qwen1.5-MoE, about 22 ms for Mistral 7B instead of 41 ms).

### Who decides what

- **The Model Component** declares the parallel semantics of its own
  graph, as node attributes (no WIT change; attributes are free-form):
  `tensor_parallel = "column"` on a matmul whose output columns may be
  split (each GPU computes its share of output features), `"row"` on a
  matmul whose input features are split (each GPU produces a partial
  sum), `"heads"` on rope/attention/kv nodes that operate per head, and
  `"local"` on elementwise nodes between them. Nothing generic guesses a
  layer's structure from weight names.
- **The Runtime** turns the declared semantics into a
  `TensorParallelPlan` (which weight is split by columns, by rows or
  replicated, and over which Devices) from the graph and the Devices'
  capacities, and hands it to the Provider before any weight is written,
  as it does a `LayerPlacement`.
- **The Provider** (`CudaMultiProvider`) stores each weight shard on its
  GPU and runs every invocation on every GPU over that GPU's shard
  (SPMD), with shapes and per-head attributes divided accordingly. After
  a `row` matmul it reduces the partial sums (peer copy and add, in the
  captured step) so the residual stream stays replicated. A final
  `column` matmul whose output the Runtime reads (the output head) is
  gathered on one GPU.

### Phases, each qualified by identical tokens to one GPU

1. MLP: `gate`/`up` column, `down` row, one reduction per layer;
   attention replicated. Dense Qwen2.5-0.5B, then Mistral 7B.
2. Attention by heads: `q`/`k`/`v` column (whole heads, K/V heads split
   with their query groups), rope, attention and the KV history per head
   shard, `o` row.
3. Mixture of experts: router replicated, each routed and shared expert's
   intermediate dimension split (column/row), one reduction per layer.
4. Output head column-split with a gather; embedding replicated.

### Evidence: phase 1 (MLP) on two RTX 3060, uncaptured

GPU Runner Smoke Test run 42762058770 (commit `cabf303`), real
Qwen2.5-0.5B-Instruct `Q8_0` GGUF, greedy, 64 tokens:

| Run | TPOT | TTFT | Load |
|---|---|---|---|
| One GPU, captured | 4.49 ms | 13.5 ms | 3.8 s |
| Two GPUs, layer segments, captured | 4.65 ms | 15.6 ms | 3.9 s |
| Two GPUs, layer segments, uncaptured | 11.60 ms | 14.6 ms | 3.9 s |
| Two GPUs, tensor parallel (MLP), uncaptured | 20.40 ms | 26.4 ms | 4.6 s |

The tensor-parallel run reports the plan applied and generates the same
token ids as one GPU (" Paris. The capital of the United States is
Washington, D. C. ..."). Correctness is established; speed is not yet:
every invocation runs on both GPUs one after the other from the host, and
each of the 24 reductions per step waits on both streams. This model is
far too small to gain from splitting (its decode is launch-bound, not
bandwidth-bound); the gain is expected on bandwidth-bound models (Mistral
7B, Qwen1.5-MoE) once steps are captured: each GPU's work captured as its
own graph, the reductions recorded between them, as the layer-segment
capture already does for its crossings.

### Evidence: phase 1 on Mistral 7B, and the first captured attempt

GPU Runner Smoke Test run 42769663567 (commit `92f7632`):

| Run | TPOT | TTFT |
|---|---|---|
| Mistral-7B-Instruct-v0.3 `Q8_0`, one GPU | 41.45 ms | 68.9 ms |
| Mistral-7B-Instruct-v0.3 `Q8_0`, two GPUs, tensor parallel (MLP) | 30.16 ms | 49.1 ms |
| Qwen2.5-0.5B `Q8_0`, two GPUs, tensor parallel, capture on | 19.85 ms | 23.1 ms |
| Qwen2.5-0.5B `Q8_0`, two GPUs, tensor parallel, capture off | 19.11 ms | 22.8 ms |

Mistral 7B is 27 % faster per token on two GPUs with only its MLP split,
answers "Paris" and generates the same token ids as one GPU (no
divergence was reported). On Qwen2.5-0.5B the captured run is no faster
than the uncaptured one: the step falls back to per-node execution, which
`MAGNETAR_STEP_CAPTURE_TRACE=1` now reports with its reason.

### Evidence: captured tensor-parallel steps

The second GPU's capture could not open while the first GPU's stream was
already capturing in thread-local mode (`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`,
reported by `MAGNETAR_STEP_CAPTURE_TRACE=1`). Each GPU now allocates what
its capture needs before any begins, and a tensor-parallel Provider
captures in relaxed mode. GPU Runner Smoke Test run 42781954300 (commit
`d438429`), 64 greedy tokens, decode step 2 captured in both models:

| Run | TPOT | TTFT |
|---|---|---|
| Qwen2.5-0.5B `Q8_0`, one GPU | 4.49 ms | 13.9 ms |
| Qwen2.5-0.5B `Q8_0`, two GPUs, tensor parallel, captured | 5.52 ms | 25.1 ms |
| Qwen2.5-0.5B `Q8_0`, two GPUs, tensor parallel, uncaptured (control) | 18.73 ms | 22.9 ms |
| Mistral-7B-Instruct-v0.3 `Q8_0`, one GPU | 41.54 ms | 68.8 ms |
| Mistral-7B-Instruct-v0.3 `Q8_0`, two GPUs, tensor parallel, captured | 28.77 ms | 49.5 ms |

Both models generate the same token ids as one GPU. Mistral 7B is 31 %
faster per token. With its MLP (about 80 % of a layer's weight bytes)
split and attention, embedding and output head replicated, a perfectly
bandwidth-bound step would take about 60 % of one GPU's time (25 ms); the
remaining gap is the replicated part and one reduction per layer. Qwen2.5-
0.5B is launch-bound, not bandwidth-bound, so splitting it cannot pay:
the captured step stays about 1 ms slower than one GPU's.

### Evidence: phase 3 (mixture of experts) on two RTX 3060

GPU Runner Smoke Test run 42789852734 (commit `9ea6b35`), Qwen1.5-MoE-A2.7B-
Chat `Q8_0` (15 GB, more than one GPU holds), 64 greedy tokens:

| Run | TPOT | TTFT | Load |
|---|---|---|---|
| Two GPUs, layer segments, captured | 14.02 ms | 33.0 ms | 98.6 s |
| Two GPUs, tensor parallel (experts whole per GPU, shared expert split), captured | 14.66 ms | 35.6 ms | 107.6 s |

The tensor-parallel run answers "Paris" with the same token ids as the
layer-segment run, and the CUDA expert-share conformance test passes on
the GPU. It is not faster. A decode step reads about 2.4 GB of `Q8_0`
weights (4 routed experts, the shared expert, attention and the output
head), about 7 ms at the RTX 3060's bandwidth, yet takes 14 ms on two GPUs
in sequence: this model's decode step is not bandwidth-bound, so halving
each GPU's expert reads does not shorten it. Where its time goes (the
expert kernels, routing, or the per-layer fixed cost) is the next thing
to measure; tensor parallelism stays opt-in (`MAGNETAR_MULTI_GPU=tensor`).

### Where a mixture-of-experts decode step's time goes

GPU Runner Smoke Test run 42797632531 timed each Kernel of a
Qwen1.5-MoE-A2.7B decode step (`moe_decode_kernel_times`, packed `Q8_0`,
one token):

| Kernel | Time | Bandwidth of the bytes read |
|---|---|---|
| `moe-experts`, 60 experts held | 144.8 us | 254 GB/s |
| attention projection `[2048, 2048]` | 32.1 us | 139 GB/s |
| shared expert up `[2048, 5632]` / down `[5632, 2048]` | 75.8 / 78.6 us | 162 / 156 GB/s |
| router `[2048, 60]` | 7.2 us | 18 GB/s |
| output head `[2048, 151936]` | 2019 us | 164 GB/s |

The same GPU's bf16 matmul reaches 285-340 GB/s. Summed per layer (four
attention projections, the shared expert, the routed experts, the router
and small Kernels) the step comes to the measured 14 ms: the `Q8_0`
matmul Kernel runs at half the bandwidth, which also explains Mistral 7B
(7.7 GB in 41 ms, 188 GB/s). Each lane now loads 16 quants per step
instead of 4; tensor parallelism can only pay off once the step is
bandwidth-bound.

### Evidence: 16-byte `Q8_0` loads

GPU Runner Smoke Test run 42804244356 (commit `edfb8dc`), every token
check passing (packed against dequantized, one GPU against two, layer
segments against tensor parallel):

| Kernel | Before | After |
|---|---|---|
| attention projection `[2048, 2048]` | 32.1 us (139 GB/s) | 21.6 us (207 GB/s) |
| shared expert up / down | 75.8 / 78.6 us | 46.2 / 51.8 us (265 / 237 GB/s) |
| output head `[2048, 151936]` | 2019 us (164 GB/s) | 1135 us (291 GB/s) |
| `moe-experts`, 60 experts | 144.8 us | 146.3 us (unchanged) |

| Run | TPOT before | TPOT after |
|---|---|---|
| Qwen2.5-0.5B `Q8_0`, one GPU | 4.49 ms | 3.31 ms |
| Mistral 7B `Q8_0`, one GPU | 41.54 ms | 28.03 ms |
| Mistral 7B `Q8_0`, two GPUs, tensor parallel | 28.77 ms | 20.67 ms |
| Qwen1.5-MoE `Q8_0`, two GPUs, layer segments | 14.02 ms | 11.36 ms |
| Qwen1.5-MoE `Q8_0`, two GPUs, tensor parallel | 14.66 ms | 12.35 ms |

`moe-experts` is now the largest Kernel of a mixture-of-experts layer: its
routing runs on one thread per token, and its down projection reads its
experts one after the other.

### Evidence: warp routing and concurrent expert reads

GPU Runner Smoke Test run 42822781072 (commit `be9fc30`): `moe-experts`
with 60 experts held takes 144.5 us (254 GB/s), unchanged; with 30 held,
77.7 us. Its time follows the routed experts it reads (about 36 us per
expert's three matrices, 255 GB/s): the routing and the order of the down
projection's reads were not its cost, and it runs near the 250-290 GB/s
the other `Q8_0` Kernels now reach. Qwen1.5-MoE TPOT: 11.33 ms (layer
segments), 12.28 ms (tensor parallel); Mistral 7B: 28.07 ms (one GPU),
20.62 ms (tensor parallel). All token checks pass.

Summed per layer, a Qwen1.5-MoE decode step is now mostly weight reads at
that bandwidth (about 10 ms of 11.3). Tensor parallelism halves each
GPU's expert and shared-expert reads but adds two reductions per layer
(48 per step), each issued from the host between per-GPU graphs: on
Qwen2.5-0.5B, 24 reductions cost 1.3 ms over one GPU. That overhead, not
the Kernels, is what keeps the tensor-parallel MoE step behind layer
segments.

### Evidence: copies between GPUs as Kernels

GPU Runner Smoke Test run 42861791878 (commit `b0805c2`): with copies and
reductions between GPUs done by Kernels reading the peer's memory (peer
access enabled), layer segments are back to their captured times
(Qwen2.5-0.5B 3.40 ms, Qwen1.5-MoE 11.31 ms) and every token check
passes. The one-graph tensor-parallel capture still fails at its first
reduction (`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`) and falls back to
per-node steps (Qwen2.5-0.5B 18.6 ms, Mistral 7B 24.4 ms, Qwen1.5-MoE
27.1 ms); a two-GPU Kernel-level test of the join now isolates it.

## Hardware limit: no peer access between the CI's two RTX 3060

The CI runner's two GeForce RTX 3060 (12 GB) most likely do not offer
direct peer-to-peer access to each other's memory: NVIDIA generally
disables peer access on GeForce cards. The evidence so far is indirect:

- a device-to-device copy between the two GPUs' contexts through unified
  addressing failed with `CUDA_ERROR_INVALID_VALUE`;
- the copy between GPUs then fell back to the driver's peer copy
  (`cuMemcpyPeerAsync`, staged by the driver), which works but cannot be
  recorded by a step capture, so the one-graph tensor-parallel capture
  failed at its first reduction;
- the two-GPU joined-capture conformance test returns early without peer
  access, and its output is not shown in CI, so it likely passed without
  testing anything.

Consequences, and what the code does about them:

- Copies and reductions between GPUs read the peer's memory in Kernels
  only when `cuDeviceCanAccessPeer` reports access and enabling it
  succeeds, per GPU (`CudaKernels::set_peer_access`).
- Without peer access, a tensor-parallel reduction crosses through a
  page-locked, portable host buffer kept per reduced tensor (two PCIe
  hops of a few KB): ordinary asynchronous copies a step capture records.
- Layer-segment crossings keep the driver's peer copy, outside captures.

To do later: confirm the missing peer access explicitly (log
`cuDeviceCanAccessPeer` in CI, and make the two-GPU joined-capture test
report when it skips); if a way to enable peer access on these GeForce
cards is found, compare direct-read reductions against the host-staged
ones.

### https://github.com/Zagitta/gpu-unlock (read 10 November 2026)

It does not unlock peer access: its README never mentions peer-to-peer
or GPU-to-GPU access. It unlocks GeForce tensor-core throughput, which
NVIDIA caps "to half speed for FP16, BF16 and TF32 math with FP32
accumulation" ("Up to 2x FP16 / BF16 / TF32 tensor throughput"):

- Mechanism: a Linux kernel module loaded from the initramfs, before the
  unmodified NVIDIA driver binds the card ("neither a UEFI application
  nor a patched NVIDIA driver"); it writes the tensor-core speed
  selectors and resets the function.
- Ampere RTX 30 is listed as "Stable, 100+ user reports", the RTX 3060
  12 GB included (27 to 51 TFLOPS BF16, user-reported); Ada and Turing
  are experimental, Blackwell unsupported. Linux only; no driver version
  requirement; no BAR or IOMMU requirement.
- Install: `make install` (root), then `gpu_unlock.targets=all` (or a
  `lspci -D` address) on the kernel command line, and a reboot; verify
  with `journalctl -k -b | grep gpu_unlock`. Suspend resets it (the module
  re-applies it on resume); hibernation is not covered. GPL-2.0-only,
  with NVIDIA firmware under NVIDIA's terms.
- The README claims bit-identical outputs, gains for prefill, fine-tuning
  and image generation, and "No change" for LLM decode and INT8/FP8/INT4.

- It does not change how the GPU identifies itself: its card table
  (`kernel/gpu_unlock/recipes.c`) only matches the PCI device ID the card
  already reports (GA106 among others); it rewrites no device, subsystem
  or vendor ID and no strap, does not present a GeForce as a professional
  card, and touches nothing related to peer access. On Ampere it opens two
  privilege masks (`WPR_CFG_PLM`, `FEAT_PLM`) and writes the speed
  selectors `FEAT_OVR_SM_SPD` (0x0082381c) and `FEAT_OVR_SM_SPD_1`
  (0x00823820).
- Its "How Ampere was unlocked" document says it reaches those registers
  through a buffer overflow in NVIDIA's signed SEC2 Booter (ROP, the
  cmpunlocker "V67" payload), a path it labels "RESEARCH-ONLY": it relies
  on exploiting a firmware vulnerability, which weighs on whether to run
  it on shared CI hardware.

For Magnetar it bears on the Tensor Core prefill path (bf16/f16 inputs,
f32 accumulation), not on decode, which is bandwidth-bound, and not on
the missing peer access. Applying it means changing the CI runner's host
kernel command line, which is the runner administrator's decision; if
done, the long-context prefill campaigns (Tensor Cores on and off) would
measure it.
