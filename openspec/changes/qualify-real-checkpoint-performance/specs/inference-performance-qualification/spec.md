## ADDED Requirements

### Requirement: A01 Real checkpoint performance harness

The qualification harness SHALL execute pinned Qwen2.5-0.5B-Instruct multi-step generation and record TTFT, TPOT, throughput, p50/p95 latency, RSS/VRAM, utilization, I/O bytes, context and batch sizes.

#### Scenario: A01 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a real checkpoint completes prefill and multiple decode steps
- **THEN** the report contains measured phase timings, token counts and resource peaks tied to the run

#### Scenario: A01 Ineligible or failed execution

- **WHEN** only synthetic fixture execution is available
- **THEN** the run is labeled fixture-only and cannot qualify production performance

### Requirement: A02 Numerical and lifecycle reference corpus

The harness SHALL maintain checkpoint-bound token/logit references, per-dtype tolerances, and streaming, resume and cancellation checks against Reference CPU and each qualified Provider.

#### Scenario: A02 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** CPU and CUDA execute the same pinned checkpoint and token inputs
- **THEN** logits and tokens are compared under the declared numerical profile and lifecycle events are checked

#### Scenario: A02 Ineligible or failed execution

- **WHEN** logits exceed the declared tolerance despite identical decoded text
- **THEN** qualification fails and preserves the differential evidence

### Requirement: A03 Hardware capability evidence

The qualification matrix SHALL distinguish AVX2, AVX-512, NEON, CUDA SM, graph capture, FP16/BF16, bandwidth, HIP and WGPU support using discovered hardware and executed tests.

#### Scenario: A03 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a target machine completes capability probes and real compute tests
- **THEN** the report names the tested Device and separates discovery, compute and production validation

#### Scenario: A03 Ineligible or failed execution

- **WHEN** required hardware is absent
- **THEN** the target is explicitly skipped with a reason and no production capability is claimed

### Requirement: A04 Repeatable cold and warm campaigns

Campaigns SHALL separate cold and warm runs, control ordering and interference, perform at least five post-warmup repetitions when reasonably affordable, and report median, dispersion and uncertainty keyed by code, model, Provider and driver identities.

#### Scenario: A04 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** repeated runs of an identical profile finish
- **THEN** raw samples and aggregate statistics are retained with the warmup and ordering protocol

#### Scenario: A04 Ineligible or failed execution

- **WHEN** five repetitions are unaffordable or results overlap measurement noise
- **THEN** the exception and uncertainty are recorded and no unsupported speedup is promoted

### Requirement: A07 Fixed budget memory campaigns

Qualification SHALL compare memory-ladder and fixed-budget split profiles with enforced budgets, OOM injection and CPU/GPU saturation, accounting for weights, KV, workspace and capture/prefetch overhead.

#### Scenario: A07 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** two residency strategies execute under the same total budget
- **THEN** reports compare throughput and tail latency alongside actual memory and I/O usage

#### Scenario: A07 Ineligible or failed execution

- **WHEN** a strategy exceeds the admitted budget or exhausts capacity
- **THEN** the run reports structured OOM/admission failure rather than an invalid speedup

### Requirement: A08 Downstream quality qualification

Numerical qualification SHALL include pinned perplexity or targeted task/conformance datasets and explicit acceptance thresholds in addition to token identity.

#### Scenario: A08 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a lower-precision or alternative model path is evaluated
- **THEN** logit errors and downstream quality deltas are compared with the declared baseline and dataset revisions

#### Scenario: A08 Ineligible or failed execution

- **WHEN** text looks plausible but the quality threshold is breached
- **THEN** the profile fails qualification and retains its higher-precision fallback

### Requirement: inference-performance-qualification promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed

### Requirement: Evidence based optimization maturity

Support reports SHALL distinguish proposed, contract-only, experimental and production-validated scope per software/hardware profile. Release claims SHALL require real multi-token checkpoint execution, measured before/after results, correct logits/downstream quality and lifecycle conformance on appropriate real hardware. Dates, version numbers and completed design artifacts SHALL NOT grant maturity.

#### Scenario: An accelerator has contracts but no accessible machine

- **WHEN** a release support matrix is produced
- **THEN** that profile remains contract-only or experimental with an explicit hardware skip

### Requirement: Comparable benchmark and release evidence

Campaigns SHALL version model and weight digests, code/Provider/driver identities, flags, sizes, warmup and budgets. They SHALL report TTFT, TPOT/ITL p50/p95/p99, aggregate/per-request throughput, memory peaks, launch/phase/transfer costs, useful cache savings and fragmentation/OOM; energy SHALL be included only with valid telemetry. Profiles SHALL exercise concurrency 1/4/16/32, context 512/4096/16384 and generation 128-512 tokens when supported, marking every unrun combination explicitly. Promotion SHALL reject unexplained numerical, quality, p95 or memory regressions relative to a declared objective. External engine comparisons SHALL be separated from internal baseline gains.

#### Scenario: A context or concurrency exceeds capacity

- **WHEN** the campaign builds its result matrix
- **THEN** that cell records unsupported, OOM or skipped with its reason rather than a zero or a passing result

#### Scenario: Aggregate throughput hides a tail latency regression

- **WHEN** a candidate violates the declared p95 or memory acceptance threshold
- **THEN** it is not promoted under that profile despite improved aggregate tokens per second

### Requirement: Optimization release safety gates

Affected implementations SHALL pass Rust formatting/checking/clippy, relevant unit/integration and Provider differential conformance, WIT and OpenSpec validation. Release evidence SHALL include cancellation, memory pressure, artifact tampering and invalid KV reuse checks, disable and rollback controls, and third-party license/SBOM/signature/provenance and reproducibility checks. Expensive hardware campaigns SHOULD run in scheduled/manual lanes while fast pull-request gates remain useful.

#### Scenario: Fast CI passes without required target evidence

- **WHEN** a change has green fixture tests but no real target campaign
- **THEN** it may land as a documented experiment but cannot claim production optimization qualification

### Requirement: Profile scoped 2.x and 3.x success evidence

An optimized 2.x profile SHALL demonstrate native F16/BF16 without global F32 weight expansion, at least one directly computed quantized format, multi-token device-resident critical state where supported, concurrent serving with paged KV and working prefix reuse, evidence-based contextual selection with rollback/pinned mode and published comparisons with the v1.0 baseline. A second non-simulated compute Provider SHALL be qualified when appropriate hardware is available; absence SHALL remain a declared portability limitation. A 3.x extension SHALL additionally require robust local multi-GPU, qualified MoE and versioned state mobility plus separately audited and real multi-node-tested Tachyon orchestration. These labels describe evidence goals, not a mandated release calendar or public API stability decision.

#### Scenario: A version is declared before its profile evidence exists

- **WHEN** release claims are reviewed
- **THEN** missing criteria remain visible and neither a version declaration nor proposed contracts count as success
