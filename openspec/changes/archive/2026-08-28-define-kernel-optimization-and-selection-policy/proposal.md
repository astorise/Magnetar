# Define Kernel Optimization And Selection Policy

## Why

Magnetar can now represent, compile, qualify, cache, prepare, promote, revoke,
and hot-swap multiple Kernel implementations of the same portable Operator.

This creates a new Runtime responsibility:

```text
Which eligible Kernel should execute this Operator now?
```

A simple fastest-kernel rule is insufficient.

Kernel suitability depends on:

- Operator semantics
- qualification
- trust
- dtype
- layout
- shape
- Device architecture
- Resource Affinity
- Provider readiness
- Device pressure
- memory availability
- workspace requirements
- determinism requirements
- latency
- throughput
- batching profile
- sequence profile
- energy goals
- preparation state
- benchmark freshness
- reproducibility policy

The Runtime therefore needs an explicit optimization and selection policy.

This policy must remain deterministic, inspectable, explainable, and subordinate
to correctness and safety.

## What Changes

This change defines:

- Kernel candidate eligibility
- hard constraints
- optimization objectives
- selection profiles
- weighted ranking
- lexicographic ranking
- deterministic tie-breaking
- runtime pressure-aware selection
- memory-aware selection
- batch-aware selection
- sequence-aware selection
- determinism-aware selection
- reproducibility modes
- policy precedence
- fallback behavior
- selection hysteresis
- anti-flapping policy
- benchmark freshness
- exploration/canary constraints
- Model Instance kernel pinning
- session/generation policy boundaries
- selection observations
- structured failure reasons
- conformance requirements

## Core Rule

Kernel selection SHALL happen in two logical phases:

```text
1. Eligibility filtering
2. Optimization ranking
```

Optimization SHALL never make an ineligible candidate eligible.

## Eligibility Before Optimization

A Kernel candidate SHALL be excluded before performance ranking when it fails
any required hard constraint.

Hard constraints MAY include:

```text
Operator semantic compatibility
Operator version compatibility
qualification status
trust policy
revocation status
dtype compatibility
layout compatibility
shape compatibility
precision policy
determinism policy
Resource Affinity
Provider readiness
Device compatibility
Device health
memory feasibility
workspace feasibility
required features
Prepared Kernel readiness
execution-mode compatibility
```

A candidate failing any required hard constraint SHALL NOT receive a positive
performance score that allows it to re-enter the candidate set.

## Optimization Objective

An optimization objective describes how eligible candidates should be ranked.

Supported portable objective dimensions MAY include:

```text
latency
throughput
tail-latency
memory
workspace
energy
determinism
startup-cost
preparation-cost
batch-efficiency
sequence-efficiency
```

The vocabulary SHOULD remain extensible.

## Optimization Profiles

Runtime SHALL support named optimization profiles.

Baseline profiles MAY include:

```text
balanced
latency
throughput
memory
deterministic
energy
reproducible
```

A profile is policy, not Provider identity.

Providers SHALL NOT create hidden selection profiles that bypass Runtime
policy.

## Balanced Profile

The balanced profile SHOULD prefer generally efficient candidates without
aggressively optimizing one metric at the expense of all others.

Its exact scoring weights MAY evolve before a stable policy version is frozen.

## Latency Profile

The latency profile SHOULD prioritize end-to-end execution latency among
eligible candidates.

It SHALL still respect:

- memory limits
- trust
- qualification
- determinism requirements
- Provider readiness
- Resource Affinity

## Throughput Profile

The throughput profile SHOULD prefer candidates that maximize useful work per
unit time for the current workload.

It MAY account for:

- batch size
- sequence length
- active sequences
- device occupancy
- continuous batching behavior

## Memory Profile

The memory profile SHOULD prefer candidates with lower:

```text
workspace
temporary allocations
persistent executable-memory cost
optional conversion pressure
```

provided correctness and required performance policy remain satisfied.

## Deterministic Profile

The deterministic profile SHALL reject candidates whose declared or qualified
behavior does not satisfy the requested determinism contract.

Performance SHALL NOT override deterministic requirements.

## Energy Profile

The energy profile MAY prefer lower-energy candidates where comparable energy
evidence exists.

Absence of reliable energy metadata SHALL be explicit.

Runtime SHALL NOT invent energy estimates.

## Reproducible Profile

The reproducible profile SHALL prefer stable, pinned Kernel selection.

It SHOULD minimize environment-sensitive re-ranking.

A reproducible execution policy MAY pin:

```text
KernelId
Kernel artifact digest
Prepared generation
Provider
Device compatibility class
qualification profile
```

## Ranking Strategies

Runtime MAY support more than one ranking strategy.

Portable strategies SHOULD include:

```text
Lexicographic
WeightedScore
PolicyOrdered
Pinned
```

## Lexicographic Ranking

Lexicographic ranking evaluates metrics in declared priority order.

Example:

```text
1. deterministic
2. p99 latency
3. workspace
4. mean latency
```

The first metric that distinguishes candidates decides the ranking.

## Weighted Ranking

Weighted ranking MAY combine normalized metrics.

Example:

```text
score =
    latency_weight    * latency_score
  + throughput_weight * throughput_score
  + memory_weight     * memory_score
```

Weights SHALL belong to Runtime policy.

Providers SHALL NOT silently redefine them.

## Comparable Metrics

Weighted ranking SHALL only combine metrics that have a defined normalization
or comparison model.

Missing metrics SHALL be handled explicitly.

Runtime SHALL NOT interpret:

```text
missing benchmark
```

as:

```text
best benchmark
```

## Missing Performance Evidence

If an eligible Kernel lacks required performance evidence, policy MAY:

```text
exclude
rank conservatively
use fallback metadata
request benchmark outside hot path
retain currently active kernel
```

The behavior SHALL be explicit.

Benchmarking SHALL NOT be silently started in token decode.

## Benchmark Context

Performance evidence SHALL be evaluated only when compatible with the current
execution context.

Compatibility SHOULD consider:

- Provider
- Device architecture
- driver/runtime compatibility
- Operator version
- Kernel artifact
- dtype
- layout
- shape
- batch
- sequence length
- execution mode
- benchmark profile version

## Benchmark Freshness

Stale benchmark evidence SHALL be identifiable.

Runtime policy MAY:

- accept stale evidence
- discount stale evidence
- exclude stale evidence
- request re-benchmarking outside hot path

It SHALL NOT silently treat incompatible evidence as current.

## Shape-Aware Ranking

Kernel ranking SHOULD account for shape specialization.

A Kernel optimized for:

```text
batch = 1
sequence = 128
```

SHALL NOT automatically be assumed optimal for:

```text
batch = 64
sequence = 8192
```

Performance evidence SHOULD be indexed by workload envelope.

## Batch-Aware Ranking

Continuous batching SHOULD expose enough workload metadata for Kernel
selection.

Ranking MAY consider:

- active sequence count
- batch width
- total active tokens
- raggedness
- prefill/decode phase
- KV cache mode

## Prefill Versus Decode

Runtime MAY select different Kernels for prefill and decode.

Example:

```text
Attention prefill:
    throughput-optimized kernel

Attention decode:
    latency-optimized kernel
```

A Model Instance SHALL NOT assume the same Kernel implementation is optimal for
both phases.

## Pressure-Aware Ranking

Provider/Device pressure MAY influence ranking among eligible candidates.

Pressure SHALL NOT bypass semantic compatibility.

Pressure inputs MAY include:

```text
queue depth
device utilization
memory pressure
workspace pressure
execution backlog
Provider admission state
```

## Device Health

An unhealthy or unavailable Device SHALL be filtered before ranking when
policy requires readiness.

A fast benchmark on an unavailable Device is irrelevant.

## Memory Feasibility

Memory Manager SHALL remain authoritative for memory feasibility.

Kernel ranking MAY use:

- workspace size
- temporary storage
- memory class
- residency compatibility

but SHALL NOT override Memory Manager rejection.

## Conversion Cost

Runtime MAY account for explicit dtype/layout/data-movement conversions when
ranking candidates.

Example:

```text
Kernel A:
    execution 20 µs
    requires layout conversion 30 µs

Kernel B:
    execution 35 µs
    no conversion
```

Policy MAY correctly prefer Kernel B.

Conversion cost SHALL correspond to explicit Runtime-visible movement or
conversion operations.

No hidden conversions SHALL be introduced.

## Preparation Cost

Preparation cost MAY influence cold-path planning.

Once a Kernel is already prepared, historical preparation cost SHOULD NOT be
blindly added to every execution decision.

Runtime SHALL distinguish:

```text
one-time cost
per-model-instance cost
per-batch cost
per-operation cost
```

## Compilation Cost

Compilation cost MAY influence artifact planning or future optimization
decisions.

Compilation cost SHALL NOT be charged as a hot-path execution metric after a
compatible compiled artifact is already cached/prepared.

## Active Kernel Preference

Runtime MAY prefer the currently active eligible Kernel when another candidate
offers only insignificant estimated benefit.

This prevents selection churn.

## Selection Hysteresis

Runtime SHOULD support hysteresis.

A candidate SHOULD replace the current active Kernel only if its expected
benefit exceeds a policy threshold.

Example:

```text
new candidate only 0.5% faster
    -> retain active kernel

new candidate 12% faster
    -> eligible for promotion
```

Exact thresholds are policy-specific.

## Anti-Flapping

Runtime SHALL avoid rapid oscillation between candidates caused by small
measurement or pressure changes.

Anti-flapping mechanisms MAY include:

- minimum active duration
- hysteresis threshold
- cooldown period
- rolling measurements
- stable ranking window
- explicit promotion events

## Selection Versus Promotion

Selection and promotion SHALL remain conceptually distinct.

Selection answers:

```text
which eligible Kernel best satisfies this policy?
```

Promotion answers:

```text
should this candidate become the preferred active generation?
```

A candidate MAY rank first without immediate promotion.

## Static Selection

Runtime MAY select a Kernel once during Model Instance loading and pin it for
the lifetime of the instance.

This mode is useful for:

- reproducibility
- low-overhead deployment
- offline environments
- strict validation

## Dynamic Selection

Runtime MAY re-evaluate selection during execution where policy allows.

Dynamic re-selection SHALL NOT violate Prepared Kernel lifetime or in-flight
stability.

## Per-Operation Selection

Runtime MAY select different Kernels for individual Operator invocations.

This SHOULD be used only where selection overhead is justified and bounded.

## Selection Caching

Runtime MAY cache selection decisions.

A cached selection key SHOULD include compatibility-relevant workload and
policy context.

Possible dimensions include:

```text
Operator
Provider/Device
dtype
layout
shape bucket
batch bucket
sequence bucket
generation phase
optimization profile
policy version
```

Selection cache entries SHALL be invalidated when relevant eligibility or
compatibility state changes.

## Model Instance Kernel Policy

A Model Instance SHALL own or reference an explicit Kernel selection policy.

It MAY define:

```text
dynamic selection
pinned selection
profile
fallback policy
determinism
allowed qualification profiles
trust policy
```

## Model Component Boundary

Model Component SHALL NOT choose concrete Kernel implementations.

It MAY declare semantic/operator requirements.

Runtime selection policy chooses Kernel implementation.

## Session Boundary

Inference Session MAY request a non-authoritative optimization preference if
allowed.

Examples:

```text
prefer-low-latency
prefer-deterministic
```

Runtime policy remains authoritative.

Session SHALL NOT provide:

```text
native Kernel handle
Provider pointer
Device pointer
unqualified KernelId override
```

## Generation Request Boundary

Generation requests MAY provide high-level policy preferences.

They SHALL NOT directly force an ineligible concrete Kernel.

## CLI Boundary

CLI MAY expose user-facing preferences such as:

```text
--profile latency
--profile throughput
--deterministic
```

CLI preferences SHALL map into Runtime policy inputs.

CLI SHALL NOT bypass Registry eligibility.

## Provider Boundary

Provider SHALL expose metrics and metadata needed for Runtime policy where
available.

Provider SHALL NOT make the final cross-Provider Kernel selection decision.

Provider MAY internally select among private implementation details only when
those distinctions do not violate advertised Kernel semantics or Runtime
policy.

## Provider-Local Variant Selection

A Provider MAY internally select a private execution variant behind one
Prepared Kernel when:

- all variants satisfy the same advertised contract
- Runtime-visible semantics remain identical
- compatibility remains unchanged
- determinism/precision commitments remain valid

If the variants differ in Runtime-relevant properties, they SHALL be modeled as
distinct Kernel candidates.

## Fallback Policy

Fallback SHALL be explicit.

Fallback MAY specify ordered classes such as:

```text
same-provider alternative
same-device alternative
other compatible Provider
Reference CPU
fail
```

Fallback SHALL NOT be silently inserted.

## Reference CPU Fallback

Reference CPU MAY be used as correctness fallback where supported.

It SHALL NOT automatically be selected when:

- policy forbids fallback
- Resource Affinity makes movement invalid
- required semantics are unsupported
- host staging is forbidden

## No Hidden Cross-Provider Movement

Kernel selection SHALL NOT silently move tensor resources across Providers.

If candidate selection requires data movement, the movement SHALL be explicit
and policy-authorized.

## Selection Explainability

Runtime SHOULD be able to explain why a Kernel was:

```text
eligible
excluded
ranked
selected
retained
promoted
not promoted
fallen back from
```

Selection explanation SHALL use redacted stable metadata.

## Candidate Exclusion Reasons

Structured exclusion reasons SHOULD include:

```text
semantic-incompatible
operator-version-incompatible
qualification-required
qualification-expired
qualification-revoked
trust-denied
dtype-incompatible
layout-incompatible
shape-incompatible
precision-incompatible
determinism-incompatible
resource-affinity-incompatible
provider-unready
device-unavailable
device-unhealthy
memory-infeasible
workspace-infeasible
required-feature-missing
prepared-kernel-unavailable
benchmark-incompatible
policy-denied
```

## Deterministic Tie-Breaking

Selection SHALL define deterministic tie-breaking.

Tie-breaking SHOULD NOT depend on:

- hash-map iteration order
- pointer values
- process IDs
- thread timing
- nondeterministic discovery order

A stable tie-break MAY use:

```text
policy rank
KernelId stable key
artifact digest
Provider stable identity
```

## Policy Versioning

Kernel selection policy SHALL be versioned.

Selection results SHOULD record the policy version that produced them.

Changing policy MAY alter Kernel choices without changing Operator semantics.

## Policy Precedence

Policy MAY be composed from multiple levels.

A recommended precedence is:

```text
Runtime safety/security constraints
        >
deployment policy
        >
Model Instance policy
        >
Session preference
        >
Generation request preference
        >
CLI/user hint
```

Lower-level preferences SHALL NOT override higher-level constraints.

## Exploration

Runtime MAY support controlled exploration of newly qualified candidates.

Exploration SHALL be explicit and disabled by default for strict/reproducible
modes.

Exploration SHALL only consider already eligible candidates.

Unqualified or untrusted Kernels SHALL NOT be explored when policy requires
qualification/trust.

## Canary Selection

A candidate MAY be assigned canary traffic or limited execution opportunities.

This change defines only local policy semantics, not distributed rollout.

Canary policy MAY constrain:

- request count
- percentage
- duration
- workload profile

## Exploration Failure

If exploratory execution fails, policy MAY:

- stop exploration
- mark candidate unhealthy
- demote candidate
- trigger rollback
- retain known-good active Kernel

Failure SHALL NOT automatically affect unrelated Kernels.

## Selection Metrics

Runtime MAY collect local execution metrics for ranking refinement.

Metrics SHALL be associated with:

- Kernel generation
- Operator
- workload bucket
- Provider/Device
- execution profile

Raw model data SHALL not be required for selection analytics.

## Online Measurement

Online observations MAY supplement offline benchmark evidence.

Online measurement SHALL NOT override correctness or trust.

## Selection Stability

Runtime SHOULD prefer stable decisions over reacting to insignificant metric
noise.

Performance optimization SHALL NOT introduce uncontrolled selection
nondeterminism.

## Selection Errors

Structured selection errors SHOULD include:

```text
kernel-selection-no-candidates
kernel-selection-no-eligible-candidates
kernel-selection-policy-invalid
kernel-selection-profile-unsupported
kernel-selection-pinned-kernel-unavailable
kernel-selection-pinned-kernel-ineligible
kernel-selection-metric-missing
kernel-selection-benchmark-stale
kernel-selection-benchmark-incompatible
kernel-selection-memory-infeasible
kernel-selection-affinity-incompatible
kernel-selection-determinism-unsatisfied
kernel-selection-fallback-denied
kernel-selection-fallback-exhausted
kernel-selection-promotion-threshold-not-met
kernel-selection-cache-stale
kernel-selection-exploration-denied
internal-kernel-selection-error
```

## Observability

Selection observability MAY include:

```text
selection-started
candidate-discovered
candidate-excluded
candidate-eligible
candidate-ranked
kernel-selected
active-kernel-retained
fallback-selected
selection-cache-hit
selection-cache-miss
selection-recomputed
promotion-suggested
promotion-threshold-not-met
exploration-started
exploration-stopped
```

Observability MAY record:

- KernelId
- artifact digest
- Provider binding
- Device binding
- optimization profile
- workload bucket
- exclusion reason
- rank
- normalized score
- policy version
- benchmark freshness

Observability SHALL NOT expose:

- native handles
- raw tensor values
- model weights
- KV cache contents
- raw prompts
- secrets
- credentials

## Conformance

Conformance SHALL validate:

- ineligible candidates never win through performance
- qualification precedes ranking where required
- trust precedes ranking where required
- Resource Affinity precedes ranking
- memory feasibility precedes ranking
- deterministic policy rejects nondeterministic candidates
- ranking is deterministic for identical inputs
- ties are stable
- missing metrics are handled explicitly
- stale/incompatible benchmarks are not silently accepted
- shape/workload context affects ranking appropriately
- active Kernel hysteresis prevents insignificant promotion
- fallback is explicit
- Reference CPU fallback obeys affinity/staging policy
- Model Component cannot choose Kernel
- session/user preferences are non-authoritative
- promotion remains distinct from ranking
- exploration uses eligible candidates only
- Provider cannot override Runtime cross-Provider selection
- selection explanation contains no native handles

## Non-Goals

This change does not:

- define one globally optimal scoring formula
- prescribe exact production weights
- implement reinforcement learning for selection
- implement a distributed scheduler
- implement fleet-wide canary rollout
- permit unqualified-kernel exploration by default
- redefine Kernel qualification
- redefine Provider compilation
- redefine Kernel Artifact cache
- make performance correctness-authoritative
- allow Components to choose Providers or Devices

## Impact

Magnetar gains an explicit policy engine for selecting among many valid Kernel
implementations.

The resulting flow is:

```text
Operator invocation
       |
       v
Candidate discovery
       |
       v
Hard eligibility filters
       |
       v
Eligible set
       |
       v
Optimization policy
       |
       v
Stable ranking
       |
       v
Selection
       |
       +-> retain active Kernel
       +-> choose prepared candidate
       +-> fallback
       +-> suggest promotion
```

This makes generative Kernel ecosystems practical without turning Kernel
selection into opaque or unsafe heuristics.