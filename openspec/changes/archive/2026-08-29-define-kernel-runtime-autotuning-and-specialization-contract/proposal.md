# Define Kernel Runtime Autotuning And Specialization Contract

## Why

Magnetar now supports the architectural lifecycle required for generative
Kernels:

```text
generate
  -> compile
  -> qualify
  -> benchmark
  -> exchange
  -> ingest
  -> cache
  -> prepare
  -> select
  -> promote
  -> execute
```

However, hardware performance is highly workload-dependent.

The best Kernel implementation may depend on:

- exact tensor shapes
- batch size
- sequence length
- prefill versus decode
- head dimension
- dtype
- layout
- quantization
- Device architecture
- Device feature set
- workspace availability

A single static benchmark result therefore cannot always identify the best
implementation for every Runtime workload.

Magnetar needs a bounded Runtime Autotuning mechanism capable of evaluating
already authorized Kernel variants and specialization configurations on the
actual execution target.

This mechanism must not become an arbitrary code-generation system.

The distinction is fundamental:

```text
Optimization Plane
    explores new implementations.

Runtime Autotuning
    evaluates a bounded specialization space already declared and authorized.
```

## What Changes

This change defines:

- Kernel Specialization Template
- Specialization Axis
- Specialization Instance
- bounded specialization domains
- specialization identity
- Runtime Autotuning Plan
- Runtime Autotuning Session
- tuning workload buckets
- tuning budgets
- tuning fixtures
- local benchmarking
- tuning records
- tuning cache
- tuning freshness
- qualification inheritance rules
- specialization preparation
- autotuning lifecycle
- warmup integration
- fallback behavior
- model-instance integration
- prefill/decode specialization
- Runtime selection integration
- Provider-local tuning boundaries
- resource-pressure protection
- observability
- conformance

## Core Principle

Runtime Autotuning SHALL operate only within a finite or explicitly bounded
candidate domain.

It SHALL NOT mutate arbitrary Kernel source.

It SHALL NOT invoke an AI code generator.

It SHALL NOT perform unconstrained search.

## Autotuning Versus Generative Optimization

Generative optimization MAY create:

```text
new algorithms
new source code
new fusion strategies
new memory strategies
new Operator implementations
```

Runtime Autotuning SHALL instead evaluate:

```text
existing Kernel candidates
existing compiled variants
authorized specialization values
Provider-declared execution configurations
```

The former belongs to the Optimization Plane.

The latter MAY occur in Magnetar cold/warm Runtime paths.

## Kernel Specialization Template

A Kernel Artifact MAY expose a Kernel Specialization Template.

The template describes which dimensions may vary without changing the portable
Operator semantics.

Conceptually:

```text
KernelSpecializationTemplate
    kernel
    axes
    constraints
    qualification_coverage
```

## Specialization Axis

An axis represents one bounded tuning dimension.

Examples MAY include:

```text
tile-m
tile-n
tile-k
block-size
warp-count
threadgroup-size
vector-width
pipeline-depth
split-k
prefetch-distance
attention-block-size
kv-page-size
execution-phase
```

Axis names SHALL be namespaced or otherwise scoped to avoid global semantic
collision.

## Axis Domain

Every tunable axis SHALL declare an explicit domain.

A domain MAY be:

```text
finite set
bounded integer range
bounded powers-of-two
enumerated symbolic values
Provider-defined bounded set
```

An unbounded integer/string domain SHALL NOT be accepted for Runtime
Autotuning.

Example:

```text
num-warps = {4, 8}
```

is valid.

An unconstrained:

```text
compiler-flags = arbitrary string
```

is not a valid Runtime Autotuning axis.

## Specialization Constraints

A template MAY define relationships between axes.

For example:

```text
block-m * block-n <= maximum-tile-elements

num-warps = 8 only when block-m >= 64
```

Constraints SHALL be deterministic and safely evaluable.

They SHALL NOT contain arbitrary executable scripts.

## Specialization Instance

A Specialization Instance is one concrete assignment of values to the declared
axes.

Example:

```text
block-m = 64
block-n = 128
num-warps = 8
phase = decode
```

A Specialization Instance SHALL have deterministic identity.

## Specialization Identity

Specialization identity SHOULD include:

```text
Kernel Artifact digest
Specialization Template version
axis/value assignments
target compatibility
```

Equivalent assignments SHALL yield stable identity independent of evaluation
order.

## Specialization Does Not Change Operator Semantics

Specialization SHALL remain implementation-level.

It SHALL NOT change:

- portable Operator identity
- Operator semantic version
- externally visible tensor semantics
- required correctness contract

If a configuration changes Runtime-visible semantics, it SHALL be represented
as a distinct Kernel contract rather than a tuning parameter.

## Source Specialization

A Kernel Source Artifact MAY require compilation for a Specialization Instance.

Example:

```text
Triton template
    +
BLOCK_M=64
BLOCK_N=128
    ->
specialized PTX/CUBIN
```

Such compilation SHALL use the Provider Kernel Compilation Capability.

It SHALL remain a cold-path operation.

## Compiled Variant Specialization

A bundle MAY already contain multiple compiled variants for Specialization
Instances.

Runtime MAY benchmark/select among them without source compilation.

## Preparation-Only Specialization

Some Providers MAY perform specialization during preparation.

Examples MAY include:

```text
pipeline configuration
graph compilation
pipeline state creation
launch metadata specialization
```

Logical specialization metadata SHALL still be explicit.

## Provider-Local Execution Parameters

Provider MAY expose bounded execution parameters that do not require a new
Compiled Kernel Artifact.

Such parameters MAY participate in Runtime Autotuning when their effects are
covered by the advertised Kernel contract.

Provider SHALL expose their allowed domain explicitly.

## Hidden Autotuning

Provider SHALL NOT perform opaque unbounded tuning on the inference hot path.

Provider-internal bounded autotuning MAY exist if:

- it is declared
- it respects Runtime deadlines/budgets
- it occurs outside active decode hot path
- its resulting behavior satisfies Kernel contract
- relevant determinism/precision properties remain valid

## Autotuning Plan

Runtime SHALL represent a bounded Autotuning Plan.

Conceptually:

```text
KernelAutotuningPlan
    candidate kernels
    specialization domain
    workload profile
    benchmark profile
    budget
    qualification policy
    fallback
```

The plan SHALL be deterministic from its inputs where policy requires
reproducibility.

## Candidate Enumeration

Before tuning begins, Runtime SHOULD be able to determine or bound the maximum
candidate space.

A policy SHALL limit candidate evaluation.

Example:

```text
total theoretical variants = 64
maximum tested variants = 16
```

Selection of the 16 MAY use deterministic pruning or Provider hints.

It SHALL NOT invoke arbitrary generation.

## Search Strategies

Autotuning MAY use bounded search strategies such as:

```text
exhaustive bounded search
ordered candidate list
Provider recommended ordering
successive elimination
bounded random sampling
deterministic sampling
simple bandit-like evaluation
```

The strategy SHALL NOT expand candidate domain beyond the declared template.

## Autotuning Is Not Qualification

A tuning benchmark answers:

```text
which eligible specialization performs better here?
```

It does not answer:

```text
is arbitrary new code correct?
```

Qualification requirements remain authoritative.

## Qualification Coverage

Specialization Template SHALL describe whether qualification evidence covers
the specialization space.

Possible conceptual coverage categories include:

```text
ExactInstance
EnumeratedInstances
DeclaredEnvelope
RequiresPerInstanceQualification
```

## Exact Instance Qualification

If qualification applies only to one exact Specialization Instance, another
instance SHALL not inherit it.

## Enumerated Qualification

Qualification MAY explicitly list multiple covered instances.

Only listed instances SHALL inherit the evidence.

## Envelope Qualification

A qualification profile MAY prove that a bounded specialization envelope
preserves required semantics.

Only if explicit evidence/policy allows such coverage MAY instances inside the
envelope inherit qualification.

## Per-Instance Qualification

If a specialization may alter numerical or memory behavior beyond existing
qualification coverage, it SHALL require qualification before production
eligibility.

## No Implicit Qualification Inheritance

Runtime SHALL NOT assume:

```text
same source template
    =>
all specializations qualified
```

without explicit coverage evidence.

## Trust Inheritance

A specialization derived deterministically from an accepted source/compiled
artifact MAY retain provenance relationships.

Trust of newly produced compiled bytes SHALL still follow artifact trust and
integrity policy.

A source artifact being trusted SHALL NOT automatically authenticate arbitrary
compiler output independently of configured build/trust policy.

## Autotuning Session

Execution of an Autotuning Plan SHALL occur within a Kernel Autotuning Session.

Suggested states:

```text
created
planning
preparing
warming-up
benchmarking
evaluating
completed
cancelled
timed-out
failed
```

## Autotuning Session Is Not Inference Session

Kernel Autotuning Session and Inference Session SHALL remain distinct concepts.

An Inference Session SHALL NOT own arbitrary tuning/compiler authority.

## Allowed Execution Points

Autotuning MAY occur at controlled points such as:

```text
Model Instance loading
Model Instance warmup
explicit management request
deployment preparation
idle-time background work
authorized optimization maintenance window
```

## Hot Path Prohibition

Normal token decode SHALL NOT synchronously start an Autotuning Session.

A cache miss for tuning information SHALL NOT cause unbounded benchmarking
inside decode.

Runtime SHALL instead use:

```text
known-good default
existing selected Kernel
static selection policy
structured not-ready/fallback behavior
```

according to policy.

## Model Warmup

Model Instance warmup MAY include bounded autotuning.

The Model Instance SHALL remain in a non-ready or explicitly warming state
until mandatory tuning is complete if deployment policy requires tuning.

Optional tuning SHALL not unnecessarily block readiness.

## Lazy Autotuning

Runtime MAY support lazy/background autotuning.

Lazy tuning SHALL:

- not block active token execution
- use bounded resources
- preserve the active known-good Kernel
- publish new tuning evidence atomically
- use normal selection/promotion rules

## Benchmark Fixtures

Autotuning SHALL use approved benchmark inputs.

Inputs SHOULD be:

```text
synthetic
deterministically generated
authorized benchmark fixtures
```

Production prompts/user content SHALL NOT be required.

## Workload Buckets

Tuning results SHALL be associated with workload context.

Typical dimensions MAY include:

```text
Operator
shape bucket
batch bucket
sequence-length bucket
prefill/decode phase
dtype
layout
quantization
Provider
Device architecture
Device feature set
```

## Exact Versus Bucketed Context

A tuning result MAY apply to:

```text
exact workload
bounded workload bucket
```

The covered domain SHALL be explicit.

Runtime SHALL not silently extrapolate a tuning result beyond its declared
workload compatibility.

## Prefill And Decode

Prefill and decode SHOULD be tunable independently.

A specialization optimal for prefill SHALL not automatically be assumed optimal
for decode.

## Continuous Batching

Autotuning MAY define workload buckets for:

- active sequences
- total active tokens
- raggedness
- KV cache layout
- paged cache use

Tuning itself SHALL not disturb active continuous batches.

## Benchmark Method

Autotuning benchmark SHALL define:

- warmup iterations
- measurement iterations
- synchronization behavior
- timeout
- metric
- outlier policy where applicable

Results SHALL be comparable only under compatible benchmark methodology.

## Primary Metric

Each Autotuning Plan SHOULD specify a primary optimization objective such as:

```text
latency
throughput
memory
energy
```

Correctness, trust, qualification, compatibility and memory feasibility remain
hard eligibility constraints.

## Secondary Metrics

A plan MAY specify secondary tie-breaking metrics.

Example:

```text
primary: latency
secondary:
    workspace
    determinism
```

## Runtime Selection Authority

Autotuning SHALL produce evidence/recommendations.

Kernel Selection Policy remains authoritative for actual execution.

```text
tuning winner != forced execution
```

## Autotuning Record

Completed tuning SHALL produce a KernelAutotuningRecord.

It SHOULD include:

- tuning plan fingerprint
- target Provider
- target Device compatibility
- Kernel candidate identities
- Specialization Instance identities
- workload bucket
- benchmark profile
- measured results
- selected tuning winner
- qualification references
- policy version
- timestamp/freshness metadata

## Tuning Cache

Magnetar SHOULD cache Autotuning Records.

This cache SHALL be logically distinct from:

- Kernel Artifact Cache
- Model Artifact Cache
- Prefix Cache
- KV Cache

## Tuning Cache Key

A tuning cache key SHOULD include:

```text
Operator semantics
candidate-set fingerprint
Kernel Artifact digests
specialization-template version
Provider version
Device architecture/features
driver/runtime compatibility
dtype
layout
workload bucket
benchmark profile
optimization objective
autotuning policy version
```

## Candidate-Set Fingerprint

If eligible specialization candidates change materially, cached tuning result
SHOULD become stale.

Adding a new candidate MAY trigger re-evaluation according to policy.

## Tuning Cache Hit

A tuning cache hit SHALL NOT bypass current eligibility validation.

Runtime SHALL re-check at least relevant:

- revocation
- trust
- qualification
- Provider readiness
- Device state
- memory feasibility
- Prepared Kernel readiness

## Tuning Freshness

Tuning evidence MAY become stale after:

```text
Provider update
driver/runtime update
Device firmware change
Kernel Artifact change
candidate set change
benchmark profile change
policy version change
```

Policy SHALL govern reuse.

## Stale Tuning Result

A stale result MAY be:

```text
ignored
used conservatively
used temporarily while retuning occurs
```

according to explicit policy.

It SHALL not be silently considered fully current.

## Bounded Runtime Cost

Autotuning SHALL have resource budgets.

Budgets MAY include:

```text
maximum candidates
maximum compilation jobs
maximum preparations
maximum benchmark invocations
wall-clock deadline
CPU time
Device time
host memory
Device memory
workspace
```

## Inference Resource Protection

Autotuning SHALL not consume unbounded resources required by active inference.

Runtime SHOULD support:

- lower-priority tuning work
- cancellation under pressure
- admission denial
- dedicated tuning windows
- separate tuning Device where available

## Memory Manager Authority

Memory Manager SHALL remain authoritative for tuning workspace feasibility.

A specialization requiring unavailable workspace SHALL not be benchmarked as if
feasible for production.

## Benchmark Memory Cleanup

Temporary benchmark allocations SHALL be released after each candidate/session
according to policy.

Autotuning SHALL not leak Tensor Resources into Model Instance execution state.

## Provider Pressure

High Provider/Device pressure MAY prevent or postpone autotuning.

Pressure SHOULD NOT invalidate an already correct cached tuning record by itself,
but it may affect whether new tuning work is admitted.

## Preparation

Each benchmarked candidate SHALL be prepared using normal Provider preparation
contracts.

Autotuning SHALL not bypass Prepared Kernel ownership rules.

## Candidate Failure

A single candidate may fail:

```text
compilation
preparation
qualification
benchmark
resource admission
```

without failing the entire Autotuning Session unless policy requires it.

## Known-Good Preservation

The currently selected known-good Kernel SHALL remain available while tuning
new specializations.

Failure of tuning SHALL NOT remove it.

## Tuning Winner Promotion

A tuning winner MAY become a candidate for normal Kernel selection/promotion.

Promotion SHALL still obey:

- current trust
- qualification
- compatibility
- selection policy
- hysteresis
- promotion policy

## No Automatic Active Replacement

Autotuning completion SHALL NOT implicitly replace an active Kernel if normal
promotion policy requires an explicit transition.

## Reproducible Mode

Reproducible Model Instance policy MAY disable Runtime Autotuning.

Alternatively, reproducible mode MAY consume a previously pinned
KernelAutotuningRecord.

Live re-tuning SHALL not silently alter a reproducible Model Instance.

## Deterministic Autotuning

Where reproducibility of tuning itself is required, the plan SHALL define:

- deterministic candidate ordering
- deterministic sampling seeds
- stable benchmark fixture
- stable tie-break rules

Performance measurements themselves MAY still contain hardware noise and
SHALL be handled according to benchmark policy.

## Autotuning Result Tie

If candidates are statistically or policy-equivalent, stable tie-breaking SHALL
be used.

Runtime SHOULD prefer existing known-good/active specialization where policy
values stability.

## Provider Recommended Defaults

Provider MAY advertise recommended default specialization values.

Defaults MAY reduce search cost.

They SHALL NOT become authoritative over Runtime eligibility/policy.

## Provider Search Hints

Provider MAY advertise bounded hints such as:

```text
preferred tile order
known-bad combinations
architecture-specific preferred values
```

Hints SHALL not expand the declared specialization domain.

## Provider Opaque Autotuner

A Provider MAY expose a bounded native autotuning capability.

If used, it SHALL accept an explicit bounded candidate/template contract and
return stable result metadata.

It SHALL NOT receive authority to generate arbitrary source or alter Runtime
selection constraints.

## Autotuner ABI

A future native implementation MAY expose tuning through Provider ABI.

Any ABI SHALL:

- be explicitly versioned
- avoid Rust trait-object ABI
- use opaque IDs
- define buffer ownership
- normalize errors
- prevent unwinding

This change defines semantics, not exact ABI declarations.

## Specialization Cache

Specialized Compiled Kernel Artifacts MAY be stored in Kernel Artifact Cache.

Their identity SHALL include specialization values.

Two materially different Specialization Instances SHALL not alias to the same
compiled artifact identity unless bytes are genuinely identical and metadata
relationships remain explicit.

## Preparation Cache

Prepared Kernel state remains ephemeral and Provider-owned.

Autotuning records SHALL not serialize PreparedKernelId as portable state.

## Cross-Device Tuning

A tuning result for one Device architecture SHALL not automatically apply to an
incompatible Device architecture.

Even identical Device models MAY require policy-controlled reuse if driver or
runtime compatibility differs.

## Device-Specific Optimization

Device-specific tuning is allowed.

Portable Operator semantics SHALL remain unchanged.

## Browser And Mobile

The same Runtime Autotuning semantics MAY apply to:

```text
WebGPU pipeline variants
Metal pipeline variants
Vulkan shader/pipeline variants
mobile AOT specialization sets
```

where platform capabilities permit.

Autotuning SHALL respect platform compilation/execution restrictions.

## Offline Deployment

Offline deployment MAY ship:

- pre-specialized Kernel Artifacts
- precomputed Autotuning Records
- pinned selections

No live tuning SHALL be required.

## Security Boundary

Autotuning SHALL operate only on accepted Kernel Artifacts and authorized
specialization templates.

A quarantined or rejected Kernel SHALL not become tunable through normal Runtime
path.

## Arbitrary Compiler Flags Prohibited

Runtime Autotuning SHALL NOT expose unrestricted compiler argument strings as a
tuning dimension.

Only explicitly modeled, bounded specialization parameters are permitted.

## Arbitrary Source Mutation Prohibited

Runtime Autotuning SHALL NOT rewrite arbitrary Kernel source.

If new source must be generated or semantically altered, work belongs to the
Optimization Plane.

## Arbitrary Network Access Prohibited

Autotuning SHALL NOT require arbitrary network access.

Required artifacts/toolchains SHALL already be available through authorized
artifact/Provider boundaries.

## Error Model

Structured errors SHOULD include:

```text
kernel-autotuning-disabled
kernel-autotuning-policy-invalid
kernel-autotuning-template-invalid
kernel-autotuning-template-unbounded
kernel-autotuning-axis-invalid
kernel-autotuning-value-out-of-domain
kernel-autotuning-constraint-unsatisfied
kernel-autotuning-no-candidates
kernel-autotuning-no-eligible-candidates
kernel-autotuning-budget-exceeded
kernel-autotuning-admission-denied
kernel-autotuning-timeout
kernel-autotuning-cancelled

kernel-specialization-invalid
kernel-specialization-identity-invalid
kernel-specialization-compilation-required
kernel-specialization-compilation-failed
kernel-specialization-preparation-failed
kernel-specialization-qualification-required
kernel-specialization-qualification-invalid
kernel-specialization-workload-incompatible

kernel-autotuning-benchmark-failed
kernel-autotuning-benchmark-invalid
kernel-autotuning-metric-unavailable
kernel-autotuning-result-inconclusive
kernel-autotuning-result-stale
kernel-autotuning-cache-miss
kernel-autotuning-cache-invalid

kernel-autotuning-hot-path-denied
kernel-autotuning-provider-pressure
kernel-autotuning-memory-infeasible
internal-kernel-autotuning-error
```

## Observability

Autotuning observability MAY include:

```text
autotuning-planned
autotuning-started
autotuning-candidate-enumerated
autotuning-candidate-pruned
specialization-compilation-started
specialization-prepared
autotuning-benchmark-started
autotuning-candidate-failed
autotuning-candidate-measured
autotuning-winner-selected
autotuning-completed
autotuning-cache-hit
autotuning-cache-stale
autotuning-cancelled
autotuning-timed-out
```

Observability MAY include:

- KernelId
- Kernel Artifact digest
- Specialization Instance ID
- workload bucket
- target architecture
- Provider/Device stable binding
- benchmark metric summary
- policy version
- budget consumption

Observability SHALL NOT expose:

- raw Kernel source
- native handles
- raw tensor fixtures by default
- model weights
- prompts
- KV contents
- secrets
- credentials

## Conformance

Conformance SHALL validate:

- autotuning candidate domain is bounded
- arbitrary source mutation is impossible
- arbitrary compiler flags are impossible
- token decode cannot synchronously start tuning
- only accepted artifacts can be tuned
- qualification coverage is explicit
- new specialization does not inherit qualification implicitly
- tuning does not replace correctness qualification
- tuning cache is context-sensitive
- stale tuning evidence is detectable
- memory feasibility is authoritative
- candidate failure preserves active known-good Kernel
- tuning winner does not bypass selection/promotion policy
- Provider hints do not override Runtime policy
- prefill/decode can tune independently
- reproducible mode can disable/pin tuning
- PreparedKernel native handles are not persisted
- autotuning observability is redacted

## Non-Goals

This change does not:

- generate arbitrary new Kernel source
- embed an AI code-generation agent
- perform evolutionary source-code search
- replace the Optimization Plane
- qualify arbitrary generated code automatically
- define one universal tuning algorithm
- require live autotuning
- require tuning on mobile/browser
- make tuning mandatory for Model Instance readiness
- allow tuning in token hot path
- allow Provider to bypass Kernel Selection Policy

## Impact

Magnetar gains a middle ground between static kernels and arbitrary generative
optimization:

```text
static implementation
        |
        v
bounded specialization space
        |
        v
target-local autotuning
        |
        v
cached tuning evidence
        |
        v
Kernel Selection Policy
        |
        v
execution
```

This allows Magnetar to adapt highly optimized Kernels to real hardware and
workload shapes without compromising its inference Runtime boundary.