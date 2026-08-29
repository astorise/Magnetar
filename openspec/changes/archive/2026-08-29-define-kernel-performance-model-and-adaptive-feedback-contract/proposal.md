# Define Kernel Performance Model And Adaptive Feedback Contract

## Why

Magnetar now supports:

```text
Kernel Artifact lifecycle
Provider compilation
qualification
benchmarking
content-addressed cache
hot swap
Kernel Selection Policy
Optimization Plane
portable exchange
artifact ingestion
bounded Runtime Autotuning
```

Runtime Autotuning can identify a suitable Kernel specialization for a given
hardware/workload context.

However, benchmark evidence is only a prediction of production behavior.

Real inference conditions evolve:

- batch composition changes
- sequence-length distributions change
- prefill/decode ratios change
- Device pressure changes
- drivers change
- firmware changes
- Provider versions change
- thermal/power behavior changes
- continuous batching changes
- memory pressure changes
- competing workloads appear

A specialization that was optimal when tuned may therefore become suboptimal or
its evidence may become stale.

Magnetar needs a bounded adaptive feedback loop that observes real execution
performance and uses aggregated evidence to:

- detect benchmark drift
- detect runtime regressions
- detect workload distribution shift
- invalidate stale tuning evidence
- recommend bounded re-tuning
- recommend selection re-evaluation
- trigger rollback/demotion policy where appropriate

This feedback loop SHALL NOT turn production inference into arbitrary
experimentation or code generation.

## What Changes

This change defines:

- Kernel Execution Performance Observation
- Workload Bucket
- performance aggregation
- Performance Model
- confidence and sample sufficiency
- baseline comparison
- benchmark drift detection
- workload drift detection
- performance regression detection
- tuning staleness
- adaptive re-tuning requests
- adaptive selection re-evaluation
- rollback/demotion signals
- online-versus-offline evidence precedence
- measurement sampling
- measurement overhead budgets
- privacy boundaries
- outlier handling
- warmup handling
- cold-start handling
- observation aging
- model versioning
- stability/hysteresis
- failure containment
- observability
- conformance

## Core Principle

Performance feedback is optimization evidence only.

```text
performance evidence
    != correctness evidence

performance evidence
    != trust evidence

performance evidence
    != qualification evidence
```

No amount of good production latency SHALL make an otherwise unqualified,
untrusted, incompatible, or revoked Kernel eligible.

## Performance Observation

Runtime MAY record a Kernel Execution Performance Observation for completed
Kernel executions.

A conceptual observation MAY include:

```text
KernelId
artifact digest
Prepared Kernel generation
Operator
Provider
Device compatibility identity
workload bucket
execution phase
latency
queue delay
workspace usage
memory pressure
Provider pressure
completion status
timestamp
```

Raw tensor contents SHALL not be required.

## Observation Scope

An observation SHALL identify enough execution context to avoid mixing
incompatible measurements.

Context SHOULD include where relevant:

- Kernel Artifact digest
- Specialization Instance
- Provider version
- Device architecture
- driver/runtime compatibility
- Operator version
- dtype
- layout
- shape bucket
- batch bucket
- sequence bucket
- prefill/decode phase
- quantization profile
- execution mode

## Workload Bucket

Performance evidence SHALL be associated with a workload bucket rather than
treated as globally applicable.

A workload bucket is a normalized representation of execution context.

Example:

```text
operator = attention
phase = decode
dtype = fp16
batch = 4..8
sequence = 2049..4096
head_dim = 128
architecture = sm90
```

## Workload Bucket Identity

Workload bucket identity SHALL be deterministic.

Equivalent Runtime contexts SHALL map to the same bucket under the same bucket
policy version.

Changing bucket policy MAY invalidate or separate historical evidence.

## Bucket Policy

Bucketing SHALL be versioned.

Policy SHOULD define:

- shape boundaries
- batch boundaries
- sequence boundaries
- phase categories
- pressure inclusion
- architecture grouping
- precision grouping

## No Raw Prompt Bucketing

Workload buckets SHALL NOT include raw prompt text, user identity, document
contents, secrets, or arbitrary user data.

## Performance Aggregation

Runtime SHOULD aggregate observations rather than retaining every individual
event indefinitely.

Aggregation MAY include:

```text
count
mean
variance
minimum
maximum
p50
p90
p95
p99
failure count
timeout count
workspace summary
pressure summary
```

Exact metrics MAY depend on policy.

## Bounded Retention

Performance observation retention SHALL be bounded.

Runtime SHALL avoid unbounded per-invocation telemetry growth.

Raw observations MAY be discarded after aggregation according to policy.

## Performance Model

A Kernel Performance Model represents aggregated evidence for a Kernel
candidate in a compatible workload/target context.

Conceptually:

```text
KernelPerformanceModel
    candidate
    workload_bucket
    metric_summary
    sample_count
    confidence
    freshness
    baseline_reference
    model_version
```

## Performance Model Is Not Machine Learning Requirement

The term Performance Model does not require an ML model.

A valid implementation MAY be:

- rolling statistics
- histograms
- EWMA
- quantile summaries
- simple regression
- another bounded statistical representation

Runtime SHALL not require an external AI model.

## Model Versioning

Performance Model computation policy SHALL be versioned.

Changing aggregation/statistical semantics SHOULD create a new model version or
invalidate incompatible prior evidence.

## Sample Sufficiency

Runtime SHALL distinguish insufficient evidence from meaningful performance
evidence.

A small number of observations SHALL not automatically trigger promotion,
rollback, or re-tuning.

Policy MAY define:

```text
minimum samples
minimum observation duration
minimum workload coverage
minimum confidence
```

## Confidence

Performance Model MAY expose confidence or evidence quality.

Confidence SHALL not be fabricated when the model cannot estimate it.

It MAY instead expose structured states such as:

```text
insufficient
low
medium
high
```

## Warmup Samples

Warmup or first-use executions MAY have different performance characteristics.

Runtime SHOULD be able to classify or exclude warmup samples from steady-state
performance evidence where appropriate.

## Cold-Start Costs

Runtime SHALL distinguish cold-start costs from steady-state execution where
possible.

Examples include:

- first pipeline invocation
- cache population
- page faults
- initial graph capture
- first memory allocation
- Provider initialization

Cold-start evidence MAY be modeled separately.

## Queue Delay Versus Execution Time

Where measurable, Runtime SHOULD distinguish:

```text
queue/admission delay
Provider submission overhead
Kernel execution duration
end-to-end operation latency
```

Selection policy MAY use different metrics.

A Provider unable to expose exact native execution timing MAY still report
coarser Runtime-observed latency.

## Measurement Capability

Providers MAY expose timing capabilities.

Examples:

```text
host-observed timing
device event timing
hardware timestamp timing
Provider-reported aggregate timing
```

Timing method SHALL be identifiable in evidence.

## Measurement Overhead

Performance measurement SHALL have bounded overhead.

Runtime MAY sample only a subset of executions.

Measurement SHALL NOT materially degrade inference without explicit policy.

## Sampling Policy

Runtime SHOULD support an observation sampling policy.

A sampling policy MAY define:

```text
all executions
one in N
probabilistic bounded sampling
per-workload-bucket budget
adaptive sampling
```

Sampling strategy SHALL not affect correctness semantics.

## Adaptive Sampling

Runtime MAY increase sampling when:

- a new Kernel generation is promoted
- a workload bucket is new
- a suspected regression exists
- benchmark evidence becomes stale

It MAY reduce sampling after evidence stabilizes.

## Sampling Independence

Performance observation sampling SHALL not change which Kernel is selected for
an already-started invocation.

## Online Evidence

Production execution observations are Online Performance Evidence.

Offline autotuning/benchmark results are Offline Performance Evidence.

These two evidence classes SHALL remain distinguishable.

## Offline Evidence Baseline

Runtime MAY use offline benchmark evidence as an expected baseline.

Example:

```text
offline p50 = 30 us
online p50 = 31 us
```

may indicate healthy agreement.

## Benchmark Drift

Benchmark drift occurs when compatible online measurements materially diverge
from offline tuning/benchmark evidence.

Drift policy SHALL define thresholds.

Example:

```text
expected p50 = 30 us
observed p50 = 45 us
sustained difference > threshold
```

MAY trigger a drift signal.

## Drift Does Not Imply Incorrectness

Performance drift SHALL NOT automatically mark Kernel numerically incorrect.

Correctness state remains governed by qualification.

## Workload Drift

Runtime MAY detect that actual workload distribution differs materially from
the workload profile used for tuning.

Examples:

```text
expected batch 1..4
actual batch mostly 16..32
```

or:

```text
expected decode seq <= 2048
actual decode mostly 8192
```

## Workload Drift Action

Workload drift MAY trigger:

- selection re-evaluation
- creation/use of new workload bucket
- bounded re-tuning request
- warning
- no action

according to policy.

## Kernel Performance Regression

Runtime MAY identify performance regression when a Kernel generation performs
materially worse than:

- its own previous compatible baseline
- previous known-good Kernel
- compatible offline benchmark
- policy-defined performance SLO

Regression SHALL require sufficient evidence.

## Regression Thresholds

Thresholds SHALL be explicit.

Examples MAY include:

```text
relative latency increase
absolute latency increase
throughput reduction
p99 increase
workspace increase
timeout-rate increase
```

## Regression Confirmation

Policy SHOULD prevent reacting to one isolated outlier.

A regression signal SHOULD require:

- enough samples
- compatible context
- minimum duration
- confidence threshold
- hysteresis

as appropriate.

## Outlier Handling

Performance Model SHOULD define outlier handling.

Outliers SHALL not be silently discarded without policy.

The policy MAY:

- retain them in tail metrics
- exclude from mean calculation
- mark them separately
- classify external interference

## External Interference

Runtime MAY distinguish degraded performance caused by broader Device pressure
from Kernel-specific regression where evidence permits.

Example:

```text
all GPU Kernels slow simultaneously
```

may indicate Device pressure rather than one bad Kernel.

## Device Pressure Correlation

Performance Model MAY correlate latency/throughput with:

- Device utilization
- queue depth
- memory pressure
- thermal/power state where exposed
- Provider pressure

These signals remain observational.

## Selection Feedback

Kernel Selection Policy MAY consume Performance Model evidence.

Online evidence MAY:

- refine ranking
- reduce confidence in stale offline benchmark
- retain current Kernel
- favor another already eligible Kernel

Online evidence SHALL not bypass hard eligibility filters.

## Online Evidence Precedence

Runtime policy SHALL define how online evidence and offline benchmark evidence
interact.

Possible policies include:

```text
offline-only
online-preferred-after-sufficient-samples
hybrid
pinned-offline
```

## Online Preferred

Where policy prefers online evidence, sufficient compatible real-world samples
MAY outrank older offline benchmark estimates.

This preference concerns performance ranking only.

## Tuning Staleness

An existing KernelAutotuningRecord MAY become stale when:

- performance drifts materially
- workload distribution shifts
- candidate set changes
- Provider changes
- driver/runtime changes
- Device behavior changes
- policy changes

## Retuning Request

Performance feedback MAY produce a KernelRetuningRequest.

Conceptually:

```text
KernelRetuningRequest
    reason
    workload_bucket
    candidate context
    evidence summary
    urgency
```

A Retuning Request SHALL NOT itself start arbitrary source generation.

## Retuning Scope

Retuning SHALL remain bounded by the Runtime Autotuning contract.

It MAY evaluate:

- existing candidates
- existing Specialization Templates
- authorized bounded Specialization Instances

It SHALL NOT generate arbitrary new Kernel source.

## Optimization Escalation

If bounded re-tuning cannot find a satisfactory result, Runtime MAY emit an
Optimization Recommendation/Request for the external Optimization Plane.

Example:

```text
bounded variants exhausted
performance still below policy target
    ->
external optimization requested
```

Runtime SHALL not execute the external optimization itself as part of inference.

## Adaptive Feedback Escalation Boundary

The path SHALL be:

```text
performance issue
    ->
bounded retuning
    ->
if unresolved:
external optimization signal
```

not:

```text
performance issue
    ->
Runtime invokes AI generator
```

## Re-Tuning Admission

Re-tuning SHALL obey normal autotuning resource/admission policy.

High inference pressure MAY postpone optional re-tuning.

## Re-Tuning Hot-Path Prohibition

A regression detected during decode SHALL NOT synchronously pause the same
decode to benchmark alternatives.

Runtime MAY:

- retain current Kernel temporarily
- switch to existing known-good candidate
- request background re-tuning
- trigger rollback policy

according to policy.

## Demotion Signal

Performance Model MAY generate a candidate demotion signal.

Demotion means the Kernel should no longer be preferred for new work.

Demotion SHALL remain subject to selection/promotion state machinery.

## Rollback Signal

Severe confirmed regression MAY generate a rollback signal.

Actual rollback remains governed by the existing rollback policy.

## Known-Good Baseline

Runtime SHOULD retain enough performance metadata about a previous known-good
Kernel to compare replacements during a configured observation window.

## Post-Promotion Observation Window

A newly promoted Kernel MAY enter a heightened observation period.

During this period Runtime MAY:

- sample more frequently
- compare against previous generation
- apply stricter regression thresholds
- retain rollback candidate

## Stabilized Kernel

After sufficient successful observations, a Kernel generation MAY be considered
performance-stable for the current workload/context.

This is not a correctness/trust status.

## Performance Health State

Runtime MAY expose performance health states such as:

```text
unknown
warming
healthy
degraded
regressed
stale
```

These states SHALL not replace Provider health or Kernel qualification.

## Failure Rate Evidence

Runtime MAY include Kernel execution failures in adaptive evidence.

A rising failure rate MAY trigger demotion/rollback investigation.

Execution failure evidence SHALL preserve structured error categories.

## Timeout Evidence

Kernel timeouts MAY participate in performance health.

Timeout rate SHALL not be hidden by good average latency.

## Memory Evidence

Performance feedback MAY include actual workspace or temporary-memory behavior
where measurable.

If actual memory requirements materially violate advertised metadata, this MAY
trigger:

- Kernel health degradation
- qualification/contract investigation
- candidate exclusion
- rollback

according to policy.

## Contract Violation Versus Performance Regression

Runtime SHALL distinguish:

```text
performance regression
```

from:

```text
Kernel contract violation
```

Contract violation MAY be stronger and may require immediate exclusion or
revocation workflow.

## Selection Stability

Adaptive feedback SHALL respect selection hysteresis.

Minor statistical variation SHALL not cause Kernel flapping.

## Re-Tuning Hysteresis

Repeated re-tuning requests SHALL be rate-limited/cooldown-controlled.

Runtime SHALL not continuously tune due to small oscillations.

## Feedback Cooldown

Policy MAY define:

```text
minimum interval between re-tuning
minimum stable observation duration
minimum regression duration
```

## Performance Model Aging

Old observations SHOULD decay or expire.

Very old performance data SHALL not dominate current evidence indefinitely.

Possible mechanisms include:

- time windows
- generation windows
- weighted decay
- explicit expiration

## Artifact And Generation Binding

Performance evidence SHALL be bound to the exact Kernel Artifact and relevant
Prepared Kernel generation or equivalent immutable identity.

A replacement Kernel SHALL not inherit performance evidence automatically.

## Specialization Binding

Performance evidence for one Specialization Instance SHALL not automatically
apply to another.

## Cross-Device Evidence

Performance evidence SHALL not automatically transfer across incompatible
Devices.

Policy MAY allow reuse across sufficiently equivalent hardware compatibility
classes where explicitly defined.

## Cross-Provider Evidence

Performance evidence from one Provider SHALL not automatically rank another
Provider implementation.

## Model Instance Interaction

A Model Instance MAY:

- consume adaptive performance evidence
- use dynamic selection policy
- remain pinned/reproducible and ignore adaptive changes

Policy SHALL be explicit.

## Reproducible Mode

Reproducible mode SHOULD disable performance-driven Kernel changes unless
explicitly permitted.

Observations MAY still be collected if policy allows, but SHALL not change the
pinned Kernel selection.

## Session Boundary

Inference Session SHALL not own the Performance Model.

Performance Model is Runtime/Model Instance policy state.

A Session SHALL not be able to falsify performance observations.

## Generation Boundary

Generation MAY produce workload context used for bucketing such as:

- prefill/decode
- batch size
- sequence length

Generation SHALL not provide raw user content to the Performance Model.

## Continuous Batching

Adaptive feedback MAY observe batch-level workload characteristics.

Metrics SHALL be attributable carefully where multiple sequences share one
Kernel invocation.

Runtime SHALL not fabricate per-session latency from batch-level kernel timing
without a defined attribution model.

## Privacy

Performance observations SHALL avoid raw inference content.

They SHOULD operate on:

- timing
- sizes
- counts
- shapes
- stable Kernel identities
- coarse workload buckets
- Device/Provider metadata

## Sensitive Correlation

Performance telemetry SHALL avoid unnecessary user/session identifiers.

If correlation is needed operationally, identifiers SHOULD be ephemeral or
redacted according to observability policy.

## Export

Aggregated Performance Model data MAY be exported to an external Optimization
Plane.

Export SHALL follow privacy/security policy.

It SHALL NOT include raw prompts, model weights, KV data, or secrets by default.

## Optimization Plane Feedback

External Optimization Plane MAY consume summaries such as:

```text
Kernel A regressed for decode seq 4k..8k on sm90
batch 8..16 dominates current workload
p99 exceeds target by 18%
```

This enables future generated candidates without coupling AI generation to
Runtime.

## Adaptive Feedback Authority

External optimization recommendations generated from exported performance data
remain non-authoritative.

Any returned Kernel Artifact follows normal ingestion, trust, qualification,
selection, and promotion pipeline.

## Error Model

Structured errors SHOULD include:

```text
kernel-performance-observation-invalid
kernel-performance-context-invalid
kernel-performance-bucket-invalid
kernel-performance-bucket-policy-unsupported
kernel-performance-model-unavailable
kernel-performance-model-insufficient-samples
kernel-performance-model-stale
kernel-performance-metric-unavailable
kernel-performance-measurement-failed
kernel-performance-measurement-overhead-exceeded

kernel-performance-drift-detected
kernel-performance-workload-drift-detected
kernel-performance-regression-detected
kernel-performance-regression-unconfirmed
kernel-performance-contract-anomaly
kernel-performance-timeout-regression
kernel-performance-memory-anomaly

kernel-performance-retuning-rate-limited
kernel-performance-retuning-denied
kernel-performance-retuning-request-failed
kernel-performance-optimization-escalation-required

kernel-performance-feedback-disabled
kernel-performance-feedback-policy-invalid
internal-kernel-performance-error
```

## Observability

Adaptive feedback observability MAY include:

```text
performance-observation-sampled
performance-model-updated
performance-model-insufficient
benchmark-drift-detected
workload-drift-detected
kernel-regression-suspected
kernel-regression-confirmed
kernel-performance-degraded
kernel-performance-recovered
retuning-requested
retuning-rate-limited
selection-reevaluation-requested
rollback-recommended
optimization-escalation-requested
```

Observability MAY include:

- KernelId
- artifact digest
- specialization identity
- workload bucket
- sample count
- aggregated latency/throughput
- benchmark delta
- confidence/evidence quality
- policy version
- reason

Observability SHALL NOT expose by default:

- raw prompts
- raw tensor values
- model weights
- KV contents
- native handles
- secrets
- credentials

## Conformance

Conformance SHALL validate:

- performance feedback cannot grant trust
- performance feedback cannot grant qualification
- performance feedback cannot make incompatible candidate eligible
- observations are workload/context bound
- evidence does not leak across incompatible artifacts/specializations
- sample insufficiency prevents premature adaptive action
- warmup samples can be distinguished
- online/offline evidence remain distinct
- benchmark drift can be detected
- workload drift can be detected
- re-tuning remains bounded
- decode does not synchronously run re-tuning
- unresolved bounded tuning escalates externally rather than invoking AI
- minor metric noise does not cause flapping
- reproducible mode prevents adaptive Kernel switching
- observation retention is bounded
- active known-good Kernel survives feedback-system failure
- exported telemetry is redacted

## Non-Goals

This change does not:

- train a neural performance model
- require machine learning
- generate Kernel source
- execute AI agents
- replace qualification
- replace trust
- change Operator semantics
- benchmark every production request
- require online adaptation
- make production traffic an unrestricted experiment
- define a centralized fleet telemetry platform
- define Tachyon-specific feedback RPC
- guarantee identical performance across hardware

## Impact

Magnetar gains a closed optimization feedback loop while keeping generative
optimization external:

```text
offline optimization
        |
        v
initial Kernel
        |
        v
Runtime execution
        |
        v
aggregated performance feedback
        |
        +-> selection refinement
        |
        +-> bounded re-tuning
        |
        +-> rollback/demotion signal
        |
        +-> external optimization request
```

This makes Kernel selection adaptive to real workloads without compromising the
inference Runtime boundary.