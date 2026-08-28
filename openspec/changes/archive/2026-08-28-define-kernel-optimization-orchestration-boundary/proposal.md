# Define Kernel Optimization Orchestration Boundary

## Why

Magnetar now defines:

```text
Kernel Artifact lifecycle
Provider Kernel Compilation
Generated Kernel Qualification
Kernel Cache and Hot Swap
Kernel Optimization and Selection Policy
```

These contracts describe how a generated Kernel becomes executable and how
Runtime chooses among eligible Kernel implementations.

They intentionally do not define who creates optimization campaigns.

Kernel generation and optimization may involve:

- AI agents
- CI systems
- developer tooling
- hardware-vendor systems
- optimization farms
- external services
- Tachyon-managed infrastructure
- future Magnetar tooling

These systems may require broader authority than an inference Runtime:

- network access
- source repositories
- compiler toolchains
- process execution
- temporary storage
- benchmark machines
- hardware reservation
- credentials
- artifact registries

Granting these authorities to Magnetar Runtime would violate the architectural
boundary:

```text
Magnetar = inference runtime
```

This change therefore defines an Optimization Plane that remains separate from
the Inference Plane.

## What Changes

This change defines:

- Optimization Plane versus Inference Plane
- Optimization Campaign
- Optimization Request
- Optimization Workload Profile
- optimization triggers
- candidate generation
- external generator boundary
- optimization worker capabilities
- compilation/qualification/benchmark composition
- campaign lifecycle
- campaign resource budgets
- cancellation and deadlines
- evidence bundles
- optimization recommendations
- artifact transfer
- privacy boundaries
- security/authority boundaries
- Runtime ingestion boundary
- promotion authority
- neutral external-orchestrator integration
- observability
- conformance

## Core Separation

Magnetar SHALL distinguish:

```text
Optimization Plane
Inference Plane
```

The Optimization Plane MAY generate and evaluate Kernel candidates.

The Inference Plane SHALL execute only Kernel implementations accepted through
Runtime policy.

## Optimization Plane

The Optimization Plane MAY own:

```text
AI kernel generators
source code generation
compiler invocation
optimization search
candidate mutation
benchmark orchestration
qualification orchestration
test generation
hardware reservation
artifact publishing
campaign history
external credentials
network access
repository access
```

These authorities SHALL NOT automatically exist inside Magnetar Runtime.

## Inference Plane

Magnetar Runtime SHALL own:

```text
artifact validation
trust/integrity evaluation
current qualification validation
Kernel eligibility
Kernel selection
promotion authorization
Prepared Kernel lifetime
Kernel Dispatch
Provider execution
Resource Affinity
Memory feasibility
inference observability
```

The Optimization Plane SHALL NOT bypass these responsibilities.

## Optimization Campaign

An Optimization Campaign represents a bounded attempt to discover or improve
Kernel implementations for one or more portable Operator requirements.

Conceptually:

```text
OptimizationCampaign
    id
    trigger
    workload_profile
    objectives
    constraints
    target_capabilities
    budgets
    policy
```

A campaign SHALL be separate from an inference request.

## Campaign Identity

Each campaign SHALL have stable identity.

Campaign identity SHOULD be usable to correlate:

- generated candidates
- compiler jobs
- qualification jobs
- benchmark results
- artifact digests
- recommendations
- promotion outcomes

It SHALL NOT encode native pointers, process handles, or secrets.

## Campaign Lifecycle

Suggested states are:

```text
planned
queued
running
generating
compiling
qualifying
benchmarking
evaluating
completed
cancelled
timed-out
failed
```

An implementation MAY collapse individual internal stages but SHALL preserve
enough evidence to explain campaign outcome.

## Optimization Trigger

An Optimization Campaign MAY be triggered by events such as:

```text
manual request
CI pipeline
new hardware target
new Provider version
new compiler version
new driver/runtime compatibility class
new Operator version
new Model workload profile
performance regression
benchmark drift
qualification suite upgrade
scheduled optimization
cache warming
```

A normal token decode SHALL NOT trigger an Optimization Campaign synchronously.

## No Hot-Path Campaign Start

The normal inference hot path SHALL NOT:

- start an AI agent
- start a kernel search loop
- invoke an optimization service
- perform candidate mutation
- start benchmarking campaigns
- wait for a remote optimization farm

If no suitable Kernel exists, Runtime SHALL use structured fallback,
admission, or failure policy.

## Optimization Workload Profile

Campaigns SHOULD use an explicit workload profile.

A workload profile MAY contain:

- Operator or fused Operator semantics
- Operator semantic version
- target Provider class
- target Device architecture
- dtype
- layout
- shape envelope
- batch envelope
- sequence envelope
- generation phase
- KV cache mode
- quantization profile
- determinism requirement
- precision requirement
- optimization objective
- memory limits
- workspace limits

## Workload Profile Is Not Raw User Data

Optimization Workload Profile SHOULD describe execution characteristics rather
than contain raw inference content.

By default it SHALL NOT contain:

- raw prompts
- conversation contents
- raw user documents
- secrets
- credentials
- raw model weights
- raw KV cache contents

## Representative Benchmark Inputs

Qualification and benchmarking MAY require tensor values.

Such inputs SHOULD come from:

- synthetic fixtures
- deterministic generated fixtures
- explicitly authorized benchmark datasets
- sanitized/reduced production-derived datasets where policy allows

Raw production inference inputs SHALL NOT be automatically exported to the
Optimization Plane.

## Workload Aggregation

Runtime or surrounding tooling MAY produce aggregate workload metadata for
optimization.

Examples:

```text
shape histogram
batch histogram
sequence-length histogram
Operator frequency
dtype distribution
layout distribution
latency distribution
Provider/Device pressure summary
```

Aggregation SHALL follow privacy and observability policy.

## External Generator Boundary

Kernel generators SHALL be considered external producers from the perspective
of Magnetar Runtime.

Generators MAY include:

```text
KernelEvolve-like systems
coding agents
LLM-based agents
human engineers
vendor optimization systems
compiler autotuners
CI scripts
```

Runtime SHALL not depend on a specific generator protocol.

## Generator Output

Generators SHOULD produce:

```text
KernelSourceArtifact
```

or a compatible precompiled Kernel Artifact.

Generator output SHALL NOT directly produce a trusted active Prepared Kernel.

## Generator Authority

A generator SHALL NOT receive ambient access to:

- Runtime tensor memory
- active KV cache
- Provider native handles
- Device native handles
- PreparedKernelId mappings
- inference secrets
- Runtime process memory

unless a future explicit privileged debugging contract defines otherwise.

## Candidate Generation

A campaign MAY create multiple candidate Kernel Artifacts.

Candidates SHALL have distinct artifact identities.

Example:

```text
candidate-001
candidate-002
candidate-003
...
```

Human-readable candidate numbering SHALL NOT replace digest-based artifact
identity.

## Optimization Search Strategy

The orchestration contract SHALL remain agnostic to search strategy.

Search MAY use:

- mutation
- evolutionary search
- LLM generation
- hill climbing
- Bayesian optimization
- exhaustive specialization
- vendor autotuning
- human iteration

Magnetar SHALL not require one algorithm.

## Optimization Worker

An Optimization Worker represents an execution environment able to perform
one or more campaign stages.

Workers MAY support:

```text
compilation
qualification
benchmarking
artifact validation
Provider preparation
```

Worker capability SHALL be explicit.

## Worker Capability Profile

A worker capability profile MAY include:

- Provider implementations
- Device architecture
- Device features
- compiler toolchains
- accepted Kernel Source Formats
- emitted Compiled Kernel Formats
- qualification profiles
- benchmark profiles
- available memory
- concurrency limits
- isolation model

It SHALL NOT expose native handles outside the worker boundary.

## Worker Selection

Optimization orchestrator MAY choose workers compatible with campaign target.

Worker selection SHALL NOT imply Runtime Provider/Device selection for
production inference.

## Compilation Composition

Optimization orchestration MAY invoke the Provider Kernel Compilation
Capability defined by the compilation contract.

The orchestration layer SHALL not redefine compiler semantics.

## Qualification Composition

Optimization orchestration SHALL use the existing Generated Kernel
Qualification contract.

It SHALL NOT treat compilation success as qualification.

## Benchmark Composition

Optimization orchestration MAY execute benchmark profiles after required
correctness gates pass.

A candidate failing mandatory correctness SHALL not be promoted merely because
it benchmarks well.

## Parallel Candidate Evaluation

A campaign MAY compile, qualify, and benchmark multiple candidates in parallel.

Parallelism SHALL obey:

- campaign budgets
- worker limits
- Provider limits
- hardware availability
- security policy

## Campaign Budgets

Campaigns SHOULD have explicit resource budgets.

Budgets MAY include:

```text
maximum candidates
maximum compiler jobs
maximum qualification jobs
maximum benchmark runs
wall-clock deadline
CPU time
GPU time
memory
temporary storage
network budget
cost budget
```

## Campaign Deadline

Campaign MAY have a deadline independent of inference deadlines.

Expiration SHALL stop or reject additional campaign work according to policy.

Campaign timeout SHALL NOT affect currently active production Kernel
automatically.

## Campaign Cancellation

Campaigns SHOULD support cancellation.

Cancellation SHALL prevent new campaign work and SHOULD cancel interruptible
jobs.

Previously active production Kernel SHALL remain unaffected.

## Candidate Failure Isolation

Failure of one candidate SHALL NOT necessarily fail the entire campaign.

Policy MAY continue evaluating remaining candidates.

Examples of isolated candidate failures include:

```text
compilation failure
qualification mismatch
benchmark crash
unsupported specialization
resource limit failure
```

## Campaign Failure

A campaign SHALL fail only according to campaign policy.

Examples MAY include:

- no candidate qualified
- required worker unavailable
- budget exhausted
- orchestration infrastructure failed
- campaign deadline expired
- mandatory security policy denied

## Evidence Bundle

Campaign output SHALL include evidence sufficient to evaluate its
recommendations.

An Optimization Evidence Bundle SHOULD reference:

- campaign identity
- source artifact digests
- compiled artifact digests
- compiler metadata
- qualification records
- benchmark records
- target context
- optimization policy/version
- workload profile
- candidate status
- trust/integrity status where known

## Evidence Immutability

Evidence associated with content-addressed artifacts SHOULD be immutable.

A corrected or rerun evaluation SHOULD create new evidence rather than silently
rewrite historical qualification results.

## Optimization Recommendation

Campaign MAY produce one or more Optimization Recommendations.

A recommendation MAY state:

```text
candidate X is recommended for profile latency
candidate Y is recommended for profile throughput
candidate Z should remain experimental
candidate A should be rejected
```

A recommendation SHALL NOT be authoritative execution policy.

## Recommendation Is Not Promotion

```text
recommended != promoted
```

Runtime SHALL independently validate:

- artifact integrity
- trust
- qualification freshness
- compatibility
- selection policy
- current Provider/Device readiness
- Resource Affinity
- memory feasibility

before promotion/execution.

## Recommendation Ranking

Optimization Plane MAY rank candidates using campaign benchmark evidence.

Runtime remains authoritative for production selection.

Campaign ranking MAY become input into Runtime policy but SHALL not bypass it.

## Artifact Transport

Optimization Plane and Runtime SHOULD exchange artifacts using stable artifact
references and digests.

Transport MAY use:

- local content-addressed store
- artifact registry
- object storage
- deployment package
- explicitly supplied bytes
- external Component/Artifact Source abstraction

This change does not prescribe one transport.

## No Pointer-Based Transport

Artifact exchange SHALL NOT use:

- raw pointers
- shared native Kernel handles
- Device handles
- Provider function pointers
- process-local PreparedKernelId mappings

as portable identity.

## External Orchestrator Neutrality

Magnetar SHALL not depend directly on a specific optimization orchestrator.

The orchestrator MAY be:

```text
CI
local developer tooling
dedicated optimization service
Tachyon-managed service
vendor infrastructure
future Magnetar optimization tooling
```

The artifact/evidence boundary SHALL remain neutral.

## Tachyon Boundary

Tachyon MAY orchestrate or distribute optimization work.

Magnetar SHALL NOT require a direct Tachyon dependency.

The boundary remains:

```text
Tachyon may distribute/orchestrate.
Magnetar validates and executes inference.
```

## CLI Boundary

`magnetar-cli` or future Magnetar tooling MAY expose optimization management
commands.

Possible future commands MAY include:

```text
magnetar kernel optimize
magnetar kernel qualify
magnetar kernel benchmark
magnetar kernel candidates
```

These commands SHALL belong to tooling/control-plane authority, not Runtime
Inference API authority.

## Runtime API Boundary

Runtime Inference API SHALL NOT expose arbitrary optimization-agent execution.

Normal inference API SHALL NOT accept:

```text
agent prompt
compiler command
kernel source injection
optimization service URL
benchmark script
shell command
repository credentials
```

## Runtime Artifact Ingestion

Runtime MAY expose an authorized artifact-ingestion/management boundary outside
normal inference requests.

Such ingestion SHALL still enforce:

- artifact validation
- trust/integrity
- qualification policy
- Provider compatibility
- Kernel selection policy

## Runtime Network Boundary

Magnetar Runtime SHALL NOT require network access to external optimization
services to execute already prepared inference.

A production deployment SHALL remain capable of running with compatible local
artifacts without contacting a generator.

## Offline Inference

Generated Kernel support SHALL preserve offline inference.

If required artifacts are already available and compatible:

```text
network unavailable
```

SHALL NOT by itself prevent execution.

## Optimization Service Credentials

Credentials used to access optimization systems, repositories, artifact
registries, or model/tool services SHALL remain outside ordinary Runtime
Inference API state.

Such credentials MAY belong to:

- CLI tooling
- CI
- external orchestrator
- deployment system
- secret-management integration

Runtime SHALL not receive ambient secret authority.

## Provider Boundary

Optimization worker MAY invoke Provider compilation/preparation APIs in a
controlled offline context.

This SHALL NOT grant the orchestrator access to Provider native handles.

## Production Provider Isolation

Production inference Provider instance SHOULD be separable from optimization
worker Provider instances.

Optimization experiments SHALL NOT require mutation of live production
Provider state.

## Production Device Isolation

Optimization benchmarks SHOULD NOT silently execute on a production inference
Device if policy prohibits interference.

Shared hardware usage SHALL be explicit and admission-controlled.

## Memory Boundary

Optimization worker memory and live Runtime inference memory SHALL remain
separate authority domains.

Optimization campaigns SHALL NOT receive raw references to active Runtime
Tensor Resources.

## Promotion Request

Optimization Plane MAY submit a candidate for production consideration.

Conceptually:

```text
PromotionCandidate
    KernelId
    artifact digest
    qualification evidence
    benchmark evidence
    requested optimization profiles
```

This is a request, not a command.

## Promotion Authority

Only Runtime/deployment policy SHALL authorize production promotion.

Optimization orchestrator SHALL NOT call a bypass equivalent to:

```text
force_active_kernel(candidate)
```

without normal eligibility validation.

## Runtime Revalidation

Runtime SHALL revalidate production-relevant state before promotion.

This MAY include:

- current trust
- current revocation status
- qualification compatibility
- benchmark compatibility
- Provider readiness
- Device availability
- memory feasibility
- current selection policy

Evidence that was valid during campaign execution MAY become stale.

## Canary Boundary

Optimization Plane MAY recommend a canary.

Runtime promotion policy controls whether local canary/exploration is
permitted.

A recommendation SHALL NOT independently route production traffic.

## Rollback Authority

Optimization Plane MAY report a regression or recommend rollback.

Runtime/deployment policy remains authoritative for actual rollback.

## Campaign Reproducibility

Campaign evidence SHOULD capture enough metadata to reproduce or explain a
candidate result where feasible.

Metadata SHOULD include:

- campaign policy version
- generator identity/version where known
- source artifact digest
- compiler fingerprint
- qualification suite
- benchmark profile
- worker hardware
- Provider version
- target architecture
- random seeds where applicable

## Generator Identity

Generator identity SHOULD be recorded as provenance.

Generator identity SHALL NOT imply trust.

## Agent Prompts

If an AI generator uses prompts or internal reasoning, Magnetar Runtime SHALL
not require those prompts for inference.

Optimization systems MAY retain generator metadata according to their own
policy.

Raw agent prompts SHALL not become mandatory Kernel Artifact metadata.

## Search History

Candidate search history MAY be retained by the Optimization Plane.

Runtime SHOULD only require artifact/evidence relevant to current production
decisions.

## Optimization Metadata Privacy

Optimization evidence SHALL avoid embedding raw user data.

Benchmark and workload evidence SHOULD prefer aggregated or synthetic data.

## Observability Separation

Optimization observability and inference observability SHALL remain
distinguishable.

Optimization events MAY include:

```text
campaign-started
candidate-generated
candidate-compilation-started
candidate-compilation-failed
candidate-qualified
candidate-rejected
candidate-benchmark-completed
recommendation-created
campaign-completed
```

Inference events remain focused on:

```text
Kernel selected
Kernel promoted
Kernel executed
Kernel rolled back
```

## Correlation

Optimization recommendation MAY carry correlation metadata linking it to:

- campaign
- artifact
- qualification
- benchmark

Runtime MAY preserve this correlation when promotion occurs.

## Redaction

Optimization observability SHALL redact by default:

- secrets
- credentials
- unrestricted source repository URLs containing credentials
- raw user prompts
- raw inference inputs
- raw model weights
- raw KV contents
- native handles
- process handles
- local sensitive paths

## Error Model

Structured errors SHOULD include:

```text
kernel-optimization-campaign-invalid
kernel-optimization-trigger-denied
kernel-optimization-budget-invalid
kernel-optimization-budget-exhausted
kernel-optimization-deadline-exceeded
kernel-optimization-cancelled
kernel-optimization-worker-unavailable
kernel-optimization-worker-incompatible
kernel-optimization-generator-unavailable
kernel-optimization-generator-failed
kernel-optimization-no-candidates
kernel-optimization-no-qualified-candidates
kernel-optimization-evidence-invalid
kernel-optimization-evidence-incomplete
kernel-optimization-recommendation-invalid
kernel-optimization-artifact-transfer-failed
kernel-optimization-policy-denied
kernel-optimization-production-boundary-violation
kernel-optimization-runtime-authority-violation
kernel-optimization-credential-boundary-violation
kernel-optimization-data-boundary-violation
kernel-optimization-hot-path-denied
internal-kernel-optimization-error
```

## Conformance

Conformance SHALL validate:

- Optimization Plane is separate from Runtime inference
- normal decode cannot start optimization campaign
- generator cannot directly execute production Kernel
- recommendation is not promotion
- Runtime revalidates candidate before promotion
- artifact exchange uses stable identity rather than native handles
- Runtime inference works without optimization-service network access
- optimization credentials do not become Runtime inference authority
- workload profiles exclude raw user data by default
- AI generator identity does not imply trust
- campaign failure does not affect active known-good Kernel
- candidate failure is isolated
- Provider native handles do not cross worker boundary
- Tachyon integration is optional and neutral
- CLI/tooling authority remains outside Runtime
- promotion still obeys Kernel Selection Policy
- optimization observability is redacted

## Non-Goals

This change does not:

- implement KernelEvolve
- implement an AI agent
- implement an optimization service
- implement distributed task scheduling
- implement a hardware reservation service
- define an artifact registry protocol
- define a network API for optimization
- define Tachyon-specific RPC
- make Runtime responsible for Git repositories
- make Runtime responsible for compiler credentials
- expose arbitrary shell/process authority to Runtime
- permit optimization during token decode
- permit generated candidates to bypass qualification
- permit recommendations to bypass promotion policy

## Impact

Magnetar gains a clean control-plane boundary for generative Kernel ecosystems:

```text
Optimization systems
     propose + evaluate
             |
             v
       artifacts/evidence
             |
             v
      Magnetar Runtime
             |
       validate + decide
             |
             v
          execute
```

This preserves Magnetar as an inference Runtime while allowing arbitrarily
advanced external optimization systems to improve the Kernel supply available
to it.