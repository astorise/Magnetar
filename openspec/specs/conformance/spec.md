# conformance Specification

## Purpose
This specification defines conformance evidence for the reference CPU provider, kernel baseline, local E2E path, release gates, and compatibility reporting.
## Requirements
### Requirement: Reference CPU Conformance Baseline

Reference CPU Provider SHALL provide or participate in conformance baselines for
supported Operators.

#### Scenario: Matmul conformance

Given Reference CPU matmul is implemented

When conformance runs

Then its output is validated against Operator semantics and tolerance profile.

---

### Requirement: Reference CPU Fixtures Avoid GPU Dependency

Reference CPU conformance fixtures SHALL not require external GPU hardware.

#### Scenario: CPU-only environment

Given tests run on CPU-only machine

When Reference CPU conformance executes

Then supported fixtures can run without GPU.

---

### Requirement: Reference CPU Can Compare Optimized Kernels

Any comparison of Reference CPU outputs against optimized Provider Kernels SHALL respect the declared tolerance profile for the Operator under test.
Reference CPU outputs MAY be used for such comparisons.

#### Scenario: CUDA comparison

Given CUDA matmul Kernel exists

When conformance compares outputs

Then Reference CPU output may be used as baseline if policy allows it.

### Requirement: First Scope Conformance Suite

Conformance SHALL include fixtures for each required-now Operator.

#### Scenario: Required operator fixture

Given `softmax` is required-now

When conformance suite runs

Then softmax fixtures are included.

---

### Requirement: First Scope Conformance Is CPU-Compatible

First scope conformance SHALL be runnable without external GPU hardware.

#### Scenario: CPU-only conformance

Given only Reference CPU Provider is available

When first scope conformance runs

Then supported required-now fixtures can execute.

---

### Requirement: First Scope Conformance Reports Placeholders

Placeholder Operators SHALL be reported as pending or unsupported rather than
passing silently.

#### Scenario: Placeholder conformance

Given `paged-attention` is placeholder

When conformance report is generated

Then it is reported as placeholder, pending, or unsupported.

### Requirement: Tensor Contract Conformance

Conformance SHALL validate Tensor Descriptor, Tensor Resource, Layout, DType, aliasing, views, Resource Affinity, and metadata safety behavior.

#### Scenario: Raw pointer exposure

Given Tensor metadata is returned during conformance

When result is inspected

Then no raw pointer or native handle is exposed.

---

### Requirement: Reference CPU Tensor Conformance

Reference CPU conformance SHALL validate host contiguous Tensor Resource support for required-now operators.

#### Scenario: CPU tensor conformance

Given host contiguous f32 tensors

When Reference CPU matmul conformance runs

Then tensor metadata and output readiness are validated.

---

### Requirement: Qwen Baseline Conformance

Conformance SHALL include Qwen baseline fixtures for config validation, tensor
inventory, graph production, operator scope, tokenizer compatibility, KV cache
metadata, adapter metadata, quantization rejection, authority, and handle
safety.

#### Scenario: Qwen conformance

Given Qwen Component claims baseline support

When conformance runs

Then it must pass Qwen baseline fixtures.

---

### Requirement: Qwen Baseline CPU Smoke Conformance

Conformance SHALL define a CPU smoke path requirement for a minimal Qwen-like
graph, which SHOULD run where all required Reference CPU kernels exist.

#### Scenario: CPU smoke graph

Given minimal Qwen-like fixture graph

When Reference CPU executes it

Then conformance validates graph planning, dispatch, and output metadata.

---

### Requirement: CLI Boundary Conformance

Conformance SHALL validate that `magnetar-cli` and Runtime preserve the
inference boundary.

#### Scenario: File access boundary

Given CLI reads file content for prompt

When Runtime receives request

Then Runtime has no filesystem authority.

---

### Requirement: Runtime Does Not Execute CLI-Owned Capabilities

Conformance SHALL validate Runtime does not execute tools, shell, Git, network,
or workspace operations.

#### Scenario: Generated shell text

Given model output contains shell command text

When Runtime emits output

Then no process execution occurs.

---

### Requirement: CLI Preserves Runtime Structured Errors

Conformance SHALL validate CLI preserves Runtime structured error categories
when displaying or wrapping errors.

#### Scenario: Runtime model loading error

Given Runtime returns model-loading-failed

When CLI displays failure

Then structured category is preserved.

---

### Requirement: Local Inference Conformance Suite

Conformance SHALL include a local inference suite that validates the full
Runtime inference path.

#### Scenario: Run local suite

Given conformance is executed

When local inference suite runs

Then the suite validates complete Runtime inference behavior.

---

### Requirement: E2E Conformance Uses Normal Runtime Contracts

E2E conformance SHALL use normal Runtime contracts and SHALL NOT use hidden
shortcuts.

#### Scenario: Shortcut detected

Given test path bypasses Model Loading

When conformance validates the path

Then the suite fails.

---

### Requirement: E2E Conformance Report

Conformance SHALL include E2E report output in machine-readable form.

#### Scenario: Report included

Given E2E suite completes

When conformance results are collected

Then E2E report is included with structured pass/fail/skipped status.

---

### Requirement: E2E Conformance Closes Baseline

E2E local inference conformance SHALL be the closing gate for the Runtime
baseline implementation.

#### Scenario: Baseline completion

Given implementation claims first baseline complete

When conformance runs

Then E2E local inference conformance must pass.

---

### Requirement: Conformance Runs Without GPU

Baseline conformance SHALL run without GPU hardware.

#### Scenario: CPU-only CI

Given CI has CPU only

When baseline conformance runs

Then required conformance suites can execute.

---

### Requirement: Conformance Detects Shortcuts

Conformance SHALL detect shortcuts that bypass Runtime contracts.

#### Scenario: Memory bypass

Given Provider writes output without Memory Manager tracking

When conformance validates output metadata

Then conformance fails.

---

### Requirement: Post-Baseline Provider Conformance

Conformance SHALL support Provider-specific profiles for optimized and
hardware-specific Providers.

#### Scenario: CUDA conformance

Given CUDA Provider is available

When CUDA conformance profile runs

Then it validates Provider, Kernel, Tensor, Memory, Operator, and observability
contracts.

---

### Requirement: Reference Comparison

Any comparison of optimized Provider output against Reference CPU fixtures SHALL use a declared tolerance profile.
Optimized Provider conformance MAY compare outputs against Reference CPU
fixtures within declared tolerance.

#### Scenario: Optimized matmul comparison

Given optimized matmul output is produced

When compared to Reference CPU output

Then difference must be within tolerance.

---

### Requirement: Benchmark Separation

Benchmarks SHALL be reported separately from correctness conformance.

#### Scenario: Benchmark fast but wrong

Given benchmark passes performance target

But correctness conformance fails

Then Provider is not accepted as conformant.

### Requirement: Model Format Conformance

Conformance SHALL include fixtures for supported model formats.

#### Scenario: safetensors conformance

Given safetensors support is enabled

When conformance runs

Then valid and invalid safetensors fixtures are validated.

---

### Requirement: Model Format Conformance Uses Normalized Artifacts

Format conformance SHALL validate that parsed files normalize into Model
Artifact, Tokenizer Artifact, or Adapter Artifact contracts.

#### Scenario: tokenizer.json conformance

Given tokenizer.json fixture is parsed

When conformance validates it

Then normalized Tokenizer Artifact metadata is produced.

---

### Requirement: Format Conformance Checks Redaction

Format conformance SHALL validate redaction of raw weights, tokenizer data,
file contents, handles, pointers, and secrets.

#### Scenario: Parser error

Given format parser fails

When diagnostics are emitted

Then raw file contents are not logged by default.

### Requirement: Source Cache Conformance

Conformance SHALL validate model source and cache behavior.

#### Scenario: Cache hit validation

Given cached artifact exists

When conformance loads it

Then trust, integrity, format, and loading validations still run.

---

### Requirement: Source Cache Boundary Conformance

Conformance SHALL validate Runtime does not gain arbitrary filesystem, network,
credential, or cache mutation authority.

#### Scenario: Arbitrary directory scan

Given Runtime is asked to scan arbitrary model directory

When conformance runs

Then request is denied.

---

### Requirement: Cache Residency Conformance

Conformance SHALL validate cache presence is distinct from memory residency.

#### Scenario: Cached but not loaded

Given artifact is cached but not loaded

When Memory Manager is inspected

Then no model tensors are resident.

### Requirement: Server API Conformance

Conformance SHALL validate Server API boundaries and Runtime API usage.

#### Scenario: Server conformance

Given server API implementation exists

When conformance runs

Then server requests use Runtime Inference API and preserve redaction.

---

### Requirement: Server Boundary Conformance

Conformance SHALL validate server does not read arbitrary files, execute tools,
execute shell/processes, execute Git, or download arbitrary models during
generation.

#### Scenario: Server filesystem violation

Given generation request asks server to read arbitrary file

When conformance runs

Then request is denied.

---

### Requirement: Server Streaming Conformance

Conformance SHALL validate server streaming preserves Runtime event ordering and
redaction.

#### Scenario: Stream order

Given Runtime emits ordered generation events

When server streams them

Then order and redaction are preserved.

### Requirement: Release Conformance Gates

Stable release SHALL pass required conformance gates.

#### Scenario: Provider conformance failure

Given Reference CPU conformance fails

When release is attempted

Then stable release is blocked.

---

### Requirement: Conformance Reports Included In Release

Release artifacts SHALL include conformance reports where applicable, or SHALL
explicitly mark them not applicable.

#### Scenario: Release artifact check

Given release candidate is prepared

When artifacts are inspected

Then conformance report is present or explicitly not applicable.

---

### Requirement: Conformance Suite Versions In Release

Release metadata SHALL include conformance suite versions.

#### Scenario: Report metadata

Given E2E report is generated

When release metadata is assembled

Then E2E suite version is included.

### Requirement: Release Security Conformance

Conformance SHALL include security release gates for dependency audit status,
license audit status, secret scanning, redaction, native handle exposure, trust
boundaries, and artifact integrity.

#### Scenario: Security conformance

Given release candidate is tested

When security conformance runs

Then release-blocking security checks pass.

---

### Requirement: Redaction Conformance Blocks Release

Redaction conformance failure SHALL block stable release.

#### Scenario: Raw KV cache leak

Given diagnostics expose raw KV cache content

When release conformance runs

Then stable release is blocked.

---

### Requirement: Trust Boundary Conformance Blocks Release

Trust boundary conformance failure SHALL block stable release.

#### Scenario: Cache trust bypass

Given cached artifact loads without trust validation

When release conformance runs

Then stable release is blocked.

### Requirement: Conformance Suite Release Mode

Conformance SHALL support release mode for `v0.1` gates.

#### Scenario: Release conformance run

Given release mode is enabled

When conformance executes

Then required baseline suites are run and optional out-of-scope suites are
skipped with reasons.

---

### Requirement: Conformance Report Redaction

Conformance reports SHALL be redacted by default.

#### Scenario: Failure report

Given conformance failure involves prompt input

When report is generated

Then raw prompt text is absent by default.

### Requirement: Conformance Reports Required For Cutover

Cutover SHALL require conformance reports for required baseline suites.

#### Scenario: Missing E2E report

Given E2E local conformance report is missing

When cutover validates artifacts

Then release is blocked.

---

### Requirement: Cutover Conformance Reports Are Redacted

Cutover conformance reports SHALL be redacted by default.

#### Scenario: Prompt in report

Given failure includes prompt text

When report is generated

Then raw prompt is absent by default.

---

### Requirement: Kernel Artifact Conformance

Conformance SHALL validate Kernel Source Artifact, Compiled Kernel Artifact,
and Prepared Kernel lifecycle separation.

#### Scenario: Lifecycle test

Given source artifact is compiled and prepared

When conformance runs

Then each stage has distinct identity and ownership.

---

### Requirement: Device Compilation Boundary Conformance

Conformance SHALL validate Device does not perform compilation.

#### Scenario: Device API audit

Given Device public contract is inspected

When conformance runs

Then arbitrary source compilation capability is absent.

---

### Requirement: Scheduler Compilation Boundary Conformance

Conformance SHALL validate Scheduler does not compile kernels.

#### Scenario: Scheduler API audit

Given Scheduler is inspected

When conformance runs

Then compiler ownership is absent.

---

### Requirement: Native Handle Conformance

Conformance SHALL validate native kernel handles remain Provider-private.

#### Scenario: Public API audit

Given Runtime, WIT, Registry, Device, and diagnostics are inspected

When conformance runs

Then no native kernel pointer or Provider executable handle is exposed.

---

### Requirement: Hot Path Compilation Conformance

Conformance SHALL validate normal decode path does not perform synchronous
kernel compilation.

#### Scenario: Unprepared kernel during decode

Given decode requires unprepared kernel

When conformance runs

Then structured readiness/admission error occurs rather than compilation.

---

### Requirement: Artifact Trust Conformance

Conformance SHALL validate artifact origin, format, AI provenance, local
location, and cache presence do not imply trust.

#### Scenario: AI-generated cached artifact

Given artifact is AI-generated and cached

When trust policy has not approved it

Then it remains untrusted.

---

### Requirement: Operator Semantics Conformance

Conformance SHALL validate Kernel Artifact semantics against portable Operator
semantics.

#### Scenario: Invalid generated MatMul

Given generated Kernel does not preserve MatMul semantics

When qualification/conformance evaluates it

Then it cannot become an eligible Kernel.

---

### Requirement: Prepared Generation Coexistence Conformance

Conformance SHALL validate multiple Prepared Kernel generations can coexist
without destroying in-flight kernel state.

#### Scenario: Replacement during execution

Given generation 1 has active invocation

When generation 2 becomes current

Then generation 1 remains valid until active references reach zero.

---

### Requirement: Provider Compilation Capability Conformance

Conformance SHALL validate optional Provider Kernel Compilation Capability.

#### Scenario: Provider without compiler

Given Provider has no compilation capability

When conformance core profile runs

Then Provider can still pass non-compilation conformance.

---

### Requirement: Source Format Negotiation Conformance

Conformance SHALL validate unsupported source formats are rejected before
compiler invocation.

#### Scenario: WGSL to CPU Provider

Given Provider does not accept WGSL

When compile is requested

Then structured unsupported format error is returned.

---

### Requirement: Compilation Job Lifecycle Conformance

Conformance SHALL validate compilation job state transitions.

#### Scenario: Successful async job

Given compilation is asynchronous

When polled

Then states progress legally to succeeded and cannot revert to compiling.

---

### Requirement: Compilation Cancellation Conformance

Conformance SHALL validate declared cancellation behavior.

#### Scenario: Cancelled compilation

Given Provider declares cooperative cancellation

When cancel is requested

Then job does not publish valid partial output.

---

### Requirement: Compilation Deadline Conformance

Conformance SHALL validate declared deadline behavior.

#### Scenario: Compiler exceeds deadline

Given deadline is enforceable

When compiler exceeds it

Then job ends timed-out without ready artifact.

---

### Requirement: Compilation Isolation Conformance

Conformance SHALL validate Runtime policy can reject insufficient isolation.

#### Scenario: Untrusted source

Given policy requires sandboxed compilation

When Provider advertises in-process compiler only

Then compilation is denied.

---

### Requirement: Compilation Trust Separation Conformance

Conformance SHALL validate compilation success does not imply trust or
qualification.

#### Scenario: Compilable untrusted source

Given untrusted source compiles

When output is created

Then output remains untrusted/unqualified according to policy.

---

### Requirement: Provider Kernel Compilation Hot Path Conformance

Conformance SHALL validate Kernel execution cannot silently invoke compiler.

#### Scenario: Missing Prepared Kernel

Given execution begins without PreparedKernel

When dispatch occurs

Then structured failure happens instead of compilation.

---

### Requirement: ABI Ownership Conformance

Conformance SHALL validate all compilation ABI buffers use declared ownership
and release paths.

#### Scenario: Result buffer

Given Provider allocates result buffer

When Runtime consumes result

Then required release callback is invoked exactly according to contract.

---

### Requirement: ABI Handle Opacity Conformance

Conformance SHALL validate job IDs and PreparedKernelIds are opaque.

#### Scenario: Numeric handle

Given handle is represented as integer

When public API/diagnostics inspect it

Then no native pointer semantics are exposed.

---

### Requirement: Compiler Failure Atomicity Conformance

Conformance SHALL validate compiler failure leaves existing known-good Kernel
state intact.

#### Scenario: Replacement compile fails

Given Kernel v1 is prepared

And compilation of v2 crashes

When job fails

Then v1 remains usable and v2 is not published.

---

### Requirement: Compilation Does Not Imply Qualification

Conformance SHALL prove successful compilation alone does not make candidate
eligible when qualification is required.

#### Scenario: Compiled-only candidate

Given artifact compiles but has no qualification

When production selection runs

Then candidate is rejected.

---

### Requirement: Qualification Does Not Imply Trust

Conformance SHALL prove qualified but untrusted Kernel is rejected where
production policy requires trust.

#### Scenario: Unknown source

Given candidate passes correctness tests but trust fails

When production policy requires both

Then candidate is ineligible.

---

### Requirement: Differential Mismatch Rejects Kernel

Conformance SHALL validate incorrect generated Kernel fails qualification.

#### Scenario: Broken MatMul

Given generated MatMul changes one result

When differential suite runs

Then candidate is rejected.

---

### Requirement: Explicit Tolerance Conformance

Conformance SHALL validate tolerance profile is explicit and enforced.

#### Scenario: Error outside tolerance

Given candidate exceeds declared tolerance

When compared

Then qualification fails.

---

### Requirement: Shape Envelope Conformance

Qualification SHALL not silently exceed tested compatibility envelope.

#### Scenario: Untested sequence length

Given qualification covers <=4096

When execution requests 8192

Then candidate is not considered qualified by that evidence.

---

### Requirement: Determinism Claim Conformance

Conformance SHALL reject Kernel that falsely advertises deterministic behavior.

#### Scenario: Repeated output differs

Given deterministic flag is true

When repeated runs differ unexpectedly

Then qualification fails.

---

### Requirement: Performance Cannot Override Correctness

Conformance SHALL validate faster incorrect Kernel never wins selection.

#### Scenario: Fastest candidate wrong

Given candidate A is incorrect but fastest and B is correct

When selection runs

Then B remains preferred/eligible.

---

### Requirement: Cache Hit Does Not Grant Eligibility

Conformance SHALL validate cached artifact is re-evaluated according to current
trust, qualification and compatibility policy.

#### Scenario: Revoked cache hit

Given revoked artifact is cached

When resolved

Then it is rejected.

---

### Requirement: Cache Corruption Fails Closed

Conformance SHALL validate corrupt cache entry is never prepared.

#### Scenario: Digest mismatch

Given cached bytes are modified

When read

Then integrity error occurs.

---

### Requirement: Atomic Promotion Conformance

Conformance SHALL validate dispatch never observes partially promoted Registry
state.

#### Scenario: Concurrent dispatch

Given promotion races with request

When Kernel resolves

Then request uses complete old or complete new generation.

---

### Requirement: In-Flight Generation Safety

Conformance SHALL validate old Prepared Kernel remains valid for in-flight work
after new generation promotion.

#### Scenario: Promotion during invocation

Given old generation is executing

When new one is promoted

Then old invocation completes safely.

---

### Requirement: Safe Retirement Conformance

Conformance SHALL validate retiring Kernel is destroyed only after quiescence.

#### Scenario: Active references

Given retiring Kernel has reference count greater than zero

When cleanup runs

Then Provider destruction does not occur.

---

### Requirement: Rollback Conformance

Conformance SHALL validate rollback can restore known-good eligible generation.

#### Scenario: New candidate fails after promotion

Given previous generation remains available

When rollback occurs

Then new dispatches use previous generation.

---

### Requirement: Revocation Conformance

Conformance SHALL validate revoked Kernel receives no new work.

#### Scenario: Active Kernel revoked

Given Kernel is revoked

When next dispatch occurs

Then another eligible Kernel is selected or structured failure is returned.

---

### Requirement: Provider Lifetime Independence Conformance

Conformance SHALL validate Kernel hot swap does not unload Provider.

#### Scenario: CUDA kernel replacement

Given new PreparedKernel generation is installed

When swap completes

Then CUDA Provider instance remains active.

---

### Requirement: Candidate Failure Atomicity

Conformance SHALL validate failure of candidate qualification, benchmark,
preparation or promotion leaves current active Kernel intact.

#### Scenario: Candidate preparation crashes

Given v1 active and v2 preparation fails

When failure completes

Then v1 remains active.

---

### Requirement: Eligibility Precedes Ranking

Conformance SHALL prove ineligible candidates are removed before performance
ranking.

#### Scenario: Fast untrusted Kernel

Given untrusted Kernel is fastest

When conformance runs

Then it is never selected.

---

### Requirement: Memory Feasibility Precedes Ranking

Conformance SHALL prove Memory Manager rejection cannot be overridden.

#### Scenario: Workspace too large

Given fastest Kernel is infeasible

When selection runs

Then feasible slower candidate wins or selection fails.

---

### Requirement: Affinity Precedes Ranking

Conformance SHALL prove Resource Affinity cannot be bypassed by performance.

#### Scenario: Cross-Provider candidate faster

Given movement is forbidden

When ranking runs

Then faster cross-Provider candidate is excluded.

---

### Requirement: Determinism Policy Conformance

Conformance SHALL prove deterministic profile excludes candidates failing
determinism.

#### Scenario: Nondeterministic fastest candidate

Given deterministic mode

When selection runs

Then candidate is not selected.

---

### Requirement: Stable Tie-Break Conformance

Conformance SHALL prove identical selection input yields identical tie result.

#### Scenario: Equal scores

Given candidates have identical score

When selection runs repeatedly

Then selected Kernel remains stable.

---

### Requirement: Benchmark Context Conformance

Conformance SHALL prove incompatible benchmark evidence is not authoritative.

#### Scenario: Different architecture

Given benchmark from sm90

When candidate targets different incompatible architecture

Then evidence is ignored/rejected.

---

### Requirement: Hysteresis Conformance

Conformance SHALL prove insignificant benefit does not force promotion.

#### Scenario: 0.1 percent improvement

Given threshold is higher

When candidate ranks slightly above active

Then active Kernel remains preferred.

---

### Requirement: Explicit Fallback Conformance

Conformance SHALL prove fallback only occurs according to policy.

#### Scenario: Fallback disabled

Given selected Provider unavailable

When policy says fail

Then Runtime fails instead of silently using CPU.

---

### Requirement: No Hidden Data Movement Conformance

Conformance SHALL prove cross-Provider selection respects explicit movement and
host staging rules.

#### Scenario: Host staging forbidden

Given CPU fallback requires staging

When policy forbids it

Then fallback fails.

---

### Requirement: Model Component Independence Conformance

Conformance SHALL prove Model Component cannot force Kernel implementation.

#### Scenario: Component attempts concrete selection

Given Component requests a specific Provider Kernel

When graph is validated

Then request is rejected/ignored according to portable contract.

---

### Requirement: User Preference Is Non-Authoritative

Conformance SHALL prove user/CLI preferences cannot force an ineligible Kernel.

#### Scenario: CLI requests latency

Given fastest candidate is revoked

When latency mode is requested

Then revoked candidate remains excluded.

---

### Requirement: Exploration Eligibility Conformance

Conformance SHALL prove exploration only includes already eligible candidates.

#### Scenario: Unqualified candidate

Given exploration enabled

When candidate lacks required qualification

Then it is not explored.

---

### Requirement: Provider Global Selection Boundary Conformance

Conformance SHALL prove Provider cannot decide cross-Provider selection.

#### Scenario: Provider advertises high score

Given Runtime policy rejects it

When selection runs

Then Provider cannot override decision.

---

### Requirement: Selection Explainability Conformance

Conformance SHALL validate selection reasoning is available and redacted.

#### Scenario: No eligible candidates

Given every candidate is excluded

When diagnostics are produced

Then structured exclusion reasons are available without native handles.

---

### Requirement: Optimization Plane Separation Conformance

Conformance SHALL validate optimization orchestration remains outside Runtime
inference authority.

#### Scenario: Runtime API audit

Given public Runtime Inference API is inspected

When conformance runs

Then arbitrary generator/optimization execution capability is absent.

---

### Requirement: No Hot-Path Optimization Conformance

Conformance SHALL prove token decode cannot synchronously launch optimization
campaign.

#### Scenario: Kernel unavailable

Given required optimized Kernel is missing

When decode runs

Then structured fallback/failure occurs rather than agent search.

---

### Requirement: Recommendation Does Not Promote Conformance

Conformance SHALL validate external recommendation cannot directly change
active Kernel.

#### Scenario: Recommended candidate

Given recommendation exists

When normal promotion validation has not run

Then active Registry state remains unchanged.

---

### Requirement: Runtime Revalidation Conformance

Conformance SHALL validate candidate is re-evaluated using current production
state.

#### Scenario: Qualification expired

Given campaign evidence was once valid

But qualification is now expired

When promotion is attempted

Then candidate is rejected.

---

### Requirement: Native Handle Boundary Conformance

Conformance SHALL prove Optimization Plane cannot transport Provider native
handles as artifacts.

#### Scenario: Worker-local PreparedKernelId

Given worker prepares candidate

When result is exported

Then production does not reuse worker-local native handle mapping.

---

### Requirement: Offline Inference Conformance

Conformance SHALL prove already prepared baseline inference can operate without
optimization-service connectivity.

#### Scenario: Network disabled

Given compatible Kernel artifacts are local

When inference runs

Then external optimizer is not contacted.

---

### Requirement: Credential Boundary Conformance

Conformance SHALL prove optimization credentials do not enter Runtime
Inference Session.

#### Scenario: Generator token configured

Given CLI/CI owns token

When Runtime session is created

Then token is absent.

---

### Requirement: Workload Privacy Conformance

Conformance SHALL validate default Optimization Workload Profile contains no raw
prompt/user content.

#### Scenario: Profile generated

Given production statistics are summarized

When profile is inspected

Then aggregate shape/sequence metadata may exist while raw prompts are absent.

---

### Requirement: Generator Identity Does Not Grant Trust Conformance

Conformance SHALL prove known generator identity alone cannot trust artifact.

#### Scenario: Trusted-name generator

Given artifact provenance says approved generator

When no authenticated trust mechanism exists

Then provenance alone is insufficient.

---

### Requirement: Campaign Failure Isolation Conformance

Conformance SHALL prove failed campaign does not disturb active known-good
Kernel.

#### Scenario: All candidate builds fail

Given production Kernel generation 8 is active

When optimization campaign fails

Then generation 8 remains active.

---

### Requirement: Tachyon Independence Conformance

Conformance SHALL validate Magnetar Runtime has no required direct Tachyon
dependency for Kernel optimization.

#### Scenario: Standalone deployment

Given Tachyon is absent

When local/CI-produced artifacts are supplied

Then Magnetar can validate and execute them.

---

### Requirement: Tooling Authority Boundary Conformance

Conformance SHALL validate CLI/external tooling authority is not ambiently
delegated to Runtime.

#### Scenario: Optimization CLI has repository access

Given CLI reads source repository

When Kernel Artifact enters Runtime

Then Runtime gains artifact data but not repository authority.

---

### Requirement: Selection Policy Still Authoritative Conformance

Conformance SHALL prove optimization ranking cannot override production Kernel
Selection Policy.

#### Scenario: Campaign says candidate fastest

Given current memory policy makes candidate infeasible

When Runtime selects Kernel

Then candidate remains excluded.

---

### Requirement: Optimization Observability Redaction Conformance

Conformance SHALL validate optimization events redact raw user data, secrets,
native handles and model internals by default.

#### Scenario: Campaign error

Given failure context contains sensitive data

When observation is exported

Then sensitive values are absent.

---

### Requirement: Canonical Manifest Conformance

Conformance SHALL prove equivalent supported manifests canonicalize
deterministically.

#### Scenario: Object key order differs

Given same fields appear in different JSON order

When canonicalized

Then digest is identical.

---

### Requirement: Duplicate-Key Conformance

Conformance SHALL reject duplicate JSON object keys.

#### Scenario: Duplicate source format

Given manifest supplies conflicting duplicate field

When parsed

Then failure is structured and fail-closed.

---

### Requirement: Blob Integrity Conformance

Conformance SHALL reject payload digest mismatch.

#### Scenario: One byte modified

Given compiled blob differs by one byte

When bundle is verified

Then preparation does not occur.

---

### Requirement: Filename Independence Conformance

Conformance SHALL prove filename/extension is not artifact format authority.

#### Scenario: CUBIN stored as digest path

Given no extension exists

When descriptor declares compatible CUBIN

Then format is resolved from metadata.

---

### Requirement: Optional Extension Forward Compatibility

Unknown optional extension SHALL not invalidate otherwise valid manifest.

#### Scenario: Vendor optional extension

Given Runtime does not understand extension

When extension marked optional

Then core manifest may still validate.

---

### Requirement: Required Extension Fail-Closed

Unknown required extension SHALL reject manifest.

#### Scenario: Required future semantic extension

Given Runtime does not support it

When manifest is loaded

Then manifest is unsupported.

---

### Requirement: Provenance Does Not Grant Trust

Conformance SHALL prove publisher/source/generator claims alone cannot grant
trusted status.

#### Scenario: Fake known publisher

Given malicious manifest writes known publisher name

When trust policy evaluates it

Then claim alone is insufficient.

---

### Requirement: Recommendation Does Not Promote

Conformance SHALL prove imported recommendation does not mutate active Kernel
Registry.

#### Scenario: Manifest says best-latency

Given candidate is imported

When no promotion occurs

Then currently active Kernel remains unchanged.

---

### Requirement: Qualification Evidence Revalidation

Conformance SHALL prove evidence reference is checked against current policy.

#### Scenario: Evidence revoked

Given valid digest references revoked evidence

When candidate is evaluated

Then it is not qualified.

---

### Requirement: External Reference Does Not Grant Network Authority

Conformance SHALL prove arbitrary external locator cannot trigger unrestricted
network access.

#### Scenario: Manifest references attacker URL

Given Runtime source policy denies it

When imported

Then no network access occurs.

---

### Requirement: Path Traversal Conformance

Conformance SHALL reject malicious bundle paths.

#### Scenario: `../../x`

Given archive contains traversal entry

When bundle is loaded

Then loading fails.

---

### Requirement: Symlink Conformance

Conformance SHALL reject symlink escape.

#### Scenario: digest path symlink

Given archive points blob path outside bundle

When loaded

Then bundle fails validation.

---

### Requirement: Repack Identity Conformance

Conformance SHALL prove archive metadata does not alter logical artifact
identity.

#### Scenario: Different ZIP compression

Given logical bytes are same

When bundle is repacked

Then manifest/blob logical digests remain identical.

---

### Requirement: Native Handle Exclusion Conformance

Conformance SHALL prove portable manifest contains no process-local native
execution authority.

#### Scenario: PreparedKernelId serialized

Given producer attempts to treat prepared ID as portable Kernel artifact

When validation runs

Then it is not accepted as executable artifact identity.

---

### Requirement: Parsing Side-Effect Conformance

Conformance SHALL prove parsing malicious manifest invokes no Provider compile,
prepare, execute or promotion operation.

#### Scenario: Parse-only validation

Given bundle is inspected

When validation fails

Then active Runtime execution state is unchanged.

---

### Requirement: Malformed Bundle Failure Atomicity

Conformance SHALL prove malformed imported bundle cannot disturb active
known-good Kernel.

#### Scenario: Replacement bundle corrupt

Given Kernel generation N is active

When invalid N+1 bundle is imported

Then N remains active.

### Requirement: Import Acceptance Separation Conformance

Conformance SHALL prove receiving/parsing artifact does not make it accepted.

#### Scenario: Valid syntax but policy denied

Given manifest parses

When trust fails

Then artifact remains outside accepted cache.

---

### Requirement: Acceptance Preparation Separation Conformance

Conformance SHALL prove accepted artifact has no PreparedKernelId merely from
ingestion.

#### Scenario: CUBIN committed

Given import succeeds

When Provider state is inspected before preparation

Then no native prepared handle exists because of ingestion alone.

---

### Requirement: Acceptance Promotion Separation Conformance

Conformance SHALL prove successful commit does not replace active Kernel.

#### Scenario: Better candidate imported

Given active generation exists

When candidate commits

Then active generation stays unchanged.

---

### Requirement: Immutable Snapshot Conformance

Conformance SHALL prove source mutation cannot change committed bytes after
validation.

#### Scenario: Local bundle replaced mid-import

Given source path is modified

When transaction commits

Then committed digest/content matches staged validated snapshot.

---

### Requirement: Quarantine Isolation Conformance

Conformance SHALL prove quarantined artifacts cannot enter normal Registry
selection.

#### Scenario: Quarantined fastest Kernel

Given benchmark says fastest

When selection runs

Then candidate is absent.

---

### Requirement: Atomic Commit Conformance

Conformance SHALL prove partial logical artifact is never observable.

#### Scenario: Commit fault injection

Given failure occurs mid-publication

When readers query cache

Then they see prior state, not half-imported Kernel.

---

### Requirement: Idempotence Conformance

Conformance SHALL prove repeated identical import preserves artifact identity.

#### Scenario: Three retries

Given same bundle imported three times

When successful

Then one content identity exists while audit contains retry transactions.

---

### Requirement: Dedup Policy Conformance

Conformance SHALL prove existing blob cannot bypass new trust/policy checks.

#### Scenario: Digest already cached

Given new manifest is untrusted

When it references cached blob

Then manifest is still subject to current policy.

---

### Requirement: Revocation Re-Import Conformance

Conformance SHALL prove deleting/re-importing revoked artifact cannot restore
eligibility.

#### Scenario: Same digest returns

Given artifact is revoked

When imported again

Then revocation still blocks it.

---

### Requirement: External Authority Conformance

Conformance SHALL prove manifest URL cannot expand Runtime network authority.

#### Scenario: Arbitrary HTTPS locator

Given source not authorized

When ingestion runs

Then no request is made.

---

### Requirement: External Digest Conformance

Conformance SHALL prove fetched data is accepted only if digest matches.

#### Scenario: Registry object replaced

Given locator returns changed bytes

When ingested

Then transaction fails.

---

### Requirement: Quota Conformance

Conformance SHALL prove oversized/over-complex input fails within configured
limits.

#### Scenario: Huge decompressed bundle

Given bundle exceeds decompressed byte budget

When processed

Then ingestion aborts without unbounded allocation.

---

### Requirement: Cancellation Conformance

Conformance SHALL prove cancellation before commit leaves accepted state
unchanged.

#### Scenario: Cancel during validation

Given transaction has staged data

When cancelled

Then staged content is cleaned and accepted cache unchanged.

---

### Requirement: Commit Cancellation Race Conformance

Conformance SHALL prove concurrent cancel and commit produce exactly one
terminal result.

#### Scenario: Race

Given cancellation races atomic commit

When both complete

Then state is either committed or cancelled, never partial/ambiguous.

---

### Requirement: Active Inference Isolation Conformance

Conformance SHALL prove failed ingestion does not invalidate active
PreparedKernel.

#### Scenario: Broken N+1 bundle

Given N is executing

When N+1 fails validation

Then N remains valid.

---

### Requirement: Ingestion Redaction Conformance

Conformance SHALL prove audit/observability contain no raw source, binary,
credential, native handle, prompt, weight or KV payload by default.

#### Scenario: Artifact rejected

Given detailed failure exists

When exported

Then sensitive payloads remain redacted.

---

### Requirement: Bounded Autotuning Conformance

Conformance SHALL prove Runtime Autotuning cannot evaluate an unbounded
specialization space.

#### Scenario: Unbounded template

Given tuning axis lacks explicit bound

When plan validates

Then it is rejected.

---

### Requirement: No Arbitrary Generation Conformance

Conformance SHALL prove Runtime Autotuning cannot invoke arbitrary Kernel source
generation.

#### Scenario: Candidate set exhausted

Given no candidate meets objective

When tuning ends

Then no external AI generator is invoked by Runtime.

---

### Requirement: No Arbitrary Compiler Flag Conformance

Conformance SHALL reject arbitrary free-form compiler arguments as tuning axes.

#### Scenario: Manifest exposes arbitrary flags

Given specialization contains unrestricted compiler command string

When validated

Then template is rejected.

---

### Requirement: No Hot-Path Tuning Conformance

Conformance SHALL prove token decode does not block on autotuning.

#### Scenario: Missing tuning cache

Given tuning record absent

When token generated

Then benchmark is not synchronously launched.

---

### Requirement: Accepted Artifact Requirement Conformance

Conformance SHALL prove quarantined/rejected artifacts cannot participate in
Runtime Autotuning.

#### Scenario: Quarantined Kernel

Given specialization template exists

When tuning candidates enumerate

Then Kernel is absent.

---

### Requirement: Qualification Coverage Conformance

Conformance SHALL prove specialization uses only appropriate qualification
evidence.

#### Scenario: Qualified exact instance differs

Given variant A is qualified and variant B is not covered

When tuning ranks both

Then B cannot become production-eligible solely from benchmark.

---

### Requirement: Tuning Cache Context Conformance

Conformance SHALL prove incompatible workload/target context invalidates tuning
reuse.

#### Scenario: Different GPU architecture

Given record came from sm90

When incompatible target uses cache

Then record is rejected/stale.

---

### Requirement: Memory Authority Conformance

Conformance SHALL prove Memory Manager may reject a tuning candidate regardless
of benchmark potential.

#### Scenario: Workspace infeasible

Given candidate would be fastest

When workspace fails admission

Then it is not benchmarked/selected as production candidate.

---

### Requirement: Known-Good Preservation Conformance

Conformance SHALL prove tuning failure cannot remove active known-good Kernel.

#### Scenario: Every candidate crashes during benchmark

Given current Kernel is healthy

When tuning fails

Then current Kernel remains active.

---

### Requirement: Tuning Winner Selection Boundary Conformance

Conformance SHALL prove tuning winner cannot bypass Kernel Selection Policy.

#### Scenario: Winner later untrusted

Given tuning identifies fastest variant

When trust policy rejects it

Then Runtime does not execute it.

---

### Requirement: Reproducible Mode Conformance

Conformance SHALL prove reproducible Model Instance cannot silently change
specialization through live tuning.

#### Scenario: Faster specialization discovered

Given Model Instance is pinned

When background tuning produces new winner

Then pinned instance remains unchanged.

---

### Requirement: Prepared State Persistence Conformance

Conformance SHALL prove Autotuning Record does not persist native
PreparedKernelId as portable tuning identity.

#### Scenario: Runtime restart

Given cached tuning record exists

When Runtime restarts

Then required Kernel is prepared again and native handle is not restored from
record.

---

### Requirement: Performance Evidence Cannot Grant Trust

Conformance SHALL prove fast Kernel remains untrusted if trust policy denies it.

#### Scenario: Untrusted fastest Kernel

Given production observations are excellent

When selection runs

Then trust denial remains authoritative.

---

### Requirement: Performance Evidence Cannot Grant Qualification

Conformance SHALL prove observations do not substitute for qualification.

#### Scenario: Unqualified specialization performs correctly in sampled runs

Given no valid qualification evidence exists

When Performance Model updates

Then candidate remains unqualified.

---

### Requirement: Performance Context Isolation

Conformance SHALL prove observations do not leak across incompatible
artifact/specialization contexts.

#### Scenario: New binary

Given artifact digest changes

When new generation executes

Then historical metrics are not automatically attributed to it.

---

### Requirement: Sample Sufficiency Conformance

Conformance SHALL prove isolated outlier cannot trigger confirmed regression
when policy requires more evidence.

#### Scenario: One slow sample

Given minimum sample count is 100

When one sample is slow

Then regression remains unconfirmed.

---

### Requirement: Drift Detection Conformance

Conformance SHALL detect sustained compatible difference from benchmark
baseline.

#### Scenario: Online latency increases materially

Given enough samples exceed threshold

When model updates

Then drift state is produced.

---

### Requirement: Workload Drift Conformance

Conformance SHALL identify substantial workload bucket distribution change.

#### Scenario: Production moves to larger batches

Given original tuning workload differs

When distribution crosses policy threshold

Then workload-drift event is emitted.

---

### Requirement: Bounded Re-Tuning Conformance

Conformance SHALL prove feedback-triggered tuning remains inside declared
specialization domain.

#### Scenario: Better code needed

Given no bounded variant meets goal

When retuning ends

Then Runtime does not invent source.

---

### Requirement: No Hot-Path Adaptive Benchmarking

Conformance SHALL prove active decode does not benchmark alternatives
synchronously.

#### Scenario: Regression during decode

Given regression is detected

When token loop continues

Then background retuning/fallback occurs according to policy.

---

### Requirement: External Escalation Boundary

Conformance SHALL prove unresolved Runtime tuning produces external optimization
signal, not direct AI generation.

#### Scenario: All specializations poor

Given policy permits external optimization

When Runtime escalates

Then no generator is invoked inside Runtime.

---

### Requirement: Adaptive Feedback Hysteresis Conformance

Conformance SHALL prove small measurement noise does not create repeated Kernel
switching.

#### Scenario: Ranking alternates slightly

Given difference stays below threshold

When observations update

Then active Kernel remains stable.

---

### Requirement: Adaptive Feedback Reproducible Mode Conformance

Conformance SHALL prove online evidence does not alter pinned Kernel choice.

#### Scenario: Another Kernel becomes faster

Given instance is pinned

When model recommends alternate Kernel

Then selection remains pinned.

---

### Requirement: Bounded Retention Conformance

Conformance SHALL prove continuous inference does not produce unbounded
performance telemetry memory growth.

#### Scenario: Million observations

Given retention limits exist

When traffic continues

Then old raw samples are aggregated/expired.

---

### Requirement: Feedback Failure Isolation

Conformance SHALL prove failure of Performance Model does not corrupt active
known-good Kernel.

#### Scenario: Aggregator fails

Given active Kernel remains healthy

When feedback subsystem errors

Then active execution remains valid.

---

### Requirement: Telemetry Redaction Conformance

Conformance SHALL prove exported performance evidence contains no raw prompt,
weight, KV, native handle, secret or credential.

#### Scenario: Export generated

Given aggregate telemetry exists

When inspected

Then sensitive inference content is absent.

### Requirement: Graph Semantic Authority Conformance

Conformance SHALL prove Prepared Execution Plan cannot redefine Execution Graph
semantics.

#### Scenario: Kernel binding substitution

Given Plan attempts to bind Kernel for different Operator semantics

When validated

Then Plan preparation fails.

---

### Requirement: Exact Binding Conformance

Conformance SHALL prove ready Plan uses the exact Kernel/specialization it
validated.

#### Scenario: Registry preference changes

Given Plan references Kernel A

When Registry later prefers B

Then executing existing Plan still uses A until safe replacement.

---

### Requirement: Native Handle Isolation Conformance

Conformance SHALL prove Plan contains no native executable pointer semantics.

#### Scenario: Provider returns opaque IDs

Given native CUDA/Metal state exists

When Plan/debug representation is inspected

Then native addresses are absent.

---

### Requirement: Session Resource Isolation Conformance

Conformance SHALL prove reusable Plan does not capture one Session's Tensor
Resources as global state.

#### Scenario: Two Sessions

Given both share Plan

When they execute concurrently

Then KV resources remain distinct.

---

### Requirement: Dynamic Guard Conformance

Conformance SHALL reject invocation outside Plan shape/workload envelope.

#### Scenario: Batch too large

Given Plan supports batch <=8

When batch=16

Then Kernel dispatch does not occur through that Plan.

---

### Requirement: Stale Versus Invalid Conformance

Conformance SHALL prove stale Plan may remain policy-safe while invalid Plan
cannot receive new work.

#### Scenario: Performance drift

Given Kernel remains qualified/trusted

When performance evidence becomes stale

Then Plan may continue temporarily.

#### Scenario: Kernel revoked

Given same Kernel is revoked

When invocation begins

Then Plan cannot execute.

---

### Requirement: Kernel Promotion Plan Isolation Conformance

Conformance SHALL prove Kernel hot swap does not mutate in-flight Plan bindings.

#### Scenario: Kernel generation changes

Given old Plan invocation active

When new Kernel promoted

Then new Plan is prepared and old invocation remains coherent.

---

### Requirement: Prepared Plan Memory Authority Conformance

Conformance SHALL prove Plan cannot override Memory Manager.

#### Scenario: Planned workspace unavailable

Given Plan requests workspace

When Memory Manager denies it

Then Plan does not force allocation.

---

### Requirement: No Hot-Path Full Planning Conformance

Conformance SHALL prove compatible ready Plan execution does not perform full
Registry/selection/autotuning/compilation pipeline.

#### Scenario: Repeated decode

Given Plan guards pass

When decode runs

Then bounded Plan execution path is used.

---

### Requirement: Atomic Plan Replacement Conformance

Conformance SHALL prove concurrent dispatch sees complete Plan generation.

#### Scenario: Replacement race

Given new Plan is published during request arrival

When each request binds Plan

Then each sees complete old or complete new generation.

---

### Requirement: In-Flight Lifetime Conformance

Conformance SHALL prove retiring Plan resources remain available until
quiescence.

#### Scenario: Old Provider segment in use

Given reference exists

When Plan retired

Then Provider destroys segment only after use completes.

---

### Requirement: Plan Cache Eligibility Conformance

Conformance SHALL prove cached Plan cannot bypass current revocation/trust/
qualification/readiness.

#### Scenario: Persisted Plan references revoked Kernel

Given Runtime restarts

When cache loads

Then Plan is not marked ready.

---

### Requirement: Provider Prepared Segment Optionality Conformance

Conformance SHALL prove Provider without graph-capture capability can execute a
Prepared Plan.

#### Scenario: Reference CPU

Given no native prepared-segment support

When Plan executes

Then individual prepared Kernel bindings are dispatched.

---

### Requirement: Adaptive Replan Conformance

Conformance SHALL prove adaptive feedback requests a new Plan rather than
mutating current Plan binding.

#### Scenario: Performance model prefers candidate B

Given current Plan uses A

When adaptation occurs

Then replacement generation is constructed.

---

### Requirement: Plan Observability Redaction Conformance

Conformance SHALL prove Plan diagnostics do not expose native handles, tensor
addresses, model weights, KV data, prompts or secrets.

#### Scenario: Plan invalidation report

Given detailed state exists

When report is exported

Then only safe logical identifiers are present.

### Requirement: Native Synchronization Isolation Conformance

Conformance SHALL prove Runtime public state contains no Provider-native
synchronization handles.

#### Scenario: GPU Provider inspection

Given Provider uses native stream/event objects

When Runtime Plan and diagnostics are inspected

Then native handles are absent.

### Requirement: Same-Stream Ordering Conformance

Conformance SHALL prove ordered submissions preserve required same-stream
effects.

#### Scenario: Write then read

Given A writes Tensor and B reads it on same ordered stream

When both execute

Then B observes completed/logically ordered A output.

### Requirement: Cross-Stream Dependency Conformance

Conformance SHALL prove cross-stream ordering occurs only through explicit
dependency or equivalent resource readiness edge.

#### Scenario: Producer and consumer streams

Given consumer depends on producer

When dependency exists

Then consumer does not race producer.

### Requirement: No Global Synchronization Requirement Conformance

Conformance SHALL prove independent streams may overlap without mandatory
Device-global wait.

#### Scenario: Independent Kernels

Given no shared resource dependency

When both execute

Then implementation may run concurrently.

### Requirement: Resource Readiness Conformance

Conformance SHALL prove unfinished output cannot be consumed improperly.

#### Scenario: Pending MatMul output

Given consumer uses different stream

When producer completion pending

Then consumer is dependency-ordered.

### Requirement: Host Readiness Conformance

Conformance SHALL prove host does not read Device output before required
completion.

#### Scenario: Final output transfer

Given transfer pending

When host wants logits

Then host waits or receives not-ready state.

### Requirement: Memory Reuse Fence Conformance

Conformance SHALL prove allocation cannot be reused while asynchronous work
references it.

#### Scenario: Temporary Tensor logically dropped

Given GPU work remains pending

When allocator wants same storage

Then storage remains unavailable.

### Requirement: Aliasing Synchronization Conformance

Conformance SHALL prove aliased resources cannot bypass synchronization.

#### Scenario: Tensor view

Given view overlaps source storage

When conflicting work executes

Then required hazard ordering exists.

### Requirement: Transfer Overlap Conformance

Conformance SHALL prove explicit dependencies do not unnecessarily serialize
independent transfer and compute.

#### Scenario: Unrelated transfer

Given transfer has no dependency on compute Kernel

When Provider supports overlap

Then contract permits concurrent execution.

### Requirement: Cross-Provider Mediation Conformance

Conformance SHALL prove Core does not exchange Provider-native events between
Providers.

#### Scenario: GPU to CPU dependency

Given Providers differ

When dependency resolves

Then Runtime mediates through logical completion/data movement.

### Requirement: KV Ordering Conformance

Conformance SHALL prove incremental decode cannot read incomplete KV mutation.

#### Scenario: Step N+1

Given step N KV append pending

When N+1 Attention is scheduled

Then required dependency prevents early read.

### Requirement: KV Page Lifetime Conformance

Conformance SHALL prove paged KV storage is not reused while in-flight access
exists.

#### Scenario: Cancelled sequence

Given page is still read by pending Kernel

When allocator reclaims sequence

Then page reuse waits for completion.

### Requirement: Continuous Batch Slot Lifetime Conformance

Conformance SHALL prove slot reuse respects asynchronous completion.

#### Scenario: Slot removed

Given previous batch still references slot

When new sequence arrives

Then slot is not reassigned unsafely.

### Requirement: Cancellation Completion Separation Conformance

Conformance SHALL prove logical cancellation does not imply physical
quiescence.

#### Scenario: Non-interruptible GPU Kernel

Given request cancelled

When Kernel remains running

Then CompletionToken stays pending and resources remain retained.

### Requirement: Deadline Safety Conformance

Conformance SHALL prove deadline expiration cannot cause premature resource
reuse.

#### Scenario: Timeout

Given request times out while Device work pending

When Runtime reports timeout

Then physical resources remain alive until safe.

### Requirement: Dependency Failure Conformance

Conformance SHALL prove failed predecessor prevents normal dependent execution.

#### Scenario: Producer fails

Given consumer requires producer output

When producer completion fails

Then consumer is not executed as successful path.

### Requirement: Device Loss Conformance

Conformance SHALL prove Device loss does not mark unfinished resources ready.

#### Scenario: Device reset

Given pending write existed

When Device becomes unavailable

Then output readiness becomes failed/lost rather than completed.

### Requirement: Stream Retirement Conformance

Conformance SHALL prove stream/native state is not destroyed prematurely.

#### Scenario: Stream draining

Given pending submissions exist

When Runtime retires stream

Then destruction waits for required completion/quiescence.

### Requirement: Synchronous Provider Conformance

Conformance SHALL prove Provider without asynchronous primitives can implement
logical model safely.

#### Scenario: Reference CPU immediate execution

Given submit executes synchronously

When returned

Then CompletionToken is terminal-completed and dependencies remain valid.

### Requirement: Prepared Plan Synchronization Conformance

Conformance SHALL prove Prepared Plan contains logical dependency information
without native synchronization state.

#### Scenario: Plan cached

Given Plan describes stream assignments

When inspected

Then no CUDA/Metal/Vulkan native queue/event object is serialized.

### Requirement: Observability Redaction Conformance

Conformance SHALL prove execution traces contain no native stream/event handles,
Tensor addresses, model weights, KV contents, prompts, secrets, or credentials.

#### Scenario: Failed synchronization trace

Given detailed internal Provider context exists

When trace is exported

Then only safe logical identities remain.

### Requirement: Same Device Residency Conformance

Conformance SHALL prove consecutive compatible Kernels can consume Device
Resource without host round-trip.

#### Scenario: MatMul then RMSNorm

Given both execute on same Device

When pipeline runs

Then intermediate is not required to materialize on host.

### Requirement: Persistent Weight Residency Conformance

Conformance SHALL prove Model weights may remain Device-resident across
inference requests.

#### Scenario: Two requests

Given same Model Instance remains loaded

When second request begins

Then normal execution does not require re-uploading unchanged weights.

### Requirement: Device Resident KV Conformance

Conformance SHALL prove incremental decode can retain KV on Device.

#### Scenario: Multiple decode steps

Given GPU memory capacity remains sufficient

When tokens are generated

Then KV remains Device-resident.

### Requirement: Zero Copy Compatibility Conformance

Conformance SHALL prove zero-copy cannot bypass dtype/layout/alignment
requirements.

#### Scenario: Wrong layout

Given Resource is directly addressable

But Kernel layout requirement differs

When candidate is bound

Then direct access is rejected/materialized explicitly.

### Requirement: Resource View Conformance

Conformance SHALL prove compatible View creation performs no byte-copy.

#### Scenario: Slice View

Given View is created

When allocation accounting is inspected

Then no mandatory new payload allocation exists.

### Requirement: View Bounds Conformance

Conformance SHALL reject invalid or overflowing View ranges.

#### Scenario: Overflow

Given malicious offset/size

When View is created

Then operation fails before native access.

### Requirement: Aliasing Conformance

Conformance SHALL prove overlapping Views preserve dependency and lifetime
safety.

#### Scenario: View write races parent read

Given overlap exists

When operations are submitted asynchronously

Then hazard is detected/ordered.

### Requirement: Mapping Readiness Conformance

Conformance SHALL prove host mapping cannot read incomplete Device writes.

#### Scenario: GPU write pending

Given host requests read mapping

When completion is pending

Then mapping is not made readable prematurely.

### Requirement: Mapping Lifetime Conformance

Conformance SHALL prove mapped Resource cannot be physically evicted/reused.

#### Scenario: Active host mapping

Given pressure requests eviction

When mapping remains active

Then storage remains valid or mapping transition is safely coordinated.

### Requirement: Coherency Conformance

Conformance SHALL prove required visibility transitions occur for non-coherent
mapping.

#### Scenario: Host writes then GPU reads

Given mapping is non-coherent

When mapping ends

Then Provider establishes required Device-visible state before execution.

### Requirement: Native Pointer Isolation Conformance

Conformance SHALL prove native addresses are absent from portable/public
Resource contracts.

#### Scenario: CUDA Tensor

Given Device pointer exists internally

When Runtime/WIT/diagnostics are inspected

Then native pointer is absent.

### Requirement: Explicit Movement Conformance

Conformance SHALL prove residency-changing copies are represented explicitly.

#### Scenario: GPU to CPU

Given no shared mapping exists

When CPU consumes Tensor

Then explicit movement operation exists.

### Requirement: Host Staging Denial Conformance

Conformance SHALL prove hidden host staging cannot bypass policy.

#### Scenario: GPU-to-GPU fallback

Given Provider needs host staging

And policy forbids it

When movement is requested

Then operation fails or alternate path is selected.

### Requirement: Async Transfer Lifetime Conformance

Conformance SHALL prove source/destination storage remains valid around
asynchronous transfer.

#### Scenario: Source released early

Given transfer pending

When owner drops source Resource

Then underlying storage remains until completion.

### Requirement: Peer Capability Conformance

Conformance SHALL prove peer access is not inferred from Device similarity.

#### Scenario: Same vendor GPUs without peer capability

Given direct peer read unsupported

When Runtime plans access

Then zero-copy peer route is denied.

### Requirement: Cross Provider Zero Copy Conformance

Conformance SHALL prove direct shared Resource access across Providers requires
explicit interoperability capability.

#### Scenario: CUDA Provider to CPU Provider

Given no interop capability exists

When direct native memory access is requested

Then Runtime uses explicit movement or rejects it.

### Requirement: Memory Manager Authority Conformance

Conformance SHALL prove Provider cannot silently relocate/spill logical Resource
against Runtime policy where such movement is Runtime-visible.

#### Scenario: Host staging forbidden

Given Provider faces memory pressure

When spill would require host staging

Then it cannot silently violate policy.

### Requirement: Prepared Plan Native Memory Isolation Conformance

Conformance SHALL prove Prepared Plan stores logical Resource bindings rather
than native memory addresses.

#### Scenario: Plan cached

Given Resource native pointer changes after restart

When Plan metadata is restored

Then native pointer was not persisted.

### Requirement: Eviction Safety Conformance

Conformance SHALL prove in-flight Resource cannot be evicted unsafely.

#### Scenario: Pending Kernel

Given Device work references Tensor

When eviction is requested

Then physical storage remains until completion/quiescence.

### Requirement: Memory Authority Separation Conformance

Conformance SHALL prove Memory Manager owns pool/allocation policy while
Provider owns native realization.

#### Scenario: Provider native pool

Given Provider supports native memory pool

When Runtime policy reserves KV capacity

Then Provider cannot override reservation.

### Requirement: Device Allocation Boundary Conformance

Conformance SHALL prove Device abstraction exposes no native allocation API.

#### Scenario: Device API inspection

Given GPU Device exists

When public Device contract is inspected

Then allocate/free/compact methods are absent.

### Requirement: Memory Pool Native Pointer Isolation Conformance

Conformance SHALL prove pools, blocks, leases, and Plans contain no native
pointer semantics.

#### Scenario: Allocation lease logged

Given Device allocation exists internally

When logical state is inspected

Then native address is absent.

### Requirement: Temporal Reuse Conformance

Conformance SHALL prove non-overlapping Tensor lifetimes SHALL share storage.

#### Scenario: A then C

Given A completion precedes C start

When AllocationPlan executes

Then same slot SHALL back both safely.

### Requirement: Asynchronous Reuse Safety Conformance

Conformance SHALL prove logical lifetime analysis does not free/reuse storage
before actual asynchronous completion.

#### Scenario: A graph node logically ended

Given Device execution still pending

When C wants same slot

Then reuse waits for CompletionToken.

### Requirement: Alignment Conformance

Conformance SHALL reject slot binding that violates Kernel/Provider alignment.

#### Scenario: 256-byte requirement

Given slot only guarantees 64-byte alignment

When Plan binds it

Then binding fails.

### Requirement: Hard Reservation Conformance

Conformance SHALL prove hard-reserved memory cannot be silently borrowed.

#### Scenario: KV reservation

Given optional tuning needs protected bytes

When pool is full outside reservation

Then tuning allocation fails.

### Requirement: Soft Borrowing Conformance

Conformance SHALL prove soft borrowing occurs only according to policy and
remains accounted.

#### Scenario: Borrowed workspace capacity

Given KV borrows soft capacity

When accounting is queried

Then borrowed amount is represented.

### Requirement: Watermark Reclamation Conformance

Conformance SHALL prove high-watermark policy can trigger reclamation without
freeing active memory.

#### Scenario: Pressure

Given pool exceeds high watermark

When reclamation runs

Then only eligible Resources are reclaimed.

### Requirement: Pending Reclaim Accounting Conformance

Conformance SHALL prove in-flight released memory is not counted as free.

#### Scenario: 1 GiB pending

Given Kernel still references released Resource

When admission checks capacity

Then 1 GiB is unavailable.

### Requirement: Fragmentation Classification Conformance

Conformance SHALL distinguish inability to realize large allocation from simple
aggregate free-byte exhaustion when allocator reports fragmentation.

#### Scenario: Fragmented pool

Given total free >= requested bytes

But no compatible region exists

When allocation fails

Then fragmentation error SHALL be emitted.

### Requirement: Compaction Safety Conformance

Conformance SHALL prove compaction does not relocate pinned, mapped, or
in-flight Resource unsafely.

#### Scenario: Active graph buffer

Given Resource requires stable address

When compaction runs

Then Resource is not moved.

### Requirement: Prepared Plan Memory Reservation Conformance

Conformance SHALL prove Plan cannot become ready under policy requiring
reservation when mandatory memory is unavailable.

#### Scenario: Workspace reservation fails

Given no compatible capacity

When Plan preparation completes

Then Plan is not marked ready.

### Requirement: KV Page Recycle Conformance

Conformance SHALL prove KV page is not recycled while any in-flight or shared
reference exists.

#### Scenario: Cancelled sequence

Given final Kernel still reads page

When Session ends

Then page remains pending reclaim.

### Requirement: Batch Memory Isolation Conformance

Conformance SHALL prove batch workspace cannot violate protected KV/weight pool
policy.

#### Scenario: Large batch spike

Given workspace request exceeds unreserved capacity

When batch is admitted

Then protected allocation class remains safe.

### Requirement: OOM Fallback Conformance

Conformance SHALL prove OOM fallback still obeys hard policy.

#### Scenario: Host spill forbidden

Given Device memory exhausted

When fallback evaluates spill

Then host-staging prohibition is respected.

### Requirement: Allocation Plan Cache Revalidation Conformance

Conformance SHALL prove cached AllocationPlan cannot bypass current pool
capacity/reservation/compatibility.

#### Scenario: Pool policy changed

Given cached Plan assumes old capacity

When reused

Then current policy is checked.

### Requirement: Memory Pool Observability Redaction Conformance

Conformance SHALL prove allocation diagnostics contain no native pointers,
native allocator handles, Tensor payloads, weights, KV contents, prompts,
secrets, or credentials.

#### Scenario: Pool OOM trace

Given Provider exposes native allocator details internally

When trace is exported

Then only safe logical/aggregate metadata remains.

