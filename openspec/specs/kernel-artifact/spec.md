# kernel-artifact Specification

## Purpose
This specification defines Kernel Source Artifact, Compiled Kernel Artifact, and Prepared Kernel identity, provenance, compatibility metadata, trust, Provider preparation, and cold-path/hot-path boundaries for generated or externally produced Kernel implementations.
## Requirements
### Requirement: Kernel Artifact Lifecycle

Magnetar SHALL distinguish Kernel Source Artifact, Compiled Kernel Artifact,
and Prepared Kernel.

#### Scenario: Generated kernel lifecycle

Given generated Triton source exists

When it becomes executable by a Provider

Then it transitions through distinct source, compiled, and prepared states.

---

### Requirement: Kernel Source Artifact

Kernel Source Artifact SHALL represent non-prepared kernel source or
intermediate representation.

#### Scenario: WGSL source

Given WGSL source implements a portable Operator

When stored as a Kernel Source Artifact

Then it is not considered executable Provider state.

---

### Requirement: Extensible Source Format Identity

Kernel Source Format SHALL use an extensible identifier rather than a closed
language enum.

#### Scenario: Future DSL

Given a future kernel DSL is introduced

When its artifact is registered

Then Magnetar can identify its format without changing a closed enum.

---

### Requirement: Source Artifact Content Identity

Content-addressed Kernel Source Artifact identity SHALL change if and only if
the immutable source bytes change. Kernel Source Artifact identity SHOULD use
immutable content-addressed identity.

#### Scenario: Source modified

Given one source byte changes

When digest is recomputed

Then artifact identity changes.

---

### Requirement: Generated Provenance Is Descriptive

Kernel Artifact provenance MAY indicate AI-generated origin but SHALL NOT imply
trust.

#### Scenario: AI-generated kernel

Given artifact provenance is `ai-generated`

When trust is evaluated

Then the artifact is not trusted solely because of that provenance.

---

### Requirement: Compiled Kernel Artifact

Compiled Kernel Artifact SHALL represent Provider-consumable compiled data
without exposing executable pointers.

#### Scenario: CUBIN artifact

Given CUDA source has been compiled to CUBIN

When represented by Runtime

Then Runtime stores artifact metadata and bytes but no `CUfunction*`.

---

### Requirement: Compiled Artifact Compatibility Metadata

Compiled Kernel Artifact SHALL carry compatibility metadata sufficient for
safe preparation decisions.

#### Scenario: Wrong target architecture

Given an artifact targets a different architecture

When preparation is attempted

Then preparation is rejected.

---

### Requirement: Prepared Kernel Is Ephemeral

Prepared Kernel SHALL represent Provider-owned executable state and SHALL NOT be
treated as a portable artifact.

#### Scenario: Runtime restart

Given Runtime restarts

When previously prepared kernel state is inspected

Then it must be prepared again or reconstructed from a persistent compiled
artifact.

---

### Requirement: Prepared Kernel Identifier Is Opaque

Runtime MAY use PreparedKernelId but SHALL treat it as opaque.

#### Scenario: Numeric identifier

Given PreparedKernelId is internally represented as integer

When Runtime receives it

Then Runtime does not reinterpret it as a pointer.

---

### Requirement: Provider Owns Native Prepared State

Provider SHALL own native executable state behind PreparedKernelId.

#### Scenario: CUDA prepared kernel

Given PreparedKernelId maps to a CUDA function internally

When Runtime dispatches it

Then only CUDA Provider resolves the native CUDA handle.

---

### Requirement: Device Does Not Compile

Device SHALL not own compilation or preparation APIs.

#### Scenario: Generated PTX requires preparation

Given PTX artifact targets a GPU Device

When preparation occurs

Then Provider performs preparation rather than Device.

---

### Requirement: Scheduler Does Not Compile

Scheduler SHALL not compile or prepare Kernel Artifacts.

#### Scenario: Required kernel missing

Given scheduling encounters an unprepared required kernel

When policy does not allow immediate readiness

Then Scheduler delays/rejects according to policy rather than compiling it.

---

### Requirement: Cold Path Compilation

Compilation and preparation SHALL be cold-path operations.

#### Scenario: Model loading

Given a model requires generated kernel

When model is loaded

Then kernel may be compiled/prepared before Model Instance becomes ready.

---

### Requirement: No Normal Hot-Path Compilation

Normal token decode SHALL NOT synchronously compile Kernel Source Artifacts.

#### Scenario: Decode discovers missing kernel

Given decode requires a kernel that is not prepared

When hot path executes

Then Runtime returns structured readiness/admission failure rather than
silently compiling it.

---

### Requirement: Kernel Artifact Trust

Kernel Source and Compiled Kernel Artifacts SHALL have explicit trust and
integrity status.

#### Scenario: Cached generated kernel

Given artifact exists in cache

When trust is evaluated

Then cache presence does not imply trust.

---

### Requirement: Operator Semantic Binding

Kernel Artifact SHALL declare which portable Operator semantics it implements.

#### Scenario: MatMul kernel

Given artifact claims MatMul implementation

When compatibility is validated

Then declared Operator semantics must match the graph requirement.

---

### Requirement: Fused Semantic Binding

Fused Kernel Artifact SHALL declare the Operator group whose semantics it
preserves.

#### Scenario: RMSNorm MatMul fusion

Given fused kernel replaces RMSNorm followed by MatMul

When Registry evaluates it

Then fusion metadata declares both portable semantics.

---

### Requirement: Explicit Shape Specialization

Shape specialization SHALL be explicit.

#### Scenario: Head dimension 128 only

Given kernel supports attention head dimension 128 only

When graph requires dimension 64

Then kernel is not compatible.

---

### Requirement: Explicit DType Specialization

DType specialization SHALL be explicit.

#### Scenario: FP16 kernel

Given kernel supports FP16 only

When graph requires FP32

Then Runtime does not silently reinterpret FP32 input.

---

### Requirement: Explicit Layout Specialization

Layout specialization SHALL be explicit.

#### Scenario: Contiguous-only kernel

Given artifact requires contiguous tensor layout

When tensor is strided

Then Runtime requires explicit conversion or selects another Kernel.

---

### Requirement: Precision Metadata

Precision metadata, when present, SHALL be visible to compatibility and
conformance policy evaluation. Kernel Artifact SHOULD expose numerical
precision behavior.

#### Scenario: Approximate math

Given generated kernel uses approximate math

When registered

Then this is visible in Kernel metadata and compatibility/conformance policy.

---

### Requirement: Provider Lifetime Independent From Kernel Artifact

Replacing a Kernel Artifact SHALL NOT require unloading Provider.

#### Scenario: Kernel v2 installed

Given Provider is active and kernel v1 has in-flight work

When kernel v2 becomes ready

Then Provider stays loaded and both prepared generations may temporarily
coexist.

---

### Requirement: Prepared Kernel Safe Retirement

Prepared Kernel SHALL NOT be destroyed while active operations reference it.

#### Scenario: Hot replacement

Given old Prepared Kernel has in-flight invocation

When replacement occurs

Then old generation remains alive until reference count reaches zero.

---

### Requirement: Runtime Tensor Ownership Preserved

Kernel preparation SHALL NOT transfer ownership of Runtime Tensor Resources to
Provider.

#### Scenario: Provider loads executable module

Given Provider allocates executable GPU memory

When inference tensors are used

Then Runtime Memory Manager continues to own tensor allocation/residency policy.

---

### Requirement: Inference API Does Not Expose Kernel Artifacts

Normal inference requests SHALL NOT expose Kernel Source Artifact,
Compiled Kernel Artifact, PreparedKernelId, or native handles.

#### Scenario: Generation request

Given client submits generation request

When Runtime validates it

Then arbitrary kernel source is outside normal inference request scope.

---

### Requirement: External Generator Independence

Magnetar Runtime SHALL not depend on a specific kernel generator.

#### Scenario: KernelEvolve-like artifact

Given artifact was produced by an external optimization system

When Runtime consumes it

Then Runtime uses generic Kernel Artifact contracts.

---

### Requirement: Kernel Artifact Structured Errors

Kernel Artifact and preparation failures SHALL use structured error categories.

#### Scenario: Unsupported format

Given Provider cannot prepare the artifact format

When preparation occurs

Then Runtime reports `kernel-artifact-format-unsupported` or equivalent
structured error.

---

### Requirement: Kernel Artifact Observability Redaction

Kernel Artifact observability SHALL redact source and native executable
internals by default.

#### Scenario: Preparation failure

Given compilation/preparation fails

When diagnostic is emitted

Then raw source, raw binary bytes, and native handles are absent by default.

---

### Requirement: Compilation Produces Compiled Kernel Artifact

Provider compilation SHALL produce a Compiled Kernel Artifact rather than a
Prepared Kernel directly at the logical contract level.

#### Scenario: Triton compiled

Given Triton source compiles successfully

When result is accepted

Then a Compiled Kernel Artifact exists before Provider preparation.

---

### Requirement: Compiled Artifact Records Compiler Identity

Compiled Kernel Artifact SHALL record compiler identity and version where available.

#### Scenario: Compiler upgrade

Given same source is compiled with different compiler version

When artifact metadata is compared

Then compiler identity difference is observable.

---

### Requirement: Compiler Options Affect Artifact Identity

Compiler settings affecting output SHALL participate in artifact identity or
compatibility fingerprint.

#### Scenario: Fast-math changed

Given compiler changes fast-math setting

When new artifact is produced

Then it is not treated as indistinguishable from previous artifact.

---

### Requirement: Compiled Artifact Records Target

Compiled Kernel Artifact SHALL identify target compatibility.

#### Scenario: sm90 artifact

Given binary was compiled for sm90

When Device is incompatible

Then preparation or Registry selection rejects it.

---

### Requirement: Compilation Output Has Digest

Compiled Kernel Artifact SHALL have integrity digest before readiness.

#### Scenario: Output mutated

Given compiled bytes change after digest calculation

When validation runs

Then integrity failure is returned.

---

### Requirement: Compilation Does Not Qualify Artifact

Compilation success SHALL remain distinct from semantic qualification.

#### Scenario: Numerically incorrect kernel

Given source compiles but produces wrong results

When compilation completes

Then artifact is not automatically production-eligible.

---

### Requirement: Qualified Kernel Artifact Metadata

When present, the QualificationRecord SHALL be immutable for the identified artifact and qualification profile.

Compiled Kernel Artifact MAY be associated with immutable QualificationRecord.

#### Scenario: Artifact qualified

Given compiled artifact passes qualification

When qualification record is stored

Then record identifies artifact digest, profile, suite, oracle and compatibility
context.

---

### Requirement: Qualification Does Not Mutate Compiled Content

Qualification SHALL NOT modify compiled artifact bytes in place.

#### Scenario: New qualification profile

Given same binary is tested against stricter profile

When result is stored

Then compiled artifact digest remains unchanged and a distinct qualification
record is produced.

---

### Requirement: Revoked Qualification

Revocation SHALL NOT alter the underlying compiled artifact bytes.

Qualification record MAY be revoked independently of artifact bytes.

#### Scenario: Test suite bug found

Given qualification procedure was invalid

When qualification evidence is revoked

Then artifact is no longer treated as qualified under that evidence.

---

### Requirement: Optimization Candidates Are Kernel Artifacts

Optimization systems SHALL communicate generated candidates using Kernel
Artifact contracts.

#### Scenario: AI-generated CUDA source

Given AI system produces CUDA source

When candidate is exported

Then it becomes KernelSourceArtifact with content identity/provenance.

---

### Requirement: Campaign Metadata Does Not Change Artifact Identity

Campaign metadata SHALL remain distinguishable from immutable artifact content.

#### Scenario: Same source evaluated twice

Given identical source is evaluated by two campaigns

When artifact digest is computed

Then content identity may remain equal while campaign evidence differs.

---

### Requirement: Generator Provenance Does Not Grant Trust

Generator/campaign identity SHALL not imply artifact trust.

#### Scenario: Known optimization service

Given trusted organization operates generator

When source artifact arrives

Then artifact trust still follows explicit trust/integrity policy.

---

### Requirement: Manifest Normalizes To Kernel Artifact Contracts

Validated portable manifest SHALL normalize into existing Kernel Artifact
domain types.

#### Scenario: Source descriptor

Given valid Triton source descriptor

When normalized

Then it becomes KernelSourceArtifact-compatible internal metadata.

---

### Requirement: Artifact Content Identity Survives Transport

Kernel Artifact identity SHALL remain based on content rather than bundle
transport representation.

#### Scenario: ZIP versus directory

Given same manifest/blobs are represented as directory and archive

When logical identities are computed

Then Kernel Artifact digests remain identical.

---

### Requirement: Multiple Compiled Variants Supported

A logical Kernel MAY carry multiple compiled variants, and Runtime SHALL treat all variants as belonging to the same logical Kernel identity regardless of count.

#### Scenario: sm80 and sm90

Given manifest contains both

When running on sm90

Then sm90-compatible variant may be selected without changing logical Operator
semantics.

---

### Requirement: Compiled Artifact Preserves Source Relationship

Compiled artifact SHOULD reference source digest where known, and when declared, the source relationship SHALL use immutable digest identity rather than a mutable location hint.

#### Scenario: Generated candidate lineage

Given source S compiled into binary B

When manifest is inspected

Then B may identify S as source artifact.

---

### Requirement: Portable Artifact Has No Runtime Handle Ownership

Kernel Artifact SHALL not own Provider/Device/native execution handle.

#### Scenario: Bundle cache import

Given bundle is cached

When Runtime process restarts

Then artifacts remain valid data but require new Provider preparation.

### Requirement: Accepted Artifact State Is Distinct

Kernel Artifact SHALL distinguish ingestion acceptance from trust,
qualification, preparation and promotion.

#### Scenario: Structurally valid source imported

Given source is accepted into cache

When inspected

Then acceptance alone does not imply qualified/trusted/prepared state.

---

### Requirement: Artifact Retains Ingestion Provenance

Accepted artifact metadata SHALL retain its content-addressed identity even
when it references the ingestion transaction/source audit record.

Such a reference MAY be included in metadata.

#### Scenario: Artifact origin investigated

Given Kernel later fails qualification

When audit is inspected

Then ingestion transaction can be correlated without changing artifact digest.

---

### Requirement: Immutable Content Identity Through Ingestion

Ingestion SHALL preserve Kernel Artifact content identity.

#### Scenario: Deduplicated blob

Given existing cache blob matches digest

When new manifest references it

Then same immutable blob identity is used.

---

### Requirement: Kernel Artifact May Declare Specialization Template

Accepted Kernel Artifact MAY expose bounded specialization metadata, and any exposed specialization metadata SHALL declare explicit bounds for each tuning axis.

#### Scenario: Compiled template

Given artifact supports multiple tile configurations

When normalized

Then specialization template is attached to the Kernel implementation metadata.

---

### Requirement: Specialized Artifact Identity

Compiled Artifact produced for one specialization SHALL record its
Specialization Instance identity.

#### Scenario: Two tile variants

Given BLOCK_M=32 and BLOCK_M=64 generate different binaries

When stored

Then compiled artifacts retain distinct specialization lineage.

---

### Requirement: Specialization Metadata Does Not Grant Qualification

Presence of valid specialization metadata SHALL not imply specialization is
qualified.

#### Scenario: Newly compiled specialization

Given artifact compiles successfully

When qualification is required

Then artifact remains unqualified until covered evidence exists.

