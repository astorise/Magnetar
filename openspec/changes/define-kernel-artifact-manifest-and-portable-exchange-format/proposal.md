# Define Kernel Artifact Manifest And Portable Exchange Format

## Why

Magnetar now defines the complete lifecycle for generated Kernels:

```text
generation
  -> KernelSourceArtifact
  -> compilation
  -> CompiledKernelArtifact
  -> qualification
  -> benchmarking
  -> cache
  -> preparation
  -> selection
  -> promotion
  -> execution
```

However, there is no portable exchange contract describing how an external
producer communicates these artifacts and their evidence to Magnetar.

Potential producers include:

```text
AI kernel generators
human engineers
CI systems
vendor tooling
optimization services
Tachyon-managed infrastructure
offline build farms
```

A portable exchange format is required so that Magnetar does not depend on:

- Triton
- CUDA
- WGSL
- Metal
- a particular optimization system
- a particular artifact registry
- filesystem layout
- process-local native handles
- one deployment platform

This change defines a versioned Kernel Artifact Manifest and a
content-addressed Kernel Exchange Bundle.

## What Changes

This change defines:

- Kernel Artifact Manifest v1
- Kernel Exchange Bundle v1
- canonical manifest serialization
- manifest identity
- content-addressed blob references
- source and compiled artifact descriptors
- Operator semantic bindings
- fused Operator bindings
- target constraints
- specialization metadata
- compiler metadata
- provenance metadata
- qualification evidence references
- benchmark evidence references
- recommendation metadata
- trust metadata
- signature envelope metadata
- extensions
- compatibility/versioning rules
- embedded and external artifact references
- deterministic bundle layout
- path and archive safety rules
- validation limits
- Runtime ingestion boundary
- redaction
- conformance

## Design Principle

Kernel exchange SHALL separate metadata from payload bytes.

```text
Kernel Artifact Manifest
        |
        +-- semantic identity
        +-- artifact descriptors
        +-- target constraints
        +-- provenance
        +-- evidence references
        +-- policy metadata
        |
        v
content-addressed blobs
        |
        +-- source
        +-- compiled binary
        +-- qualification evidence
        +-- benchmark evidence
        +-- auxiliary immutable data
```

The manifest SHALL NOT embed Provider-native executable pointers.

## Kernel Exchange Bundle

The logical portable package SHALL be called a Kernel Exchange Bundle.

Version 1 SHALL use the logical layout:

```text
kernel.manifest.json
blobs/
    sha256/
        <lowercase-hex-digest>
```

A bundle MAY be represented physically as:

- a directory
- an archive
- an object-store object set
- an OCI-like artifact
- an artifact-registry object
- another transport

Transport representation SHALL NOT change Kernel Artifact identity.

The canonical identity comes from manifest and blob content, not archive bytes.

## Bundle Manifest

The root manifest SHALL be named:

```text
kernel.manifest.json
```

for the directory/bundle representation.

The manifest SHALL be UTF-8 JSON.

JSON is chosen for the v1 interchange serialization because it is broadly
portable and does not require parsing executable configuration semantics.

YAML SHALL NOT be the canonical portable interchange encoding.

Tooling MAY render equivalent human-friendly formats, but canonical exchange
identity SHALL use the v1 JSON representation.

## Manifest Media Type

Implementations SHOULD identify the manifest using a versioned media type such
as:

```text
application/vnd.magnetar.kernel-manifest.v1+json
```

The exact transport mechanism MAY carry this media type as metadata.

## Manifest Schema Version

Manifest SHALL contain an explicit schema version.

Example:

```json
{
  "schema": "magnetar:kernel-manifest@1.0"
}
```

Schema version SHALL be independent from:

- Magnetar crate version
- Provider ABI version
- Kernel Compilation Capability version
- Operator versions
- WIT versions

## Manifest Versioning

Manifest schema SHALL use explicit major/minor compatibility.

A breaking schema change SHALL increment major version.

Additive optional fields MAY increment minor version.

A reader SHALL reject unsupported required major versions.

Unknown optional fields MAY be ignored according to extension rules.

## Canonical Manifest Representation

Manifest identity SHALL be computed from a canonical JSON representation.

Canonicalization SHALL at minimum define:

- UTF-8
- no duplicate object keys
- deterministic object key ordering
- deterministic string escaping
- deterministic integer representation
- no insignificant whitespace
- no NaN or Infinity numeric values

Manifest fields SHOULD avoid floating-point values where exact canonical
identity matters.

Performance values SHOULD normally live in referenced benchmark evidence.

## Manifest Identity

Manifest SHALL have a content digest calculated from canonical manifest bytes.

Conceptually:

```text
KernelManifestDigest {
    algorithm = sha256
    value = ...
}
```

The digest itself SHALL not be included recursively in the canonical content
unless a future envelope format defines it separately.

## Bundle Identity

Bundle identity SHOULD be derived from:

```text
manifest digest
+
referenced required blob digests
```

An archive checksum MAY additionally exist, but it SHALL NOT replace logical
artifact identity.

Repacking the same logical bundle SHALL not change its Kernel Artifact identity
merely because archive timestamps or compression differ.

## Content-Addressed Blobs

Payloads SHALL be referenced by digest.

The v1 baseline SHALL support SHA-256.

Blob references SHOULD include:

- digest algorithm
- digest value
- byte size
- media type
- artifact role
- format identity
- storage mode

Blob filename SHALL NOT determine format.

## Blob Path

An embedded SHA-256 blob SHOULD appear at:

```text
blobs/sha256/<digest>
```

where `<digest>` is lowercase hexadecimal.

Blob bytes SHALL hash to the declared digest.

Mismatch SHALL fail validation.

## Blob Roles

Portable blob roles SHOULD include:

```text
kernel-source
compiled-kernel
qualification-evidence
benchmark-evidence
auxiliary
```

Roles SHALL remain extensible.

A blob role SHALL describe purpose, not trust.

## Kernel Source Descriptor

A Kernel Source descriptor SHALL reference a KernelSourceArtifact-compatible
blob.

Metadata SHOULD include:

- blob digest
- Kernel Source Format
- optional source language/toolchain metadata
- Operator semantic binding
- specialization metadata
- generator provenance

Example conceptual descriptor:

```json
{
  "role": "kernel-source",
  "format": "triton:source@3",
  "digest": "sha256:...",
  "size": 18432
}
```

## Compiled Kernel Descriptor

A compiled artifact descriptor SHALL identify:

- compiled blob digest
- compiled format
- source artifact digest where known
- compiler metadata reference or inline structured metadata
- target compatibility
- specialization
- Operator semantics
- precision/determinism metadata

Example:

```json
{
  "role": "compiled-kernel",
  "format": "nvidia:cubin",
  "digest": "sha256:...",
  "source": "sha256:..."
}
```

## Format Identity

Artifact formats SHALL use extensible namespaced identities.

Examples:

```text
triton:source@3
nvidia:cuda-cpp
nvidia:ptx@9
nvidia:cubin
amd:hip
amd:hsaco
webgpu:wgsl
apple:msl
apple:metallib
khronos:spirv@1.6
vendor:custom-ir@4
```

Magnetar SHALL NOT require a closed `TargetLang` enum.

Unknown formats SHALL remain representable.

Provider compatibility determines whether they are usable.

## Filename Does Not Define Format

A file named:

```text
kernel.ptx
```

SHALL NOT be assumed to be PTX solely from its extension.

The manifest format identity is authoritative metadata, subject to validation.

## Operator Semantic Binding

Manifest SHALL declare which portable Operator semantics a Kernel implements.

A single-Operator Kernel SHALL identify:

- Operator ID
- Operator semantic version or compatible range

Example:

```json
{
  "operator": {
    "id": "magnetar:operator/matmul",
    "version": "1"
  }
}
```

Exact serialization MAY use structured version fields.

## Fused Semantic Binding

A fused Kernel SHALL declare the Operator group whose semantics it preserves.

The binding SHALL identify enough structure to distinguish semantically
different fusions.

For example:

```text
RMSNorm -> MatMul
```

is not automatically equivalent to:

```text
MatMul -> RMSNorm
```

The manifest SHALL NOT allow a fused Kernel to invent portable Operator
semantics that do not exist in the graph contract.

## Semantic Binding Identity

A semantic binding SHOULD have a deterministic fingerprint.

This MAY represent:

- ordered Operator IDs
- semantic versions
- relevant portable attributes

It SHALL NOT include Provider-specific handles.

## Target Constraints

Manifest MAY declare target constraints.

Target constraints MAY include:

- Provider compatibility class
- Device type
- hardware vendor
- architecture
- architecture feature requirements
- execution environment
- runtime/driver compatibility class
- memory classes
- required Device features

Target metadata SHALL remain descriptive.

It SHALL NOT contain native Device handles.

## Provider Compatibility

Compiled Kernel Artifact MAY declare compatible Provider family or ABI
requirements.

Provider name SHALL NOT alone make artifact compatible.

Runtime/Provider SHALL validate actual capability compatibility.

## Specialization

Manifest SHALL support explicit Kernel specialization metadata.

Specialization MAY include:

- exact dimensions
- dimension ranges
- batch ranges
- sequence ranges
- head count
- head dimension
- tile sizes
- alignment requirements
- dtype
- layout
- quantization profile
- execution phase
- Device features

Hidden specialization assumptions SHALL be prohibited.

## Prefill And Decode Specialization

A manifest MAY indicate that an artifact is specialized for:

```text
prefill
decode
both
```

This is optimization metadata and SHALL not redefine Operator semantics.

## Precision Metadata

Manifest MAY carry portable numerical behavior metadata including:

- accumulation dtype
- approximate math
- deterministic behavior
- tolerance profile reference
- fused semantics
- quantization behavior

Claims remain subject to qualification.

## Compiler Metadata

Compiled artifacts SHOULD record compiler metadata where known.

Metadata SHOULD include:

- compiler identity
- compiler version
- backend identity/version
- compiler flags fingerprint
- source artifact digest
- target architecture
- build fingerprint

Raw compiler command lines SHOULD NOT be mandatory.

## Reproducible Compiler Metadata

Compiler metadata SHOULD contain sufficient fingerprints to determine whether
compiled cache reuse is safe.

Compiler metadata SHALL not be interpreted as proof of trust.

## Provenance

Manifest MAY describe provenance.

Provenance MAY include:

```text
human-authored
ai-generated
ci-generated
vendor-provided
optimizer-generated
compiler-generated
imported
```

and optional generator identity/version.

Provenance SHALL NOT grant trust.

## Generator Metadata

Generator metadata MAY include:

- generator name
- generator version
- campaign ID
- source repository revision where explicitly supplied
- optimization campaign reference

Raw prompts, secrets, credentials, and internal chain-of-thought SHALL NOT be
required Kernel Manifest fields.

## Source Repository Metadata

A manifest MAY record a public or redacted source revision reference.

Repository URL SHALL be metadata only.

Runtime SHALL NOT automatically clone a repository because the manifest names
one.

Credentials SHALL NOT appear in repository locator metadata.

## Qualification Evidence References

Manifest MAY reference qualification evidence.

Qualification evidence reference SHOULD include:

- evidence digest
- qualification profile
- qualification suite version
- oracle identity/version
- target compatibility
- status

The manifest SHALL NOT make qualification evidence current merely by
referencing it.

Runtime SHALL validate evidence compatibility and current revocation status.

## Embedded Qualification Evidence

Qualification evidence MAY be embedded as a content-addressed blob.

It SHALL be treated as evidence data, not executable code.

## External Qualification Evidence

Qualification evidence MAY be external.

An external reference SHALL still carry immutable digest identity.

Location hints SHALL NOT be authoritative identity.

## Benchmark Evidence References

Manifest MAY reference benchmark evidence.

A benchmark reference SHOULD identify:

- evidence digest
- benchmark profile
- workload profile
- Device/architecture
- Provider version
- result freshness metadata

Benchmark evidence SHALL NOT be required for correctness eligibility unless
policy explicitly requires it.

## Recommendation Metadata

Optimization systems MAY attach recommendations.

Example concepts:

```text
recommended-for-latency
recommended-for-throughput
experimental
reject
```

Recommendation SHALL be advisory only.

```text
recommendation != promotion
```

## Trust Metadata

Manifest MAY carry trust-related metadata such as:

- publisher claim
- source claim
- signature envelopes
- certificate/key identifier hints
- expected trust policy labels

These are inputs to trust evaluation.

Manifest-declared trust metadata SHALL NOT itself produce trusted status.

## Publisher Claims

Publisher ID SHALL be treated as an assertion unless authenticated by an
accepted trust mechanism.

A publisher string SHALL NOT grant trust by itself.

## Source Claims

Source kind/location SHALL not grant trust by itself.

A manifest copied from one location to another SHALL not become trusted because
its text claims a trusted source kind.

## Signature Envelope

Manifest MAY contain detached signature metadata.

A signature envelope SHOULD be capable of carrying:

- signature algorithm
- key identifier
- signed-object digest
- signature blob reference
- optional certificate-chain reference
- optional transparency/provenance reference

This change defines exchange representation only.

It does not choose one mandatory cryptographic signature scheme.

## Signature Bytes

Signature material SHOULD be a content-addressed blob or compact encoded value
with explicit limits.

A signature record containing only an algorithm and digest SHALL NOT be treated
as proof that a signature was verified.

## Manifest Signatures

A signature MAY cover:

- canonical manifest digest
- selected artifact digest
- an envelope containing both

The signed scope SHALL be explicit.

## Trust Decision Outside Manifest

The final trust decision SHALL remain Runtime/deployment policy state.

The portable manifest SHALL NOT contain an authoritative field such as:

```json
{
  "trusted": true
}
```

that bypasses policy.

## Artifact Location

Artifacts MAY be embedded or external.

A descriptor SHALL indicate storage mode.

Suggested modes:

```text
embedded
external
```

## Embedded Artifacts

Embedded artifacts SHALL exist at their content-addressed bundle path.

Missing required embedded blob SHALL invalidate bundle.

## External Artifacts

External artifacts MAY provide location hints.

Location hints MAY include:

- artifact source identifier
- registry reference
- object key
- URI-like locator

Location hints SHALL NOT replace content digest identity.

## Runtime Network Boundary

Magnetar Runtime SHALL NOT automatically fetch arbitrary external artifact
locations solely because a manifest contains a URL.

External artifact acquisition SHALL use an explicitly authorized Artifact
Source / management boundary.

## Relative Paths

Manifest SHALL NOT use arbitrary relative filesystem paths for payload identity.

Embedded payloads SHALL be resolved only using deterministic digest paths.

## Absolute Paths

Portable manifest SHALL reject or ignore absolute host filesystem paths as
artifact locators according to policy.

Host-local path authority belongs outside portable artifact semantics.

## Bundle Path Safety

Bundle extraction/loading SHALL reject path traversal.

The logical bundle SHALL prohibit:

```text
../
absolute paths
drive-qualified paths
symlink escapes
hard-link escapes
device files
special filesystem entries
```

where physical transport supports such concepts.

## Symlinks

Portable Kernel Exchange Bundle SHALL NOT require symlinks.

For deterministic/safe ingestion, symlink entries SHOULD be rejected.

## Executable Permission Bits

Filesystem executable permission bits SHALL NOT be part of Kernel Artifact
semantic identity.

Payloads are data until Provider preparation.

## Archive Metadata

Archive timestamps, owners, groups and mode bits SHALL NOT contribute to
logical Kernel identity.

## Compression

Transport compression SHALL NOT alter logical artifact digest.

Digest SHALL apply to uncompressed logical blob bytes.

## Duplicate Entries

Bundle SHALL reject duplicate logical manifest/blob paths.

JSON manifest SHALL reject duplicate object keys.

Ambiguous duplicate data SHALL fail closed.

## Artifact Dependencies

A Kernel Artifact MAY declare immutable dependencies.

Dependencies SHALL themselves use content-addressed references.

Dependency semantics MAY include:

- static auxiliary constant data
- linked kernel module dependency
- Provider-specific immutable auxiliary binary

Dependencies SHALL NOT grant arbitrary filesystem/library search authority.

## Native Shared Libraries

A portable Kernel Manifest SHALL NOT treat an arbitrary `.so`, `.dll`, or
`.dylib` path as an automatically loadable trusted Provider.

Native Provider loading remains governed by Provider ABI/loading policy.

Compiled Kernel Artifact and Provider plugin are distinct concepts.

## Artifact Relationship Graph

Manifest MAY represent relationships such as:

```text
source A
   |
   +-> compiled B
   |      |
   |      +-> qualification Q
   |      +-> benchmark P
   |
   +-> compiled C
          |
          +-> qualification R
```

References SHALL use immutable artifact/evidence IDs.

Cycles in relationships that require acyclic semantics SHALL be rejected.

## Multiple Compiled Variants

A single manifest MAY describe multiple compiled variants.

Example:

```text
same logical Kernel
   ├── sm80 CUBIN
   ├── sm90 CUBIN
   ├── WGSL
   └── metallib
```

Runtime SHALL choose only compatible variants.

## Multiple Optimization Profiles

Manifest MAY reference multiple benchmark/recommendation records.

Example:

```text
variant A -> best latency
variant B -> best memory
variant C -> best throughput
```

Runtime selection policy remains authoritative.

## Bundle Completeness

Manifest SHALL distinguish required and optional artifact references.

A bundle missing a required embedded artifact SHALL fail validation.

Missing optional evidence MAY reduce eligibility/ranking according to policy
without necessarily invalidating the whole manifest.

## Manifest Limits

Runtime SHALL enforce defensive limits.

Limits SHOULD exist for:

- manifest byte size
- JSON nesting depth
- number of artifacts
- number of targets
- number of evidence references
- number of extensions
- annotation size
- string length
- total embedded artifact bytes

Limits SHALL fail with structured errors.

## Integer Safety

All sizes, counts, dimensions and offsets SHALL use validated bounded integer
representations.

Overflow SHALL be rejected.

## No Unbounded Recursion

Manifest structures SHALL avoid recursive unbounded schemas.

Parsers SHALL enforce nesting limits.

## Annotations

Manifest MAY contain non-authoritative annotations.

Annotations SHALL be namespaced.

Example:

```text
org.example:ticket
vendor.foo:tuning-strategy
```

Annotations SHALL NOT alter core semantics unless a recognized required
extension explicitly defines that behavior.

## Extensions

Manifest SHALL support namespaced extensions.

An extension SHALL declare whether it is:

```text
optional
required
```

Unknown optional extension MAY be ignored.

Unknown required extension SHALL make the manifest unsupported.

## Extension Isolation

Extensions SHALL NOT silently override core fields.

Core semantics such as:

- artifact digest
- Operator identity
- trust policy
- Provider binding
- native handle rules

cannot be replaced by an annotation.

## Manifest Compatibility

Reader SHOULD preserve unknown optional extension fields when round-tripping
where practical.

Reader SHALL reject unsupported required semantic extensions.

## Normalized Internal Representation

Runtime SHALL normalize the portable manifest into internal Kernel Artifact
contracts.

Portable exchange types SHALL not become Provider-native execution types.

Conceptually:

```text
KernelManifestV1
      |
      v
validation
      |
      v
normalized KernelSourceArtifact
normalized CompiledKernelArtifact
QualificationRecord references
BenchmarkRecord references
```

## Parsing Does Not Prepare

Parsing a manifest SHALL NOT:

- compile source
- load executable code
- call Provider.prepare
- execute Kernel
- start qualification
- start benchmarking
- promote candidate

These are later policy-controlled stages.

## Validation Order

Ingestion SHOULD proceed conceptually as:

```text
parse
  -> structural validation
  -> schema validation
  -> canonical identity
  -> blob integrity
  -> semantic validation
  -> trust/integrity evaluation
  -> evidence validation
  -> compatibility evaluation
  -> optional preparation
```

## Fail-Closed Integrity

Digest mismatch SHALL fail before Provider preparation.

Malformed semantic bindings SHALL fail before execution.

Unsupported required extensions SHALL fail before preparation.

## Distribution Neutrality

Kernel Exchange Bundle SHALL be transport-neutral.

Magnetar SHALL not require:

- GitHub
- OCI
- S3
- Hugging Face
- Tachyon
- filesystem
- one cloud registry

to use Kernel Artifacts.

## External Component Source Relationship

A generic external Artifact Source MAY deliver Kernel Exchange Bundles.

Source implementation SHALL provide bytes/metadata.

Runtime remains responsible for validation.

Source type SHALL NOT imply trust.

## Kernel Cache Integration

After successful structural/integrity validation, individual blobs MAY be
inserted into the content-addressed Kernel Cache according to policy.

Cache insertion SHALL preserve digest identity.

Cache presence SHALL NOT grant trust.

## Prepared Kernel Exclusion

PreparedKernelId and native Provider state SHALL NOT appear in the portable
manifest.

Prepared Kernel is process/runtime-specific ephemeral state.

## Device Handle Exclusion

Portable manifest SHALL NOT contain native Device pointers or handles.

Stable Device compatibility metadata is allowed.

## Provider Handle Exclusion

Portable manifest SHALL NOT contain Provider-native object pointers, function
pointers, compiler handles, pipeline objects, streams, events or contexts.

## Runtime Inference API Boundary

Normal generation requests SHALL NOT directly carry arbitrary Kernel Exchange
Bundles.

Kernel Artifact ingestion is a management/loading concern, not per-token
generation authority.

## CLI And Tooling

CLI or external tooling MAY:

- inspect manifests
- validate bundles
- import bundles
- export bundles
- show artifact metadata

Tooling SHALL still rely on Runtime/library validation rather than treating the
manifest as trusted configuration.

## Human Inspection

The format SHOULD remain reasonably human-inspectable.

Human readability SHALL NOT override deterministic parsing or security.

## Error Model

Structured errors SHOULD include:

```text
kernel-manifest-invalid-json
kernel-manifest-duplicate-key
kernel-manifest-schema-missing
kernel-manifest-schema-unsupported
kernel-manifest-required-extension-unsupported
kernel-manifest-too-large
kernel-manifest-limit-exceeded
kernel-manifest-canonicalization-failed
kernel-manifest-semantic-binding-invalid
kernel-manifest-target-invalid
kernel-manifest-specialization-invalid
kernel-manifest-artifact-reference-invalid
kernel-manifest-dependency-cycle
kernel-manifest-provenance-invalid
kernel-manifest-evidence-reference-invalid
kernel-manifest-signature-envelope-invalid

kernel-bundle-manifest-missing
kernel-bundle-duplicate-entry
kernel-bundle-path-invalid
kernel-bundle-symlink-denied
kernel-bundle-blob-missing
kernel-bundle-blob-size-mismatch
kernel-bundle-blob-digest-mismatch
kernel-bundle-total-size-exceeded
kernel-bundle-required-artifact-missing

kernel-exchange-format-unsupported
kernel-exchange-artifact-format-unsupported
kernel-exchange-external-reference-denied
kernel-exchange-external-artifact-unavailable
kernel-exchange-trust-denied
kernel-exchange-qualification-invalid
kernel-exchange-benchmark-evidence-invalid
kernel-exchange-compatibility-failed

internal-kernel-manifest-error
```

## Observability

Manifest/bundle observability MAY include:

- manifest discovered
- manifest parsed
- manifest digest
- schema version
- artifact count
- artifact formats
- blob validation success/failure
- semantic binding
- target compatibility
- qualification evidence count
- benchmark evidence count
- provenance summary
- trust evaluation result
- import/cache status

Observability SHALL redact by default:

- raw Kernel source
- raw compiled binary
- raw signature private material
- credentials
- sensitive URLs
- local filesystem paths
- raw benchmark tensors
- model weights
- native handles

## Conformance

Conformance SHALL validate:

- canonical manifest identity is deterministic
- duplicate JSON keys are rejected
- bundle blobs are content-addressed
- filenames do not determine artifact format
- blob digest mismatch fails closed
- unknown optional extensions are tolerated
- unknown required extensions are rejected
- publisher/source claims do not grant trust
- recognized format does not grant trust
- recommendation does not grant promotion
- qualification references are revalidated
- external URL does not trigger ambient Runtime network access
- traversal/symlink bundle attacks are rejected
- archive metadata does not alter logical identity
- compression does not alter blob identity
- PreparedKernelId cannot appear as portable execution state
- native Provider/Device handles cannot appear
- multiple hardware variants remain distinguishable
- parsing does not trigger compilation/preparation/execution
- malformed bundle cannot modify current active Kernel

## Non-Goals

This change does not:

- define one artifact registry
- define an OCI profile
- define HTTP APIs
- define S3 APIs
- define Tachyon transport
- define mandatory cryptographic signature algorithm
- authenticate publisher identity by itself
- implement Kernel compilation
- implement qualification
- implement benchmarking
- implement promotion
- implement generated Kernel agents
- make arbitrary URLs fetchable by Runtime
- transport Prepared Kernel native state

## Impact

Magnetar gains a neutral interchange layer:

```text
producer-specific world
        |
        v
Kernel Exchange Bundle
        |
        v
portable Magnetar artifact model
        |
        v
Runtime validation/policy
        |
        v
Provider-specific execution world
```

This permits an ecosystem of independent Kernel generators and optimization
systems without coupling Magnetar Runtime to any specific producer, language,
registry, or hardware vendor.