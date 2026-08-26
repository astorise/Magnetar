# Define Post-Baseline Model Source And Cache Roadmap

## Why

Magnetar now has a roadmap for real-world model formats.

Those formats define how external files normalize into:

- Model Artifact
- Tokenizer Artifact
- Adapter Artifact
- Tensor metadata
- quantization metadata
- manifest metadata

The next concern is where these artifacts come from and how they are cached.

After the baseline, Magnetar needs a controlled model source and artifact cache
roadmap.

This roadmap must support future user workflows such as:

```text
magnetar model pull qwen/...
magnetar model list
magnetar model inspect ...
magnetar run qwen3.5 "..."
```

But it must not give Magnetar Runtime arbitrary filesystem or network authority.

Model source resolution and cache mutation must remain explicit, validated, and
policy-controlled.

## What Changes

This change defines the post-baseline roadmap for model sources and artifact
cache.

It introduces controlled source kinds, cache semantics, identity rules,
integrity validation, trust metadata, eviction policy, and CLI/Runtime
boundaries.

Supported or planned source kinds include:

```text
development-fixture
client-provided-artifact
local-cache
local-directory-source
external-registry-source
model-hub-source
tachyon-provided-source
```

The exact implementation order is implementation-defined.

## Source And Cache Principle

A source provides artifact bytes and metadata.

A cache stores validated artifact bytes and metadata.

Neither source nor cache grants trust by itself.

Canonical flow:

```text
ModelRef
    |
    v
Source Resolution
    |
    v
Artifact Candidate
    |
    v
Format Normalization
    |
    v
Artifact Validation
    |
    v
Trust / Integrity Validation
    |
    v
Digest-addressed Cache
    |
    v
Model Loading
```

## Runtime Boundary

Runtime SHALL not perform arbitrary network downloads during inference.

Runtime SHALL not scan arbitrary directories.

Runtime MAY consume artifacts from explicitly authorized sources.

Runtime MAY read from Runtime-owned local cache where policy allows.

Runtime MAY accept client-provided artifact references.

Runtime MAY participate in source validation, normalization, trust, integrity,
and loading.

## CLI Boundary

`magnetar-cli` MAY own user-facing source operations such as:

- resolving friendly model aliases
- downloading model artifacts
- authenticating to registries where policy allows
- selecting local directories
- managing cache UX
- pruning cache
- displaying cache metadata
- importing local artifacts
- exporting cache entries

CLI SHALL not bypass Runtime artifact validation.

CLI download or import does not imply artifact trust.

## Source Kinds

Model source kinds SHOULD include:

```text
development-fixture
client-provided-artifact
local-cache
local-directory-source
external-registry-source
model-hub-source
tachyon-provided-source
```

Each source kind SHALL be represented explicitly.

Source kind SHALL not imply trust.

## Development Fixture Source

Development fixture source is used for tests and conformance.

It MAY provide deterministic fixture artifacts.

Fixture artifacts SHALL still pass normal artifact, format, trust, and loading
validation, using explicit test trust policy.

## Client-Provided Artifact Source

Client-provided artifact source represents artifacts supplied explicitly by a
caller such as CLI or test harness.

It MAY reference local files or in-memory data through authorized contracts.

Runtime SHALL validate the provided artifact before loading.

## Local Cache Source

Local cache source represents previously stored artifact content.

Cache entries SHALL be addressed by digest or equivalent content identity.

Cache hit SHALL not bypass trust, integrity, compatibility, or policy checks.

## Local Directory Source

Local directory source MAY represent a user-selected model directory.

Directory selection SHALL be explicit and policy-controlled.

Runtime SHALL not recursively scan arbitrary directories during inference.

A local directory source SHALL be normalized into an explicit artifact candidate
before loading.

## External Registry Source

External registry source MAY represent future artifact registries.

Registry access SHALL be explicit and policy-controlled.

Runtime SHALL not perform arbitrary registry access during inference.

CLI or a dedicated source manager may fetch artifacts, then Runtime validates
them.

## Model Hub Source

Model hub source MAY represent future integrations with public or private model
hubs.

Model hub support SHALL remain outside the core inference path.

Model hub metadata SHALL normalize into Model Artifact metadata.

Authentication and network policy SHALL remain outside core Runtime inference.

## Tachyon-Provided Source

Tachyon-provided source MAY allow Tachyon to supply artifacts or source metadata.

Tachyon remains responsible for distributed orchestration.

Magnetar Runtime still validates artifact identity, integrity, trust,
compatibility, memory, and execution contracts.

Tachyon SHALL not bypass Model Loading.

## ModelRef Resolution

ModelRef resolution SHALL be explicit.

A ModelRef MAY resolve to:

- existing Model Instance
- cached Model Artifact
- client-provided artifact
- local directory source
- development fixture
- future registry entry
- future hub entry
- future Tachyon source

Ambiguous ModelRefs SHALL fail or require policy-defined disambiguation.

## Model Aliases

User-facing aliases MAY be owned by `magnetar-cli`.

Runtime aliases, if any, SHALL be explicit and policy-controlled.

Aliases SHALL not bypass validation.

Alias resolution SHALL produce a ModelRef or source candidate, not a loaded
model by magic.

## Artifact Identity

Artifact identity SHALL be digest-based where possible.

Identity metadata SHOULD include:

- content digest
- manifest digest
- part digests
- shard digests
- tokenizer digest
- adapter digest
- config digest
- normalized manifest digest
- source annotations
- version metadata

Human names SHALL not be authoritative identity.

## Cache Addressing

Cache entries SHALL be addressed by digest or normalized artifact identity.

Cache path layout is implementation-defined.

Public APIs SHALL not expose raw cache paths by default.

Cache lookup SHALL return stable metadata and controlled artifact references.

## Cache Entry Metadata

Cache entries SHOULD include:

- artifact identity
- normalized manifest reference
- source kind
- source annotations
- trust status
- integrity status
- format metadata
- size estimate
- part list
- shard list
- tokenizer references
- adapter references
- last used timestamp
- created timestamp
- eviction eligibility
- validation status

## Cache Trust Model

Cache presence SHALL not imply trust.

A cached artifact SHALL still be checked against trust and policy before use.

Trust metadata MAY be cached, but policy SHALL determine whether cached trust is
still acceptable.

Revocation checks MAY invalidate cached trust.

## Cache Integrity

Cache integrity SHALL be validated.

Integrity validation MAY include:

- digest check
- shard digest check
- manifest consistency
- file size consistency
- normalized manifest consistency
- tokenizer/artifact compatibility
- adapter/base model compatibility
- corruption detection

Corrupt entries SHALL not load.

## Cache Mutation

Cache mutation SHALL be explicit and policy-controlled.

Mutations MAY include:

- insert
- update metadata
- mark validated
- mark untrusted
- mark revoked
- evict
- prune
- pin
- unpin
- repair placeholder

Runtime SHALL not mutate cache arbitrarily during inference except according to
approved policy.

## Cache Eviction

Cache eviction SHALL be policy-controlled.

Eviction policy MAY consider:

- size
- age
- last used time
- pin status
- trust state
- source kind
- validation status
- artifact type
- active Model Instance references

Eviction SHALL not remove artifacts required by active Model Instances.

## Cache Pinning

Cache entries MAY be pinned.

Pinned entries SHOULD be protected from automatic eviction.

Pinning SHALL not bypass trust, integrity, or compatibility validation.

## Partial Cache Entries

Partial cache entries MAY exist during downloads or imports.

Partial entries SHALL not be used for Model Loading unless explicitly supported
and validated.

Partial entries SHOULD have lifecycle state.

## Cache Lifecycle

Cache entry lifecycle states SHOULD include:

```text
discovered
resolving
fetching
partial
normalizing
validating
ready
untrusted
revoked
corrupt
evicting
evicted
failed
```

## Offline Mode

Runtime and CLI MAY support offline mode.

Offline mode SHALL use only local cache, client-provided artifacts, or
development fixtures.

Offline mode SHALL not attempt network access.

Offline failure SHALL be structured.

## Authentication Boundary

Authentication for remote sources SHALL not be owned by core Runtime inference.

CLI or source manager may handle credentials according to policy.

Secrets SHALL not be stored in Runtime cache metadata by default.

Observability SHALL not log credentials.

## Source Policy

Source policy SHOULD define allowed source kinds.

Policy MAY restrict:

- remote sources
- local directory sources
- unsigned artifacts
- untrusted cache entries
- large artifacts
- quantized artifacts
- license-restricted artifacts
- development fixtures in production
- Tachyon-provided sources

## License And Provenance

Source and cache metadata SHOULD preserve license and provenance metadata.

Policy MAY reject artifacts based on license or provenance.

License metadata SHALL not be treated as verified unless validated by policy.

## Compatibility With Model Formats

Model source/cache roadmap SHALL integrate with model format normalization.

Cache MAY store raw source files, normalized manifests, and derived metadata.

Model Loading SHALL consume normalized artifacts, not arbitrary raw source
directories.

## Compatibility With Adapter And Tokenizer Artifacts

Cache SHALL support Model Artifact, Tokenizer Artifact, and Adapter Artifact
entries.

Adapter cache entries SHALL preserve base model compatibility metadata.

Tokenizer cache entries SHALL preserve tokenizer/model compatibility metadata.

## Compatibility With Memory Manager

Cache presence SHALL not mean artifact is memory-resident.

Memory residency is owned by Model Loading and Memory Manager.

Cache stores bytes and metadata; Memory Manager owns loaded resources.

## Diagnostics

Source and cache diagnostics SHALL be redacted by default.

Diagnostics MAY include:

- source kind
- cache hit/miss
- artifact digest prefix
- validation status
- trust status
- integrity status
- size estimate
- missing parts
- revoked status
- policy denial reason

Diagnostics SHALL not expose credentials, raw file contents, raw model weights,
secrets, raw cache paths by default, Provider handles, Device handles, Kernel
handles, or memory pointers.

## Error Model

Model source/cache errors SHALL be structured.

Error categories SHOULD include:

- model-source-unsupported
- model-source-invalid
- model-source-ambiguous
- model-source-policy-denied
- model-source-network-denied
- model-source-authentication-failed
- model-source-not-found
- model-source-offline-unavailable
- model-cache-unavailable
- model-cache-miss
- model-cache-entry-invalid
- model-cache-entry-corrupt
- model-cache-entry-untrusted
- model-cache-entry-revoked
- model-cache-integrity-failed
- model-cache-insert-denied
- model-cache-eviction-denied
- model-cache-active-reference
- model-cache-partial-entry
- model-cache-path-redacted
- model-alias-not-found
- model-alias-ambiguous
- internal-model-source-cache-error

## Observability

Runtime and CLI SHOULD emit source/cache observations.

Observations MAY include:

- model source resolved
- model source rejected
- model source ambiguous
- cache lookup started
- cache hit
- cache miss
- cache entry validating
- cache entry ready
- cache entry corrupt
- cache entry untrusted
- cache entry revoked
- cache entry evicted
- cache insertion started
- cache insertion completed
- cache pruning started
- cache pruning completed
- offline mode active
- source policy denied

Observability SHALL not expose raw model weights, raw tokenizer data, raw file
contents, credentials, secrets, raw cache paths by default, Provider handles,
Device handles, Kernel handles, or memory pointers.

## Non-Goals

This change does not:

- implement model downloads
- define final `magnetar model pull` UX
- define registry protocol
- define model hub API
- define Tachyon distribution protocol
- define cache directory layout
- define production credential storage
- bypass Model Artifact validation
- bypass Model Loading
- make cached artifacts automatically trusted
- load models directly from arbitrary directories
- expose raw cache paths by default
- define HTTP server behavior

## Impact

Magnetar gains a controlled roadmap for model sources and artifact caching.

The intended post-baseline path becomes:

```text
ModelRef
  -> explicit source resolution
  -> artifact candidate
  -> format normalization
  -> trust/integrity validation
  -> digest-addressed cache
  -> Model Loading
```

while preserving:

```text
Runtime = inference validation and execution
CLI/source manager = user-facing source and cache UX
```