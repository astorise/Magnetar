# Define Component Artifact and Trust Model

## Why

Magnetar now has a Component Runtime boundary and a path toward concrete
WebAssembly Component execution.

The next architectural risk is accepting arbitrary Component bytes without a
stable artifact identity and trust model.

A WebAssembly Component is executable code.

Before Magnetar prepares, links, instantiates, or invokes such code, the
Runtime must be able to answer:

- what artifact is this?
- what content digest identifies it?
- what logical Component identity does it claim?
- which WIT imports does it require?
- which WIT exports does it provide?
- which Magnetar Runtime version or Capability versions does it target?
- who published or signed it, if known?
- which source provided it?
- is it trusted, rejected, quarantined, or pending policy?
- has its content changed since trust was evaluated?
- is this the same artifact Tachyon, local install, or another source claimed
  to provide?

Without this model, future Component distribution would be unsafe and ambiguous.

Magnetar must not treat a filename such as:

```text
qwen.wasm
```

as a trustworthy identity.

Magnetar must also not trust a manifest merely because it appears beside a
`.wasm` file.

The Runtime needs a canonical artifact model where executable bytes are
identified by content digest and validated against declared metadata.

This is required before later changes define:

- Component authority scoping
- Component distribution contract
- Tachyon-supplied Magnetar Components
- model Components
- tool Components
- agent Components

This change defines local artifact identity, manifest structure, compatibility
metadata, trust status, and validation behavior.

It does not define network distribution.

It does not define Tachyon as a required source.

It does not define final value-level filesystem, network, process, or secret
authority.

## What Changes

This change defines the Component Artifact and Trust Model.

A Component Artifact represents executable WebAssembly Component code.

It is distinct from:

- a Model Artifact
- a Provider binary
- a Runtime module
- a Tachyon deployment artifact
- a local configuration file
- a trust policy

The core distinction is:

```text
Component Artifact = executable portable WASM Component code
Model Artifact     = model data such as weights, tokenizer, config, metadata
Provider Artifact  = trusted native implementation binary or package
```

### Component Artifact Identity

A Component Artifact SHALL have a Runtime-recognized artifact identity.

The identity SHALL include at minimum:

- artifact kind
- content digest
- digest algorithm
- logical Component identity
- Component version
- manifest version

The content digest SHALL be computed over canonical executable artifact content.

For an initial implementation, the digest SHOULD be computed over the Component
`.wasm` bytes.

A future packaging format MAY define a canonical archive digest.

### Component Manifest

A Component Artifact SHALL be accompanied by or contain a manifest.

The initial manifest format SHALL be YAML with:

```yaml
schema: magnetar-component-artifact
schema_version: 1

artifact:
  kind: component
  digest:
    algorithm: sha256
    value: "sha256:0123456789abcdef..."

component:
  name: "magnetar.examples.hello"
  version: "0.1.0"
  description: "Minimal Magnetar Component fixture"
  role: "test-fixture"

runtime:
  magnetar:
    min_version: "0.1.0"

wit:
  imports:
    - package: "magnetar:test"
      interface: "echo"
      version: "1.0.0"
  exports:
    - package: "magnetar:test"
      interface: "run"
      version: "1.0.0"

capabilities:
  requires:
    - id: "magnetar:test/echo"
      version: "1.0.0"

authority:
  requires: []

publisher:
  id: "local-dev"
  name: "Local Development"

source:
  kind: "local"
  uri: "./fixtures/hello.component.wasm"

signatures: []
```

The manifest SHALL describe:

- artifact kind
- manifest version
- Component name
- Component version
- Component description
- Component role
- WIT imports
- WIT exports
- required Magnetar Runtime compatibility
- required Capability compatibility
- declared authority requirements
- optional publisher identity
- optional source identity
- optional signature metadata
- optional build/provenance metadata
- optional license metadata
- optional tags/categories

The manifest SHALL NOT be trusted until it is validated against the actual
Component artifact.

### Manifest Location

The initial implementation MAY support sidecar manifests.

For example:

```text
component.wasm
component.magnetar-component.yaml
```

or:

```text
component.wasm
component.magnetar-component.json
```

The exact filename convention is implementation-defined.

A future change MAY define embedded metadata, OCI packaging, or registry
metadata.

### Manifest Validation

Manifest validation SHALL check that:

- required fields are present
- manifest version is supported
- artifact kind is `component`
- Component name is valid
- Component version is valid
- declared WIT imports match or are compatible with the executable Component
- declared WIT exports match or are compatible with the executable Component
- digest matches the executable bytes
- declared Runtime and Capability compatibility can be evaluated
- declared authority requirements are syntactically valid
- signature metadata, if present, refers to the correct digest

A manifest mismatch SHALL fail closed.

### Digest

A Component Artifact SHALL be content-addressable.

The Runtime SHALL compute the digest of executable Component content before
trusting metadata.

The digest algorithm SHALL be explicit.

The initial digest algorithm SHOULD be:

```text
sha256
```

A digest mismatch SHALL reject the artifact before preparation or instantiation.

### Logical Identity

Logical identity is not the same as content identity.

For example:

```text
name: magnetar.examples.hello
version: 1.0.0
digest: sha256:...
```

The same logical Component version SHALL NOT be assumed to have the same bytes
unless the digest matches.

The Runtime MAY allow multiple digests for the same logical Component identity
when policy permits.

### Compatibility Metadata

The manifest SHALL declare compatibility requirements.

These MAY include:

- minimum Magnetar Runtime version
- maximum supported Magnetar Runtime version
- required WIT packages
- required WIT interfaces
- required Capability versions
- required Component Runtime features
- required engine features
- required WASI interfaces, if any
- declared authority requirements

Compatibility metadata SHALL be used before Component preparation and
instantiation.

### WIT Compatibility

The Runtime SHALL inspect the executable Component's actual WIT imports and
exports.

The manifest's WIT declarations SHALL not be authoritative unless they are
consistent with the executable artifact.

If the manifest declares fewer imports than the Component actually requires,
validation SHALL fail.

If the manifest declares exports that the Component does not provide,
validation SHALL fail.

### Authority Declaration

The manifest SHALL declare requested authority.

Authority declarations describe what the Component may need, such as:

- filesystem
- network
- environment
- process execution
- secrets
- clock
- randomness
- source-control access
- tool access
- external services

This change only defines declaration and validation of authority requirements.

Granting, scoping, denial, and runtime enforcement are defined by
`define-component-authority-scoping-model`.

Until that later change, unrecognized or unsupported authority requirements
SHALL fail closed or remain ungranted.

### Trust Status

The Runtime SHALL represent trust status explicitly.

Initial statuses SHOULD include:

```text
unknown
trusted
rejected
quarantined
revoked
```

The exact enum names are implementation-defined.

Semantics:

- `unknown`: artifact has not been trusted by policy
- `trusted`: artifact is allowed by current policy
- `rejected`: artifact failed validation or policy
- `quarantined`: artifact is retained for inspection but not executable
- `revoked`: artifact was previously known but is now explicitly disallowed

Only trusted artifacts MAY be prepared for execution.

### Trust Evaluation

Trust evaluation SHALL consider:

- digest
- manifest validity
- WIT compatibility
- Runtime compatibility
- source policy
- publisher policy
- signature policy when configured
- revocation state
- local administrator decision
- future distribution metadata

The exact policy mechanism MAY start simple.

For the initial implementation, local allowlists and denylists by digest MAY be
sufficient.

### Signature Metadata

This change defines signature metadata as optional.

If signature metadata is present, it SHALL bind to the executable artifact
digest or canonical package digest.

The Runtime SHALL NOT treat an unsupported signature as proof of trust.

A signature may be recorded as:

```text
present but unverified
```

unless the Runtime has a configured trust root and verification implementation.

Full signature format and public-key infrastructure MAY be refined in a later
change.

### Publisher Identity

Publisher identity SHALL be metadata.

It SHALL NOT imply trust by itself.

For example:

```text
publisher: tachyon
publisher: local
publisher: astorise
publisher: unknown
```

does not automatically mean trusted.

Policy decides whether a publisher identity is accepted.

### Source Identity

Source identity describes where the artifact came from.

Examples:

- local path
- local cache
- file library
- Tachyon source
- registry source
- OCI source
- development fixture

Source identity SHALL NOT imply trust by itself.

### Trust Store

The Runtime SHALL maintain or consume a trust store.

The trust store MAY include:

- trusted artifact digests
- rejected artifact digests
- revoked artifact digests
- trusted publisher identities
- allowed sources
- required signature policies
- local administrative decisions

The initial trust store MAY be file-based.

The initial trust store format SHALL support this minimal YAML shape:

```yaml
schema: magnetar-component-trust
schema_version: 1

trusted_digests:
  - "sha256:0123456789abcdef..."

rejected_digests: []

revoked_digests: []

trusted_publishers: []

trusted_sources: []

development:
  allow_unsigned_local: false
```

The trust store SHALL be separate from the artifact manifest.

A Component Artifact SHALL NOT declare itself trusted.

### Artifact Cache

Magnetar MAY maintain a local Component Artifact cache.

The cache SHALL be keyed by digest or include digest-indexed metadata.

Cache lookup SHALL verify digest integrity before use.

A cache entry SHALL NOT be trusted merely because it exists locally.

### Validation Pipeline

The Runtime SHALL validate a Component Artifact before preparation.

The pipeline SHALL conceptually be:

```text
Artifact bytes
      |
      v
compute digest
      |
      v
load manifest
      |
      v
validate manifest structure
      |
      v
compare manifest digest
      |
      v
inspect WIT imports/exports
      |
      v
validate manifest vs actual WIT
      |
      v
evaluate Runtime and Capability compatibility
      |
      v
validate authority declarations
      |
      v
evaluate trust policy
      |
      v
trusted Component Artifact
      |
      v
ComponentEngine::prepare
```

Preparation SHALL NOT occur before artifact validation succeeds unless explicitly
running in a controlled test mode.

### Artifact and Prepared Component Separation

A trusted Component Artifact is still not a Prepared Component.

Artifact validation answers:

```text
Is this executable artifact acceptable to prepare?
```

Preparation answers:

```text
Can the selected ComponentEngine compile/prepare this artifact?
```

These are separate states.

### Artifact and Instance Separation

A Component Artifact is not a Component Instance.

The lifecycle is:

```text
Component Artifact
      |
      v
validated / trusted artifact
      |
      v
prepared Component
      |
      v
Component Instance
      |
      v
invocation
```

Trusting an artifact SHALL NOT automatically instantiate it.

### Revocation

The Runtime SHALL support rejecting a previously accepted artifact by digest.

Revocation SHALL prevent future preparation and instantiation.

If an already-running Component Instance was created from a later-revoked
artifact, Runtime policy SHALL define whether it is:

- allowed to finish
- drained
- interrupted
- destroyed

The default SHOULD be to prevent new instances and allow Runtime policy to
decide active instance handling.

### Quarantine

Invalid or suspicious artifacts MAY be quarantined.

A quarantined artifact SHALL NOT be prepared or instantiated.

Quarantine MAY retain metadata for diagnostics.

### Development Mode

A development mode MAY allow locally built unsigned Components.

Development mode SHALL still compute digests and validate manifests.

Development mode SHALL be explicit.

It SHALL NOT be enabled silently in production configuration.

### Test Fixtures

Repository tests MAY mark fixture Components as trusted through test-only trust
configuration.

Fixture trust SHALL NOT become production default trust.

### Tachyon Boundary

Tachyon MAY later provide Component Artifacts to Magnetar.

This change SHALL NOT make Tachyon a required source.

The trust model SHALL remain vendor-neutral:

```text
External source
      |
      v
Component Artifact
      |
      v
Magnetar validation and trust policy
      |
      v
execution only if trusted
```

If Tachyon supplies an artifact, Magnetar still validates digest, manifest,
compatibility, authority declaration, and trust policy locally.

Tachyon distributes.

Magnetar validates and executes.

### Observability

Artifact validation SHOULD emit observations for:

- artifact discovered
- digest computed
- manifest loaded
- manifest validation failed
- WIT mismatch
- compatibility failure
- trust decision
- revocation
- quarantine
- preparation allowed

Observability SHALL not leak secrets or private signature material.

### Documentation

The repository SHALL document:

- Component Artifact identity
- manifest fields
- digest behavior
- trust status
- trust policy basics
- development mode
- artifact lifecycle
- separation from Model Artifacts
- separation from Provider binaries
- Tachyon boundary

## Non-Goals

This change does not:

- define Component network distribution
- define registry protocol
- define OCI packaging
- define Tachyon distribution contract
- define final signature format
- define complete PKI
- define transparency logs
- define SBOM requirements
- define value-level authority scopes
- grant filesystem/network/process/secrets access
- define model artifact trust
- define Provider binary trust
- implement hot reload
- implement remote attestation
- implement cluster-wide revocation

## Impact

Magnetar gains a stable local trust boundary before executable Components are
prepared or instantiated.

The Runtime no longer treats arbitrary `.wasm` bytes as sufficient to create a
Component.

Component execution becomes gated by:

```text
artifact identity
+ digest integrity
+ manifest validity
+ WIT compatibility
+ Runtime compatibility
+ Capability compatibility
+ authority declaration validity
+ trust policy
```

The central security rule is:

```text
Manifest says "trusted"        no
Filename looks familiar        no
Came from Tachyon              no
Exists in local cache          no
Publisher name is known        no

Digest + manifest + WIT + compatibility + trust policy = trusted
```

This creates a secure foundation for:

- local installed Components
- development Components
- future Tachyon-provided Components
- future Component registries
- future model and agent Components
- authority-scoped tool Components
