# Define Component Distribution Contract

## Why

Magnetar can validate, trust, prepare, and execute local WebAssembly Components.

Magnetar now needs a contract describing how Component Artifacts may be supplied
by external or local sources without coupling the Runtime to a specific
distribution system.

This is required before Magnetar can safely accept Components from:

- a local development directory
- a local artifact cache
- a package or registry mirror
- a client such as `magnetar-cli`
- a future Tachyon integration
- another external Component source

However, Magnetar's scope has been narrowed to inference execution.

Therefore, this distribution contract is only for Magnetar-compatible
Inference Components.

It SHALL NOT define distribution of general-purpose agent tools.

The following are out of scope for Magnetar Component distribution:

- filesystem tool Components
- Git tool Components
- shell/process tool Components
- secret reader Components
- network fetcher Components
- workspace editor Components
- arbitrary external service tools

Those belong to clients or orchestrators such as `magnetar-cli`.

Magnetar's distribution contract must preserve the central rule:

```text
External systems may provide artifacts.

Magnetar validates, trusts, and executes inference only.
```

A distributed Component is not trusted merely because it came from a known
source.

For example:

```text
Tachyon provided it        ❌ not sufficient
Local cache contains it    ❌ not sufficient
Registry lists it          ❌ not sufficient
Client requested it        ❌ not sufficient
Manifest says trusted      ❌ not sufficient
```

The Runtime must still validate:

- artifact digest
- manifest structure
- WIT imports and exports
- Runtime compatibility
- Capability compatibility
- inference-scoped authority
- trust policy
- revocation state

## What Changes

This change defines the Component Distribution Contract.

A Component Distribution Source is any local or external source that can make
Component Artifact bytes and metadata available to Magnetar.

The contract SHALL remain vendor-neutral.

Tachyon MAY implement this contract later, but Magnetar SHALL NOT depend on
Tachyon to distribute or execute Components.

### Distribution Unit

The unit of distribution SHALL be a Component Artifact Package.

A Component Artifact Package SHALL contain or reference:

- executable WebAssembly Component bytes
- Component Artifact manifest
- content digest
- optional signature metadata
- optional provenance metadata
- optional source metadata
- optional cache metadata

The package SHALL describe one Magnetar-compatible Inference Component.

The package SHALL NOT bundle arbitrary client-side tools as Magnetar Runtime
Components.

### Source Contract

A Component Distribution Source SHALL be able to provide, at minimum:

- artifact identity
- artifact bytes or resolvable byte location
- manifest bytes or resolvable manifest location
- declared digest
- source identity
- optional publisher identity
- optional signature metadata
- optional provenance metadata

The source SHALL NOT be trusted as authoritative for execution.

It is an input to Magnetar validation.

### Pull Model

Magnetar MAY support a pull model where the Runtime asks a source for an
artifact by identity.

Conceptually:

```text
Runtime
  -> source.resolve(component identity)
  -> source.fetch(artifact digest)
  -> validate locally
```

### Push Model

Magnetar MAY support a push model where an external system provides an artifact
package to the Runtime.

Conceptually:

```text
external source
  -> provide artifact package
  -> Runtime validates locally
```

Push delivery SHALL NOT bypass validation.

### Source Neutrality

The contract SHALL support multiple source kinds.

Initial source kinds MAY include:

```text
local-directory
local-cache
client-provided
development-fixture
external-registry
tachyon
```

The exact serialized names may differ.

Source kind is metadata.

Source kind SHALL NOT imply trust.

### Tachyon Boundary

Tachyon MAY later distribute Magnetar-compatible Inference Components.

If Tachyon supplies a Component Artifact Package, Magnetar SHALL still validate
it locally.

Tachyon SHALL NOT grant Magnetar broad authority.

Tachyon SHALL NOT turn a filesystem/Git/shell tool into a valid Magnetar
Runtime Component.

The invariant remains:

```text
Tachyon distributes.

Magnetar validates and executes inference.
```

### magnetar-cli Boundary

`magnetar-cli` MAY obtain Component packages from local paths, caches, or
registries and submit them to Magnetar.

When it does so, `magnetar-cli` acts as a Component Distribution Source.

Its workspace, Git, filesystem, network, secret, or shell authorities SHALL NOT
become Magnetar Component authorities.

### Distributed Artifact Identity

A distributed package SHALL identify its artifact by digest.

Logical names and versions are not sufficient.

The Runtime SHALL compute the executable content digest locally after receiving
the bytes.

A source-provided digest SHALL be treated as a claim until locally verified.

### Manifest Consistency

The distributed manifest SHALL be validated against the actual executable
Component.

The Runtime SHALL reject a package when:

- manifest digest does not match computed digest
- manifest WIT imports do not match actual imports
- manifest WIT exports do not match actual exports
- manifest declares unsupported Runtime compatibility
- manifest declares unsupported Capability compatibility
- manifest requests out-of-scope authority
- trust policy rejects the package

### Inference-Scoped Authority

Distributed Components SHALL request only inference-scoped authority.

Allowed Magnetar Component authority categories are those defined by the
Inference-Scoped Component Authority Model.

Examples include:

- model-artifact-read
- tokenizer-artifact-read
- prompt-template-read
- adapter-artifact-read
- quantization-artifact-read
- inference-session-state
- generation-session-state
- kv-cache-access
- prefix-cache-access
- compute-capability
- generation-capability
- sampling-capability
- observability-emit
- runtime-diagnostics

Distributed packages requesting broad authority SHALL be rejected.

### Package Does Not Grant Authority

A distribution package may declare required authority.

It does not grant that authority.

Authority is granted only by Runtime policy during validation and Link Plan
construction.

### Cache Contract

Magnetar MAY cache distributed Component Artifacts.

The cache SHALL be keyed by digest or otherwise verify digest before use.

Cache presence SHALL NOT imply trust.

A cached package SHALL be revalidated according to Runtime policy before
preparation unless policy explicitly allows reuse of a still-valid trust
decision.

### Revocation

The distribution contract SHALL support revocation by digest.

A Runtime SHALL reject packages with revoked digests.

A source MAY provide revocation metadata, but Runtime trust policy remains
authoritative.

### Version Selection

The distribution contract MAY support resolving a logical Component identity and
version requirement to one or more candidate artifact digests.

For example:

```text
name = magnetar.tokenizer.qwen
version requirement = >=1.0,<2.0
```

may resolve to:

```text
sha256:aaa...
sha256:bbb...
```

The Runtime SHALL validate the selected digest locally.

The source SHALL NOT be trusted to select a safe artifact without Runtime
validation.

### Compatibility Selection

A source MAY report compatibility metadata.

Runtime SHALL still evaluate compatibility locally.

Compatibility may include:

- Magnetar Runtime version
- WIT package versions
- Capability versions
- Component Engine feature requirements
- inference authority requirements

### Provenance

A package MAY include provenance metadata.

Provenance metadata MAY include:

- builder identity
- source repository
- commit digest
- build timestamp
- build system
- reproducibility metadata
- SBOM reference

This change records provenance metadata but does not require a full SLSA,
transparency log, or reproducible-build policy.

Provenance SHALL NOT imply trust by itself.

### Signatures

A distribution source MAY provide signature metadata.

Signature metadata SHALL bind to the artifact digest or package digest.

A signature SHALL NOT imply trust unless Runtime policy has a configured trust
root and verification support.

Unsupported signatures SHALL be recorded as unverified or rejected according to
policy.

### Package Integrity

If a future package format contains multiple files, the package SHALL have a
canonical digest or manifest-bound file digests.

The initial implementation MAY use separate `.wasm` and sidecar manifest files.

### Offline Operation

The distribution contract SHALL allow local/offline operation.

Magnetar SHALL be able to validate and execute locally trusted Component
Artifacts without contacting Tachyon or a remote registry.

### No Remote Execution

This contract defines artifact distribution only.

It SHALL NOT define remote Component execution.

A distributed Component becomes local executable code only after validation,
trust, preparation, linking, and instantiation by Magnetar.

### Observability

Runtime SHOULD emit observations for distribution events, including:

- source resolution
- package received
- fetch success
- fetch failure
- digest mismatch
- manifest mismatch
- trust rejection
- revocation
- cache hit
- cache verification failure
- preparation allowed

Observability SHALL NOT leak secrets, private keys, raw credentials, or
unauthorized source URLs.

### Error Model

Distribution failures SHALL be structured.

Failure categories SHOULD include:

- source unavailable
- artifact not found
- version not found
- digest mismatch
- manifest missing
- manifest invalid
- WIT mismatch
- compatibility failure
- forbidden authority
- trust rejected
- revoked artifact
- cache integrity failure
- unsupported signature
- policy denied

### Future Registry Protocol

This change does not standardize a registry API or network protocol.

A future registry-specific change MAY define:

- HTTP API
- OCI artifact mapping
- authenticated registries
- mirror behavior
- index format
- transparency logs
- signed repository metadata

### Future Tachyon Contract

A future Tachyon integration change MAY define how Tachyon implements this
distribution contract.

That future contract SHALL not weaken Magnetar validation.

## Non-Goals

This change does not:

- define a public network registry protocol
- define OCI packaging
- define Tachyon implementation details
- require Tachyon
- define remote execution
- define cross-node execution
- define CLI workspace authority
- define filesystem tool distribution
- define Git tool distribution
- define shell tool distribution
- define secret tool distribution
- define network tool distribution
- grant authority to Components
- implement model artifact distribution
- implement Provider binary distribution
- define complete signature PKI
- define transparency logs
- define full SBOM requirements
- implement hot reload

## Impact

Magnetar gains a vendor-neutral way to accept Component Artifact Packages from
different sources.

The Runtime remains responsible for validation and trust.

The distribution source becomes a supplier of bytes and metadata, not an
execution authority.

The safe flow becomes:

```text
Distribution Source
        |
        v
Component Artifact Package
        |
        v
Magnetar local validation
        |
        v
Trust policy
        |
        v
ComponentEngine preparation
        |
        v
Runtime Link Plan
        |
        v
Inference execution
```

This prepares Magnetar for future Tachyon integration while keeping Magnetar
scoped to inference.