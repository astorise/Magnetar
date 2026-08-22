# component Specification

## Purpose

Define the host-side contracts and lifecycle management for portable WebAssembly
Components, independently of concrete hardware implementations.
## Requirements
### Requirement: Component Discovery

The Runtime SHALL support discovery of local WebAssembly Component artifacts during
initialization or through explicit registration.

Discovered artifacts SHALL be validated and prepared by the Component Runtime
before instantiation.

Discovery alone SHALL NOT grant authority to execute imports.

#### Scenario: Discover Component file

Given a valid Component file exists in a configured directory

When the Runtime discovers it

Then the Component is known to the Runtime

But it is not instantiated with unauthorized imports.

---

### Requirement: Component Contracts

Every Component SHALL declare its portable dependencies and exports through WIT
interfaces.

Component imports SHALL identify required interfaces rather than concrete native
implementations or Component instance names.

The Runtime SHALL validate required imports before instantiation.

#### Scenario: Validate contracts

Given a Component imports one or more WIT interfaces

When the Runtime prepares the Component for instantiation

Then every required interface is validated for compatibility and authorization

And unresolved mandatory imports prevent instantiation.

---

### Requirement: Component Isolation

Components SHALL execute through a real WebAssembly Component Model engine while
remaining isolated from native Providers, Devices, and engine-native handles.

#### Scenario: Execute with native Provider present

Given a CUDA Provider is registered in the Runtime

And a Component imports an authorized Runtime Capability

When the Component executes

Then it cannot access CUDA-native handles unless a future explicit portable
contract permits an opaque resource.

---

### Requirement: Component Lifecycle

The Runtime SHALL manage lifecycle for engine-backed Component definitions and
instances.

A Component definition SHALL be validated and prepared before instantiation.

A Component Instance SHALL be destroyed before its engine Store state is
released.

#### Scenario: Runtime shutdown

Given an engine-backed Component Instance exists

When Runtime shutdown occurs

Then the Runtime prevents new invocations and destroys the instance according
to Runtime policy.

### Requirement: Dependency Resolution

Component dependencies SHALL be expressed through WIT imports.

Components SHALL NOT require direct dependency on another Component's logical
name as the canonical dependency mechanism.

The Runtime SHALL resolve and authorize required interfaces before
instantiation.

#### Scenario: Resolve Capability dependency

Given a Component imports `magnetar:compute/run`

When the Runtime constructs its Link Plan

Then the import is linked to an authorized Runtime Compute Capability endpoint

And the Component does not name the Component or Provider implementing the
underlying behavior.

---

### Requirement: Component Runtime Observability

The Component Runtime SHALL support structured Runtime observations for
important lifecycle and execution events.

Observations MAY include:

- definition identity
- instance identity
- preparation
- instantiation
- invocation
- interruption
- trap
- resource-limit violation
- destruction

#### Scenario: Component traps

Given a Component invocation traps

When Runtime observability records the failure

Then the observation identifies the relevant Component instance and stable trap
category

Without exposing engine-native handles or secret data.

### Requirement: Concrete WASM Component Engine

Magnetar SHALL provide at least one concrete WebAssembly Component Model engine
implementation behind the engine-neutral ComponentEngine boundary.

The initial implementation SHOULD use Wasmtime unless implementation evidence
requires another engine.

#### Scenario: Instantiate with concrete engine

Given a valid WASM Component artifact

And an authorized Link Plan

When the Runtime instantiates the Component

Then the concrete engine creates an executable Component Instance

And public Magnetar APIs remain engine-neutral.

---

### Requirement: Wasmtime Is an Implementation Detail

If Wasmtime is used, Wasmtime-native types SHALL remain private to the engine
adapter.

Canonical Magnetar Component APIs SHALL NOT expose concrete Wasmtime objects.

#### Scenario: Use Component Runtime API

Given application code uses Magnetar Component Runtime APIs

When Wasmtime is the concrete engine

Then application code does not require `wasmtime::Store`,
`wasmtime::component::Linker`, `wasmtime::component::Instance`, or
`wasmtime::Trap`.

---

### Requirement: Engine-Backed Component Preparation

The concrete Component Engine SHALL validate and prepare Component bytes before
instantiation.

Preparation MAY include engine parsing, validation, compilation, and
optimization.

#### Scenario: Invalid Component bytes

Given invalid Component bytes

When preparation is attempted

Then preparation fails with a stable Magnetar Component error.

---

### Requirement: Prepared Component Opaqueness

Prepared engine representation SHALL remain opaque outside the engine adapter.

Prepared state SHALL NOT cross WIT and SHALL NOT become a portable artifact.

#### Scenario: Cache prepared Component

Given the engine compiles a Component

When the prepared representation is cached

Then the cache remains internal to the Runtime and engine adapter.

---

### Requirement: WIT Import Inspection

The concrete engine integration SHALL support inspection or validation of WIT
imports required by a Component.

Required imports SHALL be matched against Runtime-owned Link Plans.

#### Scenario: Missing import

Given a Component imports an interface absent from the approved Link Plan

When instantiation is attempted

Then instantiation fails before execution.

---

### Requirement: WIT Export Inspection

The concrete engine integration SHALL support identifying Component exports
needed for invocation or validation.

An export SHALL NOT automatically become a globally available Capability.

#### Scenario: Component exports helper interface

Given a Component exports interface X

When the Component is registered

Then X is recorded as an export

But it is not globally linked to other Components without explicit Runtime
policy.

---

### Requirement: Runtime-Owned Link Plan Execution

The concrete engine adapter SHALL translate the Runtime-owned Link Plan into
engine-specific linker configuration.

Only approved imports SHALL be linked.

#### Scenario: Unauthorized import

Given a Component imports filesystem access

And the Runtime Link Plan does not authorize filesystem

When the adapter constructs the engine linker

Then filesystem is not linked.

---

### Requirement: No Ambient WASI

The concrete Component Engine SHALL NOT provide a broad ambient WASI
environment by default.

WASI interfaces SHALL be linked only when explicitly authorized.

#### Scenario: Component expects filesystem

Given a Component expects WASI filesystem access

And filesystem was not authorized

When the Component is linked or instantiated

Then the operation fails closed.

---

### Requirement: Capability Host Adapter

The concrete Component Runtime SHALL support host adapters that expose
Magnetar Runtime endpoints to Component imports.

A host adapter SHALL not expose native Provider or Device handles.

#### Scenario: Component imports test Capability

Given a Component imports an authorized Magnetar test Capability

When it invokes the host function

Then the call reaches the Runtime endpoint

And returns through the WASM Component boundary.

---

### Requirement: Capability Linking Does Not Resolve Provider

Linking a Provider-backed Capability import SHALL not select a concrete
Provider or Device.

#### Scenario: Link Compute import

Given a Component imports `magnetar:compute/run`

When the Component is instantiated

Then the import is linked to a Runtime Compute endpoint

And concrete Provider resolution is deferred until Compute work is submitted.

---

### Requirement: Async Host Call Scope

The concrete engine integration SHALL keep async host-call support internal to
the adapter boundary and SHALL NOT expose a concrete async runtime through
public Magnetar APIs.

The first implementation MAY support only synchronous unit-shaped host
adapters. If a linked Magnetar Capability requires asynchronous execution and
no typed async adapter exists, the adapter SHALL fail closed rather than
blocking a long-running Provider operation on an engine thread.

#### Scenario: Async Runtime endpoint

Given a linked host Capability completes asynchronously

When the Component invokes it

Then the engine adapter either coordinates completion through a typed Runtime
adapter

Or rejects the unsupported async host signature before execution.

---

### Requirement: Instance Store Isolation

Each Component Instance SHALL execute with isolated engine Store state.

#### Scenario: Two instances from one Component

Given one prepared Component definition

When the Runtime creates two Component Instances

Then each instance receives distinct engine execution state.

---

### Requirement: Engine Resource Tables Are Private

Engine resource table entries SHALL remain private implementation details.

They SHALL not become stable Magnetar resource identifiers.

The first implementation SHALL reject WIT resource imports unless an explicit
Runtime resource mapping exists for the linked host adapter.

#### Scenario: Engine creates resource entry

Given a Component call creates a WIT resource

When the engine stores the resource internally

Then the table entry is not exposed as a stable public Magnetar handle.

#### Scenario: Resource import lacks Runtime mapping

Given a Component imports a WIT resource

And no Runtime resource mapping exists for that resource type

When the Component is linked

Then instantiation fails closed.

---

### Requirement: Interruption Support

The concrete engine adapter SHALL support Runtime-requested interruption where
the engine can enforce it.

#### Scenario: Deadline expires

Given a Component invocation exceeds its configured deadline

When the Runtime requests interruption

Then the concrete engine attempts to interrupt execution

And the result is normalized to a Magnetar Component error.

---

### Requirement: Engine Trap Normalization

Engine traps SHALL be mapped into stable Magnetar Component trap errors.

#### Scenario: Component traps

Given Component execution traps inside the concrete engine

When the adapter reports the error

Then callers receive a Magnetar Component trap classification

And not the raw engine trap object.

---

### Requirement: Resource Limit Enforcement

The concrete engine adapter SHALL enforce configured resource limits where
supported.

If a required safety limit cannot be enforced, the adapter SHALL fail closed.

#### Scenario: Required memory limit unsupported

Given Runtime policy requires a Component memory limit

And the engine configuration cannot enforce that limit

When instantiation is attempted

Then instantiation fails rather than silently ignoring the policy.

---

### Requirement: Component Fixture Execution

The repository SHALL include at least one real WASM Component fixture that can
be prepared, linked, instantiated, and invoked by Magnetar tests.

#### Scenario: Execute fixture

Given the test fixture Component imports an authorized test Capability

When the end-to-end test runs

Then the Component invokes the Runtime host adapter successfully.

---

### Requirement: Unauthorized Import Fixture

The repository SHALL include a fixture or test proving that unauthorized imports
fail closed.

#### Scenario: Unauthorized filesystem import

Given a fixture Component requires filesystem access

And the Runtime does not authorize that import

When instantiation or linking occurs

Then the operation fails.

---

### Requirement: Trap Fixture

The repository SHALL include a fixture or test proving that Component traps are
normalized.

#### Scenario: Trap fixture executes

Given a fixture Component intentionally traps

When the Runtime invokes it

Then the error is reported as a stable Magnetar Component trap.

---

### Requirement: Multiple Instance Fixture

The repository SHALL include a fixture or test proving that multiple Component
instances created from one definition do not implicitly share mutable Store
state.

#### Scenario: Isolated instances

Given two instances are created from the same prepared Component

When one instance mutates its local state

Then the other instance does not observe that mutation.

---

### Requirement: Feature-Gated Engine Implementation

If the concrete engine is feature-gated, the repository SHALL provide a CI path
that enables and tests the feature.

#### Scenario: CI runs Component engine tests

Given the Wasmtime engine feature is optional

When CI validates the repository

Then at least one job enables the feature and runs end-to-end Component tests.

---

### Requirement: Component Artifact

A Component Artifact SHALL represent executable WebAssembly Component code.

A Component Artifact SHALL be distinct from a Model Artifact, Provider binary,
Runtime module, and trust policy.

#### Scenario: Classify artifacts

Given a `.wasm` Component and a model weights file

When Magnetar classifies artifacts

Then the `.wasm` file is a Component Artifact

And the weights file is a Model Artifact.

---

### Requirement: Component Artifact Identity

Every Component Artifact SHALL have a Runtime-recognized identity.

The identity SHALL include at least:

- artifact kind
- digest algorithm
- content digest
- logical Component name
- Component version
- manifest version

#### Scenario: Identify Component bytes

Given a local Component artifact

When Magnetar evaluates it

Then its executable bytes are identified by computed digest

And not merely by filename.

---

### Requirement: Content Digest

A Component Artifact SHALL be identified by a content digest.

The digest algorithm SHALL be explicit.

The initial digest algorithm SHOULD be `sha256`.

#### Scenario: Same name different bytes

Given two artifacts have the same logical name and version

But different executable bytes

When Magnetar computes their digests

Then they are different Component Artifact identities.

---

### Requirement: Component Manifest

A Component Artifact SHALL have a manifest describing its declared metadata.

The manifest SHALL include required fields for identity, compatibility, WIT
contracts, and authority declarations.

The initial manifest SHALL use `schema: magnetar-component-artifact` and
`schema_version: 1`.

The initial manifest SHALL include `artifact.kind: component`, a canonical
`artifact.digest`, `component` metadata, `runtime.magnetar.min_version`, `wit`
imports and exports, `capabilities.requires`, `authority.requires`, `source`,
and `signatures`.

The manifest MAY declare `runtime.magnetar.max_version` and per-Capability
`max_version` fields to bound compatibility.

The manifest MAY mark WIT import metadata as optional. Optional WIT import
metadata SHALL remain distinct from the executable Component's required imports.

#### Scenario: Load manifest

Given a Component artifact has a sidecar manifest

When Magnetar loads the artifact

Then the manifest is parsed and validated before execution preparation.

#### Scenario: Manifest schema version

Given a Component artifact manifest uses schema `magnetar-component-artifact`

And schema version `1`

When Magnetar validates the manifest

Then the manifest is accepted as the initial Component Artifact manifest format.

---

### Requirement: Manifest Is Not Trust

A Component manifest SHALL NOT be treated as proof of trust.

A Component Artifact SHALL NOT be trusted merely because its manifest says it is
trusted.

#### Scenario: Manifest claims trust

Given a manifest contains a field or text claiming the artifact is trusted

When Magnetar evaluates the artifact

Then trust is determined by Runtime trust policy

And not by the artifact's own claim.

---

### Requirement: Manifest Digest Match

A manifest SHALL identify the executable artifact digest.

The Runtime SHALL reject a Component Artifact when the manifest digest does not
match the computed digest.

#### Scenario: Digest mismatch

Given a Component artifact's manifest declares digest A

And the Runtime computes digest B

When A and B differ

Then the artifact is rejected before preparation.

---

### Requirement: WIT Manifest Consistency

Declared WIT imports and exports in the manifest SHALL be consistent with the
actual executable Component.

Required WIT imports SHALL match the executable Component's actual required
imports.

Optional WIT import metadata SHALL NOT satisfy or hide an actual required
executable import.

#### Scenario: Manifest omits import

Given a Component executable imports interface X

And its manifest does not declare X

When artifact validation runs

Then validation fails.

#### Scenario: Manifest claims nonexistent export

Given a manifest declares export Y

And the executable Component does not export Y

When artifact validation runs

Then validation fails.

---

### Requirement: Runtime Compatibility Declaration

A Component manifest SHALL declare Runtime compatibility where required.

The Runtime SHALL reject artifacts that declare incompatible Runtime
requirements.

The Runtime SHALL reject artifacts whose optional maximum Runtime version is
lower than the current Runtime version.

#### Scenario: Future Runtime required

Given a Component requires a future Magnetar Runtime version

When the current Runtime cannot satisfy that requirement

Then the artifact is rejected before preparation.

---

### Requirement: Capability Compatibility Declaration

A Component manifest SHALL declare required Magnetar Capabilities.

Capability compatibility SHALL be evaluated before instantiation.

Capability compatibility SHALL reject unsupported major versions and declared
version ranges that do not include the actual WIT import version.

#### Scenario: Unsupported Capability major version

Given a Component requires a Capability major version unsupported by the Runtime

When validation runs

Then the artifact is rejected.

---

### Requirement: Authority Requirement Declaration

A Component manifest SHALL declare requested inference authority.

Authority declarations SHALL be limited to inference-scoped Runtime resources
and Capabilities:

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

Broad workspace, filesystem, network, process, secret, Git, source-control, or
tool-execution authorities SHALL NOT be valid Magnetar Runtime Component
authorities.

#### Scenario: Tokenizer Component authority requested

Given a Component manifest declares `tokenizer-artifact-read`,
`compute-capability`, and `observability-emit`

When artifact validation runs

Then the authority declaration is accepted if all other validation and trust
rules pass.

#### Scenario: Filesystem authority requested

Given a Component manifest declares `filesystem`

When artifact validation runs

Then Magnetar rejects the artifact as outside Runtime scope.

---

### Requirement: Unsupported Authority Fails Closed

Unsupported, unknown, or broad non-inference authority declarations SHALL fail
closed.

Trusted digest, trusted publisher, development mode, or trusted source SHALL
NOT override forbidden Magnetar authority.

#### Scenario: Unknown authority

Given a Component manifest declares an unknown authority kind

When validation runs

Then the artifact is rejected or marked not executable according to policy.

#### Scenario: Trusted artifact requests network

Given a Component digest is trusted

And its manifest requests `network`

When Magnetar validates authority

Then validation fails.

---

### Requirement: Inference-Scoped Authority

Magnetar Component authority SHALL be inference-scoped.

Allowed authority categories MAY include `model-artifact-read`,
`tokenizer-artifact-read`, `prompt-template-read`, `adapter-artifact-read`,
`quantization-artifact-read`, `inference-session-state`,
`generation-session-state`, `kv-cache-access`, `prefix-cache-access`,
`compute-capability`, `generation-capability`, `sampling-capability`,
`observability-emit`, and `runtime-diagnostics`.

#### Scenario: Sampling Component authority

Given a sampling Component requests `sampling-capability`

When validation runs

Then the authority is considered within Magnetar inference scope.

---

### Requirement: Broad Authority Is Forbidden In Magnetar

Magnetar SHALL reject Component authority declarations for filesystem, network,
environment, process, shell, secrets, workspace, git, source-control,
tool-execution, and external-service.

#### Scenario: Git authority requested

Given a Component manifest requests `git`

When Magnetar validates authority

Then the artifact is rejected before preparation.

---

### Requirement: Model Artifact Authority Is Not Filesystem Authority

`model-artifact-read` SHALL allow access only to Runtime-registered model
artifacts authorized for the inference context.

It SHALL NOT grant arbitrary filesystem read access.

#### Scenario: Model artifact access

Given a Component requests a model artifact

And the artifact is registered in the Runtime model artifact registry

When access is granted

Then the Component receives Runtime-mediated model artifact access

And not an unrestricted filesystem path.

---

### Requirement: Tokenizer Artifact Authority Is Not Filesystem Authority

`tokenizer-artifact-read` SHALL allow access only to Runtime-registered
tokenizer artifacts.

#### Scenario: Tokenizer access

Given a Component requests tokenizer data

When Magnetar authorizes access

Then the access is mediated by Runtime artifact identity

And does not expose arbitrary local files.

---

### Requirement: Prompt Template Authority Is Not Filesystem Authority

`prompt-template-read` SHALL allow access only to Runtime-registered prompt or
chat templates.

#### Scenario: Prompt template access

Given a Component requests a chat template

When access is granted

Then it is granted through Runtime-managed inference metadata

And not arbitrary path access.

---

### Requirement: Adapter And Quantization Authority Are Artifact-Scoped

Adapter and quantization authority SHALL refer to Runtime-registered inference
artifacts.

#### Scenario: LoRA adapter access

Given a Component requests an adapter artifact

When Magnetar evaluates the request

Then the artifact must be registered and authorized for the inference context.

---

### Requirement: Cache Authority Is Inference Scoped

KV cache and prefix cache authority SHALL be scoped to authorized inference or
generation sessions.

#### Scenario: KV cache access

Given a Component has `kv-cache-access`

When it accesses cache state

Then access is limited to the session or model context authorized by Runtime
policy.

---

### Requirement: Observability Authority Does Not Grant Network

`observability-emit` SHALL allow Runtime-mediated observation emission only.

It SHALL NOT grant direct network export authority.

#### Scenario: Emit observation

Given a Component emits an inference observation

When the Runtime receives it

Then Runtime observability policy decides whether and where it is exported.

---

### Requirement: Diagnostics Authority Is Redacted

`runtime-diagnostics` SHALL provide inference-related diagnostics only.

Diagnostics SHALL be redacted according to Runtime policy.

#### Scenario: Diagnostic contains client path

Given a diagnostic would include a client workspace path

When Magnetar emits diagnostics

Then the path is omitted or redacted unless an external client policy
explicitly allows disclosure outside Magnetar.

---

### Requirement: No Magnetar Tool Components

Magnetar SHALL NOT authorize Components whose purpose is general tool
execution.

#### Scenario: Shell tool Component

Given a Component's declared role is shell command execution

When Magnetar validates the Component manifest

Then validation fails because the Component is outside inference Runtime scope.

---

### Requirement: Client-Owned Authority Metadata

A manifest SHALL keep client-intended metadata separate from Magnetar Runtime
authority.

Client-owned authority metadata SHALL NOT be interpreted by Magnetar as granted
Runtime authority.

Magnetar SHALL ignore client-owned authority metadata for Runtime authority
granting.

#### Scenario: CLI metadata present

Given a manifest contains a client-specific section describing workspace needs

When Magnetar validates the Component

Then Magnetar ignores that section for Runtime authority

And still validates only Magnetar inference authority.

---

# Trust

### Requirement: Component Trust Status

The Runtime SHALL represent Component Artifact trust status explicitly.

Trust status SHALL include at least:

- unknown
- trusted
- rejected
- quarantined
- revoked

Only trusted Component Artifacts MAY be prepared for execution.

#### Scenario: Unknown artifact

Given a valid Component artifact with no matching trust policy

When artifact validation completes

Then the artifact remains unknown or rejected according to policy

And is not prepared as trusted executable code.

---

### Requirement: Trust Policy Determines Executability

A Component Artifact SHALL be executable only when Runtime trust policy permits
it.

Trust policy MAY evaluate:

- digest
- source
- publisher
- signature metadata
- revocation
- local administrator decision

#### Scenario: Digest allowlist

Given a Component artifact digest is present in the trust allowlist

And no rejection or revocation rule applies

When validation succeeds

Then the artifact may be marked trusted.

---

### Requirement: Rejection Overrides Trust

Rejected or revoked artifact status SHALL override allowlist or publisher trust
unless policy explicitly defines a stronger administrative override.

#### Scenario: Digest both allowed and revoked

Given an artifact digest appears in both allowlist and revoked list

When trust is evaluated

Then revoked status wins

And the artifact is not executable.

---

### Requirement: Publisher Metadata Is Not Sufficient Trust

Publisher identity SHALL be metadata.

It SHALL NOT imply trust unless Runtime policy explicitly trusts that publisher
and all other validation succeeds.

#### Scenario: Known publisher

Given a manifest declares a known publisher

But Runtime policy does not trust that publisher

When trust is evaluated

Then the artifact is not trusted solely because of the publisher field.

---

### Requirement: Source Metadata Is Not Sufficient Trust

Source identity SHALL describe where the artifact came from.

It SHALL NOT imply trust unless Runtime policy explicitly trusts that source and
all other validation succeeds.

#### Scenario: Local file source

Given a Component artifact is loaded from a local directory

When trust is evaluated

Then local presence alone does not make it trusted.

---

### Requirement: Signature Metadata Is Optional and Non-Authoritative

Signature metadata SHALL be optional and non-authoritative.

Signature metadata MAY be present.

An unsupported or unverified signature SHALL NOT make an artifact trusted.

#### Scenario: Signature present but unsupported

Given a manifest contains signature metadata

And the Runtime has no configured verifier for that signature

When trust is evaluated

Then the signature is recorded as unverified

And does not by itself grant trust.

---

### Requirement: Revocation

The Runtime SHALL support revoking Component Artifacts by digest.

A revoked artifact SHALL NOT be prepared or instantiated.

Revocation SHALL prevent future preparation and new instances. Existing active
instances are outside this initial enforcement path and remain governed by
Runtime instance lifecycle policy.

#### Scenario: Previously trusted artifact revoked

Given an artifact digest was previously trusted

And the digest is later revoked

When new preparation is requested

Then preparation is denied.

---

### Requirement: Quarantine

The Runtime SHALL treat quarantined Component Artifacts as non-executable.

The Runtime MAY quarantine invalid or suspicious Component Artifacts.

A quarantined artifact SHALL NOT be prepared or instantiated.

Quarantine SHALL preserve diagnostic status without granting executability.

#### Scenario: Suspicious artifact

Given an artifact fails a trust or integrity check

When policy chooses quarantine

Then diagnostic metadata may be retained

But executable preparation is prohibited.

---

### Requirement: Development Mode

Development mode SHALL be explicit when enabled.

Development mode MAY allow unsigned local Component Artifacts.

Development mode SHALL be explicit.

Development mode SHALL still validate digest, manifest structure, WIT
consistency, and compatibility.

#### Scenario: Local development Component

Given development mode is enabled

And a local unsigned Component has a valid manifest and digest

When trust policy evaluates it

Then it may be accepted according to development policy.

---

### Requirement: File-Based Trust Store

The initial Runtime trust store SHALL support YAML with
`schema: magnetar-component-trust` and `schema_version: 1`.

The trust store SHALL support `trusted_digests`, `rejected_digests`,
`revoked_digests`, `quarantined_digests`, `trusted_publishers`,
`trusted_sources`, and
`development.allow_unsigned_local`.

The trust store SHALL remain separate from the Component Artifact manifest.

---

### Requirement: Artifact Cache

The Runtime MAY maintain a local Component Artifact cache.

A Component Artifact cache SHALL key entries by digest.

A Component Artifact cache SHALL verify digest integrity when loading entries.

A Component Artifact cache SHALL NOT make an artifact trusted merely because an
entry exists locally.

Validation metadata SHALL remain separate from executable trust.

#### Scenario: Cached artifact

Given Component bytes exist in a local cache

When Magnetar retrieves the cached entry

Then Magnetar verifies the digest before use

And cache presence alone does not mark the artifact trusted.

#### Scenario: Minimal trust store

Given a trust store lists a digest under `trusted_digests`

And the same digest is not rejected or revoked

When a matching artifact passes manifest, digest, WIT, compatibility, and
authority validation

Then policy may mark the artifact trusted.

#### Scenario: Manifest cannot self-trust

Given a manifest includes publisher metadata or trust-like text

When the trust store has no matching trust rule

Then the artifact is not trusted.

---

# Lifecycle

### Requirement: Artifact Validation Before Preparation

A Component Artifact SHALL be validated and trusted before ComponentEngine
preparation.

Validation SHALL follow this order: compute digest, load manifest, validate
manifest schema, compare digest, inspect actual WIT imports and exports, compare
manifest WIT to actual WIT, check Runtime compatibility, check Capability
compatibility, validate authority declarations, evaluate trust policy, and only
then produce a trusted Component Artifact for `ComponentEngine::prepare`.

#### Scenario: Prepare Component

Given a local `.wasm` file has not been validated

When preparation is requested

Then Runtime first performs artifact validation and trust evaluation.

---

### Requirement: Artifact State Is Distinct from Prepared State

A trusted Component Artifact SHALL NOT automatically be a Prepared Component.

#### Scenario: Trust succeeds but compilation fails

Given artifact validation and trust evaluation succeed

But the ComponentEngine cannot prepare the artifact

When preparation runs

Then the artifact remains trusted

And preparation fails with a separate Component preparation error.

---

### Requirement: Artifact State Is Distinct from Instance State

Trusting or preparing a Component Artifact SHALL NOT automatically instantiate a
Component.

#### Scenario: Trust artifact

Given a Component Artifact is marked trusted

When no instantiation is requested

Then no Component Instance is created.

---

### Requirement: Artifact Digest Attached to Component Definition

A Component Definition created from a Component Artifact SHALL retain the
artifact digest.

#### Scenario: Inspect Component definition

Given a Component Definition was created from a trusted artifact

When Runtime observability or diagnostics refer to it

Then the artifact digest can be included as identity metadata.

---

# Tachyon Boundary

### Requirement: Vendor-Neutral Component Source

The Component Artifact model SHALL be independent from any one distribution
source.

Tachyon MAY be a future source of Component Artifacts, but Magnetar SHALL NOT
require Tachyon to validate or execute local Components.

#### Scenario: Local Component install

Given a Component artifact is installed locally without Tachyon

When Magnetar validates and trusts it

Then it can be prepared according to the same artifact model.

---

### Requirement: Tachyon Distribution Does Not Imply Trust

If Tachyon supplies a Component Artifact, Magnetar SHALL still validate the
artifact locally.

#### Scenario: Tachyon-provided artifact

Given Tachyon provides a Component Artifact

When Magnetar receives it

Then Magnetar computes digest, validates manifest consistency, evaluates
compatibility, and applies trust policy before execution.

---

# Observability

### Requirement: Component Artifact Observability

The Runtime SHALL keep Component Artifact observations non-authoritative.

The Runtime SHOULD emit structured observations for Component Artifact
validation and trust decisions.

Observations MAY include:

- artifact source
- digest algorithm
- digest
- manifest validation result
- WIT consistency result
- compatibility result
- trust decision
- revocation
- quarantine

#### Scenario: Artifact rejected

Given a Component Artifact fails digest validation

When Runtime observability records the event

Then the observation identifies the stable failure category

And does not expose secrets or private signature material.

# Component Distribution Contract

### Requirement: Component Artifact Package

A Component Artifact Package SHALL be the distribution unit for a
Magnetar-compatible Inference Component.

It SHALL contain or reference:

- executable Component bytes
- Component manifest
- declared digest
- source identity
- optional publisher identity
- optional signature metadata
- optional provenance metadata

#### Scenario: Receive package

Given a Component Distribution Source provides a package

When Magnetar receives it

Then the Runtime treats it as untrusted input until local validation completes.

---

### Requirement: Component Distribution Source

A Component Distribution Source SHALL provide Component Artifact Packages or
resolvable package metadata.

Source identity SHALL be metadata and SHALL NOT imply trust.

#### Scenario: Source is known

Given a package comes from a known source

When trust is evaluated

Then source identity may be considered by policy

But does not automatically mark the package trusted.

---

### Requirement: Source-Declared Digest Is A Claim

A digest provided by a Component Distribution Source SHALL be verified locally.

#### Scenario: Source digest mismatch

Given a source declares digest A

And Magnetar computes digest B from received bytes

When A and B differ

Then the package is rejected.

---

### Requirement: Manifest Validation For Distributed Packages

A distributed Component Package SHALL pass Component manifest validation before
preparation.

#### Scenario: Invalid distributed manifest

Given a package contains malformed manifest data

When Magnetar validates the package

Then the package is rejected before ComponentEngine preparation.

---

### Requirement: Distributed Package WIT Consistency

A distributed package SHALL pass WIT consistency validation between manifest and
actual executable Component.

#### Scenario: Manifest hides import

Given the executable Component imports interface X

And the distributed manifest omits X

When validation runs

Then the package is rejected.

---

### Requirement: Distributed Package Authority Is Inference Scoped

A distributed package SHALL request only Magnetar inference-scoped Component
authority.

#### Scenario: Distributed Component requests Git

Given a distributed package manifest requests Git authority

When Magnetar validates the package

Then the package is rejected as outside Runtime scope.

---

### Requirement: Package Does Not Grant Authority

A Component Artifact Package SHALL NOT grant authority even when it declares
requested authority.

It SHALL NOT grant that authority.

#### Scenario: Package declares compute

Given a package declares `compute-capability`

When Magnetar validates it

Then Runtime policy still decides whether the Component receives a Compute
endpoint in its Link Plan.

---

### Requirement: Package Does Not Imply Trust

A Component Artifact Package SHALL NOT be trusted merely because it exists,
came from a source, or contains a manifest.

#### Scenario: Package from cache

Given a package exists in the local cache

When Magnetar loads it

Then the package still requires integrity and trust validation.

---

### Requirement: Distributed Package Revocation

A distributed package SHALL be rejected if its artifact digest is revoked.

#### Scenario: Revoked package received

Given a source provides a package whose digest is revoked

When Magnetar validates it

Then validation fails before preparation.

---

### Requirement: Distributed Package Compatibility

A distributed package SHALL be checked for Runtime, Capability, Component
Engine, WIT, and inference authority compatibility.

#### Scenario: Package requires unsupported Compute major version

Given a package requires an unsupported Compute major version

When compatibility validation runs

Then the package is rejected.

---

### Requirement: Optional Provenance Metadata

A distributed package SHALL treat provenance metadata as optional,
non-authoritative metadata when present.

Provenance metadata SHALL NOT imply trust by itself.

#### Scenario: Provenance present

Given a package includes source repository and build commit metadata

When trust is evaluated

Then the metadata may be recorded

But trust still depends on Runtime policy.

---

### Requirement: Optional Signature Metadata

A distributed package SHALL treat signature metadata as optional,
non-authoritative metadata when present.

Unsupported or unverified signatures SHALL NOT imply trust by themselves.

#### Scenario: Signed package without trust root

Given a package includes signature metadata

And Runtime has no configured verifier or trust root

When trust is evaluated

Then the package is not trusted solely because a signature is present.

---

### Requirement: Cache Integrity

If Magnetar caches Component packages, cached content SHALL be verified before
use.

#### Scenario: Corrupted cache entry

Given cached bytes no longer match the expected digest

When Magnetar loads the cache entry

Then the cache entry is rejected.

---

### Requirement: Offline Distribution

The distribution contract SHALL support local/offline Component packages.

#### Scenario: Offline local package

Given a trusted local Component package is available

And no external network is available

When Magnetar validates it

Then validation can succeed without contacting a remote service.

---

### Requirement: Distribution Is Not Instantiation

Receiving or resolving a package SHALL NOT instantiate a Component.

#### Scenario: Package fetched

Given Magnetar fetches a Component package

When no instantiation is requested

Then no Component Instance is created.

---

### Requirement: Distribution Is Not Trust

Resolving a logical Component identity to an artifact digest SHALL NOT mark the
artifact trusted.

#### Scenario: Version resolved

Given a source resolves a version request to a digest

When Magnetar receives the candidate

Then Magnetar still validates and applies trust policy before preparation.
