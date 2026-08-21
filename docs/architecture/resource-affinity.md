# Resource Affinity

Resource Affinity is host-side metadata that records which runtime resources
may safely participate in the same native call. It prevents an opaque handle
created by one Provider, Device, capability implementation, artifact bundle,
or Runtime from being passed to an incompatible implementation.

Affinity is a runtime constraint, not a scheduling hint. Once a live resource
contains a binding, that binding remains authoritative until the resource is
explicitly destroyed, recreated, or transferred by a future transfer
contract. A fallback classification never permits the Runtime to ignore a
live binding.

This change provides the runtime foundation only. It does not change
`magnetar:compute@1.1.0`, expose Provider or Device identity through WIT, or
create model, tokenizer, prompt-template, or generation WIT contracts. The
adapter examples below describe how a future WASM host integration should use
the runtime types around its native handles.

## Data model

`ResourceAffinity` is an immutable set of facts. Its builder-style `with_*`
methods return a new value instead of mutating an affinity already attached to
a resource. `AffinityResource<T>` keeps those facts beside an opaque native
value without placing the value, a pointer, a queue, or Provider-private state
inside the affinity descriptor.

| Dimension | Stable value | Meaning |
| --- | --- | --- |
| Execution context | `ExecutionContextId` | Process-local identity of the `Runtime` that created or resolved the resource. It is not a persistence or migration identity. |
| Provider | `ProviderBinding` | Registered Provider name that owns the resource. Provider names are unique only within a Runtime. |
| Device | `DeviceBinding` containing a `DeviceId` | Device on which the resource resides. The Provider registry remains authoritative for Device ownership. |
| Capability | `CapabilityBinding` | Package-qualified `CapabilityId` and the exact version used to create the live resource. |
| Artifact | `ArtifactBinding` | A role and canonical content fingerprint. The role states which artifacts are comparable. |
| Affinity group | `AffinityGroupId` | Runtime-local identity of one dependent state chain, such as a submitted operation or generation session. |
| Recovery | `FallbackClass` | The most restrictive recovery behavior required by the resource chain. |

The three recovery classes are ordered from least to most restrictive:

```text
Transparent < Restartable < ProviderPinned
```

- `Transparent` means an equivalent implementation may be selected before
  Provider-owned state or observable output exists.
- `Restartable` means the caller may explicitly recreate the state from
  replayable inputs. It does not make a live handle portable.
- `ProviderPinned` means the current resource must stay with its Provider; an
  unavailable Provider causes the operation to fail or the resource to be
  explicitly torn down.

## Core rules

1. A host adapter attaches affinity when it wraps a newly created opaque
   resource. A Provider-owned resource records its Provider, and a
   device-resident resource also records its Device.
2. Every resource created by a resolved Capability records that Capability's
   exact ID and version. Semantic-version matching is used only before the
   resource exists.
3. All input affinities are aggregated before a Provider-specific handle is
   dereferenced or passed to native code. Aggregation never resolves conflicts
   by letting the last input win.
4. A missing binding is unconstrained. When another input supplies that
   binding, aggregation preserves it.
5. Distinct Capability IDs and distinct artifact roles are cumulative, not
   conflicting. They allow one chain to describe several contracts and
   artifacts.
6. A Device binding also constrains the Provider indirectly. The registry must
   contain the Device, and its registered owner must match any explicit
   Provider binding.
7. Execution-context IDs and affinity-group IDs are runtime-local. They must
   never be serialized as globally meaningful identities or reused to justify
   cross-Runtime handle access.
8. Errors contain stable identifiers, versions, roles, and fingerprints for
   diagnostics. They do not contain native handles or Provider-private
   objects.
9. Existing stateless resolution remains available. Code that consumes live
   resources uses `Runtime::resolve_with_affinity` instead.

## Compatibility matrix

The matrix applies while `AffinityConstraints` aggregates all resources for a
single dependent call.

| Dimension | Only one input binds it | Both bind the same value | Both bind different values | Result |
| --- | --- | --- | --- | --- |
| Execution context | Preserve the binding | Compatible | Incompatible | Context-mismatch error with both IDs |
| Provider | Preserve the binding | Compatible | Incompatible | Provider-mismatch error with both names |
| Device | Preserve the binding | Compatible | Incompatible | Device-mismatch error with both IDs |
| Capability, same ID | Preserve the exact version | Compatible only when versions are exactly equal | Incompatible, even when versions are semver-compatible | Capability-mismatch error with the ID and both exact versions |
| Capability, different IDs | Preserve both entries | Not applicable | Compatible | Both bindings remain in the aggregate |
| Artifact, same role | Preserve the fingerprint | Compatible | Incompatible | Artifact-mismatch error with the role and both fingerprints |
| Artifact, different roles | Preserve both entries | Not applicable | Compatible | Both bindings remain in the aggregate |
| Affinity group | Preserve the binding | Compatible | Incompatible | Group-mismatch error with both IDs |
| Fallback class | Keep the declared class | Keep that class | Compatible | Keep the more restrictive class |

There are two additional cross-dimension checks:

- A bound Device must be registered. If both Device and Provider are bound,
  the Device's registered Provider must equal the Provider binding.
- A Provider selected for a dependent call must implement the requested
  Capability version and every constraint needed by that call. A bound but
  unavailable Provider is an affinity failure, not an invitation to fall back.

### Capability versions

Version handling deliberately differs before and after resource creation:

| Phase | Version rule |
| --- | --- |
| No live binding for the requested Capability ID | Select the best semantically compatible registered version. |
| Provider already bound | Search compatible versions implemented by that Provider; do not select a global version and filter afterward. |
| Live resource binds the requested Capability ID | Require its exact version. A newer compatible version is not interchangeable with the live handle. |

For example, a tensor created by `magnetar:compute/run@1.1.0` remains bound to
`1.1.0`. A Provider advertising `1.2.0` cannot consume it merely because
`1.2.0` satisfies an initial request for `1.1.0`.

### Artifact roles

Artifact fingerprints are compared only within the same role. A model digest
and a tokenizer digest normally differ and therefore use different roles. If
their compatibility depends on belonging to the same release, both resources
also declare an identical shared role, for example `model-bundle` or
`compatibility-manifest`.

```text
loaded model:  model = sha256:<model digest>
               model-bundle = sha256:<manifest digest>

tokenizer:     tokenizer = sha256:<tokenizer digest>
               model-bundle = sha256:<manifest digest>
```

The unequal `model` and `tokenizer` fingerprints do not conflict. Unequal
`model-bundle` fingerprints do. A follow-up contract must define the canonical
fingerprint format and the authoritative source of each shared role; file
paths, timestamps, cache keys, and display names are not content identity.

## Resolution lifecycle

For a dependent call, the Runtime performs the following logical sequence:

1. Aggregate every input `ResourceAffinity`. Reject a conflict before selecting
   or invoking a Provider.
2. Confirm that any execution-context binding is the current Runtime's ID.
3. Resolve a Device binding through `ProviderRegistry::provider_for_device` and
   reconcile it with any explicit Provider binding.
4. If a Provider is bound, require that live Provider and search compatible
   Capability versions inside it. Do not consider another Provider.
5. If no Provider is bound, use normal compatible Provider selection. This is
   the only stage where transparent Provider fallback is possible.
6. Return one `AffinityResolution` containing the selected Provider, exact
   Capability, and the affinity to attach to the newly created resource.

The public entry point is additive:

```rust,ignore
let resolution = runtime.resolve_with_affinity(
    &required_capability,
    &dependency_affinities,
    FallbackClass::ProviderPinned,
)?;

let resource = AffinityResource::new(native_value, resolution.into_affinity());
```

An adapter that only needs to validate or combine facts can use
`AffinityConstraints::try_from_affinities(...)`. It must still use
`resolve_with_affinity` before invoking a Provider for a dependent call.

### Affinity groups

Groups identify dependent state, not every reusable input:

- A resolution with no resource dependencies creates no group. Independently
  created tensors, graphs, loaded models, tokenizers, and templates can
  therefore remain reusable.
- A resolution with dependencies inherits their one existing group. If none
  of the dependencies has a group, the Runtime creates a new group for the
  composite resource.
- Input resources are immutable and are not retroactively added to the new
  group. A shared tensor or loaded model can feed several independent
  operations or sessions.
- Dependencies carrying different groups are incompatible. Combining two
  active operations or generation sessions is rejected even if their Provider
  and Device happen to match.

For example, submitting an ungrouped graph and ungrouped tensors creates an
operation with a new group. The inputs remain ungrouped. Output tensors derived
from that operation carry the operation's group.

## Compute host-adapter examples

The following Rust-like sketches use the runtime API names but leave
WASM-engine and Provider-private calls schematic. A concrete host adapter owns
the native types and maps them to the existing WIT `tensor`, `graph`, and
`operation` resources.

```rust,ignore
type HostTensor = AffinityResource<NativeTensor>;
type HostGraph = AffinityResource<NativeGraph>;
type HostOperation = AffinityResource<NativeOperation>;
```

The Component sees only the opaque WIT resource and portable records such as
`tensor-descriptor`. The `AffinityResource<T>` envelope and all affinity
inspection remain native host concerns.

### Tensor creation

A tensor created without resource dependencies is resolved without a group.
The adapter chooses a Device owned by the selected Provider, creates the native
tensor there, and adds that Device binding to the returned affinity.

```rust,ignore
fn create_tensor(
    runtime: &Runtime,
    descriptor: TensorDescriptor,
) -> Result<HostTensor, AdapterError> {
    let resolution = runtime.resolve_with_affinity(
        &compute_capability(),
        &[],
        FallbackClass::ProviderPinned,
    )?;

    // `select_device` must return a Device registered to this Provider.
    let device = select_device(resolution.provider(), &descriptor)?;
    let native = compute_adapter(resolution.provider())
        .create_tensor(device.id(), &descriptor)?;

    let affinity = resolution
        .into_affinity()
        .with_device(DeviceBinding::new(device.id().clone()));
    Ok(AffinityResource::new(native, affinity))
}
```

The resulting affinity records the Runtime execution context, selected
Provider, selected Device, and exact `magnetar:compute/run` version. No Device
identifier is added to the WIT tensor descriptor.

If tensor contents are replayable from host memory, application policy may
remember how to recreate the tensor. That does not make the current native
tensor movable: a call using it remains bound to its Provider and Device.

### Graph creation

A reusable Provider-owned graph that has no resource dependencies can be
created with an empty dependency list and remains ungrouped. If graph
compilation consumes tensors or another opaque resource, their affinities must
participate in resolution first:

```rust,ignore
fn compile_graph(
    runtime: &Runtime,
    source: &GraphSource,
    constants: &[&HostTensor],
) -> Result<HostGraph, AdapterError> {
    let dependencies = constants
        .iter()
        .map(|tensor| tensor.affinity())
        .collect::<Vec<_>>();

    let resolution = runtime.resolve_with_affinity(
        &compute_capability(),
        &dependencies,
        FallbackClass::ProviderPinned,
    )?;

    // Resolution has already proven that every native constant is consumable
    // by this exact Provider and capability implementation.
    let native = compute_adapter(resolution.provider())
        .compile_graph(source, constants.iter().map(|tensor| tensor.value()))?;

    Ok(AffinityResource::new(native, resolution.into_affinity()))
}
```

When `constants` is non-empty and ungrouped, the new graph receives a group;
the constant tensors remain ungrouped. If one constant already has a group,
the graph inherits it. Constants from different groups fail before native
compilation starts.

### Operation submission and outputs

`submit(graph, inputs)` is the critical aggregation boundary. The adapter
collects the graph and every input tensor affinity, resolves the complete set,
and only then exposes native handles to the selected Provider.

```rust,ignore
fn submit(
    runtime: &Runtime,
    graph: &HostGraph,
    inputs: &[&HostTensor],
) -> Result<HostOperation, AdapterError> {
    let mut dependencies = Vec::with_capacity(inputs.len() + 1);
    dependencies.push(graph.affinity());
    dependencies.extend(inputs.iter().map(|tensor| tensor.affinity()));

    let resolution = runtime.resolve_with_affinity(
        &compute_capability(),
        &dependencies,
        FallbackClass::ProviderPinned,
    )?;

    let native = compute_adapter(resolution.provider()).submit(
        graph.value(),
        inputs.iter().map(|tensor| tensor.value()),
    )?;

    Ok(AffinityResource::new(native, resolution.into_affinity()))
}
```

Provider, Device, context, exact Capability version, artifact, or group
conflicts are therefore reported before `submit` receives any native handle.
If the inputs are ungrouped, the returned operation receives a fresh group.
If the graph already belongs to a group, the operation inherits it.

`status`, `await-completion`, and `cancel` route to the Provider that owns the
operation; they do not perform stateless Provider selection. When
`take-outputs` returns native tensors, the adapter preserves the operation's
affinity on every output:

```rust,ignore
fn wrap_outputs(operation: &HostOperation, outputs: Vec<NativeTensor>)
    -> Vec<HostTensor>
{
    outputs
        .into_iter()
        .map(|output| {
            AffinityResource::new(output, operation.affinity().clone())
        })
        .collect()
}
```

The output tensors consequently remain on the operation's Provider, Device,
context, exact Compute version, and group. Switching Providers requires an
explicit download/copy and recreation flow defined by a future transfer
contract; relabeling the existing handles is never valid.

## Future model and generation guidance

The types in this section do not exist as WIT contracts in this change. The
examples are requirements for future proposals and host adapters, not claims
that model loading, tokenization, prompt formatting, or generation is already
implemented.

### Reusable model, tokenizer, and template resources

Initial reusable resources should normally be resolved without resource
dependencies so that they remain ungrouped. Each adapter then attaches its own
artifact role plus an explicit shared compatibility role.

| Future resource | Own artifact facts | Shared compatibility facts | Typical live-state classification |
| --- | --- | --- | --- |
| Loaded model | `model` fingerprint; exact model-loading Capability | `model-bundle` or `compatibility-manifest` | `ProviderPinned`; also Device-bound when weights are resident |
| Tokenizer | `tokenizer` and, when separately defined, vocabulary/normalization fingerprints; exact tokenization Capability | Same bundle or manifest role as the model | `Transparent` for a pure stateless call with identical fingerprints; an opaque tokenizer or incremental decoder remains bound |
| Prompt template | `prompt-template` fingerprint and exact prompt-formatting Capability | Same bundle or manifest role; message/tool schema identity when relevant | `Transparent` only for deterministic stateless rendering with identical inputs and fingerprints |

A loaded model and tokenizer can therefore have different own-content digests
while still proving compatibility through one equal `model-bundle` binding.
Future changes must specify where that shared fingerprint comes from and which
normalization, special-token, vocabulary, template, tool-schema, and reasoning
policy facts it covers.

An incremental tokenizer decoder is stateful even if whole-input encode or
decode is stateless. Its adapter should create a dependent resource from the
tokenizer affinity, preserve the resulting group, and classify recovery as
`Restartable` only when replaying the complete token history is explicitly
supported. Otherwise it is `ProviderPinned`.

### Generation sessions

Before creating a generation session, a future adapter must aggregate the
loaded model, tokenizer, prompt template, and any other opaque input resources
in one affinity-aware resolution:

```rust,ignore
let dependencies = [
    loaded_model.affinity(),
    tokenizer.affinity(),
    prompt_template.affinity(),
];

let resolution = runtime.resolve_with_affinity(
    &causal_generation_capability,
    &dependencies,
    FallbackClass::ProviderPinned,
)?;

let native_session = generation_adapter(resolution.provider())
    .start(loaded_model.value(), tokenizer.value(), prompt_template.value(), request)?;

let session = AffinityResource::new(
    native_session,
    resolution.into_affinity(),
);
```

If the reusable inputs are ungrouped, the session receives a fresh group while
the model, tokenizer, and template remain reusable by other sessions. A
session derived from an already grouped dependency inherits that group.

The aggregation must reject, before session creation:

- a model and tokenizer with different shared bundle or manifest
  fingerprints;
- a template incompatible with the declared bundle or request schema;
- resources from another Runtime execution context;
- different bound Providers or Devices;
- different exact versions for the same Capability ID; and
- dependencies belonging to different active groups.

The current affinity-aware resolver selects one coherent Provider. A future
architecture that intentionally splits model execution, tokenization, or
formatting across Providers needs an explicit multi-Provider transfer and
trust-boundary design; it must not bypass the single-Provider checks.

Once a generation session contains model state, KV cache, RNG state, or has
emitted an event, it is Provider-pinned. Provider loss ends the current
operation. An explicit restart may create a new session only from replayable
inputs and must define prompt replay, accepted-prefix handling, sampling policy
and seed, event sequence numbers, and duplicate-output behavior. Even then,
the contract must not promise bit-identical continuation unless it can prove
all relevant execution semantics.

### Requirements for follow-up contracts

Each future model, tokenizer, template, or generation proposal must define:

1. The WIT resource lifecycle and which objects are stateless, restartable, or
   Provider-pinned.
2. The exact Capability IDs and versions recorded on each live resource.
3. Canonical artifact roles, fingerprint formats, and shared manifest source.
4. Device placement and explicit upload, download, copy, unload, or recreation
   boundaries.
5. Which dependency set creates or inherits an affinity group.
6. Replay requirements and the point after which fallback is no longer
   transparent.
7. Structured portable errors, while keeping native handles, paths,
   credentials, queues, caches, and backend diagnostics out of WIT.

Until those contracts exist, `ResourceAffinity` and `AffinityResource<T>` are
host-side building blocks only. They must not be used to invent placeholder
model or generation resources under `magnetar:compute@1.1.0`.
