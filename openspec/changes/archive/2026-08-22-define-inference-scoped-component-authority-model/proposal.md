# Define Inference-Scoped Component Authority Model

## Why

The previous Component Artifact and Trust Model introduced authority
declarations for executable WebAssembly Components.

That model correctly established that Component Artifacts must declare the
authority they request and that the Runtime must not grant authority merely
because a manifest declares it. However, the declared authority categories were
too broad for Magnetar's intended responsibility.

Magnetar is an inference runtime. It must not become a general-purpose agent
tool runtime.

The client, such as `magnetar-cli`, owns agent orchestration and workspace
interaction. Broad authorities such as filesystem access, network access,
process execution, shell execution, secrets access, workspace access, Git
access, source-control access, arbitrary tool execution, and external service
access SHALL NOT be part of Magnetar's Component authority model.

Those authorities belong to clients and orchestrators outside the Magnetar
inference Runtime.

Final boundary:

```text
magnetar-cli
   |
   +-- reads files
   +-- inspects Git
   +-- calls network services
   +-- manages secrets
   +-- executes tools
   +-- orchestrates agents
   |
   v
Magnetar
   |
   +-- receives prompt / context
   +-- loads authorized models
   +-- tokenizes
   +-- generates
   +-- streams tokens
   +-- manages KV cache
   +-- executes compute
   +-- returns result
```

This change narrows the Component authority model to inference-scoped
authorities only.

Magnetar Components may participate in inference execution. They may not become
arbitrary tools with ambient access to the user's machine, workspace, network,
secrets, or source-control state.

## What Changes

This change defines the Magnetar Component authority model as
inference-scoped.

The canonical rule becomes:

```text
Magnetar SHALL provide inference authority only.

Clients such as magnetar-cli MAY provide workspace, tool, filesystem, network,
Git, and secret authority outside Magnetar.
```

### Magnetar Runtime Scope

Magnetar SHALL be scoped to local inference execution.

Magnetar MAY own:

- model artifact access
- tokenizer artifact access
- prompt template access
- adapter artifact access
- quantization artifact access
- inference session state
- generation session state
- KV cache access
- prefix cache access
- compute Capability access
- generation Capability access
- sampling/logits processing
- Provider access through Runtime resolution
- Device access through Runtime resolution
- observability emission
- Runtime diagnostics
- inference-local temporary resources

Magnetar SHALL NOT own general-purpose authority for:

- arbitrary filesystem access
- workspace traversal
- Git operations
- shell execution
- process spawning
- arbitrary network access
- secret retrieval
- source-code editing
- external service calls
- user project mutation
- general agent tool execution

### Client Scope

A client such as `magnetar-cli` MAY own workspace selection, file reading and
writing, Git integration, network access, secret access, command execution,
tool orchestration, coding-agent behavior, project mutation, user prompts and
approvals, and external API calls.

When such a client needs inference, it calls Magnetar. When such a client needs
tools, it executes or authorizes them itself.

Magnetar SHALL NOT need to know how the client obtained the prompt, retrieved
files, invoked Git, or managed secrets.

### Component Authority Categories

Magnetar Component manifests SHALL use inference-scoped authority categories:

```text
model-artifact-read
tokenizer-artifact-read
prompt-template-read
adapter-artifact-read
quantization-artifact-read
inference-session-state
generation-session-state
kv-cache-access
prefix-cache-access
compute-capability
generation-capability
sampling-capability
observability-emit
runtime-diagnostics
```

### Forbidden Magnetar Component Authorities

A Component Artifact manifest SHALL NOT request the following authorities as
Magnetar Runtime authorities:

```text
filesystem
network
environment
process
shell
secrets
workspace
git
source-control
tool-execution
external-service
```

If these appear in a Magnetar Component manifest, validation SHALL fail. The
default behavior SHALL be fail-closed.

### Artifact Manifest Correction

The previous broad tool-like authority example is no longer valid for Magnetar
Runtime execution:

```yaml
authority:
  requires:
    - filesystem
    - network
    - secrets
    - git
    - workspace
```

The target Magnetar Component Artifact manifest is:

```yaml
schema: magnetar-component-artifact
schema_version: 1

artifact:
  kind: component
  digest:
    algorithm: sha256
    value: "sha256:0123456789abcdef..."

component:
  name: "magnetar.examples.tokenizer"
  version: "0.1.0"
  description: "Tokenizer Component fixture"
  role: "tokenizer"

runtime:
  magnetar:
    min_version: "0.1.0"

wit:
  imports:
    - package: "magnetar:compute"
      interface: "run"
      version: "2.0.0"
  exports:
    - package: "magnetar:tokenizer"
      interface: "tokenize"
      version: "1.0.0"

capabilities:
  requires:
    - id: "magnetar:compute/run"
      version: "2.0.0"

authority:
  requires:
    - tokenizer-artifact-read
    - compute-capability
    - observability-emit

publisher:
  id: "local-dev"
  name: "Local Development"

source:
  kind: "local"
  uri: "./fixtures/tokenizer.component.wasm"

signatures: []
```

### No Ambient Authority

The previous fail-closed rule remains. A Component receives only
inference-scoped interfaces explicitly linked by Magnetar.

Unlinked inference authority is unavailable. Non-inference authority is outside
Magnetar and unavailable through Magnetar.

### Capability Linking

Inference authority is realized through Runtime-owned Capability linking.

For example:

```text
compute-capability
    -> magnetar:compute/run

generation-capability
    -> magnetar:generation/...

observability-emit
    -> magnetar:observability/...
```

Linking a Capability SHALL still not select a concrete Provider or Device.
Provider and Device resolution remain Runtime-owned.

### Inference Resource Boundaries

Model, tokenizer, prompt-template, adapter, and quantization artifact
authorities are not arbitrary filesystem authorities. Components may access
only Runtime-registered inference artifacts authorized for the inference
context.

KV cache and prefix cache authority SHALL be scoped to Runtime inference state.
A Component SHALL NOT inspect unrelated sessions, unrelated users, unrelated
models, or client workspace state.

Observability authority MAY allow a Component to emit inference-related
observations, but it SHALL NOT grant network export authority. Runtime
observability policy controls export destinations.

Runtime diagnostics authority MAY allow a Component to produce or consume
inference diagnostics. Diagnostics SHALL be redacted according to Runtime
policy and SHALL NOT expose secrets, raw prompts beyond policy, filesystem
paths beyond policy, native handles, Provider internals, or client workspace
contents unless explicitly allowed by an external client policy.

### magnetar-cli Boundary

`magnetar-cli` MAY implement Codex-like behavior outside the Magnetar
inference Runtime.

For example, `magnetar-cli` may:

1. read workspace files
2. inspect Git state
3. call Magnetar for inference
4. receive generated text or tool suggestions
5. ask the user for approval
6. write files or execute Git commands

Magnetar itself only performs inference.

### Tachyon Boundary

Tachyon MAY distribute Magnetar-compatible inference Components. Tachyon
distribution SHALL NOT imply that Magnetar accepts broad tool authority.

If Tachyon supplies a Component, Magnetar still validates digest, manifest, WIT
imports/exports, Runtime compatibility, Capability compatibility,
inference-authority declarations, and trust policy.

Tachyon distributes. Magnetar validates and executes inference.

### Compatibility With Previous Change

This change supersedes the broad authority examples from
`define-component-artifact-and-trust-model`.

That change remains implemented. This change narrows its authority taxonomy.

## Non-Goals

This change does not:

- implement `magnetar-cli`
- define CLI workspace authority
- define CLI Git integration
- define CLI secret access
- define CLI process execution
- define CLI network policy
- define a general agent tool runtime
- define file editing tools
- define shell tools
- define arbitrary WASI access
- grant filesystem access to Magnetar Components
- grant network access to Magnetar Components
- grant secrets access to Magnetar Components
- grant Git access to Magnetar Components
- define remote distribution protocol
- define Tachyon Component distribution protocol
- implement model inference itself

## Impact

Magnetar's scope becomes clearer and safer.

Magnetar is an inference Runtime. `magnetar-cli` or another client is
responsible for agentic interaction with the outside world.

The Component authority model becomes simpler:

```text
Allowed in Magnetar:
    inference resources and inference Capabilities

Not allowed in Magnetar:
    general tools and external-world authority
```

This prevents Magnetar from accidentally becoming a broad sandboxed operating
environment and keeps future security policy smaller and more auditable.
