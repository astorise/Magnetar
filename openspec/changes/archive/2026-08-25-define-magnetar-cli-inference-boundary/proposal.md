# Define magnetar-cli Inference Boundary

## Why

Magnetar Runtime now exposes a Runtime Inference API.

The next risk is architectural drift between `magnetar-cli` and
`magnetar-runtime`.

The CLI needs to be powerful:

- read files
- inspect workspace
- assemble prompts
- run Git operations
- call tools
- manage secrets
- use network where allowed
- provide user interaction
- orchestrate agent workflows
- call Runtime Inference API

But the Runtime must remain inference-only.

If workspace, Git, tools, secrets, shell, or agent orchestration leak into the
Runtime, Magnetar will no longer be a clean inference substrate.

This change defines the `magnetar-cli` inference boundary.

## What Changes

This change defines `magnetar-cli` as the first-party client/runtime that owns
user-facing workflows around inference.

`magnetar-cli` MAY:

- parse CLI commands
- resolve user-facing model names
- read files where user requested
- inspect workspace
- assemble prompt context
- manage chat UX
- call Git
- call network services where policy allows
- access secrets through CLI-owned secret policy
- invoke tools
- execute shell/processes where policy allows
- orchestrate agents
- call Runtime Inference API
- display streaming generation output
- store CLI session metadata

`magnetar-runtime` SHALL NOT absorb those responsibilities.

## Primary Boundary

The authoritative boundary is:

```text
Magnetar Runtime = inference runtime
magnetar-cli = client/workspace/tool/agent runtime
```

The CLI prepares inference inputs.

Runtime performs inference.

The CLI decides what to do with inference outputs.

## CLI Commands

The first CLI-facing commands MAY include:

```text
magnetar run <model> <prompt>
magnetar chat <model>
magnetar model list
magnetar model inspect <model>
magnetar model load <model>
magnetar model unload <model>
magnetar devices
magnetar providers
magnetar sessions
magnetar serve
```

This change defines boundary and behavior, not final command syntax.

## `magnetar run`

`magnetar run` SHOULD perform one-shot inference.

Conceptual flow:

```text
CLI parses args
CLI resolves prompt input
CLI may read files if user requested
CLI builds prompt payload
CLI calls Runtime Inference API
Runtime resolves/loads model
Runtime tokenizes/generates
CLI renders output
```

Runtime SHALL not read workspace files for `magnetar run`.

## `magnetar chat`

`magnetar chat` SHOULD manage interactive conversation UX.

The CLI MAY keep chat transcript, user turns, command history, and UI state.

Runtime SHALL only receive inference-scoped prompt input, session policy, and
generation requests.

Runtime SHALL not own CLI conversation UX or shell interaction.

## Model Commands

CLI model commands MAY provide user-facing model management.

The CLI MAY map friendly names to Runtime model references.

The CLI MAY call Runtime model resolution/loading APIs.

The CLI SHALL not bypass Model Artifact trust, Model Loading, Model Component,
Memory Manager, or Provider validation.

## Provider And Device Commands

CLI provider and device commands MAY inspect Runtime diagnostics.

Examples:

```text
magnetar providers
magnetar devices
```

They SHALL display redacted Runtime metadata.

They SHALL not expose raw Provider handles, Device handles, Kernel handles, or
native pointers.

## Workspace And File Access

`magnetar-cli` MAY access workspace files when explicitly requested by the user
or allowed by CLI policy.

Runtime SHALL not access arbitrary workspace files.

When file content is needed for inference, CLI SHALL read and provide selected
content as prompt input or structured context.

Runtime receives the resulting prompt/context, not filesystem authority.

## Git Access

`magnetar-cli` MAY run Git operations according to CLI policy.

Runtime SHALL not own Git.

Git-derived information may be included in prompts only after CLI gathers it.

Runtime SHALL not call Git directly.

## Network Access

`magnetar-cli` MAY access network services according to CLI policy.

Runtime SHALL not perform arbitrary network operations.

Future model distribution sources remain governed by the validated distribution
contract, not arbitrary inference-time network authority.

## Secret Access

`magnetar-cli` MAY access secrets according to CLI policy.

Runtime SHALL not own user secrets.

Secrets SHALL not be sent to Runtime unless explicitly required by an
inference-scoped contract and policy permits it.

Secrets SHALL not appear in Runtime observability by default.

## Tool Execution

`magnetar-cli` MAY execute tools.

Runtime SHALL not execute tools.

If model output contains tool-call-like text, Runtime SHALL stream or return
text only.

CLI decides whether and how to interpret or execute tool calls.

## Shell And Process Execution

`magnetar-cli` MAY execute shell or processes according to CLI policy.

Runtime SHALL not execute shell commands or processes.

Runtime outputs SHALL not automatically trigger process execution.

## Agent Orchestration

`magnetar-cli` MAY orchestrate agent workflows.

Runtime SHALL not own agent planning, tool loops, workspace mutation, or task
execution.

Agent orchestration may call Runtime Inference API repeatedly.

## Prompt Assembly

CLI SHALL own prompt/context assembly that depends on:

- files
- workspace
- Git
- network retrieval
- tool outputs
- user interaction
- agent memory
- command history

Runtime MAY own inference-scoped prompt formatting such as tokenizer chat
templates where already authorized by Model/Tokenizer contracts.

## Chat Template Boundary

If chat messages are passed to Runtime, Runtime MAY apply authorized chat
template formatting through the Tokenizer/Prompt Template contracts.

CLI MAY also pre-render prompt text.

The boundary SHALL be explicit.

Runtime SHALL not fetch templates from arbitrary filesystem or network during
inference.

## Sessions

CLI MAY create, use, list, and close Runtime Inference Sessions through the API.

CLI session metadata is separate from Runtime Inference Session state.

Runtime Session SHALL not store workspace, Git, tool, shell, network, or secret
state.

## Streaming

CLI SHALL consume Runtime streaming events and render them to the user.

Runtime streaming events SHALL be inference-scoped.

CLI may add UI behaviors such as:

- progress display
- token rendering
- cancellation shortcut
- transcript storage
- terminal formatting

Runtime shall not own terminal UX.

## Cancellation

CLI MAY expose user cancellation.

Cancellation SHALL call Runtime Inference API cancellation.

CLI may also cancel CLI-owned work such as file scanning, Git, tools, or network
tasks separately.

Runtime cancellation applies only to inference-owned work.

## Diagnostics

CLI MAY display Runtime diagnostics.

Diagnostics SHALL be redacted by default.

CLI MAY enrich diagnostics with CLI-side context such as command name, user
flags, workspace path, or Git branch, but Runtime SHALL not own that context.

## Configuration

CLI MAY own user configuration for:

- default model alias
- default generation parameters
- default context limits
- default workspace behavior
- default tool policy
- default network policy
- secret providers
- output formatting
- profiles

Runtime MAY own inference policy defaults.

The boundary between CLI config and Runtime policy SHALL be explicit.

## Model Aliases

CLI MAY maintain friendly model aliases.

Runtime SHALL receive resolved ModelRef or model resolution request.

CLI aliases SHALL not bypass Runtime trust, loading, or compatibility checks.

## Local Model Files

If CLI supports local model file paths, path resolution SHALL occur in CLI.

Runtime SHALL receive a validated artifact source reference or client-provided
artifact reference according to the distribution/model artifact contracts.

Runtime SHALL not scan arbitrary directories.

## Serve Mode

`magnetar serve` MAY expose an API server.

If implemented in CLI crate or companion binary, serve mode SHALL still call
Runtime Inference API and SHALL not bypass Runtime validation.

HTTP/server behavior is not defined by this change.

## Error Model

CLI boundary errors SHALL be structured.

Error categories SHOULD include:

- cli command invalid
- cli prompt input invalid
- cli file read failed
- cli workspace access denied
- cli git failed
- cli network denied
- cli secret unavailable
- cli tool failed
- cli shell denied
- cli model alias not found
- cli model reference invalid
- cli runtime unavailable
- cli runtime request failed
- cli stream interrupted
- cli cancellation requested
- cli diagnostics redacted
- cli boundary violation
- internal cli error

Runtime errors SHALL remain Runtime Inference API errors.

CLI SHALL preserve Runtime structured errors when displaying or wrapping them.

## Observability

CLI MAY emit CLI-side observations.

Runtime MAY emit Runtime-side observations.

CLI-side observations MAY include:

- command received
- command parsed
- file context collected
- Git context collected
- tool executed
- runtime request submitted
- stream rendered
- command completed
- command failed

Runtime-side observations remain inference-only.

Observability SHALL not log raw prompts, secrets, file contents, model weights,
tokens, tensor values, Provider handles, Device handles, Kernel handles, or
memory pointers by default.

## Security Boundary

The CLI may have broader authority than Runtime.

That authority SHALL not be implicitly delegated to Runtime.

Runtime SHALL not receive ambient authority from CLI.

All data sent from CLI to Runtime SHALL be explicit request data.

## Browser Target

`magnetar-cli` is primarily native.

The Runtime Inference API remains platform-neutral.

Browser clients may exist later, but this change does not define browser CLI.

## Non-Goals

This change does not:

- finalize CLI command syntax
- implement CLI UX
- define terminal UI
- define HTTP server API
- define tool protocol
- define agent loop semantics
- define Git command set
- define workspace indexing
- define secret storage
- define model download UX
- expose Runtime internals
- move workspace responsibility into Runtime

## Impact

Magnetar gets a clean first-party usage boundary.

The intended path becomes:

```text
magnetar-cli
  -> gather/prepare user context
  -> call Runtime Inference API
  -> render or act on output
```

while Runtime remains:

```text
magnetar-runtime
  -> validate model/session/generation
  -> execute inference
  -> stream inference events
```

This prepares:

- end-to-end local inference conformance
- concrete `magnetar run`
- concrete `magnetar chat`
- future serve mode