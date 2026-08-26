## ADDED Requirements

### Requirement: magnetar-cli Boundary

`magnetar-cli` SHALL be the first-party client/runtime around Magnetar
inference.

`magnetar-runtime` SHALL remain the inference runtime.

#### Scenario: CLI calls Runtime

Given user runs a CLI inference command

When inference is needed

Then `magnetar-cli` calls Runtime Inference API instead of bypassing Runtime.

---

### Requirement: CLI Owns Workspace Responsibilities

`magnetar-cli` SHALL own workspace and file responsibilities.

Runtime SHALL not read arbitrary workspace files.

#### Scenario: File prompt

Given user asks CLI to include a file in prompt

When CLI prepares inference

Then CLI reads the file and sends selected content as explicit prompt/context.

---

### Requirement: CLI Owns Git Responsibilities

`magnetar-cli` SHALL own Git operations.

Runtime SHALL not execute Git.

#### Scenario: Git context

Given CLI includes Git diff in prompt

When Runtime receives request

Then Runtime receives explicit text/context, not Git authority.

---

### Requirement: CLI Owns Network Responsibilities

`magnetar-cli` SHALL own arbitrary network operations according to CLI policy.

Runtime SHALL not perform arbitrary inference-time network operations.

#### Scenario: Retrieval context

Given CLI fetches remote context

When inference runs

Then Runtime receives explicit retrieved context only.

---

### Requirement: CLI Owns Secret Responsibilities

`magnetar-cli` SHALL own user secret access according to CLI policy.

Runtime SHALL not receive ambient secret authority.

#### Scenario: Secret available to CLI

Given CLI can access a token

When Runtime request is built

Then the token is not sent unless explicitly required and policy allows it.

---

### Requirement: CLI Owns Tool Execution

`magnetar-cli` SHALL own tool execution.

Runtime SHALL not execute tools.

#### Scenario: Tool-call-like output

Given Runtime streams text that resembles a tool call

When output is received

Then CLI decides whether to interpret it; Runtime does not execute the tool.

---

### Requirement: CLI Owns Shell And Process Execution

`magnetar-cli` SHALL own shell/process execution according to CLI policy.

Runtime SHALL not execute shell commands or processes.

#### Scenario: Generated shell command

Given model generates `rm -rf tmp`

When Runtime streams the text

Then Runtime does not execute it.

---

### Requirement: CLI Owns Agent Orchestration

Runtime SHALL not own agent planning or workspace mutation.

`magnetar-cli` MAY own agent orchestration and tool loops.

#### Scenario: Agent loop

Given CLI runs an agent workflow

When it needs model output

Then it calls Runtime Inference API and keeps tool decisions in CLI.

---

### Requirement: CLI Owns Prompt Assembly From External Context

`magnetar-cli` SHALL assemble prompt/context when it depends on files,
workspace, Git, network retrieval, tool outputs, user interaction, or agent
memory.

#### Scenario: Workspace prompt

Given CLI builds prompt from workspace files

When Runtime is called

Then Runtime receives explicit prompt/context only.

---

### Requirement: Runtime May Apply Authorized Chat Template

Runtime SHALL not fetch templates from arbitrary filesystem or network.

Runtime MAY apply authorized chat template formatting through Tokenizer/Prompt
Template contracts.

#### Scenario: Chat messages

Given CLI sends chat messages

When Runtime prepares tokens

Then Runtime may apply authorized template if available.

---

### Requirement: CLI Session Metadata Is Separate From Runtime Session State

CLI chat transcript, terminal state, command history, and UI state SHALL remain
CLI-owned.

Runtime Inference Session SHALL remain inference-scoped.

#### Scenario: Runtime session

Given CLI opens chat session

When Runtime creates Inference Session

Then Runtime session does not store workspace, Git, tool, shell, network, or
secret state.

---

### Requirement: CLI Renders Runtime Streams

`magnetar-cli` SHALL consume Runtime streaming events and render them.

Runtime SHALL not own terminal UI.

#### Scenario: Decode event

Given Runtime emits decoded text event

When CLI receives it

Then CLI renders it to the terminal.

---

### Requirement: CLI Cancellation Calls Runtime Cancellation

CLI user cancellation SHALL call Runtime cancellation for inference work.

CLI-owned file, Git, network, and tool work SHALL be cancelled separately by CLI.

#### Scenario: Ctrl-C during generation

Given user cancels generation

When CLI receives cancellation

Then CLI calls Runtime cancellation for active inference.

---

### Requirement: CLI Displays Redacted Runtime Diagnostics

CLI MAY display Runtime diagnostics but SHALL not expose raw Runtime handles or
redacted internals.

#### Scenario: Provider status

Given CLI runs `magnetar providers`

When Runtime returns diagnostics

Then CLI displays redacted Provider metadata only.

---

### Requirement: CLI Configuration Is Separate From Runtime Policy

CLI configuration and Runtime inference policy SHALL have explicit boundary.

#### Scenario: Default model alias

Given CLI config sets default model alias

When inference starts

Then CLI resolves alias and Runtime still validates resolved model reference.

---

### Requirement: CLI Model Aliases Do Not Bypass Runtime Validation

CLI model aliases SHALL not bypass Runtime trust, loading, compatibility, or
policy validation.

#### Scenario: Alias to untrusted model

Given alias points to untrusted artifact

When CLI calls Runtime

Then Runtime rejects the artifact.

---

### Requirement: Local Model Paths Are Resolved By CLI

If local model paths are supported, CLI SHALL resolve paths and pass authorized
artifact source references to Runtime.

Runtime SHALL not scan arbitrary directories.

#### Scenario: Local path

Given user provides local model path

When CLI prepares load request

Then Runtime receives client-provided artifact reference, not directory scanning
authority.

---

### Requirement: Serve Mode Uses Runtime Inference API

`magnetar serve` SHALL call Runtime Inference API and SHALL not bypass Runtime
validation.

#### Scenario: Serve request

Given serve mode receives generation request

When inference runs

Then serve mode submits Runtime Inference API request.

---

### Requirement: CLI Boundary Error Categories

CLI boundary failures SHALL use structured error categories and preserve Runtime
structured errors when wrapping them.

#### Scenario: Runtime error

Given Runtime returns model-loading-failed

When CLI displays it

Then CLI preserves the structured Runtime error category.

---

### Requirement: CLI Observability Is Separate From Runtime Observability

CLI-side authority and context SHALL not be implicitly logged by Runtime.

CLI MAY emit CLI-side observations and Runtime MAY emit Runtime-side
observations.

#### Scenario: File context

Given CLI collects file context

When Runtime emits inference observations

Then Runtime does not log file path or file content unless explicitly included
and policy allows it.

---

### Requirement: CLI Authority Is Not Runtime Authority

CLI authority SHALL not become Runtime ambient authority.

All data sent from CLI to Runtime SHALL be explicit request data.

#### Scenario: CLI has network access

Given CLI has network access

When Runtime request is created

Then Runtime does not receive network authority.

---

### Requirement: Runtime Remains Inference-Only

Runtime SHALL NOT absorb CLI responsibilities such as workspace, Git, network,
secrets, tools, shell/process execution, or agent orchestration.

#### Scenario: Runtime tool execution request

Given CLI or model output requests tool execution through Runtime

When Runtime validates it

Then Runtime rejects it as outside inference scope.