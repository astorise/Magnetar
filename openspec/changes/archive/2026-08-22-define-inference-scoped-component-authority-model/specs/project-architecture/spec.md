## ADDED Requirements

### Requirement: Magnetar Is Scoped To Inference Runtime

Magnetar SHALL be scoped to local AI inference execution.

Magnetar SHALL NOT be the owner of general-purpose agent tool authority.

#### Scenario: Classify responsibility

Given a feature requires reading a user workspace, executing Git, or accessing
secrets

When architectural ownership is determined

Then the feature belongs to a client or orchestrator such as `magnetar-cli`

And not to the Magnetar inference Runtime.

---

### Requirement: Client Owns Workspace And Tool Authority

Magnetar SHALL treat workspace, filesystem, network, Git, secret, process, and
tool authority as client-owned when exposed by a client such as `magnetar-cli`.

The client MAY call Magnetar for inference.

Magnetar SHALL treat this authority as client-owned and outside the inference
Runtime boundary.

#### Scenario: Coding assistant edits files

Given a coding assistant needs to inspect and modify a workspace

When it runs through `magnetar-cli`

Then `magnetar-cli` owns workspace and file mutation authority

And Magnetar provides inference output only.

---

### Requirement: Magnetar Does Not Own Filesystem Authority

Magnetar SHALL NOT provide arbitrary filesystem authority to Components.

Inference artifacts SHALL be accessed through Runtime-managed artifact
registries, not unrestricted filesystem paths.

#### Scenario: Component requests model

Given a Component needs a model artifact

When Magnetar grants access

Then access is to a Runtime-registered model artifact

And not to arbitrary local filesystem paths.

---

### Requirement: Magnetar Does Not Own Network Authority

Magnetar SHALL NOT provide arbitrary network authority to Components.

Network access for agents, tools, package managers, Git remotes, APIs, or web
fetching belongs to clients or orchestrators.

#### Scenario: Agent wants web data

Given an agent workflow needs to fetch network data

When ownership is determined

Then the client performs or authorizes the network operation

And Magnetar remains limited to inference.

---

### Requirement: Magnetar Does Not Own Secret Authority

Magnetar SHALL NOT provide Components direct access to user secrets.

Clients MAY manage secrets and decide what prompt or inference input is sent to
Magnetar.

#### Scenario: API key needed by tool

Given a tool needs an API key

When the workflow executes

Then the client or orchestrator manages the secret

And Magnetar does not expose a secret-reading Capability.

---

### Requirement: Magnetar Does Not Own Git Authority

Magnetar SHALL NOT provide Git authority to Components.

Git integration belongs to clients such as `magnetar-cli`.

#### Scenario: Generate commit message

Given a workflow needs Git diff context

When `magnetar-cli` runs the workflow

Then `magnetar-cli` may read Git state and send inference input to Magnetar

And Magnetar does not execute Git commands.

---

### Requirement: Magnetar Does Not Own Process Execution

Magnetar SHALL NOT provide shell or process execution authority to Components.

#### Scenario: Test command requested

Given a coding workflow wants to run tests

When execution occurs

Then the client may decide whether to run the command

And Magnetar does not spawn the process.

---

### Requirement: Inference Components Only

Magnetar Components SHALL be limited to inference-related responsibilities.

Valid responsibilities MAY include:

- model architecture logic
- tokenizer logic
- prompt-template logic
- sampling logic
- logits processing
- generation helpers
- observability emission for inference
- inference diagnostics

#### Scenario: Add tokenizer Component

Given a tokenizer implementation is portable and inference-scoped

When it is added as a Component

Then it may be accepted by Magnetar subject to artifact trust and authority
validation.

---

### Requirement: Tool Components Are Out Of Magnetar Scope

Components whose primary purpose is general tool execution SHALL be outside
Magnetar Runtime scope.

Out-of-scope Components include:

- filesystem tools
- Git tools
- shell tools
- network fetchers
- secret readers
- workspace editors

#### Scenario: Filesystem tool Component

Given a Component exists to read and write arbitrary workspace files

When it is classified

Then it belongs to a client-side tool system

And not to Magnetar Runtime.
