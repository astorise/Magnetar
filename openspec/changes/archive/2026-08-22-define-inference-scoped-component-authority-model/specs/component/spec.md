## MODIFIED Requirements

### Requirement: Authority Requirement Declaration

A Component manifest SHALL declare requested inference authority.

Authority declarations SHALL be limited to inference-scoped Runtime
capabilities and resources.

Broad workspace, filesystem, network, process, secret, Git, source-control, or
tool-execution authorities SHALL NOT be valid Magnetar Runtime Component
authorities.

#### Scenario: Inference authority requested

Given a Component manifest declares `model-artifact-read` and
`compute-capability`

When artifact validation runs

Then the authority declaration is accepted if all other validation and trust
rules pass.

#### Scenario: Target tokenizer manifest authority requested

Given a Component manifest uses schema `magnetar-component-artifact`

And schema version `1`

And declares Component `magnetar.examples.tokenizer` with role `tokenizer`

And imports `magnetar:compute/run` version `2.0.0`

And exports `magnetar:tokenizer/tokenize` version `1.0.0`

And declares `tokenizer-artifact-read`, `compute-capability`, and
`observability-emit`

When artifact validation runs

Then the manifest is accepted if digest, WIT, compatibility, and trust rules
also pass.

#### Scenario: Filesystem authority requested

Given a Component manifest declares `filesystem`

When artifact validation runs

Then Magnetar rejects the artifact as outside Runtime scope.

---

### Requirement: Unsupported Authority Fails Closed

Unsupported, unknown, or broad non-inference authority declarations SHALL fail
closed unless explicitly marked as client-owned metadata ignored by Magnetar.

Trusted digest, trusted publisher, development mode, or trusted source SHALL NOT
override forbidden Magnetar authority.

#### Scenario: Trusted artifact requests network

Given a Component digest is trusted

And its manifest requests `network`

When Magnetar validates authority

Then validation fails.

---

## ADDED Requirements

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

Then the path is omitted or redacted unless an external client policy explicitly
allows disclosure outside Magnetar.

---

### Requirement: No Magnetar Tool Components

Magnetar SHALL NOT authorize Components whose purpose is general tool execution.

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
