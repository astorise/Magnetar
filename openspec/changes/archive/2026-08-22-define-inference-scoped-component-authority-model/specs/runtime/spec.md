## ADDED Requirements

### Requirement: Runtime Grants Only Inference Authority

Magnetar Runtime SHALL grant only inference-scoped Component authority.

It SHALL NOT grant broad external-world authority.

#### Scenario: Component requests network

Given a Component manifest requests network authority

When Runtime validates the artifact

Then Runtime rejects the artifact before ComponentEngine preparation.

---

### Requirement: Runtime Rejects Broad Tool Authority

Runtime validation SHALL reject Component manifests requesting broad tool
authority, including filesystem, network, secrets, process, Git, workspace, and
external service access.

#### Scenario: Trusted digest with forbidden authority

Given a Component digest is trusted

And its manifest requests filesystem authority

When Runtime validates authority

Then the artifact is rejected despite the trusted digest.

---

### Requirement: Runtime Separates Inference Artifacts From Filesystem Paths

Runtime-mediated inference artifact access SHALL use artifact identity rather
than unrestricted filesystem paths.

#### Scenario: Model artifact read

Given a Component has `model-artifact-read`

When it requests model content

Then Runtime resolves the model through its artifact registry

And does not expose arbitrary filesystem authority.

---

### Requirement: Runtime Does Not Link General WASI For Components

Runtime SHALL NOT link broad WASI filesystem, environment, process, or network
interfaces to Magnetar inference Components.

#### Scenario: Component imports WASI filesystem

Given a Component imports WASI filesystem interfaces

When Runtime builds the Link Plan

Then those imports are rejected as outside Magnetar inference scope.

---

### Requirement: Runtime Links Inference Capabilities Only

Runtime Link Plans for Magnetar Components SHALL include only authorized
inference-scoped endpoints.

#### Scenario: Link Compute

Given a Component is trusted and requests `compute-capability`

When Runtime builds the Link Plan

Then a Runtime Compute endpoint may be linked.

#### Scenario: Link Git

Given a Component requests Git access

When Runtime builds the Link Plan

Then no Git endpoint is linked because Git belongs to the client.

---

### Requirement: Runtime Does Not Execute Agent Tools

Runtime SHALL NOT execute general-purpose agent tools.

Clients MAY call Runtime for inference and execute tools externally according to
their own policy.

#### Scenario: Coding agent asks to edit file

Given a client is running a coding-agent workflow

When a file edit is needed

Then the client handles file authority and mutation

And Magnetar only provides inference output.

---

### Requirement: Runtime Treats CLI Context As Input Only

Context gathered by a client from files, Git, network, or tools SHALL be
treated by Magnetar as inference input only.

Magnetar SHALL NOT infer from that input that it has the authority to access
the same source directly.

#### Scenario: Prompt contains file content

Given `magnetar-cli` sends file content in a prompt

When Magnetar generates a response

Then Magnetar does not gain filesystem access to that file.

---

### Requirement: Runtime Observability Does Not Bypass Scope

Runtime observability SHALL not be used as a channel for Components to obtain
network, filesystem, secret, workspace, Git, or process authority.

#### Scenario: Component emits observation

Given a Component emits an observation

When Runtime exports observability data

Then export behavior is controlled by Runtime observability configuration

And the Component cannot choose arbitrary network destinations.

---

### Requirement: Runtime Trust Policy Cannot Permit Out-Of-Scope Authority

Trust policy SHALL NOT mark a Component executable when its requested authority
is outside Magnetar inference scope.

#### Scenario: Trusted publisher requests shell

Given Runtime policy trusts a publisher

And that publisher's Component requests shell authority

When the artifact is validated

Then validation fails because shell authority is out of scope.

---

### Requirement: Runtime Development Mode Cannot Permit Out-Of-Scope Authority

Development mode SHALL NOT silently allow broad external-world authority.

#### Scenario: Local dev Component requests secrets

Given development mode is enabled

And a local Component requests secrets authority

When validation runs

Then Magnetar rejects the Component as outside inference scope.
