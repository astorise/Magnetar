## ADDED Requirements

### Requirement: Recurrent state is a session resource that starts from zeros

The Runtime SHALL hold each validated recurrent state binding of a session's graphs as a Runtime-owned resource of that session, admitted at the layout's fixed size under the session's memory owner. The binding's input edge SHALL read the committed value. Before the session's first commit, it SHALL read zeros of the layout's shape. Two sessions SHALL never share a recurrent resource.

#### Scenario: A new sequence

- **WHEN** a session runs its first step on a graph with a recurrent binding
- **THEN** the binding's input edge reads a zero tensor of the declared shape, admitted under that session

#### Scenario: Two sessions

- **WHEN** two sessions run the same model and graphs
- **THEN** each step of one session reads only the state committed by that session

### Requirement: Recurrent state commits atomically with KV

The Runtime SHALL write each recurrent output edge's value to a pending resource, distinct from the committed one. It SHALL publish all pending recurrent values and KV updates of a step together when the step is accepted. If the step fails or is aborted, it SHALL release the pending values and leave the committed state and KV unchanged.

#### Scenario: Accepted steps accumulate

- **WHEN** a graph whose state output is its state input plus the step input runs three accepted steps
- **THEN** the third step reads the sum of the first two inputs

#### Scenario: A failed step

- **WHEN** a step fails after a recurrent output was written
- **THEN** the next step reads the state committed before the failed step, and the KV history has the same rows as before it

### Requirement: Prefill and decode continue the same recurrent state

The Runtime SHALL commit the recurrent state produced by a prefill, or by each chunk of a chunked prefill, so that the next chunk or decode step reads it. A chunked prefill SHALL produce the same state and outputs as the one-pass prefill, within the operator's tolerance.

#### Scenario: Chunked prefill equals one pass

- **WHEN** a prompt is prefilled in chunks and, separately, in one pass, then decoded for one token
- **THEN** the committed states and the decoded logits agree within tolerance

#### Scenario: Gated-delta mixer continuation

- **WHEN** a graph with one `gated-delta-mixer` node prefills the reference fixture's prefix and then decodes its next token
- **THEN** the output and both states equal the fixture's full-sequence values within tolerance

### Requirement: Unsupported paths refuse or bypass recurrent graphs explicitly

Until a path supports recurrent state, the Runtime SHALL NOT run a graph with recurrent bindings through it with stale, shared or missing state:
- **Prefix reuse:** bypassed, with an observation.
- **Step capture and replay:** not used, with a traced reason.
- **Continuous batching and speculative verification:** refused with an explicit error.

#### Scenario: A prefix-cache hit on a recurrent model

- **WHEN** a request on a model with recurrent bindings shares a prefix with an earlier request
- **THEN** the prompt is prefilled from zero state, and an observation records that prefix reuse was bypassed

#### Scenario: Decode step capture

- **WHEN** decode would capture a step for a graph with recurrent bindings, before paired-buffer replay is implemented
- **THEN** no capture is recorded, the trace gives the reason, and decode results equal those without capture
