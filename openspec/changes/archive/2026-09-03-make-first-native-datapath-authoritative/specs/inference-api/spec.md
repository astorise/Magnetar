## ADDED Requirements

### Requirement: Chat Uses Persistent Runtime Session
The chat CLI SHALL execute all turns of a ChatSession through its persistent Runtime and InferenceSession.

#### Scenario: Two chat turns execute
- **WHEN** two turns are submitted through one ChatSession
- **THEN** both turns use the same Runtime InferenceSession identifier.

#### Scenario: Chat is cancelled
- **WHEN** ChatSession cancellation is requested
- **THEN** cancellation targets the Runtime session used by chat turns and blocks new turns.
