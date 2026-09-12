## MODIFIED Requirements

### Requirement: Prompt Input Boundary

Runtime Inference API MAY accept plain text, chat messages, already-tokenized input, or test token sequences. It SHALL not perform external retrieval, file reading, workspace scanning, or tool execution.

#### Scenario: Chat messages

Given chat messages are submitted

When Runtime prepares input

Then chat-template formatting occurs only through authorized Runtime prompt
contracts.

#### Scenario: Chat messages rendered through a real artifact-declared template
- **WHEN** chat messages are submitted for a Model Instance whose Model Artifact declares a real chat template
- **THEN** Runtime renders those messages through that real template before tokenization, not through a generic placeholder that ignores the artifact's own template.
