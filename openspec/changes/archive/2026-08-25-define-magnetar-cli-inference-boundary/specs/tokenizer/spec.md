## ADDED Requirements

### Requirement: CLI May Send Text Or Chat Messages

Runtime SHALL accept plain text, chat messages, or already-tokenized input from
`magnetar-cli`.

`magnetar-cli` MAY send plain text, chat messages, or already-tokenized input to
Runtime.

#### Scenario: Chat input

Given CLI has chat transcript

When generating response

Then CLI sends appropriate chat messages or prompt text through Runtime
Inference API.

---

### Requirement: Runtime Tokenization Does Not Read CLI Files

Tokenizer execution through Runtime SHALL not read CLI workspace files.

#### Scenario: Template path

Given request attempts to make Runtime read template from workspace path

When Runtime validates it

Then access is denied unless template is already an authorized Model/Tokenizer
Artifact component.