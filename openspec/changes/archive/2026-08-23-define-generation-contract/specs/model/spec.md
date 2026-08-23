## ADDED Requirements

### Requirement: Model Supports Generation Metadata

A Model Artifact or future Model Instance SHALL expose metadata required for
generation validation.

Metadata MAY include context length, supported generation modes, supported
dtypes, supported tokenizer, EOS token metadata, and architecture generation
support.

#### Scenario: Model has context length

Given a model declares context length

When generation validates prompt and max new tokens

Then the model context length is used.

---

### Requirement: Model Artifact Alone Is Not Executable Generation

A Model Artifact SHALL not be treated as an executable generation context by
itself.

#### Scenario: Artifact trusted

Given a Model Artifact is trusted

When generation is requested

Then Runtime still requires a loaded model context or future Model Instance.