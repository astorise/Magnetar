## ADDED Requirements

### Requirement: Model References Tokenizer Artifact

A text-generation Model Artifact SHALL validate tokenizer references when tokenizer metadata is required.

#### Scenario: Model bundle tokenizer

Given a model bundle includes tokenizer metadata

When Runtime validates the model bundle

Then the referenced tokenizer artifact identity is validated.

---

### Requirement: Tokenizer Compatibility Is Part Of Model Validation

Model validation SHALL include tokenizer compatibility when a tokenizer is
required.

#### Scenario: Wrong tokenizer

Given a model expects tokenizer digest A

And a tokenizer artifact with digest B is selected

When validation runs

Then Runtime rejects the pairing unless explicit policy permits override.

---

### Requirement: Tokenizer Metadata Does Not Define Generation

Tokenizer metadata SHALL NOT define generation behavior beyond tokenization
defaults and special token metadata.

#### Scenario: EOS metadata

Given tokenizer metadata defines EOS token ID

When generation later uses EOS as a stop condition

Then generation behavior is defined by Generation Contract

And tokenizer only supplies token metadata.
