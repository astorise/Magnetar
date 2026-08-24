## ADDED Requirements

### Requirement: Model Metadata May Constrain Sampling

Runtime SHALL treat Model Artifact or loaded model metadata declarations for
supported or default sampling parameters as policy input during Sampling
validation.

#### Scenario: Unsupported sampling mode

Given a model metadata declares it does not support a requested sampling mode

When generation validates parameters

Then Runtime rejects the request or applies policy fallback.

---

### Requirement: Model Vocabulary Must Match Sampling

Loaded model output logits SHALL be compatible with tokenizer vocabulary
metadata used by Sampling.

#### Scenario: Logits vocabulary mismatch

Given loaded model produces logits for vocabulary size X

And tokenizer reports vocabulary size Y

When Sampling validates logits

Then Runtime reports vocabulary-mismatch.
