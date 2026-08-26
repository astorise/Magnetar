## ADDED Requirements

### Requirement: Tokenizer Artifacts May Use Source Cache

Tokenizer Artifacts MAY be resolved through source/cache workflow, and Runtime SHALL validate the tokenizer cache entry before use.

#### Scenario: Cached tokenizer

Given tokenizer artifact is cached

When model loading validates tokenizer compatibility

Then cached tokenizer metadata is validated before use.

---

### Requirement: Tokenizer Cache Does Not Override Compatibility

Tokenizer cache hit SHALL not bypass tokenizer/model compatibility validation.

#### Scenario: Wrong tokenizer cached

Given cached tokenizer is incompatible with model

When loading runs

Then Runtime rejects compatibility.