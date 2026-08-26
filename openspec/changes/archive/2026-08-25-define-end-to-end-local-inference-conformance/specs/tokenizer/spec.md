## ADDED Requirements

### Requirement: E2E Uses Tokenizer Contract

E2E conformance SHALL use Tokenizer Contract for text prompt encoding and output
decoding.

#### Scenario: Text prompt

Given fixture text prompt is submitted

When inference runs

Then Runtime tokenizes it through Tokenizer Contract.

---

### Requirement: E2E Validates Tokenizer Failure

E2E conformance SHALL include tokenizer failure cases.

#### Scenario: Incompatible tokenizer

Given tokenizer vocabulary is incompatible with fixture model

When compatibility validation runs

Then Runtime reports structured tokenizer incompatibility.