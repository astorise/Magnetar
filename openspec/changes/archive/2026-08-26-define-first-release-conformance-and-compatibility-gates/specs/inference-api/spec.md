## ADDED Requirements

### Requirement: Inference API Release Gate Coverage

Runtime Inference API SHALL have release gate coverage for baseline inference.

#### Scenario: One-shot inference

Given one-shot inference is included in `v0.1`

When release gate runs

Then one-shot path is tested through normal Runtime contracts.