## ADDED Requirements

### Requirement: Extend graphs for multi-token verification

A Model Component supporting speculation SHALL export, through an additive minor version of the graph contract, a graph for `sequence-length` new tokens appended to `cached-token-count` committed tokens, with positions offset by the committed count, KV appended in place and causal attention over the committed history and the new tokens. Components that do not export it SHALL keep linking and generating without speculation.

#### Scenario: Old Component

- **WHEN** a Component built against an earlier graph-contract version is loaded and speculation is requested
- **THEN** speculation is reported Unsupported for its Model Instance, and ordinary generation proceeds

### Requirement: Declared multi-token-prediction heads

A Model Component whose model carries multi-token-prediction heads MAY declare them, through an additive minor version of the graph contract, with the number of future tokens they predict and the graph outputs carrying their candidate logits. The Component SHALL NOT accept, reject or commit tokens, and SHALL NOT select Providers or Devices; the Runtime treats declared heads as a draft source.

#### Scenario: Declared heads

- **WHEN** a Component declares MTP heads predicting two future tokens
- **THEN** the Runtime can read two candidate tokens per step from the declared outputs and verifies them like any other draft
