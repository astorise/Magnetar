## ADDED Requirements

### Requirement: Model Component Graph Contract Supports Per-Segment Graph Building

A Model Component implementing `model-component-graph-producer` SHALL be able to build a graph for one decoder-layer range `[start_layer, end_layer)` instead of always the whole stack, through `build-prefill-graph-segment`/`build-decode-graph-segment` (`model-component-graph.wit` `1.3.0`, purely additive over `1.2.0`). A segment whose `start_layer` is `0` SHALL begin from the same `token_ids`-plus-embedding input a full graph's own first segment uses; a segment whose `start_layer` is not `0` SHALL instead begin from a declared `hidden_states_in` input. A segment whose `end_layer` equals the real `num_hidden_layers` SHALL end with the real final-norm/lm-head projection, exactly as a full graph does; otherwise it SHALL end with the raw post-layer hidden state as the graph's own declared output.

#### Scenario: A segment starting at layer zero produces the same embedding lookup as a full graph

- **GIVEN** a segment graph built with `start_layer == 0`
- **WHEN** it is compared to the corresponding prefix of a full graph built for the same prompt
- **THEN** both declare the same `token_ids` input and produce the token embedding through the same real embedding lookup

#### Scenario: A segment ending at the real final layer produces the same real logits as a full graph

- **GIVEN** a segment graph built with `end_layer == num_hidden_layers`
- **WHEN** it is dispatched with the correct upstream hidden state as its own input
- **THEN** its output matches the full graph's own final logits for the equivalent computation

#### Scenario: An internal segment's output is a raw hidden state, not logits

- **GIVEN** a segment graph built with `end_layer != num_hidden_layers`
- **WHEN** its declared output edge is inspected
- **THEN** it carries a raw post-layer hidden state of shape `[sequence_length, hidden_size]`, not a vocabulary-sized logits tensor
