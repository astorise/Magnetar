## ADDED Requirements

### Requirement: Step capture on the optimized CPU Provider

The optimized CPU Provider SHALL support step capture unless `MAGNETAR_CPU_STEP_CAPTURE=0`. A capture SHALL record the Kernel invocations the step dispatches while running them. A host write during a capture SHALL make `end_step_capture` fail. A replay SHALL write the given inputs and run the recorded invocations with every advancing integer attribute increased by the offset. A released capture SHALL no longer replay.

#### Scenario: Replay at an offset

- **WHEN** a step that projects an input and writes it to KV row 1 is captured with `row_offset` advancing, then replayed at offset 3 with a new input
- **THEN** row 1 holds the captured projection, row 4 the replayed one, and the rows between are untouched

#### Scenario: Failures

- **WHEN** a replay's offset puts a write past its store, a host write happens during a capture, or a released capture is replayed
- **THEN** the replay fails, the capture fails, or the replay is reported invalidated, respectively

#### Scenario: A real checkpoint

- **WHEN** Qwen2.5-0.5B generates 16 greedy tokens on the optimized CPU Provider with capture on and off
- **THEN** the tokens are identical

### Requirement: Greedy selection in place

The optimized CPU Provider SHALL select the greedy column of a logits row in place, skipping non-finite scores and excluded ranges, with each score divided by a nonzero temperature.

#### Scenario: Exclusions

- **WHEN** the best column of a row is excluded
- **THEN** the best remaining column is selected

### Requirement: Replayed prefills skip their padding rows

While a captured step whose advancing attributes include `padding_rows` replays with offset `p`, the optimized CPU Provider SHALL compute a row matmul of `m > max(p, 1)` rows on its leading `m - p` rows only, leaving the padding rows zero, and SHALL compute every row of any other row matmul.

#### Scenario: A padded prefill and its LM head

- **WHEN** an 8-row matmul and a 1-row matmul run during a replay with 3 padding rows
- **THEN** the first 5 rows of the 8-row product equal the Reference CPU product and the last 3 are zero, and the 1-row product is whole
