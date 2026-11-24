## ADDED Requirements

### Requirement: Zero-centered RMSNorm weight and per-head split

The `rmsnorm` Operator SHALL accept an optional finite `weight_offset` float. With it, the normalized row SHALL be multiplied by `weight_offset + weight`, and a node without a weight input SHALL be refused. The `split` Operator SHALL accept an optional positive `head_count` integer that divides its last dimension into `head_count` equal blocks, each of an even width. With it, the first output SHALL hold the first half of every block in order, and the second output the second halves. A node using either form SHALL require the `rmsnorm-weight-offset` or `split-per-head` Kernel feature. A Runtime SHALL send such a node only to a Kernel advertising that feature, and SHALL pass the attribute to it.

#### Scenario: Qwen3.5's gated attention block

- **WHEN** a graph splits a `[T, H * 2D]` query projection per head, normalizes the query and key per head with `weight_offset` 1, rotates them, attends, and multiplies by the sigmoid of the gate
- **THEN** its output equals the pinned upstream `Qwen3_5Attention` output for the same synthetic weights within 2e-6 plus 1e-5 of the value

#### Scenario: A zero offset

- **WHEN** `weight_offset` is 0 or absent and `head_count` is 1 or absent
- **THEN** the node needs no added Kernel feature and computes the baseline form

#### Scenario: Invalid forms

- **WHEN** `weight_offset` is not finite, or `head_count` is not positive or does not divide the last dimension into even blocks
- **THEN** graph validation refuses the node
