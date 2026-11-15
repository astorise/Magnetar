## Decisions

### Only row-wise unary nodes, each the only reader of its input

Every output row of `activation`, `gelu` and `silu` depends on the same input row alone. Selecting one row before them therefore gives the same row after them.

Each node must be the only reader of its input. Otherwise another reader would see a one-row tensor where it expects every row, so the gather is refused instead.

A chain longer than the graph's node count is a cycle and is refused.

### No new state

The gather only changes edge shapes. Neither the replay keys nor the recordings change. The capture of a bucketed prefill now covers a one-row softcap, which needs no row skipping.

## Evidence

- `the_logits_row_gather_reaches_the_head_behind_a_softcap` checks the rewrite:
  - the head reads the gathered hidden row;
  - the head's output and `"logits"` hold one row;
  - another reader of the head's output prevents the gather.
- `a_softcapped_bucketed_prefill_matches_its_capped_plain_logits` runs a bucket graph with a softcap after the head on the Reference CPU, gathered and not gathered. Not gathered, the head reads the sampled row only. Over the prefill and three decode steps, its logits equal `cap * tanh(plain / cap)` within 1e-5.
