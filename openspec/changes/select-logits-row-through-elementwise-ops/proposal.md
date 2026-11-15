## Why

A bucketed prefill selects the sampled prompt row before the LM head (`replay-bucketed-prefill`). Likewise, an exact-length prefill computes the LM head on its last row only. Both looked for a plain or bias `matmul` producing `"logits"` directly.

Layered models (issue #187) cap their logits: `"logits"` is an `activation` (`kind = "softcap"`) over the head's output. For those graphs:
- the gather was refused, so every bucket row went through the head and the cap;
- the last-row read did not apply, so a prefill computed every row's logits only to sample one.

## What changes

- **`logits_head`:**
  - It finds the LM head behind `"logits"`: a plain or bias matmul, reached directly or through row-wise unary nodes (`activation`, `gelu`, `silu`).
  - Each of those nodes must be the only reader of its input.
- **Logits row gather:**
  - `insert_logits_row_gather` gathers the row before that head.
  - The head's output and every row-wise node after it become one row.
- **Last-row read:** the exact-length prefill's head reads only the last row behind such nodes, and the nodes after it process one row.

Graphs whose head produces `"logits"` directly behave as before.

## Impact

- **`magnetar-runtime`:** `prefill_buckets` and the first-native engine's last-row read, with tests.
