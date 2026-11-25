## 1. Runtime

- [x] 1.1 `logits_head`: the LM head behind `"logits"` through row-wise unary nodes.
- [x] 1.2 `insert_logits_row_gather` and the exact-length prefill's last-row read use it.

## 2. Tests

- [x] 2.1 The rewrite through a softcap and its refusal for a shared head output.
- [x] 2.2 Softcapped bucket graphs, gathered or not, against the capped plain logits.

## 3. Validation

- [x] 3.1 `cargo fmt`, `clippy -D warnings`, `magnetar-runtime` tests.
