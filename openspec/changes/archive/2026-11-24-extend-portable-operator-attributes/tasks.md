## 1. Catalog

- [x] 1.1 `OperatorSpec::min_inputs` and input-count validation between it and `inputs`; `rmsnorm` 1..=2 and `moe-experts` 5..=6.
- [x] 1.2 `ShapeRule::SameShapeOrScalar` for `mul`; RMSNorm shape rule with an optional weight and the per-head `[head_dimension]` weight.
- [x] 1.3 Attribute rules: `attention.scale`, `rope.partial_rotary_factor`, `gelu.approximate`, `rmsnorm.head_count` / `head_dimension`, `activation.cap` / `factor`, `embedding.scale`, `moe-experts.activation`.

## 2. Shared semantics and Reference CPU

- [x] 2.1 `operator_semantics`: softcap, scale, per-head and scale-less RMSNorm, scalar `mul`, MoE expert activation and per-expert scale.
- [x] 2.2 Reference CPU: attention scale, partial RoPE, embedding scale, and dispatch of every new form, refusing unsupported values.
- [x] 2.3 Conformance fixtures: each new form against an independent computation, plus the refusals.

## 3. Validation

- [x] 3.1 `cargo fmt`, `clippy -D warnings` and tests for `magnetar-core`, `magnetar-provider-api` and `magnetar-runtime`.
