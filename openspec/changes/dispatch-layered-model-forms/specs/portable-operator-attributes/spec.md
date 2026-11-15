## ADDED Requirements

### Requirement: Kernel features of the optional forms

The operator catalog SHALL name the Kernel feature each optional form needs:
- `attention.scale`: `attention-scale`;
- `embedding.scale`: `embedding-scale`;
- per-head `rmsnorm`: `rmsnorm-per-head`;
- weightless `rmsnorm`: `rmsnorm-without-weight`;
- `rope.partial_rotary_factor` other than 1: `partial-rotary`;
- `moe-experts.activation` other than `silu`: `moe-experts-activation`;
- a sixth `moe-experts` input: `per-expert-scale`.

`rmsnorm` SHALL accept an optional boolean `with_scale`, and a Kernel SHALL refuse one that contradicts its input count.

#### Scenario: Uniform graph
- **WHEN** a node uses no optional form
- **THEN** it needs no Kernel feature
