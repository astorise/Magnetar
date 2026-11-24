## ADDED Requirements

### Requirement: Optional trailing inputs

The operator catalog SHALL let an Operator accept between a minimum and a maximum number of inputs, the omitted ones being trailing. `rmsnorm` SHALL accept 1 or 2 inputs and `moe-experts` 5 or 6.

#### Scenario: Scale-less RMSNorm validates
- **WHEN** a graph declares `rmsnorm` with one input
- **THEN** the catalog accepts it, and the Reference CPU normalizes with scale 1

### Requirement: Portable attributes for layered attention models

The catalog SHALL accept, and the Reference CPU SHALL implement:
- `attention.scale`;
- `rope.partial_rotary_factor`;
- `gelu.approximate = "tanh"`;
- `rmsnorm.head_count` and `rmsnorm.head_dimension`;
- `activation` kinds `softcap` (`cap`) and `scale` (`factor`);
- `embedding.scale`;
- `moe-experts.activation = "gelu_tanh"` and its `per_expert_scale` input;
- `mul` by a `[1]` operand.

An absent attribute SHALL keep the existing behavior.

#### Scenario: Each new form matches an independent computation
- **WHEN** the Reference CPU runs each new attribute or form
- **THEN** its output matches a fixture computed independently, within f32 tolerance

#### Scenario: Unsupported values fail closed
- **WHEN** an attribute carries a value outside its definition (a non-positive `cap`, a `partial_rotary_factor` outside `(0, 1]`, an unknown `approximate` or `activation`)
- **THEN** the kernel refuses the invocation instead of computing something else
