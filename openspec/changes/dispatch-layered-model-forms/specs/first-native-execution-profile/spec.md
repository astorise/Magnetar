## ADDED Requirements

### Requirement: Optional operator forms reach only Providers that declare them

The first-native engine SHALL forward each layered-model form to the Kernel:
- `embedding.scale`;
- `rmsnorm` with one input and its per-head attributes;
- `rope.partial_rotary_factor`;
- `gelu` with its attributes;
- `moe-experts` with its `activation` and a sixth input.

It SHALL first require each Kernel feature `required_kernel_features` names for the node from every active Kernel of the bound Provider for that operator, and SHALL otherwise refuse the dispatch.

#### Scenario: Provider ignoring partial rotation
- **WHEN** a `rope` node sets `partial_rotary_factor = 0.25` and the bound Provider's `rope` Kernels do not advertise `partial-rotary`
- **THEN** dispatch fails with a Kernel-unavailable error

#### Scenario: Reference CPU
- **WHEN** a graph uses every layered-model form on the Reference CPU
- **THEN** it runs, and its logits match an independent host computation
