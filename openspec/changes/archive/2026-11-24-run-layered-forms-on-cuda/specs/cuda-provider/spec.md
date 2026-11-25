## ADDED Requirements

### Requirement: CUDA Kernels honour the layered models' optional forms

The CUDA Provider SHALL implement these optional forms, and SHALL declare each one's Kernel feature on the Kernels that honour it:
- the attention score scale;
- the scaled embedding;
- per-head and weightless RMSNorm;
- partial RoPE;
- GELU-gated routed experts;
- per-expert output scales.

An attention Kernel given no scale SHALL compute with `1/sqrt(head_dimension)`, as before.

#### Scenario: Scaled attention on CUDA
- **WHEN** an `attention` node sets `scale = 1.0` and runs on the CUDA Provider
- **THEN** the scores use that factor, and the output matches the Reference CPU's
