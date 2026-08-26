## ADDED Requirements

### Requirement: Fused Kernels Declare Semantic Equivalence

Post-baseline fused Kernels SHALL declare semantic equivalence to portable
Operator sequences or graph fragments.

#### Scenario: Fused MLP

Given Provider advertises fused MLP Kernel

When Runtime validates it

Then metadata identifies the equivalent portable Operator sequence.

---

### Requirement: Advanced Kernels Declare Specialized Requirements

Advanced Kernels SHALL declare dtype, layout, memory class, precision,
determinism, and Resource Affinity requirements.

#### Scenario: Flash attention kernel

Given Provider advertises flash attention

When Kernel metadata is inspected

Then required layout, dtype, memory class, and precision tolerance are explicit.