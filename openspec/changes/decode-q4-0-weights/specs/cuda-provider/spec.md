## ADDED Requirements

### Requirement: CUDA Provider Runs Packed Q4_0 Weights

The CUDA Provider SHALL accept packed `Q4_0` weights. Downloading or expanding one SHALL give back the Runtime's dequantization exactly, and its matmuls SHALL match the Reference CPU's.

#### Scenario: Q4_0 decode matmul
- **WHEN** a decode matmul reads a packed `Q4_0` weight with int8 activations
- **THEN** its result matches the Reference CPU's on the rounded activations and the dequantized weight
