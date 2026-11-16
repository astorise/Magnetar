## ADDED Requirements

### Requirement: Optimized CPU Provider Runs Packed Q4_0 Weights

The optimized CPU Provider SHALL accept packed `Q4_0` weights, and its matmuls over them SHALL match the Reference CPU over their dequantized values.

#### Scenario: Q4_0 matmul
- **WHEN** a matmul reads a packed `Q4_0` weight with int8 activations
- **THEN** its result matches the Reference CPU's on the rounded activations and the dequantized weight
