## ADDED Requirements

### Requirement: Quantized Tables Stay Packed

Weight loading SHALL offer a block-quantized two-dimensional tensor stored untransposed, such as an embedding table, to the Provider in its packed encoding when its stored rows are whole blocks, as it offers a transposed projection. A Provider that cannot read it SHALL decline it and receive it dequantized to `f32`.

#### Scenario: A k-quant embedding table
- **WHEN** a checkpoint holds a `Q6_K` embedding table whose rows are whole super-blocks
- **THEN** it is offered packed, untransposed, and its resident size is its storage size
