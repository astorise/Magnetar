## MODIFIED Requirements

### Requirement: Qwen Architecture Metadata Validation

Qwen Model Component SHALL validate decoder-only architecture metadata: every dimension non-zero, and a KV head count that divides the attention head count. The hidden size SHALL NOT be required to equal the attention head count times the head dimension, since a checkpoint may declare a head width apart from it; the query and output projections' expected shapes carry both widths.

#### Scenario: Invalid head configuration
- **WHEN** Qwen config validation runs on a zero head dimension, or a KV head count that does not divide the attention head count
- **THEN** Runtime returns qwen-config-invalid or qwen-architecture-unsupported

#### Scenario: A declared head width apart from the hidden size
- **WHEN** Qwen config validation runs on a hidden size that is not the attention head count times the head dimension
- **THEN** the configuration is accepted
