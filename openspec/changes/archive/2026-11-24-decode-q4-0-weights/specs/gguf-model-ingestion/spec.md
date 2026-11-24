## ADDED Requirements

### Requirement: GGUF Ingestion Reads Q4_0 Tensors

GGUF ingestion SHALL read a `Q4_0` tensor (GGML type 2) as storage dtype `Q4`.

A `Q4_0` projection SHALL keep its stored bytes, packed, and SHALL declare transposed storage, as a `Q8_0` projection does.

#### Scenario: A Q4_0 projection
- **WHEN** a GGUF file stores a projection weight as `Q4_0`
- **THEN** its tensor keeps storage dtype `Q4` with transposed storage declared, and Model Loading reads its values as `d * (q - 8)`, transposed
