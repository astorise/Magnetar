## MODIFIED Requirements

### Requirement: KV Cache Layout Metadata

KV cache SHALL expose or track layout metadata sufficient for Runtime planning.

Metadata MAY include layer count, head count, head dimension, token capacity,
current token length, block/page metadata, dtype, layout format, and position
range. Metadata SHALL include the KV storage precision of K and of V when the
history is stored in a precision other than its logical dtype.

#### Scenario: Capacity exceeded

- **WHEN** a cache has capacity for 1024 tokens and decode would append token 1025
- **THEN** Runtime reports cache-capacity-exceeded or reallocates according to policy

#### Scenario: Stored precision tracked

- **WHEN** a history's K is stored in `q8_0` and its V in `bf16`
- **THEN** its layout metadata records both precisions and its admitted size follows them
