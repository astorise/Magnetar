## ADDED Requirements

### Requirement: Parse Result Exposes The Tensor Data Section's Absolute Offset

The GGUF parser SHALL expose the absolute byte offset, from the start of the parsed file, at which the tensor data section begins, so a caller can compute each tensor's real absolute byte range from its already-returned data-section-relative `offset_bytes` without re-deriving the parser's own alignment and header-layout computation independently.

#### Scenario: The exposed offset addresses real tensor bytes

- **WHEN** a well-formed GGUF file is parsed
- **THEN** adding the exposed tensor data section offset to any tensor's own `offset_bytes` yields that tensor's real, correct absolute position in the original file
