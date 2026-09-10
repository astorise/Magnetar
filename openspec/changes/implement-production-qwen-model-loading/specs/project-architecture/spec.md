## ADDED Requirements

### Requirement: Concrete Production Model Artifact Ingestors Are Externalized

Concrete production Model Artifact ingestion implementations that understand source/bundle formats such as Hugging Face directory structure, Safetensors index conventions, or other distribution-specific layouts SHALL live outside `magnetar-runtime` and depend on Runtime-owned generic contracts, never the reverse.

The Runtime MAY define generic ingestion, payload-access, validation, and registration contracts.

#### Scenario: Hugging Face production ingestor
- **WHEN** Magnetar adds production Hugging Face bundle loading
- **THEN** the concrete implementation is an external/pinned module that may depend on `magnetar-runtime` public contracts and external format modules
- **AND** `magnetar-runtime` has zero compile-time dependency on that implementation.

#### Scenario: Runtime uses ingestor
- **WHEN** an embedder registers a compatible concrete ingestor
- **THEN** Runtime may consume the normalized output through its generic contract without learning Hugging Face/Safetensors-specific types.
