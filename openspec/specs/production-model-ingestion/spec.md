# production-model-ingestion Specification

## Purpose
This specification defines the Runtime-owned, format-neutral contract external ingestors implement to convert an already-authorized production model source (e.g. a Hugging Face-style bundle) into normalized artifact/config/tokenizer metadata plus bounded payload access, without granting trust and without introducing a concrete format/source dependency into `magnetar-runtime` itself.
## Requirements
### Requirement: Production Model Artifact Ingestion Contract

Magnetar SHALL define a Runtime-owned, format-neutral contract for converting an already-authorized model source into normalized artifact/config/tokenizer metadata plus bounded access to required artifact payloads.

The contract SHALL NOT itself define Hugging Face, Safetensors, GGUF, OCI, CUDA, or model-family-specific identities.

#### Scenario: External ingestor registered
- **WHEN** an embedder registers a compatible concrete Model Artifact ingestor
- **THEN** Runtime can consume its normalized output through generic contracts
- **AND** `magnetar-runtime` does not import that concrete implementation.

---

### Requirement: Production Ingestion Does Not Grant Trust

A production ingestor SHALL NOT grant Model Artifact trust merely because parsing or normalization succeeds.

#### Scenario: Parsed artifact is not trusted
- **WHEN** an ingestor successfully normalizes an artifact whose digest is not trusted by Runtime policy
- **THEN** Runtime Model Loading rejects it before materialization.

---

### Requirement: Production Ingestion Uses Authorized Source Boundaries

A concrete ingestor SHALL only access bytes/files authorized by the source capability supplied for that ingestion operation.

#### Scenario: Arbitrary sibling file requested
- **WHEN** bundle metadata attempts to resolve a file outside the authorized source boundary
- **THEN** the ingestor rejects the access with a structured source/boundary error.

---

### Requirement: Production Ingestion Exposes Bounded Payload Access

The generic ingestion contract SHALL allow Model Loading to obtain required artifact/tensor payload bytes by validated logical identity and bounds without exposing unrestricted raw filesystem/network handles to Model Components or callers.

#### Scenario: Tensor payload requested
- **WHEN** Runtime requests the payload for validated tensor metadata
- **THEN** the ingestor/payload source returns only bytes belonging to that authorized tensor/part range or a structured error.

---

### Requirement: Production Ingestion Preserves Runtime Authority Boundaries

Concrete ingestors SHALL NOT select Providers, Devices, Kernels, Runtime Tensor Resource identities, memory allocations, or Model Instance readiness state.

#### Scenario: Hugging Face bundle targets CUDA
- **WHEN** a Hugging Face ingestor normalizes a Qwen bundle later executed on CUDA
- **THEN** the ingestor output contains no authoritative CUDA/Device selection
- **AND** Provider/Device resolution remains Runtime-owned.

---

### Requirement: Production Ingestion Errors Are Structured

The ingestion boundary SHALL expose stable structured failure categories for unsupported format/profile, unauthorized source, required part missing, malformed metadata, payload unavailable/out-of-bounds, integrity mismatch, and implementation unavailable.

#### Scenario: Required config absent
- **WHEN** the selected production profile requires model configuration and the authorized bundle contains none
- **THEN** ingestion fails with a structured missing/invalid artifact error rather than constructing placeholder configuration.

