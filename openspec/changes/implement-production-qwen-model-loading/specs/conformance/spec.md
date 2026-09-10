## ADDED Requirements

### Requirement: Production Model Loading Conformance Uses Real Format Parsers And No Fixture Manifest

Conformance for the production Qwen loading profile SHALL include a deterministic small bundle that uses the real production ingestion path and real format parsers.

The bundle MAY use a tiny architecture for test cost, but SHALL contain real production-layout files and SHALL NOT use fixture-only manifest or tensor-inventory constructors to enter Model Loading.

#### Scenario: Per-PR production bundle
- **WHEN** the production loading conformance test runs
- **THEN** `config.json`, tokenizer files, Safetensors files/index, normalized manifest creation, Model Loading, Component graph production, weight materialization, and generation all traverse the production path.

---

### Requirement: Production Qwen Loading Has Real Checkpoint Smoke Evidence

Release evidence for production Qwen Model Loading SHALL include at least one public Qwen-compatible checkpoint pinned by immutable revision and/or digest.

#### Scenario: Reference CPU real-checkpoint smoke
- **WHEN** the real-checkpoint smoke profile runs on Reference CPU
- **THEN** the checkpoint is ingested, loaded, executed, and produces the expected bounded output/logit evidence without fixture shortcuts.

#### Scenario: CUDA real-checkpoint smoke
- **WHEN** the CUDA production profile is selected on a compatible hardware runner
- **THEN** the same normalized Model Artifact executes through the CUDA Provider
- **AND** the test proves CUDA Device residency/execution rather than silently falling back to Reference CPU.

---

### Requirement: GPU Production Loading Gates Cannot Pass Vacuously

A required GPU production loading test SHALL fail its release gate if the hardware-required test was skipped, filtered to zero tests, or did not prove the expected CUDA Provider/Device path.

#### Scenario: Zero GPU tests selected
- **WHEN** a GPU release job selects zero required production-loading tests
- **THEN** the job fails rather than reporting successful CUDA conformance.

---

### Requirement: Production Loading Cleanup Is Conformance-Tested

Conformance SHALL inject failures during production weight loading and prove cleanup at both Runtime Memory Manager and Provider storage boundaries.

#### Scenario: Provider failure after earlier weights staged
- **WHEN** a Provider failure occurs after one or more required weights were staged
- **THEN** conformance proves there are no orphan Provider tensors, leaked Memory Manager allocations, ready bindings, or ready Model Instance state.
