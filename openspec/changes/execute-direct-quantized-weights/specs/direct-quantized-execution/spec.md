## ADDED Requirements

### Requirement: B03 Direct GGUF quantized execution

Providers SHALL compute directly from validated Q8_0 and then Q4_K/Q5_K packed weights, preserving block scales, zero conventions and layouts without mandatory whole-model F32 dequantization.

#### Scenario: B03 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a real checkpoint in a supported quantized profile generates multiple tokens
- **THEN** kernel and residency evidence proves packed-weight computation and quality passes

#### Scenario: B03 Ineligible or failed execution

- **WHEN** a block is truncated or quantization metadata is incompatible
- **THEN** ingestion or preparation rejects it before dispatch and does not silently relabel dequantized execution as direct

### Requirement: B04 Direct GPTQ and AWQ matrix execution

Qualified GPTQ/AWQ profiles SHALL validate packing, group sizes, scales, zeros and calibration provenance and execute direct matrix kernels on real checkpoints.

#### Scenario: B04 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a declared GPTQ or AWQ checkpoint completes generation
- **THEN** logits, downstream quality and memory are compared against its calibrated reference

#### Scenario: B04 Ineligible or failed execution

- **WHEN** a group layout or calibration profile is unsupported
- **THEN** the loader or planner returns an explicit incompatibility and any dequantized fallback is opt-in and separately measured

### Requirement: direct-quantized-execution promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
