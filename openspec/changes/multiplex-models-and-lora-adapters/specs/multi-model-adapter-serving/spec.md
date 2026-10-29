## ADDED Requirements

### Requirement: F06 Multiplexed LoRA inference

LoRA/QLoRA inference SHALL validate base-model and target-module compatibility, use dedicated bounded adapter slots and avoid copying the full base model for each adaptation.

#### Scenario: F06 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** concurrent authorized sessions use distinct compatible adapters
- **THEN** outputs match isolated adapted references and KV/prefix keys distinguish adapter identities

#### Scenario: F06 Ineligible or failed execution

- **WHEN** an adapter changes during active work or an unsupported composition is requested
- **THEN** the change is rejected or performed at an explicit safe boundary with incompatible cache invalidation

### Requirement: F08 Concurrent model instance management

Runtime SHALL manage multiple Model Instances with explicit warmup, per-model quotas/reservations and model-bound kernel compatibility and cache identities.

#### Scenario: F08 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** multiple real models share admitted local resources
- **THEN** warmup and request latency are measured without cross-model state or weight aliasing

#### Scenario: F08 Ineligible or failed execution

- **WHEN** one model reloads, exceeds quota or changes generation
- **THEN** only dependent plans/caches are invalidated and unrelated admitted models retain valid leases

### Requirement: multi-model-adapter-serving promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
