## ADDED Requirements

### Requirement: A05 Production CLI and minimal service

The CLI and an external minimal service SHALL use the existing Runtime Inference API for authorized real local models, token streaming and concurrent requests, without introducing transport into Runtime Core.

#### Scenario: A05 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** N authorized requests load a real model and generate through the CLI or service
- **THEN** each stream and cancellation is attributed to its Runtime session and results match the API reference

#### Scenario: A05 Ineligible or failed execution

- **WHEN** a client requests an unauthorized model path or exceeds admission limits
- **THEN** the boundary rejects it without granting filesystem, network, tool or shell authority to Components

### Requirement: production-inference-serving-boundary promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
