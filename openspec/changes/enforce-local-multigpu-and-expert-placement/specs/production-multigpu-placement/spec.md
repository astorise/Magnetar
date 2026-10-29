## ADDED Requirements

### Requirement: G05 Enforced local multi GPU placement

Production multi-GPU profiles SHALL enforce versioned layer/tensor placement, per-Device reservations and explicit P2P, host-copy or collective edges before submission.

#### Scenario: G05 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a real model is partitioned across distinct local GPUs
- **THEN** logits and multi-step tokens match the unsplit reference and movement/capacity costs are recorded

#### Scenario: G05 Ineligible or failed execution

- **WHEN** a Device is lost or a required peer link/collective is unavailable
- **THEN** dependent plans invalidate and only an explicitly authorized replan may resume with compatible state

### Requirement: G06 Conditional expert parallelism

Multi-GPU MoE SHALL qualify expert placement, duplication and load balancing under explicit interconnect capacity, routing ownership and per-Device budgets.

#### Scenario: G06 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a real multi-GPU MoE workload changes expert demand
- **THEN** expert balance, communication, duplication memory and accepted throughput are measured

#### Scenario: G06 Ineligible or failed execution

- **WHEN** rebalancing would violate affinity or exceed link/memory budgets
- **THEN** the new placement is rejected and the current valid generation remains or fails explicitly

### Requirement: production-multigpu-placement promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
