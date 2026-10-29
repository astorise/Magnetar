## ADDED Requirements

### Requirement: G08 Tachyon prefill decode integration proposal

An optional Tachyon adapter SHALL own inter-node routing, transport, global admission and cost-aware prefill/decode transfer while Magnetar retains local validation/execution and standalone operation.

#### Scenario: G08 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** audited and approved interfaces run on real multiple nodes
- **THEN** P/D is compared with colocated execution for KV transfer cost, TTFT, throughput and backpressure

#### Scenario: G08 Ineligible or failed execution

- **WHEN** transport costs dominate, authorization fails or an interface is unapproved
- **THEN** the integration remains an unvalidated proposal or is disabled without coupling Runtime Core to Tachyon

### Requirement: G09 Distributed session recovery and drain

The distributed adapter SHALL coordinate checkpoint/resume, draining, quotas and generation rollback with explicit ownership handoff and fencing of the previous token producer.

#### Scenario: G09 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a node drains or fails after an acknowledged token boundary
- **THEN** a compatible checkpoint resumes under one authorized owner with a declared output replay policy

#### Scenario: G09 Ineligible or failed execution

- **WHEN** the source is partitioned, transfer is partial or the latest state is uncertain
- **THEN** fencing prevents duplicate producers and recovery uses a verified boundary or explicit failure rather than silent token loss

### Requirement: G10 Evidence gated remote pooling

Remote memory pooling, external KV and RDMA/NVLink transport experiments SHALL require real topology evidence, explicit authorization, bounded quotas and a measured advantage over simpler transfer or re-prefill.

#### Scenario: G10 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a justified infrastructure profile runs a pooling experiment
- **THEN** latency, throughput, integrity, isolation and failure behavior are measured on the actual interconnect

#### Scenario: G10 Ineligible or failed execution

- **WHEN** the link is unavailable, unauthorized or slower than the baseline
- **THEN** the feature stays disabled and local execution or an explicitly supported transport remains usable

### Requirement: distributed-inference-integration-proposal promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
