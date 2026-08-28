## ADDED Requirements

### Requirement: Runtime Remains Inference-Only

Magnetar Runtime SHALL not become an AI/kernel optimization-agent host.

#### Scenario: Generator requires network/model API

Given generator needs remote LLM

When Runtime serves inference

Then Runtime does not invoke generator.

---

### Requirement: Runtime Consumes Artifacts And Evidence

Runtime MAY consume validated Kernel Artifacts and Optimization Evidence, but SHALL validate them using existing trust and qualification contracts before use.

#### Scenario: Optimization completed externally

Given candidate artifact/evidence are available

When Runtime considers candidate

Then it validates them using existing contracts.

---

### Requirement: Runtime Does Not Trust External Recommendation

Runtime SHALL treat optimization recommendation as non-authoritative.

#### Scenario: Recommendation says production-ready

Given candidate lacks current trust

When Runtime evaluates it

Then recommendation does not bypass trust.

---

### Requirement: Runtime Does Not Require Optimization Network

Runtime SHALL execute compatible prepared inference without external
optimization-service connectivity.

#### Scenario: Offline deployment

Given all required artifacts are local

When network is unavailable

Then inference can continue.

---

### Requirement: Runtime Owns Production Promotion Decision

Runtime/deployment policy SHALL remain authoritative for Kernel promotion.

#### Scenario: Optimization campaign ends

Given new candidate is recommended

When production policy denies promotion

Then current active Kernel remains active.