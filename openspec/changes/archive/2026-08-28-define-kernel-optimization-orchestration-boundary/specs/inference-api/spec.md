## ADDED Requirements

### Requirement: Inference API Excludes Optimization Orchestration

Runtime Inference API SHALL NOT expose arbitrary Kernel optimization-agent
execution.

#### Scenario: Generate request contains optimizer URL

Given caller submits optimization-service URL with generation request

When request is validated

Then this authority is rejected as outside inference scope.

---

### Requirement: Inference API Does Not Accept Arbitrary Kernel Source

Normal inference requests SHALL NOT inject executable Kernel source.

#### Scenario: Prompt request carries CUDA source

Given generation request attempts to override MatMul with supplied CUDA code

When Runtime validates request

Then source injection is rejected.

---

### Requirement: Inference Session Does Not Own Optimization Credentials

Inference Session SHALL NOT hold generator/compiler-service credentials.

#### Scenario: External optimizer has API token

Given optimizer token exists

When inference session is inspected

Then token is absent.