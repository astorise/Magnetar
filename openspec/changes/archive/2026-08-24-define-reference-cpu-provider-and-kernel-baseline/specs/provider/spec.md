## ADDED Requirements

### Requirement: Provider May Be Reference CPU

Provider model SHALL support a built-in or test-enabled Reference CPU Provider.

#### Scenario: Reference Provider

Given Runtime policy enables the Reference CPU Provider

When Provider registry initializes

Then Reference CPU Provider may register.

---

### Requirement: Reference Provider Still Uses Provider Contract

Reference CPU Provider SHALL follow the normal Provider, Device, Kernel,
Registry, Dispatch, health, readiness, pressure, and observability contracts.

#### Scenario: CPU Provider status

Given Reference CPU Provider is registered

When Runtime queries Provider status

Then it returns standard Provider status metadata.

---

### Requirement: Provider Fallback Must Be Policy Controlled

Fallback from another Provider to Reference CPU SHALL be explicit and
policy-controlled.

#### Scenario: Silent fallback attempt

Given CUDA dispatch fails

When Runtime considers CPU fallback

Then it uses Reference CPU only if fallback policy permits it.