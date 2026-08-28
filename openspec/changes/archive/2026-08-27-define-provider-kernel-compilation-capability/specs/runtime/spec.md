## ADDED Requirements

### Requirement: Runtime Coordinates Compilation Cold Path

Runtime MAY coordinate Provider compilation jobs during loading/preparation, and any such coordination SHALL occur outside the decode hot path.

#### Scenario: Model load requires generated kernel

Given compatible compiled artifact does not exist

And policy allows source compilation

When Model Loading runs

Then Runtime may submit Provider compilation job.

---

### Requirement: Runtime Respects Compilation Capability

Runtime SHALL compile only using Provider-advertised compatible capability.

#### Scenario: Provider lacks source format

Given source format is unsupported

When Runtime plans compilation

Then request is rejected before submit.

---

### Requirement: Runtime Does Not Compile On Decode Hot Path

Normal decode path SHALL not synchronously submit compilation jobs.

#### Scenario: Prepared Kernel lost

Given decode finds kernel not ready

When request executes

Then Runtime surfaces structured readiness failure rather than compiler latency.

---

### Requirement: Runtime Applies Compilation Policy

Runtime SHALL evaluate trust, isolation, resource and target policy before
submitting compilation.

#### Scenario: Untrusted source requires sandbox

Given Provider only supports unsandboxed in-process compilation

When policy requires sandbox

Then Runtime denies compilation.

---

### Requirement: Runtime Preserves Provider Device Authority Boundary

Runtime SHALL select target; Provider SHALL implement compilation for that
target.

#### Scenario: Provider receives DeviceBinding

Given Runtime selected Device A

When compilation completes

Then resulting target metadata remains bound to Device A compatibility.