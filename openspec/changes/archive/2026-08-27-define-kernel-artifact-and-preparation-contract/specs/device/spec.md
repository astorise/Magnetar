## ADDED Requirements

### Requirement: Device Remains Hardware Abstraction

Device SHALL remain a representation of hardware identity, capabilities,
availability, health, pressure, and execution limits.

#### Scenario: Kernel compilation needed

Given Kernel Source Artifact needs compilation

When architecture assigns responsibility

Then Device is not the compiler.

---

### Requirement: Device Does Not Own Kernel Source

Device SHALL not accept arbitrary kernel source through its public contract.

#### Scenario: Triton source

Given Triton source exists

When Provider prepares it

Then Device receives no source-management responsibility.

---

### Requirement: Device Does Not Expose Executable Pointer

Device SHALL not expose native kernel executable pointers.

#### Scenario: GPU device metadata requested

Given diagnostics inspect Device

When metadata is returned

Then native loaded kernel addresses are absent.