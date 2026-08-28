## ADDED Requirements

### Requirement: Runtime Coordinates Kernel Preparation

Runtime SHALL NOT mark a Model Instance ready when required Kernel
preparation has failed. Runtime MAY coordinate Kernel Artifact validation and
Provider preparation as part of model/execution readiness.

#### Scenario: Model load

Given required Kernel has compiled artifact but is not prepared

When Model Instance is being loaded

Then Runtime may request Provider preparation before marking instance ready.

---

### Requirement: Runtime Does Not Compile On Normal Decode Hot Path

Runtime SHALL NOT synchronously compile Kernel Source Artifact in the normal
token decode loop.

#### Scenario: Kernel unavailable during decode

Given required Kernel is not prepared

When decode reaches it

Then Runtime returns structured readiness/admission failure according to policy.

---

### Requirement: Runtime Treats Prepared Kernel Id As Opaque

Runtime SHALL treat PreparedKernelId as opaque.

#### Scenario: Dispatch

Given PreparedKernelId is passed to Provider

When Runtime handles it

Then Runtime does not reinterpret its numeric value.

---

### Requirement: Runtime Remains Generator Independent

Runtime SHALL not depend on external AI/kernel generation system APIs.

#### Scenario: Human-authored artifact

Given human-generated compiled kernel is valid

When Runtime prepares it

Then it follows same lifecycle as AI-generated artifact.