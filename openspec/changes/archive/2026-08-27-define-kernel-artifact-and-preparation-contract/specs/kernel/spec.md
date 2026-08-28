## ADDED Requirements

### Requirement: Kernel May Be Artifact-Backed

An artifact-backed Kernel SHALL implement the same portable Operator
semantics as a statically defined Kernel. A Kernel implementation MAY be
backed by a Kernel Artifact lifecycle.

#### Scenario: Generated MatMul

Given generated MatMul artifact is prepared

When Kernel Registry advertises it

Then the prepared implementation remains a Kernel implementing portable
MatMul semantics.

---

### Requirement: Kernel Identity Is Separate From Prepared State

KernelId SHALL remain logical implementation identity and SHALL NOT be the
native prepared handle.

#### Scenario: Same Kernel prepared twice

Given same KernelId is prepared for two Devices

When Registry tracks them

Then each PreparedKernelId is distinct while KernelId semantics remain the
same.

---

### Requirement: Kernel Advertisement May Reference Artifact Metadata

Artifact metadata referenced by KernelAdvertisement SHALL NOT replace
KernelId as the authoritative logical identity. KernelAdvertisement MAY
reference artifact identity and preparation metadata.

#### Scenario: Generated kernel advertisement

Given generated kernel is advertised

When Registry evaluates it

Then artifact identity and build fingerprint may participate in selection.

---

### Requirement: Kernel Native State Remains Provider Private

Kernel contracts SHALL not expose Provider-native executable pointers.

#### Scenario: CUDA Kernel

Given CUDA Provider owns CUfunction

When Kernel metadata is returned

Then CUfunction address is absent.