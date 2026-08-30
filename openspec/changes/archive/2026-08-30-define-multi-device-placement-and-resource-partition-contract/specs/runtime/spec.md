## ADDED Requirements
### Requirement: Runtime Coordinates Local Multi Device Placement

Runtime SHALL coordinate graph, Kernel, memory, Resource, and Device bindings
for local multi-Device execution.

#### Scenario: Two GPU model

Given Model Instance spans GPU0/GPU1

When Plan builds

Then Runtime owns the combined placement decision.

### Requirement: Runtime Couples Kernel And Device Compatibility

Final binding SHALL pair each Operator/segment with a Kernel compatible with
its selected Provider/Device.

#### Scenario: Kernel only supports sm90

Given GPU0 is sm80 and GPU1 is sm90

When Kernel is selected

Then it cannot bind to GPU0.

### Requirement: Runtime Makes Device Movement Explicit

Resource transitions between Devices SHALL not be hidden by ordinary Kernel
dispatch.

#### Scenario: Stage boundary

Given activation changes Device

When Plan executes

Then explicit movement/access edge exists.

### Requirement: Runtime Keeps Multi Device Re Planning Off Hot Path

Normal decode SHALL not synchronously rebuild global placement due to small
dynamic changes.

#### Scenario: GPU pressure increases

Given current Plan remains valid

When token executes

Then Runtime may request background replacement while continuing current Plan.

### Requirement: Runtime Fails Closed On Hard Placement Invalidation

A Plan requiring unavailable Device/path SHALL not receive new work.

#### Scenario: Required peer path lost

Given Plan has no alternate movement path

When next invocation begins

Then Runtime uses fallback/replan/failure rather than unsafe execution.

### Requirement: Runtime Keeps Multi Host Out Of Core Placement Contract

Remote host Devices SHALL not be represented as ordinary local Devices by this
contract.

#### Scenario: Tachyon offers remote GPU

Given remote node exists

When local placement runs

Then remote scheduling remains external/distributed scope.