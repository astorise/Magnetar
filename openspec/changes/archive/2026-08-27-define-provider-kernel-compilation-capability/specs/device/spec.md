## ADDED Requirements

### Requirement: Device Trait Remains Compilation-Free

Device public contract SHALL NOT gain arbitrary kernel source compilation
methods.

#### Scenario: PM proposes Device::compile

Given Provider requires Triton compilation

When architecture is implemented

Then capability is added to Provider, not Device.

---

### Requirement: Device Binding Is Sufficient Target Reference

Runtime SHALL identify compilation target using portable Device binding and
metadata.

#### Scenario: Native CUDA device exists

Given Provider maps DeviceBinding internally to native CUDA device

When compilation occurs

Then native mapping remains private.