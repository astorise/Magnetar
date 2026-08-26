## ADDED Requirements

### Requirement: CLI Model Instance Operations Use Runtime API

`magnetar-cli` SHALL inspect, warm, suspend, resume, drain, or unload Model
Instances through Runtime Inference API.

#### Scenario: CLI unload

Given user runs model unload

When CLI performs unload

Then Runtime validates active usage and policy before unloading.

---

### Requirement: CLI Does Not Access Instance Internals

CLI SHALL not access raw Model Instance internals such as Provider handles,
Device handles, Kernel handles, Tensor pointers, or raw weights.

#### Scenario: Instance inspect

Given CLI inspects instance

When metadata is displayed

Then only redacted Runtime metadata is shown.