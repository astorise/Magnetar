## ADDED Requirements

### Requirement: Model Instance Operations Are Exposed Safely

Runtime Inference API SHALL expose Model Instance lifecycle operations such as create, inspect, warm, suspend, resume, drain, and unload.

#### Scenario: Inspect instance

Given caller inspects Model Instance

When Runtime returns metadata

Then it does not expose Provider handles, Device handles, Kernel handles, or raw
tensor pointers.

---

### Requirement: Inference API Respects Active Instance Use

Model Instance lifecycle operations through Runtime Inference API SHALL respect active sessions and generations.

#### Scenario: Unload active instance

Given Model Instance has active generation

When unload is requested

Then Runtime drains, rejects, waits, or forces according to policy.