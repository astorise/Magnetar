## ADDED Requirements

### Requirement: Server Facade Uses Inference API

Server API SHALL use Runtime Inference API for model, session, generation,
streaming, cancellation, diagnostics, and usage operations.

#### Scenario: Server generation

Given server receives generation request

When Runtime work is required

Then Runtime Inference API is called.

---

### Requirement: Server Facade Does Not Expose Runtime Internals

Server API SHALL not expose Runtime internal handles through Inference API
responses.

#### Scenario: Provider diagnostic

Given server returns diagnostics

When response is inspected

Then Provider handles, Device handles, Kernel handles, and memory pointers are
absent.