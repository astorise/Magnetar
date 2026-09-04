## ADDED Requirements

### Requirement: Model Loading Is Reachable Only Through the Runtime-Sealed Loading API

The Model Loading operation that evaluates trust and produces a loaded model context SHALL NOT be reachable by a caller outside the Runtime-owned Model Loading API, so that the Runtime-sealed trust policy that API enforces cannot be bypassed by supplying a trust decision directly to the underlying loading operation.

#### Scenario: No external direct-load path

Given a caller outside the Runtime-owned Model Loading API

When that caller attempts to supply their own trust decision directly to the underlying Model Loading operation

Then no such path is reachable outside the crate implementing Runtime
