## ADDED Requirements

### Requirement: Generation Fails Closed For An Unsupported Request Shape

Generation SHALL reject a request whose required execution shape the resolved Provider cannot perform, with a structured `Unsupported` error identifying the real constraint, before committing real execution work toward that request. Every Provider defaults to supporting a request shape unless it explicitly declares otherwise; a Provider declaring it cannot service a shape SHALL do so through an explicit, generic capability signal Generation checks once, not through Generation inferring the limitation from a specific Provider's identity.

#### Scenario: Provider cannot service a multi-step decode request
- **WHEN** a generation request requires more than one decode step against a Provider that declares it cannot supply host-readable historical KV state for incremental decode
- **THEN** Generation rejects the request with a structured `Unsupported` error before running prefill, rather than allowing prefill to run and failing partway through decode with an internal residency error.

#### Scenario: Provider supports the requested shape
- **WHEN** a generation request requires more than one decode step against a Provider that declares support for it
- **THEN** Generation proceeds normally; this requirement introduces no new restriction for a Provider capable of the requested shape.

---

### Requirement: Generation Usage Reports Real Measured Throughput

Generation usage metadata's `tokens_per_second` field SHALL be populated from real measured prefill/decode wall-clock time when generation completes, for every Provider, rather than left unset.

#### Scenario: Generation completes successfully
- **WHEN** a generation request completes and produced at least one token
- **THEN** the returned usage metadata's `tokens_per_second` reflects real measured time for that request, not an absent value and not an estimate derived from a fixed assumed rate.
