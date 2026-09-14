## ADDED Requirements

### Requirement: A Resident Component Instance Serializes Its Own Generation

A resident, loaded inference Component instance SHALL allow only one generation (prefill through its last decode step) in flight at a time. A second generation request against the same instance while one is already in flight SHALL wait for the first to finish rather than being rejected, silently dropped, or interleaved against the same instance's execution state. This constraint applies per instance, not process-wide: an embedder that needs concurrent generation across independent requests SHALL do so by loading and routing across multiple independent instances, not by expecting one instance to serve overlapping generations itself.

#### Scenario: A second generation call on the same instance waits, not fails

- **GIVEN** a resident Component instance with a generation already in flight
- **WHEN** a second generation request arrives for the same instance
- **THEN** the second request waits for the first to complete rather than returning an error or running concurrently against the same instance's execution state

#### Scenario: Independent instances generate independently

- **GIVEN** two independently loaded resident Component instances
- **WHEN** a generation request is in flight on one
- **THEN** a generation request on the other instance is unaffected by it
