## ADDED Requirements

### Requirement: Component-Served Observed Source

The ingestion gateway SHALL record a bundle served by a Component as observed from a Component-served source identified by that Component Artifact's digest, distinct from the manifest's source claim. Like every observed source, it SHALL NOT grant trust.

#### Scenario: Observed and claimed sources differ

- **WHEN** a Component serves a bundle whose manifest claims a vendor source
- **THEN** the transaction records the Component-served source and the claim separately, and reports them as conflicting
