## ADDED Requirements
### Requirement: Provider Advertises Device Pair Capabilities

Provider SHALL expose capabilities relevant to local multi-Device execution per
Device pair.

#### Scenario: GPU peer topology

Given GPU0 can peer-copy to GPU1 but not peer-read

When capabilities are queried

Then distinction is explicit.

### Requirement: Provider Does Not Own Placement Policy

Provider MAY supply capability/performance metadata but SHALL not make final
cross-Device placement decision.

#### Scenario: Provider prefers GPU1

Given Runtime memory policy excludes GPU1

When placement occurs

Then Provider preference cannot override Runtime eligibility.

### Requirement: Provider Executes Runtime Placement

Provider SHALL execute or prepare work for the Device binding selected by
Runtime.

#### Scenario: Stage bound to GPU0

Given Prepared Plan names GPU0-compatible binding

When Kernel executes

Then Provider does not silently move it to GPU1.

### Requirement: Provider Reports Hidden Staging Requirement

If native peer operation requires host staging, Provider SHALL expose that fact
to Runtime policy.

#### Scenario: Peer API unavailable

Given Provider could internally stage through host

When host staging is forbidden

Then Provider cannot hide fallback.

### Requirement: Provider Native Peer State Is Private

Native peer handles/topology objects SHALL remain Provider-private.

#### Scenario: CUDA peer access enabled

Given Provider has native peer state

When placement metadata is inspected

Then raw CUDA Device handles are absent.
