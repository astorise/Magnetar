## ADDED Requirements
### Requirement: KV Pages Have Explicit Device Ownership

Device-resident KV page SHALL identify authoritative Device residency.

#### Scenario: Session decodes on GPU1

Given pages allocated on GPU1

When KV state is inspected

Then GPU1 ownership/residency is explicit.

### Requirement: Decode Placement Favors KV Locality

Runtime SHALL account for KV movement cost when selecting decode placement.

#### Scenario: GPU0 slightly faster

Given Session KV is on GPU1

And moving KV would cost more than expected gain

When placement ranks options

Then GPU1 may remain preferred.

### Requirement: Session Device Migration Moves KV Explicitly

Changing decode Device SHALL not magically retarget existing KV memory.

#### Scenario: GPU1 to GPU0 migration

Given Session migrates

When GPU0 decode begins

Then required KV is transferred/replicated according to explicit policy first.

### Requirement: KV Partition Requires Supported Attention Contract

Runtime SHALL not partition KV across Devices unless selected execution contract
supports the partitioning.

#### Scenario: Standard local Attention Kernel

Given Kernel expects complete local KV

When KV is split across GPU0/GPU1

Then Kernel is ineligible without explicit reconstruction/compatible path.

### Requirement: KV Device Failure Invalidates Affected Session State

Loss of authoritative KV Device SHALL make affected Session unable to continue
unless valid replica/migration recovery exists.

#### Scenario: GPU1 lost

Given Session KV exists only on GPU1

When Device fails

Then Runtime does not fabricate valid KV on GPU0.
