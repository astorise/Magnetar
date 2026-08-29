## ADDED Requirements

### Requirement: Manifest Normalizes To Kernel Artifact Contracts

Validated portable manifest SHALL normalize into existing Kernel Artifact
domain types.

#### Scenario: Source descriptor

Given valid Triton source descriptor

When normalized

Then it becomes KernelSourceArtifact-compatible internal metadata.

---

### Requirement: Artifact Content Identity Survives Transport

Kernel Artifact identity SHALL remain based on content rather than bundle
transport representation.

#### Scenario: ZIP versus directory

Given same manifest/blobs are represented as directory and archive

When logical identities are computed

Then Kernel Artifact digests remain identical.

---

### Requirement: Multiple Compiled Variants Supported

A logical Kernel MAY carry multiple compiled variants, and Runtime SHALL treat all variants as belonging to the same logical Kernel identity regardless of count.

#### Scenario: sm80 and sm90

Given manifest contains both

When running on sm90

Then sm90-compatible variant may be selected without changing logical Operator
semantics.

---

### Requirement: Compiled Artifact Preserves Source Relationship

Compiled artifact SHOULD reference source digest where known, and when declared, the source relationship SHALL use immutable digest identity rather than a mutable location hint.

#### Scenario: Generated candidate lineage

Given source S compiled into binary B

When manifest is inspected

Then B may identify S as source artifact.

---

### Requirement: Portable Artifact Has No Runtime Handle Ownership

Kernel Artifact SHALL not own Provider/Device/native execution handle.

#### Scenario: Bundle cache import

Given bundle is cached

When Runtime process restarts

Then artifacts remain valid data but require new Provider preparation.