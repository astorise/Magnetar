## ADDED Requirements

### Requirement: Accepted Artifact State Is Distinct

Kernel Artifact SHALL distinguish ingestion acceptance from trust,
qualification, preparation and promotion.

#### Scenario: Structurally valid source imported

Given source is accepted into cache

When inspected

Then acceptance alone does not imply qualified/trusted/prepared state.

---

### Requirement: Artifact Retains Ingestion Provenance

Accepted artifact metadata SHALL retain its content-addressed identity even
when it references the ingestion transaction/source audit record.

Such a reference MAY be included in metadata.

#### Scenario: Artifact origin investigated

Given Kernel later fails qualification

When audit is inspected

Then ingestion transaction can be correlated without changing artifact digest.

---

### Requirement: Immutable Content Identity Through Ingestion

Ingestion SHALL preserve Kernel Artifact content identity.

#### Scenario: Deduplicated blob

Given existing cache blob matches digest

When new manifest references it

Then same immutable blob identity is used.