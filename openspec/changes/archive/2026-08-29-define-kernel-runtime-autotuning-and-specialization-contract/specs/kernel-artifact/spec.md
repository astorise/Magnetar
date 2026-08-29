## ADDED Requirements

### Requirement: Kernel Artifact May Declare Specialization Template

Accepted Kernel Artifact MAY expose bounded specialization metadata, and any exposed specialization metadata SHALL declare explicit bounds for each tuning axis.

#### Scenario: Compiled template

Given artifact supports multiple tile configurations

When normalized

Then specialization template is attached to the Kernel implementation metadata.

---

### Requirement: Specialized Artifact Identity

Compiled Artifact produced for one specialization SHALL record its
Specialization Instance identity.

#### Scenario: Two tile variants

Given BLOCK_M=32 and BLOCK_M=64 generate different binaries

When stored

Then compiled artifacts retain distinct specialization lineage.

---

### Requirement: Specialization Metadata Does Not Grant Qualification

Presence of valid specialization metadata SHALL not imply specialization is
qualified.

#### Scenario: Newly compiled specialization

Given artifact compiles successfully

When qualification is required

Then artifact remains unqualified until covered evidence exists.