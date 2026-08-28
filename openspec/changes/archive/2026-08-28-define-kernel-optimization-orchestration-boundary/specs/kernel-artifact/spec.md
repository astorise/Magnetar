## ADDED Requirements

### Requirement: Optimization Candidates Are Kernel Artifacts

Optimization systems SHALL communicate generated candidates using Kernel
Artifact contracts.

#### Scenario: AI-generated CUDA source

Given AI system produces CUDA source

When candidate is exported

Then it becomes KernelSourceArtifact with content identity/provenance.

---

### Requirement: Campaign Metadata Does Not Change Artifact Identity

Campaign metadata SHALL remain distinguishable from immutable artifact content.

#### Scenario: Same source evaluated twice

Given identical source is evaluated by two campaigns

When artifact digest is computed

Then content identity may remain equal while campaign evidence differs.

---

### Requirement: Generator Provenance Does Not Grant Trust

Generator/campaign identity SHALL not imply artifact trust.

#### Scenario: Known optimization service

Given trusted organization operates generator

When source artifact arrives

Then artifact trust still follows explicit trust/integrity policy.