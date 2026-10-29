## RENAMED Requirements

- FROM: `### Requirement: Kernel Registry Candidate Selection Is Provider-Ranked, Not Provider-Filtered By Affinity`
- TO: `### Requirement: Kernel Candidate Enumeration Is Distinct From Affinity Safe Selection`

## MODIFIED Requirements

### Requirement: Kernel Candidate Enumeration Is Distinct From Affinity Safe Selection

Kernel Registry MAY enumerate structurally compatible candidates across registered Providers for diagnostics and explicit movement planning. Enumeration SHALL NOT authorize execution. Production selection SHALL filter hard trust, ResourceAffinity, Device, dtype/layout and memory constraints before performance ranking; a caller SHALL NOT be required to repair an unsafe top-ranked selection. Explicitly authorized movement MAY create a newly eligible destination resource before a new selection.

#### Scenario: Faster foreign Provider conflicts with live residency

- **GIVEN** live tensor affinity bound to Provider A and a faster candidate on Provider B
- **WHEN** production selection runs without an authorized movement edge
- **THEN** B is excluded before ranking and only an eligible candidate may dispatch

#### Scenario: Diagnostic enumeration across Providers

- **WHEN** tooling requests all structurally compatible kernels
- **THEN** both Providers may be listed with eligibility reasons and listing does not grant dispatch authority
