## ADDED Requirements

### Requirement: Runtime Validates Bundle Before Preparation

Runtime SHALL complete structural/integrity validation before Provider
preparation.

#### Scenario: CUBIN digest mismatch

Given binary does not match manifest digest

When imported

Then Provider.prepare is never called.

---

### Requirement: Runtime Normalizes Portable Manifest

Runtime SHALL convert validated portable metadata into existing Kernel domain
contracts.

#### Scenario: Multi-target manifest

Given multiple variants exist

When normalized

Then each candidate retains explicit target compatibility.

---

### Requirement: Runtime Re-evaluates Trust

Manifest trust claims SHALL pass Runtime/deployment trust policy.

#### Scenario: Self-declared publisher

Given manifest claims known publisher

When no authenticated mechanism proves it

Then Runtime does not infer trust.

---

### Requirement: Runtime Re-evaluates Qualification

Portable evidence SHALL be checked against current qualification policy.

#### Scenario: Qualification expired

Given evidence exists but is stale

When Runtime considers Kernel

Then Kernel is not treated as currently qualified.

---

### Requirement: Runtime Does Not Automatically Fetch External Locations

External artifact references SHALL use explicitly authorized sources.

#### Scenario: Offline deployment

Given manifest references external source but required artifact exists in local
cache

When Runtime resolves it

Then no network fetch is required.

---

### Requirement: Runtime Manifest Ingestion Is Not Generation Request Authority

Normal inference request SHALL not import arbitrary Kernel bundles.

#### Scenario: User prompt carries bundle

Given inference request attempts Kernel injection

When request is validated

Then operation is outside inference scope.