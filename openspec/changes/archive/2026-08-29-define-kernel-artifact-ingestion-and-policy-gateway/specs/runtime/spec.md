## ADDED Requirements

### Requirement: Runtime Consumes Only Accepted Kernel Artifacts

Normal Runtime Kernel planning SHALL not consume staged/quarantined/rejected
artifacts.

#### Scenario: Quarantine has optimal benchmark

Given quarantined Kernel is fastest

When Runtime selects Kernel

Then candidate is unavailable to normal selection.

---

### Requirement: Runtime Inference Is Independent From Ingestion Failure

Ingestion failures SHALL not abort unrelated active Runtime inference.

#### Scenario: Invalid bundle imported during generation

Given generation is using known-good Kernel

When bundle parsing fails concurrently

Then current generation continues according to normal Runtime behavior.

---

### Requirement: Ingestion Does Not Modify Active Registry

Successful import SHALL not automatically alter active Kernel Registry
preference.

#### Scenario: Accepted replacement

Given replacement is committed

When no promotion has occurred

Then active Registry generation is unchanged.

---

### Requirement: Runtime Does Not Expose Ingestion Through Generation Request

Normal generation request SHALL not possess arbitrary Kernel artifact import
authority.

#### Scenario: Client supplies remote Kernel URL

Given generation request contains Kernel bundle URL

When request is validated

Then Runtime rejects it as outside inference authority.