## ADDED Requirements

### Requirement: Inference API Does Not Expose Raw Tensor Storage

Runtime Inference API SHALL not expose raw tensor storage, pointers, native handles, or Provider-owned opaque internals.

#### Scenario: Diagnostics include tensor

Given diagnostic includes tensor metadata

When Runtime returns it

Then only stable Tensor Resource metadata is included.

---

### Requirement: Inference API May Report Tensor Usage Metadata

Runtime Inference API SHALL report tensor usage metadata such as memory estimate or residency summary when policy allows.

#### Scenario: Usage report

Given usage report includes memory estimate

When caller receives it

Then it does not include raw tensor values or memory addresses.