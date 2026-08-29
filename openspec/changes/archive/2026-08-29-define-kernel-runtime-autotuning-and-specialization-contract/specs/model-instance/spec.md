## ADDED Requirements

### Requirement: Model Instance May Have Autotuning Policy

Model Instance MAY be configured with disabled, optional, required, or pinned Kernel Autotuning behavior, and Runtime SHALL enforce exactly one active policy at a time.

#### Scenario: Required warmup tuning

Given deployment requires tuning before readiness

When Model Instance loads

Then it remains warming until required tuning completes or fails according to
policy.

---

### Requirement: Optional Autotuning Does Not Necessarily Block Readiness

Optional tuning MAY occur after Model Instance is usable with known-good Kernels, and optional tuning SHALL NOT block Model Instance readiness.

#### Scenario: Background tuning

Given baseline Kernel is ready

When optional tuning begins

Then Model Instance may remain inference-ready.

---

### Requirement: Reproducible Model Instance May Pin Tuning Record

Model Instance MAY pin a known Autotuning Record, and a pinned Autotuning Record SHALL NOT change without explicit reconfiguration.

#### Scenario: Reproducible deployment

Given pinned record remains compatible

When Model Instance executes

Then Runtime uses its authorized specialization policy without live retuning.