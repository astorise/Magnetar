## ADDED Requirements

### Requirement: Qwen Baseline Operators Are First-Scope Operators

Qwen baseline SHALL use first-scope required Operators unless explicit policy
allows additional implemented Operators.

#### Scenario: Required operator set

Given Qwen baseline graph is inspected

When operator scope validation runs

Then all Operators are required-now or explicitly supported.

---

### Requirement: Qwen Operator Requirements Are Portable

Qwen operator requirements SHALL reference portable Operator IDs, not
Provider-specific Kernel names.

#### Scenario: Kernel-specific requirement

Given Qwen Component declares `cuda.flash_attention_v2`

When Runtime validates requirements

Then validation fails or marks it non-authoritative invalid metadata.