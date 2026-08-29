## ADDED Requirements

### Requirement: Provider May Advertise Bounded Specialization Parameters

Provider MAY expose tunable implementation parameters with explicit bounded domains, and Provider SHALL declare explicit bounds for every advertised parameter.

#### Scenario: Warp count

Given CUDA Kernel supports warp counts 4 and 8

When Provider advertises tuning metadata

Then arbitrary warp count is not accepted.

---

### Requirement: Provider Hints Are Non-Authoritative

Provider MAY recommend specialization defaults/order, but Runtime SHALL retain
final authority over specialization selection.

#### Scenario: Provider recommends fastest default

Given default is memory-infeasible

When Runtime evaluates it

Then recommendation does not override Memory Manager.

---

### Requirement: Provider Autotuning Is Not Arbitrary Generation

Provider-local autotuning SHALL not expand declared candidate domain or invent
new source.

#### Scenario: Native vendor tuner

Given Provider tests launch configurations

When tuning runs

Then tested configurations remain within advertised bounded domain.

---

### Requirement: Provider Autotuning Avoids Decode Hot Path

Provider SHALL not silently start expensive tuning during normal execute.

#### Scenario: First token execution

Given native pipeline has not been tuned

When execute is called

Then Provider follows prepared/default behavior or structured failure rather
than blocking with undeclared autotuning.