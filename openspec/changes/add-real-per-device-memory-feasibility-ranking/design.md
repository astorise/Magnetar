## Context

`magnetar_runtime::multi_device_placement` already contained a real, working eligibility-and-cost-ranking algorithm (`PlacementCandidate::is_eligible`, `select_lowest_cost_eligible`) -- 24 of the module's own unit tests exercise it, but every one of them constructs `PlacementCandidate` values by hand with synthetic numbers. No test anywhere in this repository's history had populated `available_memory_bytes`/`required_memory_bytes` from a real Device's own real, discovered capacity.

## Goals / Non-Goals

**Goals:**
- Real `available_memory_bytes` sourced from a real GPU's real `DeviceMetadata.memory_capacity` for at least one candidate.
- A deterministic, real infeasibility rejection, honestly labeled as using an artificially constrained budget where genuine heterogeneous hardware is unavailable.
- Confirmation that eligibility is checked before cost ranking, using real candidate data.

**Non-Goals:**
- Any change to `magnetar_runtime`'s existing `PlacementCandidate`/`select_lowest_cost_eligible` implementation -- it needed none.
- Testing against genuinely heterogeneous real GPU capacities (24 GiB vs 8 GiB, as the spec's own example scenario describes) -- this repository's real hardware is two identical GPUs.
- Wiring this feasibility check into any real dispatch decision -- like the rest of `multi_device_placement`, it remains a callable, verified primitive, not yet consulted by any real execution path.

## Decisions

- **Constrain one candidate's budget rather than skip the infeasibility case entirely.** An alternative considered: only prove the "both real, both feasible" case, since no genuinely heterogeneous real budget exists. Rejected: the "Per Device Memory Feasibility" requirement is specifically about *rejection* under infeasibility, and a same-capacity-always-fits test would never exercise that branch at all. An artificially constrained value, honestly documented as such (matching this repository's own precedent, `check_weight_materialization_failure_never_reaches_ready`'s constrained `max_runtime_bytes`), gives a real, deterministic proof of the rejection path without overstating what hardware was actually available.
- **Give the infeasible candidate a lower cost than the feasible one.** Makes the test also a real proof of requirement ordering (eligibility before ranking) using the same real data, rather than needing a second, separate test for that property.

## Risks / Trade-offs

- **The infeasibility case is not driven by genuine hardware heterogeneity** -- a future real test against actually different-capacity real GPUs would be strictly stronger evidence, but no such hardware exists in this repository's tooling.
