## ADDED Requirements

### Requirement: Per-Device Memory Feasibility Ranking Is Verified Against Real Device Capacity

The existing `PlacementCandidate`/`select_lowest_cost_eligible` eligibility-and-ranking logic SHALL be exercised with at least one real Device's own real, discovered memory capacity, not only synthetic fixture values, and SHALL correctly reject a candidate whose required bytes exceed its available budget regardless of that candidate's own ranking cost.

#### Scenario: A real feasible candidate is selected over a cheaper infeasible one

- **GIVEN** two placement candidates for the same required byte size -- one backed by a real Device's real, sufficient memory capacity, one backed by an insufficient budget -- where the insufficient candidate has a lower ranking cost
- **WHEN** the candidates are evaluated
- **THEN** the real, sufficient candidate is selected
- **AND** the insufficient candidate is rejected specifically for memory infeasibility, not any other reason
