## ADDED Requirements
### Requirement: Generation First Baseline Uses Scoped Graphs

Generation using the first executable baseline SHALL execute only graphs whose
operators are within first scope.

#### Scenario: Decode first baseline

Given decode graph uses required-now operators

When Generation runs

Then Runtime may execute it through Reference CPU kernels.

---

### Requirement: Generation Does Not Require Provider-Assisted Sampling In First Scope

The first operator scope SHALL not require Provider-assisted sampling.

#### Scenario: Logits produced

Given decode graph produces logits

When token selection is needed

Then Sampling Contract may run separately from Provider-assisted graph operators.