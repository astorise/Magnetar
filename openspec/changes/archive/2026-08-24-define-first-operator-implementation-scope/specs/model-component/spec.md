## ADDED Requirements
### Requirement: First Baseline Model Component Uses Scoped Operators

A Model Component used for the first baseline SHALL declare only operators that
are implemented or explicitly allowed by first scope policy.

#### Scenario: Unsupported operator requirement

Given Model Component declares flash-attention as mandatory

When first baseline validates the Component

Then Runtime rejects it or requires a non-flash attention graph alternative.

---

### Requirement: Model Component May Provide Unfused Graph

For first baseline execution, Model Component SHALL be able to provide unfused
graphs using required-now operators.

#### Scenario: Unfused MLP

Given fused MLP is unavailable

When graph production is requested

Then Model Component may emit matmul, SiLU, mul, and matmul sequence.
