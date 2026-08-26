## ADDED Requirements

### Requirement: Tensor Release Gate

Tensor Resource and Layout contracts SHALL have release gate coverage.

#### Scenario: Tensor pointer leak

Given Tensor Resource API exposes pointer

When release validation runs

Then stable release is blocked.