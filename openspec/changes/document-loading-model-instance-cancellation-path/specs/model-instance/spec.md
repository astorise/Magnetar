## ADDED Requirements

### Requirement: Loading Model Instance Can Be Canceled

A Model Instance in `Loading` or any other non-terminal lifecycle state SHALL be releasable without first reaching `Ready`: explicitly failing it SHALL be accepted regardless of its current lifecycle state, and a failed instance SHALL then be unloadable through the normal unload path. Unloading an instance that never had any resources bound SHALL report an empty release set rather than an error.

#### Scenario: Abandoned Loading instance is failed and unloaded

Given a Model Instance was created and is still in `Loading`, with no weights ever materialized for it

When the instance is explicitly failed

Then it transitions to `Failed` regardless of having been in `Loading`, not `Ready`

#### Scenario: A failed, never-materialized instance unloads cleanly

Given a Model Instance was failed while still in `Loading`

When it is unloaded

Then unload succeeds, the instance reaches `Unloaded`, and the unload report's released weight resources and memory allocations are both empty
