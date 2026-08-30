## ADDED Requirements
### Requirement: Generation Integration Gate Uses Model Logits

Generation SHALL not be declared integrated until Sampling consumes logits
produced by Runtime model execution.

#### Scenario: Fake logits still configured

Given Generation can operate from test callback

When native profile runs

Then callback is disabled/not used.

### Requirement: Prefill Occurs Once Per Generation Start

The normal first-profile generation flow SHALL not rerun full prompt prefill for
every generated token.

#### Scenario: Three generated tokens

Given prompt is already prefetched

When subsequent decode runs

Then historical prompt is represented through KV state.

### Requirement: Greedy Golden Sequence Is Exit Gate

At least one deterministic prompt SHALL produce a versioned expected greedy
token sequence.

#### Scenario: Clean CI runs

Given same Model Artifact and fixture version

When generation executes

Then expected token sequence is reproduced within deterministic semantics.

### Requirement: Cancellation Stops Future Generation Work

First-profile cancellation SHALL prevent new decode steps after cancellation is
recognized.

#### Scenario: Cancellation between synchronous steps

Given current CPU Kernel completed

When cancellation is observed before next decode

Then no new model step is submitted.