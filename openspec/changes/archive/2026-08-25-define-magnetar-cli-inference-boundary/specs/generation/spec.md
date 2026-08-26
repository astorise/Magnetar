## ADDED Requirements

### Requirement: CLI Generation Uses Runtime Contract

Generation requested by `magnetar-cli` SHALL use Runtime Generation Contract.

#### Scenario: CLI generate

Given CLI submits prompt input

When Runtime runs generation

Then Generation Contract owns prefill, decode, Sampling, stop conditions, and
streaming events.

---

### Requirement: Runtime Does Not Act On Generated Commands

Runtime SHALL not execute generated commands, tool calls, shell text, Git
instructions, or network requests.

#### Scenario: Generated Git command

Given generated output says `git commit`

When Runtime streams it

Then Runtime only emits text.