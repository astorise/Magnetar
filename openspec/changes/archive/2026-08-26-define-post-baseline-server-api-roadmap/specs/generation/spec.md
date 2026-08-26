## ADDED Requirements

### Requirement: Server Generation Uses Generation Contract

Server generation endpoint SHALL execute through Runtime Generation Contract.

#### Scenario: Server generate

Given prompt is submitted to server

When generation runs

Then Runtime owns prefill, decode, Sampling, stop conditions, and usage.

---

### Requirement: Server Generation Does Not Execute Side Effects

Core Server generation SHALL not execute tools, shell, Git, network, filesystem,
or external service side effects from generated output.

#### Scenario: Generated Git instruction

Given generated text says to run Git

When server returns it

Then no Git command is executed.