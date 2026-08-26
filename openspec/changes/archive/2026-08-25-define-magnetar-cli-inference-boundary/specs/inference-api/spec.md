## ADDED Requirements

### Requirement: Runtime Inference API Is Called By CLI

`magnetar-cli` SHALL call Runtime Inference API for inference operations.

#### Scenario: CLI run

Given user executes `magnetar run`

When generation is needed

Then CLI calls Runtime Inference API.

---

### Requirement: Runtime Inference API Receives Explicit Context

Runtime Inference API SHALL receive explicit prompt/context data from CLI and
not CLI authority.

#### Scenario: File context

Given CLI reads a file

When Runtime request is built

Then request contains selected content or references allowed by contract, not
filesystem authority.

---

### Requirement: Runtime Inference API Does Not Execute CLI Tools

Runtime Inference API SHALL not execute CLI tools or shell commands.

#### Scenario: Tool-like output

Given generated output contains tool syntax

When Runtime returns it

Then Runtime does not execute the tool.