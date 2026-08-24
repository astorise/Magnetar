## ADDED Requirements

### Requirement: Generation Is Exposed Through Inference API

Runtime Inference API SHALL expose generation through the Generation Contract.

#### Scenario: Generate

Given session and input tokens are valid

When generation request is accepted

Then Runtime executes generation through prefill, decode, sampling, and stop
contracts.

---

### Requirement: Inference API Generation Does Not Own Tools

Generation through Runtime Inference API SHALL not execute tools, shell commands, network calls, Git operations, or workspace file operations.

#### Scenario: Tool request in generated text

Given model output contains a tool-call-like string

When Runtime emits output

Then Runtime does not execute any tool.