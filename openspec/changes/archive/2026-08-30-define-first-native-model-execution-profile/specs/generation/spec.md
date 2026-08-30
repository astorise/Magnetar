## ADDED Requirements
### Requirement: First Profile Generation Starts From Text

Mandatory E2E generation SHALL begin with prompt text and use Tokenizer contract.

#### Scenario: Hello prompt

Given caller supplies `"Hello"`

When generation starts

Then Runtime tokenizes it before model execution.

### Requirement: Prefill Executes Real Model

Prompt prefill SHALL execute through Qwen graph and Reference CPU Kernels.

#### Scenario: Prompt has multiple tokens

Given Session starts

When prefill occurs

Then model logits and KV result from actual graph execution.

### Requirement: Decode Is Incremental

First-profile decode SHALL reuse existing KV rather than recomputing entire
history as mandatory strategy.

#### Scenario: Five tokens already exist

Given sixth token is decoded

When model runs

Then decode consumes cached history and new-token input according to Qwen
semantics.

### Requirement: Greedy Sampling Is Mandatory

First profile SHALL support deterministic greedy selection.

#### Scenario: Logits available

Given greedy sampling configured

When token is selected

Then highest eligible logit determines output according to Sampling contract.

### Requirement: Generation Does Not Accept Caller Logits

Generation SHALL obtain logits from Runtime model execution.

#### Scenario: Generate call

Given caller supplies prompt/configuration

When generation runs

Then no external model-logits producer is used.