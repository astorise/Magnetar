## ADDED Requirements
### Requirement: Adapters May Modify Execution Graphs

Adapters SHALL represent any adapter modification or extension to Execution Graphs through explicit graph metadata,
additional operators, modified paths, merge graphs, or fused adapter metadata.

#### Scenario: LoRA active

Given LoRA adapter is active

When Runtime builds an MLP graph

Then LoRA path or fused adapter metadata is represented explicitly.

---

### Requirement: Adapter Graph Changes Affect Cache Compatibility

Adapter-induced graph semantic changes SHALL affect KV Cache and Prefix Cache
compatibility where model outputs change.

#### Scenario: Adapter changed graph

Given Prefix Cache entry was created without adapter graph path

When adapter graph path is active

Then Runtime rejects reuse unless policy proves compatibility.
