## ADDED Requirements

### Requirement: Production Model Instance Placement Across Two Real Devices Is Achieved Via Two Separate Instances And Explicit Movement

A production Qwen `ModelInstance`'s real decoder stack SHALL be placeable across two real, distinct Devices by loading two separate, ordinary `ModelInstance`s -- each bound to its own real Provider/Device and each materializing only its own real decoder-layer range's weights -- and moving the boundary hidden-state tensor between them explicitly. This requirement does not itself restructure `ModelInstancePlacement`, which remains structurally single-Device per instance.

#### Scenario: A real forward pass splits bit-for-bit identically across two segment Model Instances

- **GIVEN** a real Qwen forward pass and a chosen decoder-layer split point `mid`
- **WHEN** the same prompt is run once through the one, full, unsegmented graph, and separately through two segment `ModelInstance`s (layers `[0, mid)` then `[mid, num_hidden_layers)`), the boundary hidden state handed explicitly from the first into the second
- **THEN** the segmented result matches the full-graph result bit-for-bit

#### Scenario: The same split executes correctly across two real, physically distinct GPUs

- **GIVEN** two real GPUs, one segment Model Instance loaded and executed on each
- **WHEN** the boundary hidden-state tensor is moved between them via an explicit Host round trip
- **THEN** the two-real-GPU segmented result matches the real full graph dispatched on one real GPU alone, within numeric tolerance

#### Scenario: A real, multi-step decode generation splits correctly across two real GPUs against a real, public checkpoint

- **GIVEN** the real, public Qwen2.5-0.5B-Instruct checkpoint, split by real decoder-layer range across two real GPUs, each segment Model Instance loaded once and streaming only its own weights from the real checkpoint file
- **WHEN** a real multi-step greedy generation runs (one real prefill step, then several real decode steps), each segment's own per-layer KV state threaded forward from its own prior step
- **THEN** the generated token ids exactly match the real full, unsegmented graph's own generation on one real GPU alone, for the identical prompt
