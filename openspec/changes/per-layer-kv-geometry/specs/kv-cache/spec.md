## ADDED Requirements

### Requirement: KV geometry per decoder layer

The Runtime SHALL size and keep each decoder layer's K/V at that layer's own row width (`kv_head_count * head_dimension` of the layer). A layer that attends over an earlier layer's K/V SHALL keep no K/V of its own and SHALL read the producing layer's full history.

#### Scenario: Layers of different widths
- **WHEN** a model's layers declare different KV head counts or head widths
- **THEN** each layer's KV store has its layer's row width, and the KV cache estimate and pooled store bytes sum the widths of the layers that keep K/V

#### Scenario: Shared KV layer during decode
- **WHEN** a layer reuses an earlier layer's K/V and a decode step runs over a prompt's history
- **THEN** its attention reads the producer's history including the step's new row, and the step's logits match a full-sequence computation
