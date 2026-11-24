## ADDED Requirements

### Requirement: Layered artifacts are validated per layer

When the architecture configuration declares a layer layout, tensor inventory and shape validation SHALL use each layer's own geometry:
- a layer reusing an earlier layer's K/V SHALL NOT require K or V projections;
- a layer deriving V from K SHALL NOT require a V projection;
- each layer's projections and MLP SHALL be sized with that layer's head width, KV head count and feed-forward width.

A layer computing its own K/V SHALL still require both projections.

#### Scenario: Shared-KV layer without K/V projections
- **WHEN** a layer's layout entry names a K/V source layer and the artifact has no K or V projection for it
- **THEN** inventory validation accepts the artifact

#### Scenario: Wider full-attention heads
- **WHEN** a layer's layout entry declares a wider head than the architecture-wide one
- **THEN** that layer's `q_proj` is expected with the wider width, and the architecture-wide width is refused
