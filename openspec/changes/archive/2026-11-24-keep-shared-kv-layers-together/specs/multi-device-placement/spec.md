## ADDED Requirements

### Requirement: Layers Sharing K/V Stay With Their Source Layer

The Runtime's multi-Device layer placement SHALL place every layer that reuses an earlier layer's K/V in the same segment as that source layer, so that no segment boundary falls between them.

#### Scenario: A balanced split between shared-KV layers
- **WHEN** a balanced placement would start a segment at a layer between a source layer and a layer reusing its K/V
- **THEN** the segment grows past those layers, and the source and every layer reusing it are on one Device

#### Scenario: A Device too small for the source and its readers
- **WHEN** a Device's budget ends inside a source layer's group
- **THEN** that Device's segment ends before the source layer, and the next Device holds the whole group
