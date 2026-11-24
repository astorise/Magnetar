## ADDED Requirements

### Requirement: Hybrid decoder configuration

The `model-config` interface SHALL expose, from version 1.13.0:
- `linear-attention`: per decoder layer, whether it is a gated-delta linear-attention layer, with the mixer's key and value head counts and widths and its convolution kernel;
- `decoder-variants`: the RMSNorm weight offset and whether attention outputs are gated by a per-head query gate.

Each SHALL be `none` when the checkpoint does not declare it. When the bound configuration declares either, the Runtime SHALL refuse a graph whose Component did not read it. The Runtime SHALL keep linking Components compiled against earlier versions.

#### Scenario: A Qwen3.5 checkpoint

- **WHEN** a Qwen3.5 configuration is bound and a Component builds a graph without calling `linear-attention` or `decoder-variants`
- **THEN** the unread declared features list names both functions, and the graph is refused

#### Scenario: A plain decoder

- **WHEN** a configuration without linear layers, norm offset or output gate is bound
- **THEN** both functions return `none` and neither is required to be read

### Requirement: Linear-attention layers keep no KV history

The first-native Runtime SHALL give a linear-attention layer no K/V store and no KV row width. Its state SHALL be the recurrent state its graph declares.

#### Scenario: KV geometry of a hybrid decoder

- **WHEN** a configuration marks every layer but the last as linear attention
- **THEN** only the last layer has K and V row widths
