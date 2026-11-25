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

### Requirement: Residual-stream, n-gram embedding and token-selection configuration

The `model-config` interface SHALL expose, from version 1.14.0:
- `hyper-connections`: the number of residual streams and the rank of their input mixer;
- `ngram-embeddings`: per decoder layer, whether it adds a hashed n-gram embedding, with the embedding width, convolution kernel, n-gram size, heads per order, vocabulary base and multiple, seed and delimiter token;
- `token-selection`: the indexer's query heads, key heads, head width, token budget and block size;
- `linear-attention-gate`: the activation (`silu` or `sigmoid`) of the linear-attention layers' output gate, `none` meaning `silu`.

Each SHALL be `none` when the checkpoint does not declare it. When the bound configuration declares one, the Runtime SHALL refuse a graph whose Component did not read it. The Runtime SHALL refuse a declaration whose counts are inconsistent with the architecture (fewer than two streams, a layer list that does not name every layer, an n-gram embedding that does not split evenly over its heads, an indexer with more than one key head or a budget that is not whole blocks). The Runtime SHALL keep linking Components compiled against earlier versions.

#### Scenario: A Qwen4-Exp checkpoint

- **WHEN** a configuration with residual streams, n-gram embeddings, token selection and a sigmoid gate is bound and a Component builds a graph without calling those four functions
- **THEN** the unread declared features list names all four, and the graph is refused

#### Scenario: A plain decoder

- **WHEN** a configuration without these forms is bound
- **THEN** the four functions return `none` and none is required to be read

#### Scenario: An inconsistent indexer

- **WHEN** a configuration declares an indexer with two key heads or a budget of 30 tokens in blocks of 4
- **THEN** the Runtime refuses the configuration before any graph is built

