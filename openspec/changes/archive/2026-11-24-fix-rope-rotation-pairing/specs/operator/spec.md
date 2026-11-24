## MODIFIED Requirements

### Requirement: RoPE Operator

Rotary position embedding SHALL be represented as a position-encoding
operator or explicit attention attribute. When RoPE applies independently
across multiple head-sized blocks of a row, the number of such blocks
SHALL be represented as an explicit `head_count` attribute set by whichever
component constructs the graph; Runtime SHALL NOT infer it from a node
identifier or other naming convention. The element pairing RoPE rotates
SHALL be represented by an optional `rotation_pairing` attribute set by the
component that constructs the graph: `interleaved` rotates elements
`(2i, 2i+1)` and `half_split` rotates elements `(i, i + dimension/2)` of
each head block. An absent attribute SHALL mean `interleaved`. Providers
SHALL reject any other value rather than substitute a pairing, and Runtime
SHALL NOT infer the pairing from a model family, artifact format or node
identifier.

#### Scenario: RoPE metadata

Given model architecture uses RoPE

When graph is created

Then RoPE base, scale, dimension, and position mode are represented.

#### Scenario: Multi-head RoPE metadata

Given model architecture applies RoPE independently across multiple
attention heads in one row

When graph is created

Then the RoPE operator node also represents an explicit `head_count`
attribute equal to the number of independently-rotated head blocks for
that node.

#### Scenario: Half-split RoPE pairing

Given a checkpoint trained with half-split (`rotate_half`) rotary embedding,
such as a Hugging Face Qwen2 or Llama checkpoint

When its Model Component creates the graph

Then every RoPE operator node carries `rotation_pairing` equal to
`half_split`, and every Provider rotates elements `(i, i + dimension/2)`
of each head block.

#### Scenario: Unknown RoPE pairing

Given a RoPE operator node whose `rotation_pairing` is neither
`interleaved` nor `half_split`

When a Provider is asked to execute it

Then the dispatch fails with an unsupported-attribute error and no output
is produced.
