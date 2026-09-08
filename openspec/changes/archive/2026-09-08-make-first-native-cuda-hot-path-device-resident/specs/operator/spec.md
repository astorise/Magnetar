## MODIFIED Requirements

### Requirement: RoPE Operator

Rotary position embedding SHALL be represented as a position-encoding
operator or explicit attention attribute. When RoPE applies independently
across multiple head-sized blocks of a row, the number of such blocks
SHALL be represented as an explicit `head_count` attribute set by whichever
component constructs the graph; Runtime SHALL NOT infer it from a node
identifier or other naming convention.

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
