## ADDED Requirements

### Requirement: RoPE Multi-Head Scope

RoPE SHALL support rotating multiple, independently-sized head blocks of a
row in a single Kernel invocation via an explicit `head_count` graph
attribute, in addition to the single-block baseline mode `RoPE Scope`
already defines. `head_count` SHALL be set by the graph builder as
authoritative graph data (e.g. a model's own attention head count for a
Query projection and its own key/value head count for a Key projection
under grouped-query attention), not inferred by the Runtime from a node
identifier or other naming convention. Each head's own rotation width MAY
be less than its block width (partial RoPE); Runtime SHALL NOT require a
head's full block to be rotated. Runtime SHALL NOT require one Kernel
invocation per head when a Provider's `rope` Kernel advertises multi-head
support.

#### Scenario: Multi-head RoPE within scope

Given a graph node rotates Q or K across multiple attention heads in one
row, with an explicit `head_count` attribute set by the graph builder

When first scope validates it

Then a single `rope` Kernel invocation with that `head_count` is within
scope, and per-head decomposition into separate Kernel invocations is not
required.

#### Scenario: Grouped-query attention uses distinct head counts for Q and K

Given a model's Query projection has more attention heads than its Key/
Value projection (grouped-query or multi-query attention)

When the graph builder constructs each projection's RoPE node

Then the Query RoPE node's `head_count` equals the attention head count and
the Key RoPE node's `head_count` equals the key/value head count,
independently of each other.

#### Scenario: Partial RoPE within scope

Given a model rotates fewer columns per head than that head's own width

When first scope validates a multi-head RoPE node whose rotation width is
less than its block width

Then the node is within scope, and the unrotated remainder of each head's
block is preserved unchanged rather than treated as an error or as
requiring the full block to be rotated.
