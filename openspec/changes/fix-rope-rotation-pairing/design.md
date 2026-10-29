## Context

RoPE rotates each head's query/key vector in two-dimensional pairs by a position-dependent angle `position * base^(-2i/dimension)`. Two layouts of those pairs exist in practice. GPT-J and GGML's "normal" mode rotate adjacent elements `(x[2i], x[2i+1])`; GPT-NeoX, Hugging Face Llama/Qwen2 (`rotate_half`) and llama.cpp's NeoX mode rotate `(x[i], x[i + dimension/2])`. They are not interchangeable: applying one to weights trained for the other scrambles position information while still producing plausible-looking text, which is why the defect survived until A02's independent reference.

## Decisions

### Explicit graph attribute, owned by the Component

The pairing is model-family semantics, so the Model Component that builds the graph declares it, exactly like `head_count`: Runtime never infers it from a family name, artifact format or node id. `rotation_pairing` is an optional `String` attribute with values `interleaved` and `half_split`. Absent means `interleaved`, preserving every existing graph, fixture and conformance case bit-for-bit; Providers reject any other value with `KernelAttributeUnsupported` instead of guessing.

### Kernel semantics

For each row, each of the `head_count` blocks of width `head_width`, and pair index `i < dimension/2`, with `theta = position * base^(-2i/dimension)` (position = `position_offset + row`, multiplied by `scale`):

- `interleaved`: `(a, b) = (x[2i], x[2i+1])`, written back to the same indices.
- `half_split`: `(a, b) = (x[i], x[i + dimension/2])`, written back to the same indices.

In both cases `a' = a cos(theta) - b sin(theta)` and `b' = a sin(theta) + b cos(theta)`, and columns `>= dimension` inside a block are unchanged. This matches Hugging Face's `q * cos + rotate_half(q) * sin`.

### Rollout

Runtime and Providers land first (attribute accepted, default unchanged), then the Components emit `half_split` and the compiled fixtures are rebuilt and re-pinned, then submodule pins move. The Runtime's Component-graph dispatch forwards the node's `rotation_pairing` to the Kernel; `magnetar-runtime`'s own Rust-side Qwen graph builder emits `half_split` too, so the oracle/graph-equivalence tests compare like with like.

## Risks / Trade-offs

- Every real-checkpoint output changes. That is the fix, and A02's independent corpus is the evidence.
- A Provider that does not implement `half_split` fails the dispatch explicitly instead of producing wrong numbers.

## Migration and rollback

Older Components (no attribute) keep the interleaved behavior. Rolling back the Component fixtures restores the previous (incorrect) outputs; rolling back a Provider while new Components are deployed fails dispatch explicitly.
