## Decisions

### Derive in `[out, in]` orientation, then transpose as usual

Each loader produces the derived tensors before its projection transpose, in `nn.Linear`'s `[out, in]` orientation:
- the per-layer projection slice is `[P, hidden]`;
- the stacked gate and up projections are `[experts * intermediate, hidden]`, with expert `e` owning rows `[e * intermediate, (e + 1) * intermediate)`;
- the stacked down projection is `[experts * hidden, intermediate]`;
- the router is `[experts, hidden]`.

The existing transpose then orients them like every stored projection. A float tensor gets its bytes transposed, and a block-quantized one keeps its bytes and declares transposed storage. This is the layout the Runtime's `moe-experts` Operator and weight shapes already use for `qwen2moe`.

### Cut in stored bytes, on rows or blocks

Slices and stacks copy stored bytes; they never re-encode them:
- **Per-layer embedding:** each row's columns of one layer. In a block-quantized table, the cut must fall on a block boundary, and the real E2B file's `Q6_K` table has one 256-element block per layer per row. A cut inside a block is refused.
- **Per-layer projection and experts:** whole rows.
- **Reads:** a derivation reads only the part of the stored tensor it needs, by chunks of rows or by expert.

### Gate before up; router scale copied by the converter

The orders and factors were checked at their sources:
- **Gate/up order:** `transformers`' `Gemma4TextExperts` splits each expert's `gate_up_proj` output into gate, then up (`chunk(2)`).
- **Router:** `Gemma4TextRouter` computes `proj(norm(x) * scale * hidden^-0.5)`, so the scale and the factor fold into the projection's input columns exactly.
- **llama.cpp's converter:** it copies `router.scale` to `ffn_gate_inp.scale` and `per_expert_scale` to `ffn_down_exps.scale` unchanged, and copies the fused expert tensor as is. The GGUF loader therefore folds the same factor.

## Evidence

- **Hugging Face, unit tests:** a bundle's per-layer slices, split and stacked experts, folded router and per-expert scale are checked against values computed by hand, and inconsistent packed tensors are refused.
- **GGUF, unit test:** a whole `gemma4` ingest with a `Q8_0` per-layer table checks the same derivations. Each layer gets its rows' block, copied byte for byte. Three mutations of the derivation make it fail:
  - every layer reading layer 0's slice;
  - gate and up swapped;
  - the router factor dropped.
- **Logits, Hugging Face path:** the Component's comparison with `transformers` 5.23 ran through this loader (Reference CPU, f32, 12 greedy tokens).
  - **Bundles:** tiny random bundles with per-layer inputs (`ple`) and routed experts (`moe`).
  - **Component:** built locally with the `moe.*` names; the Component's own switch to them is pending.
  - **Result:** the tokens are identical, and the maximum logit differences are 4.9e-5 (`ple`) and 1.8e-5 (`moe`).
  - **Mutations:** reading layer 0's embedding slice for every layer makes the `ple` comparison fail by 13.7, and swapping gate and up makes the `moe` one fail by 12.6.
- **Not run:**
  - The real E2B and 26B-A4B GGUF files: their attention and MLP weights are `Q4_0`, which `formats/gguf` does not decode yet.
  - The real Hugging Face checkpoints, which were not downloaded.
  - The resident memory of the per-layer embeddings, which was not measured.
