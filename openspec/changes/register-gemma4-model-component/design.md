## Context

Like the Llama and Qwen Components, the Gemma Component is checked in as a
compiled artifact. An embedder registers it under its digest, and the
family a checkpoint's loader declares selects it. The Component side ran
two kinds of checks outside Magnetar's CI: engine smoke tests against a
local checkout, and a logit comparison with `transformers`.

## Decisions

- **The digest of our own build.** The Component author built with Rust
  1.97 and `--ignore-rust-version`. The artifact here is built with the
  repository's pinned toolchain, and only its digest is recorded.
- **Families.** The loaders take a Hugging Face bundle's own `model_type`
  as its family: `gemma4` for a nested text config, `gemma4_text` for a
  text-only export, `gemma4_unified` for the 12B's type. A GGUF file
  declares `gemma4`. The manifest lists the three.
- **Checked-in checkpoints, not generated in CI.** The references need
  torch and `transformers`, which the quality jobs do not install. The six
  checkpoints and their references are checked in (about 3 MB), with the
  commands that made them in their README.
  - The four GGUF files are small (150–340 KB) and cover every variant
    through the full Runtime path.
  - The Hugging Face bundles are larger. Only the dense one and the
    `gemma4_unified` one, which tests the family and the vision tensor left
    out, are kept.
- **Pinned node counts** in the engine tests. A change to the Component or
  to the graph-builder's validation shows up as a count change.
