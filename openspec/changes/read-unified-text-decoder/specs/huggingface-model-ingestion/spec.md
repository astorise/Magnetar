## ADDED Requirements

### Requirement: Hugging Face Ingestion Keeps Only A Multimodal Bundle's Text Decoder

A Hugging Face bundle whose `config.json` nests the decoder under
`text_config` SHALL be ingested as that text decoder: the layout comes from
`text_config`, the Artifact family is the bundle's own `model_type`, and
tensors under the vision and audio towers, their embedders
(`model.embed_vision.`, `model.embed_audio.`, `model.vision_embedder.`) and
the multimodal projector SHALL stay out of the text inventory.

#### Scenario: A unified bundle

- **WHEN** the loader parses `gemma-4-12B-it`'s `config.json`
  (`gemma4_unified`)
- **THEN** the family is `gemma4_unified`, every sixth layer attends over the
  whole context with 512-wide heads, one K/V head, V derived from K, RoPE
  base 1e6 and a quarter of the head rotated, and the other layers slide over
  1024 tokens with 256-wide heads, eight K/V heads and RoPE base 1e4

#### Scenario: The image-patch embedder

- **WHEN** the bundle stores `model.vision_embedder.patch_dense.weight`
- **THEN** the tensor is not part of the text inventory
