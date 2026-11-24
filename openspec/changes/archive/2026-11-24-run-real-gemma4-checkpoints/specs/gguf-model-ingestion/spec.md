## ADDED Requirements

### Requirement: GGUF Loader Reads The Gemma 4 Tokenizer

The GGUF loader SHALL build the tokenizer of a file whose `tokenizer.ggml.model` is `gemma4` from its own metadata: its pieces and `tokenizer.ggml.merges` as a byte-fallback BPE, spaces encoded as `▁` with no space added before the text, `▁` decoded back to spaces. Its encoding and decoding SHALL match the checkpoint's upstream `tokenizer.json`.

#### Scenario: Gemma 4 E2B tokenizer parity
- **WHEN** the tokenizer is built from the Gemma 4 E2B QAT `Q4_0` file's metadata
- **THEN** it encodes and decodes the parity texts exactly as the upstream `tokenizer.json` does, allowing only the beginning-of-sequence token it adds before an encoding with special tokens
