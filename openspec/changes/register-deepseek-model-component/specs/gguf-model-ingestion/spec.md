## MODIFIED Requirements

### Requirement: GGUF Ingestion Builds A Real Tokenizer From The File's Own Embedded Vocabulary

The GGUF ingestor's tokenizer construction SHALL build a real tokenizer directly from the GGUF file's own embedded vocabulary and merge rules, without requiring a separate tokenizer file, and SHALL satisfy the same special-token uniqueness constraint every tokenizer in the Runtime satisfies regardless of source format.

#### Scenario: A real tokenizer builds from embedded vocabulary alone

- **WHEN** a GGUF file declares a byte-level BPE tokenizer model with embedded token and merge arrays
- **THEN** a real tokenizer builds from those arrays alone, without reading any file other than the GGUF file itself

#### Scenario: A shared beginning-of-sequence and end-of-sequence id does not fail construction

- **WHEN** a GGUF file declares the same token id for both its beginning-of-sequence and end-of-sequence special tokens
- **THEN** tokenizer construction succeeds, registering that id under one role rather than failing on a special-token conflict

#### Scenario: A byte-level BPE adds no space before the text

- **WHEN** a byte-level BPE tokenizer built from a GGUF file encodes text that does not start with a space
- **THEN** the first token is the text's own first piece, not one carrying a leading space, as llama.cpp and the checkpoint's upstream `tokenizer.json` encode it

