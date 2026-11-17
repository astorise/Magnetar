## ADDED Requirements

### Requirement: Production Generation Encodes Plain-Text Prompts With Special Tokens

A production generation SHALL encode a plain-text prompt with its tokenizer's special tokens, so that the prompt starts with the beginning-of-sequence token of a model whose tokenizer adds one, as `transformers`' `tokenizer(text)` and llama.cpp encode it. It SHALL encode the rendered text of chat messages as it is, since their template writes those tokens, and SHALL use token ids as given.

#### Scenario: Plain text gains the beginning-of-sequence token
- **WHEN** a production generation receives a plain-text prompt and its tokenizer declares a beginning-of-sequence token it adds
- **THEN** the prompt's first token is that token

#### Scenario: Chat messages and token ids are unchanged
- **WHEN** the prompt is chat messages or token ids
- **THEN** no special token is added to the rendered text or to the ids
