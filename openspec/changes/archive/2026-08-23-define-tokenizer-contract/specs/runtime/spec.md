## ADDED Requirements

### Requirement: Runtime Owns Tokenizer Contract

Runtime SHALL expose tokenization through a stable Tokenizer Contract.

#### Scenario: Runtime encode

Given a caller submits text for inference

When Runtime prepares model input

Then Runtime uses the Tokenizer Contract to encode text.

---

### Requirement: Runtime Validates Tokenizer Before Use

Runtime SHALL validate tokenizer artifact identity, metadata, trust, and model
compatibility before tokenization.

#### Scenario: Untrusted tokenizer

Given a tokenizer artifact is not trusted

When Runtime attempts to use it

Then Runtime rejects tokenization.

---

### Requirement: Runtime Separates Tokenizer From Generation

Runtime SHALL keep tokenization separate from generation.

#### Scenario: Decode generated token

Given generation produces token IDs

When text output is needed

Then Runtime uses tokenizer decode rather than generation owning text decoding.

---

### Requirement: Runtime Counts Prompt Tokens

Runtime SHALL use tokenizer output for prompt length accounting.

#### Scenario: Context window exceeded

Given tokenized prompt length exceeds model context window

When inference request validation runs

Then Runtime rejects or truncates according to explicit policy.

---

### Requirement: Runtime Does Not Log Raw Prompt By Default

Runtime observability SHALL not log raw prompt text by default during tokenizer
operations.

#### Scenario: Tokenization observed

Given encode succeeds

When Runtime emits observability

Then it records metadata such as token count

And not raw prompt content unless explicit policy enables it.

---

### Requirement: Runtime Supports Streaming Detokenization

Runtime SHALL support streaming detokenization for generated token streams.

#### Scenario: Token stream

Given generation emits tokens incrementally

When Runtime streams output to the client

Then Runtime uses tokenizer streaming decode state to emit valid text chunks.
