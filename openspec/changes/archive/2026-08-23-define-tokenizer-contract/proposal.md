# Define Tokenizer Contract

## Why

Magnetar is an inference Runtime.

The Model Artifact model now distinguishes model data from executable
Components and native Providers.

Text-generation inference requires a stable tokenizer contract.

The tokenizer is the boundary between user-visible text and model-visible token
IDs.

Without a tokenizer contract, Magnetar cannot safely define:

- prompt encoding
- chat-template output tokenization
- special token handling
- BOS/EOS/PAD behavior
- stop token matching
- streaming decode
- partial UTF-8 decode
- token offsets
- tokenizer artifact validation
- tokenizer compatibility with a model
- generation request validation
- prompt length accounting
- KV cache prefix matching
- batch collation
- detokenization for clients

A tokenizer must not be confused with:

- model weights
- chat templates
- prompt templates
- generation
- sampling
- Provider execution
- filesystem access
- client workspace tools

This change defines Magnetar's Tokenizer Contract.

## What Changes

This change introduces a Tokenizer domain contract.

The contract SHALL support:

- tokenizer artifact identity
- tokenizer compatibility with Model Artifacts
- tokenizer vocabulary metadata
- special token metadata
- encoding text to token IDs
- decoding token IDs to text
- incremental streaming decode
- token offsets where supported
- normalization metadata where available
- pre-tokenization metadata where available
- added tokens
- chat/template boundary
- prompt length accounting
- stable tokenizer errors
- tokenizer observability

The tokenizer contract is part of Magnetar inference.

It SHALL remain separate from client-side workspace, filesystem, Git, network,
or secret authority.

## Tokenizer Artifact

A tokenizer may be represented as a Model Artifact part.

Tokenizer-related Model Artifact kinds include:

```text
tokenizer
tokenizer-config
vocabulary
special-tokens
```

The Tokenizer Contract consumes validated Runtime-registered tokenizer
artifacts.

It SHALL NOT read arbitrary filesystem paths directly.

## Tokenizer Implementation

A tokenizer implementation MAY be provided by:

- Runtime-native code
- a Magnetar Component
- a future Provider-backed implementation
- a test fixture implementation

The implementation mechanism is not the public contract.

The contract is the Runtime-facing behavior.

If the tokenizer implementation is a Component, then:

```text
Tokenizer Component = executable Component Artifact
Tokenizer data      = Model Artifact
```

They SHALL have separate identity, trust, validation, and caching.

## Tokenizer Capability

Magnetar MAY define a tokenizer Capability.

A tokenizer Capability SHOULD be named in the Magnetar namespace.

Example conceptual Capability:

```text
magnetar:tokenizer/tokenize
magnetar:tokenizer/decode
```

or a combined tokenizer interface.

Exact WIT package/interface names are implementation-defined by this change's
implementation.

The Capability SHALL be portable and SHALL NOT expose Provider or Device
selection.

## Encode

The tokenizer SHALL support encoding text into token IDs.

Encoding input MAY include:

- text
- optional add_special_tokens flag
- optional truncation policy
- optional max token length
- optional return_offsets flag
- optional normalization policy where supported
- optional special token policy
- optional tokenizer mode

Encoding output SHALL include:

- token IDs
- token count
- optional token type IDs where supported
- optional attention mask where supported
- optional offsets where requested and supported
- special token markers where available
- diagnostics or warnings where relevant

## Decode

The tokenizer SHALL support decoding token IDs into text.

Decode input MAY include:

- token IDs
- optional skip_special_tokens flag
- optional clean_up_tokenization_spaces flag where supported
- optional streaming state reference
- optional partial decode mode

Decode output SHALL include:

- decoded text
- consumed token count
- optional pending partial state
- diagnostics or warnings where relevant

## Streaming Decode

The tokenizer SHALL support incremental decode semantics for generation.

Streaming decode must handle cases where token boundaries do not map cleanly to
valid output text.

For example:

- partial UTF-8 bytes
- byte-fallback tokens
- subword continuation
- special token suppression
- whitespace normalization
- tokenizer-specific merge behavior

Streaming decode SHALL be stateful or expose explicit decode state.

The Runtime SHALL not emit invalid text chunks merely because a token arrived.

## Token IDs

Token IDs SHALL be represented using a stable integer type.

The contract SHALL define the valid token ID range for a tokenizer.

The tokenizer SHALL reject token IDs outside the vocabulary or added-token
range unless policy defines fallback behavior.

## Vocabulary Metadata

Tokenizer metadata SHOULD include:

- vocabulary size
- added token count
- model max length where known
- token ID range
- unknown token ID
- BOS token ID
- EOS token ID
- PAD token ID
- separator token ID where relevant
- mask token ID where relevant
- additional special tokens
- byte fallback support
- normalization behavior
- pre-tokenizer behavior
- tokenizer family

## Special Tokens

Special token handling SHALL be explicit.

The contract SHALL represent:

- BOS
- EOS
- PAD
- UNK
- SEP
- CLS
- MASK
- additional special tokens
- stop tokens where model metadata defines them

Encoding and decoding SHALL specify whether special tokens are added, skipped,
or preserved.

## Tokenizer Compatibility With Model Artifact

A Model Artifact that references a tokenizer SHALL validate tokenizer
compatibility.

Compatibility MAY include:

- tokenizer digest
- tokenizer vocabulary size
- expected special token IDs
- expected chat template
- expected model max length
- expected tokenizer family
- expected added tokens
- expected normalization behavior

A model bundle SHALL not silently use an incompatible tokenizer.

## Chat Template Boundary

Chat templates are not tokenizers.

A chat template transforms structured messages into text or tokenization input.

The tokenizer transforms text into token IDs.

Conceptual flow:

```text
messages
    |
    v
chat template rendering
    |
    v
prompt text
    |
    v
tokenizer encode
    |
    v
input tokens
```

This change MAY define the boundary but does not fully define chat template
rendering behavior.

A later generation or prompt-template change may define template rendering
semantics in more detail.

## Prompt Length Accounting

Tokenizer output SHALL support prompt length accounting.

Generation needs prompt token count for:

- context window validation
- KV cache planning
- batching
- truncation
- max_new_tokens calculation
- stop condition planning
- memory feasibility

Prompt length SHALL be counted in token IDs after all relevant special tokens
and template rendering are applied.

## Truncation

The tokenizer contract MAY support truncation policy.

Truncation policies SHOULD be explicit.

Examples:

```text
none
left
right
middle
model-default
client-policy
```

The Runtime SHALL not silently truncate prompts unless policy explicitly allows
it.

If a prompt exceeds model context limits and truncation is not allowed, encoding
or request validation SHALL fail.

## Offsets

The tokenizer MAY support offsets mapping tokens back to input text spans.

Offsets are useful for diagnostics, UI, and future constrained editing.

Offsets SHALL be optional because not all tokenizer implementations can provide
accurate offsets.

If requested but unsupported, the tokenizer SHALL return a structured
unsupported-offsets error or omit offsets according to explicit policy.

## Batch Encoding

The tokenizer contract SHOULD support batch encoding.

Batch encoding SHALL define:

- input ordering
- per-input token output
- optional padding behavior
- optional truncation behavior
- attention mask behavior
- error behavior for partial failures

Batch encoding SHALL not silently reorder inputs.

## Padding

Padding SHALL be explicit.

Padding policy MAY include:

```text
none
longest
max-length
model-default
```

Padding token ID SHALL be known when padding is requested.

If padding is requested and no PAD token exists, encoding SHALL fail unless
policy defines fallback.

## Attention Masks

Tokenizers MAY produce attention masks.

Attention masks are useful for batching.

If produced, they SHALL align with token IDs and padding behavior.

## Token Type IDs

Some model families use token type IDs.

Token type IDs MAY be produced where supported.

If unsupported, the tokenizer SHALL report unsupported token type IDs when
requested.

## Stop Token Compatibility

Tokenizer metadata SHALL support resolving textual stop sequences to token
patterns where feasible.

Stop conditions may be token-based, text-based, or both.

This change prepares stop token compatibility but does not fully define
generation stopping semantics.

## Memory Manager Relationship

Tokenizer artifacts and tokenizer execution SHALL integrate with Memory Manager
where relevant.

The Memory Manager MAY manage:

- tokenizer artifact residency
- vocabulary memory
- added token memory
- temporary encoding buffers
- batch token buffers
- output token buffers
- streaming decode state buffers

Tokenizer execution SHALL not allocate unbounded memory outside Runtime policy.

## Provider Relationship

Tokenization is not Provider selection.

A tokenizer may be implemented natively, as a Component, or through another
Runtime mechanism.

A tokenizer contract SHALL NOT expose Provider or Device selectors to clients
or Components.

If a future Provider accelerates tokenization, Runtime Resolution remains
authoritative.

## Browser Relationship

Tokenizer contract SHALL work on native and browser targets where supported.

Browser targets may use:

- Runtime-native wasm tokenizer logic
- Component-based tokenizer logic
- JavaScript-mediated tokenizer implementation
- browser-compatible memory placement

Browser tokenization SHALL not require Wasmtime.

## Error Model

Tokenizer errors SHALL be structured.

Error categories SHOULD include:

- tokenizer artifact missing
- tokenizer artifact invalid
- tokenizer incompatible with model
- unsupported tokenizer family
- invalid token ID
- unknown token
- invalid UTF-8
- decode pending partial
- offsets unsupported
- padding token missing
- truncation required
- truncation forbidden
- prompt too long
- batch input invalid
- special token missing
- special token conflict
- vocabulary mismatch
- added token mismatch
- memory allocation failed
- streaming state invalid
- implementation unavailable

## Observability

Runtime SHOULD emit observations for:

- tokenizer loaded
- tokenizer compatibility checked
- encode requested
- encode completed
- encode failed
- decode requested
- decode completed
- decode failed
- streaming decode chunk emitted
- streaming decode pending partial
- prompt too long
- truncation applied
- tokenizer memory pressure
- tokenizer implementation unavailable

Observability SHALL not leak raw prompts unless policy explicitly allows prompt
content logging.

The default SHOULD avoid logging raw prompt text.

## Non-Goals

This change does not:

- implement full generation
- define sampling
- define KV cache semantics
- define model loading fully
- define chat template rendering fully
- define client workspace behavior
- define filesystem access
- define network access
- define Git access
- define secrets access
- define Provider-specific tokenizer ABI
- require tokenizers to be Components
- require tokenizers to be native code
- require browser tokenization implementation
- define UI text rendering
- define moderation or safety filters

## Impact

Magnetar gains a stable tokenizer contract.

Future generation can depend on:

- prompt tokenization
- token count
- streaming detokenization
- special token handling
- tokenizer/model compatibility
- prompt length validation
- batch tokenization
- stop token preparation

The inference pipeline becomes:

```text
Model Artifact
    |
    +-- tokenizer artifact
    |
    v
Tokenizer Contract
    |
    +-- encode text -> token IDs
    +-- decode token IDs -> text
    +-- streaming decode
    |
    v
Generation Contract
```