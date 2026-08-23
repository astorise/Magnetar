# Tokenizer Contract

Magnetar treats tokenization as an inference Runtime contract.

A tokenizer converts rendered text to model-visible token IDs, decodes token IDs
back to text, and maintains streaming decode state when generation emits tokens
incrementally. It does not render structured chat messages, sample or generate
tokens, select Providers or Devices, or access client workspace resources.

## Artifact Relationship

Tokenizer data is Model Artifact data. The Runtime recognizes tokenizer-related
artifact kinds for tokenizer data, tokenizer configuration, vocabulary data, and
special-token metadata. The Tokenizer Contract consumes Runtime-registered
artifact identities only; it does not open arbitrary filesystem paths.

Tokenizer Component Artifacts are different. If a tokenizer implementation is
packaged as a Component, that executable Component Artifact has separate
identity, trust, validation, authority, and cache policy from tokenizer data.
The Component can receive inference-scoped tokenizer artifact authority, but not
filesystem, network, Git, secrets, workspace, process, Provider, or Device
authority.

## Encode

Encode input contains rendered text plus explicit options for adding special
tokens, truncation, maximum token length, offsets, padding, and special-token
policy. Encode output contains token IDs, token count, optional offsets,
optional attention mask, optional token type IDs, and diagnostics.

Prompt token counts are measured after chat-template rendering and after any
configured special tokens are applied. Overlong prompts fail unless an explicit
truncation policy allows reduction. When truncation applies, the result carries
a diagnostic.

## Decode And Streaming Decode

Decode input contains token IDs plus explicit skip-special-token and cleanup
options. Output contains decoded text, consumed token count, optional pending
streaming state, and diagnostics.

Streaming decode exists because token boundaries do not always map to complete
valid text. Partial UTF-8, byte fallback, subword continuation, suppressed
special tokens, and whitespace normalization can require pending state. Runtime
must not emit invalid text chunks just because a token arrived.

## Special Tokens

Special tokens are explicit metadata. The contract represents token kind,
string, token ID, whether the token may be added during encode, and whether it
is skipped during decode. Known roles include UNK, BOS, EOS, PAD, SEP, CLS,
MASK, additional special tokens, and stop-token metadata where a model declares
it. Compatibility validation detects missing required special tokens and
conflicting token identities.

## Model Compatibility

A Model Artifact that requires a tokenizer validates compatibility before use.
Compatibility can include tokenizer digest, vocabulary size, special-token IDs,
model maximum length, family, added token count, and normalization behavior.
Generation behavior is not defined by tokenizer metadata; tokenizer metadata
only supplies tokenization facts used by later generation contracts.

## Batch, Padding, Masks, Token Types, And Stops

Batch encode preserves input ordering. Padding is explicit and requires a PAD
token when padding is requested. Attention masks, when returned, match token ID
length and padding. Token type IDs are returned only when supported; otherwise a
structured unsupported result is used. Textual stop sequence preparation can
resolve token patterns where feasible, but final stop-condition evaluation
belongs to Generation.

## Memory And Browser Targets

Tokenizer artifact residency, vocabulary memory, added-token memory, encode
buffers, batch buffers, and streaming decode state buffers are Runtime memory
concerns. Tokenizer execution follows Memory Manager admission and fails with a
structured memory error when allocation is not feasible.

The contract is platform-neutral. Browser-compatible tokenization must not
require Wasmtime or native Provider loading.

## Observability

Tokenizer observations include load, compatibility check, encode/decode
request, completion, failure, streaming chunks, pending partials, prompt length
failures, truncation, memory pressure, and implementation availability. Raw
prompt text is not logged by default.

## Non-Goals

This contract does not define full generation, sampling, KV cache semantics,
complete model loading, chat-template rendering, client workspace behavior,
filesystem access, network access, Git access, secrets access, Provider-specific
tokenizer ABI, UI rendering, moderation, or safety filters.
