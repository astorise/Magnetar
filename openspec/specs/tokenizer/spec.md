# tokenizer Specification

## Purpose
This specification defines tokenizer artifact identity, loading, model compatibility, encode/decode behavior, truncation/padding, offsets, and memory policy.
## Requirements
### Requirement: Tokenizer Contract

Magnetar SHALL define a Tokenizer Contract for converting between text and
token IDs.

The contract SHALL be part of Magnetar inference Runtime.

#### Scenario: Encode prompt

Given a text prompt

When Runtime encodes it through the Tokenizer Contract

Then the output is token IDs and token metadata suitable for generation.

---

### Requirement: Tokenizer Artifact Relationship

A tokenizer SHALL consume Runtime-registered tokenizer artifacts.

It SHALL NOT read arbitrary filesystem paths directly.

#### Scenario: Tokenizer artifact registered

Given a tokenizer artifact is registered and trusted

When tokenization is requested

Then Runtime uses the registered artifact identity.

---

### Requirement: Tokenizer Is Not Chat Template

A tokenizer SHALL NOT own structured chat message rendering.

Chat templates render messages into tokenizer input.

#### Scenario: Chat messages encoded

Given structured chat messages

When inference input is prepared

Then a chat template renders the messages before tokenizer encode runs.

---

### Requirement: Tokenizer Is Not Generation

A tokenizer SHALL NOT generate new model tokens.

Generation consumes token IDs and produces token IDs.

#### Scenario: Generate response

Given a prompt is tokenized

When generation starts

Then generation uses token IDs produced by the tokenizer.

---

### Requirement: Tokenizer Implementation Is Hidden Behind Contract

Tokenizer implementations SHALL remain hidden behind the public Tokenizer Contract.

They MAY be native, Component-based, browser-compatible, or test fixtures.

#### Scenario: Component tokenizer

Given a tokenizer implementation is packaged as a Component

When Runtime validates it

Then the executable Component Artifact and tokenizer data artifacts remain
separate.

---

### Requirement: Encode

The Tokenizer Contract SHALL support encoding text into token IDs.

#### Scenario: Encode text

Given input text `hello`

When encode runs

Then the tokenizer returns token IDs and token count.

---

### Requirement: Decode

The Tokenizer Contract SHALL support decoding token IDs into text.

#### Scenario: Decode tokens

Given valid token IDs

When decode runs

Then the tokenizer returns decoded text and consumed token count.

---

### Requirement: Streaming Decode

The Tokenizer Contract SHALL support incremental streaming decode semantics.

Streaming decode SHALL handle pending partial output.

#### Scenario: Partial decode

Given a generated token does not yet form a complete valid text chunk

When streaming decode processes it

Then the tokenizer records pending partial state instead of emitting invalid
text.

---

### Requirement: Token ID Validation

The tokenizer SHALL validate token IDs according to its vocabulary and added
tokens.

#### Scenario: Invalid token

Given a token ID outside the tokenizer's valid range

When decode runs

Then it returns a structured invalid-token-id error.

---

### Requirement: Vocabulary Metadata

Tokenizer metadata SHALL define required compatibility metadata.

Tokenizer metadata SHOULD include vocabulary size, added token count, token ID
range, model max length where known, special token IDs, tokenizer family, byte
fallback support, normalization behavior, and pre-tokenizer behavior.

#### Scenario: Vocabulary mismatch

Given a model expects vocabulary size X

And tokenizer metadata reports vocabulary size Y

When compatibility is checked

Then Runtime rejects the tokenizer/model pairing unless policy explicitly
allows it.

---

### Requirement: Special Token Metadata

Special token handling SHALL be explicit.

The tokenizer SHALL represent BOS, EOS, PAD, UNK, and additional special tokens
where available.

#### Scenario: Missing EOS token

Given a model requires an EOS token

And tokenizer metadata has no EOS token

When compatibility is checked

Then Runtime reports a missing-special-token error.

---

### Requirement: Encode Special Token Policy

Encoding SHALL specify whether special tokens are added.

#### Scenario: Add BOS/EOS

Given encode is requested with special tokens enabled

When the tokenizer supports BOS and EOS

Then output may include the configured special tokens according to tokenizer
policy.

---

### Requirement: Decode Special Token Policy

Decoding SHALL specify whether special tokens are skipped or preserved.

#### Scenario: Skip EOS

Given decode is requested with `skip_special_tokens`

When EOS appears in the token stream

Then the decoded text excludes EOS.

---

### Requirement: Tokenizer Model Compatibility

Runtime SHALL validate tokenizer compatibility with the Model Artifact that
references it.

#### Scenario: Compatible tokenizer

Given a model bundle references tokenizer digest D

And the registered tokenizer artifact has digest D

When validation runs

Then compatibility may succeed if metadata also matches.

---

### Requirement: Prompt Length Accounting

Tokenizer output SHALL include or support prompt token count.

#### Scenario: Prompt too long

Given an encoded prompt exceeds the model context length

When request validation runs

Then Runtime returns a prompt-too-long error unless truncation policy allows
reduction.

---

### Requirement: No Silent Truncation

Runtime SHALL NOT silently truncate tokenizer input unless explicit policy
allows truncation.

#### Scenario: Truncation forbidden

Given input exceeds maximum token length

And truncation policy is none

When encode runs

Then the tokenizer returns a truncation-required or prompt-too-long error.

---

### Requirement: Token Offsets

The tokenizer SHALL define behavior for token offsets.

It MAY support offsets.

If offsets are requested but unsupported, the tokenizer SHALL return a
structured unsupported-offsets result or error.

#### Scenario: Offsets unsupported

Given offsets are requested

And the tokenizer cannot provide offsets

When encode runs

Then Runtime reports offsets unsupported.

---

### Requirement: Batch Encoding

The tokenizer SHALL preserve input ordering for batch encoding.

It SHOULD support batch encoding.

#### Scenario: Batch encode

Given inputs A, B, and C

When batch encode runs

Then outputs correspond to A, B, and C in the same order.

---

### Requirement: Padding Policy

Padding SHALL be explicit.

If padding is requested and no PAD token is available, encoding SHALL fail.

#### Scenario: Padding without PAD

Given padding is requested

And tokenizer metadata has no PAD token

When batch encoding runs

Then Runtime returns a padding-token-missing error.

---

### Requirement: Attention Mask

When tokenizer output includes attention masks, mask length SHALL match token ID
length.

#### Scenario: Attention mask length

Given batch encoding returns token IDs and attention masks

When Runtime validates output

Then each attention mask has the same length as the corresponding token list.

---

### Requirement: Token Type IDs

Token type IDs SHALL have explicit unsupported behavior.

They MAY be returned where supported.

If requested but unsupported, the tokenizer SHALL report unsupported behavior.

#### Scenario: Token type IDs unsupported

Given token type IDs are requested

And the tokenizer does not support them

When encode runs

Then Runtime reports unsupported token type IDs.

---

### Requirement: Stop Sequence Token Preparation

The tokenizer SHALL keep stop sequence preparation separate from generation stopping semantics.

It SHOULD support resolving textual stop sequences to token patterns
where feasible.

Full stop condition evaluation is defined by Generation.

#### Scenario: Stop text

Given a stop sequence string is configured

When Runtime prepares generation

Then tokenizer may provide token pattern metadata for that stop sequence.

---

### Requirement: Tokenizer Memory Management

Tokenizer execution SHALL respect Runtime Memory Manager policy.

#### Scenario: Batch token buffer

Given batch encoding requires output buffers

When Memory Manager denies allocation

Then tokenization fails with a structured memory error.

---

### Requirement: Browser-Compatible Tokenizer

The Tokenizer Contract SHALL not require Wasmtime or native Provider loading.

#### Scenario: Browser target

Given Magnetar runs on a browser target

When tokenizer support is enabled

Then tokenizer implementation uses a browser-compatible path.

---

### Requirement: Tokenizer Observability

Runtime tokenizer observations SHALL avoid raw prompt content by default.

Runtime SHOULD emit tokenizer observations without logging raw prompt text by
default.

#### Scenario: Encode observation

Given encode completes

When observability records it

Then it may record token count and duration

But does not record raw prompt text by default.

---

### Requirement: Tokenizer Error Categories

Tokenizer failures SHALL use structured error categories.

#### Scenario: Invalid UTF-8

Given tokenizer decode encounters invalid UTF-8 or incomplete byte fallback
state

When decode is requested

Then Runtime reports a stable tokenizer decode error or pending partial state.

### Requirement: Tokenizer Provides Generation Input

Tokenizer output SHALL provide token IDs and prompt token count usable by the
Generation Contract.

#### Scenario: Prompt encoded

Given a prompt is encoded

When generation request is constructed

Then token IDs and token count come from tokenizer output.

---

### Requirement: Tokenizer Provides Streaming Decode For Generation

Tokenizer streaming decode SHALL be used to convert generated token streams into
valid text chunks.

#### Scenario: Generated token stream

Given generation emits token IDs

When text streaming is requested

Then tokenizer streaming decode produces text chunks.

---

### Requirement: Tokenizer Provides Stop Sequence Preparation

Tokenizer SHALL define token pattern preparation for textual stop sequences
where feasible.

#### Scenario: Text stop sequence

Given a user configures a textual stop sequence

When generation validates stop conditions

Then Runtime asks tokenizer to prepare token matching metadata where possible.

### Requirement: Session References Tokenizer

An Inference Session SHALL reference a tokenizer compatible with its model
context.

#### Scenario: Session tokenizer

Given session creation requests model M and tokenizer T

When T is incompatible with M

Then session creation fails.

---

### Requirement: Session May Own Tokenizer Streaming State

A session operation SHALL be able to own tokenizer streaming decode state.

#### Scenario: Streaming decode state

Given generated tokens are being decoded incrementally

When a token produces partial output

Then the session operation preserves tokenizer decode state.

### Requirement: Tokenizer Metadata Supports Sampling

Tokenizer metadata SHALL provide token ID validity, vocabulary size, special
token metadata, and stop token preparation used by Sampling.

#### Scenario: Vocabulary size

Given logits length differs from tokenizer vocabulary size

When Sampling validates input

Then Sampling reports vocabulary-mismatch.

---

### Requirement: Text Constraints Become Token Constraints

Textual constraints SHALL be converted through Tokenizer before affecting
Sampling.

#### Scenario: Text stop sequence

Given a textual stop sequence is configured

When Sampling or Generation needs token constraints

Then Runtime uses Tokenizer-derived token metadata.

---

### Requirement: Qwen Tokenizer Compatibility

Tokenizer compatibility for Qwen baseline SHALL validate vocabulary and special
token metadata required by the Qwen Model Component.

#### Scenario: EOS missing

Given Qwen generation metadata requires EOS token

When tokenizer compatibility is checked

Then Runtime rejects tokenizer if EOS metadata is unavailable.

---

### Requirement: Qwen Component Does Not Own Tokenization

Qwen Model Component SHALL not own tokenization execution.

#### Scenario: Encode prompt

Given user prompt needs tokenization

When Runtime processes it

Then Tokenizer Contract performs encoding before Qwen graph execution.

---

### Requirement: Tokenizer Is Exposed Through Inference API

Runtime Inference API SHALL expose encode, decode, and streaming decode through the Tokenizer Contract.

#### Scenario: Tokenize prompt

Given prompt text is supplied

When API tokenization runs

Then Tokenizer Contract produces token IDs.

---

### Requirement: Inference API Tokenization Is Redacted By Default

Raw prompt logging SHALL be disabled by default for tokenization API use.

#### Scenario: Tokenization failed

Given tokenization fails

When diagnostics are emitted

Then raw prompt text is not logged by default.

---

### Requirement: CLI May Send Text Or Chat Messages

Runtime SHALL accept plain text, chat messages, or already-tokenized input from
`magnetar-cli`.

`magnetar-cli` MAY send plain text, chat messages, or already-tokenized input to
Runtime.

#### Scenario: Chat input

Given CLI has chat transcript

When generating response

Then CLI sends appropriate chat messages or prompt text through Runtime
Inference API.

---

### Requirement: Runtime Tokenization Does Not Read CLI Files

Tokenizer execution through Runtime SHALL not read CLI workspace files.

#### Scenario: Template path

Given request attempts to make Runtime read template from workspace path

When Runtime validates it

Then access is denied unless template is already an authorized Model/Tokenizer
Artifact component.

---

### Requirement: E2E Uses Tokenizer Contract

E2E conformance SHALL use Tokenizer Contract for text prompt encoding and output
decoding.

#### Scenario: Text prompt

Given fixture text prompt is submitted

When inference runs

Then Runtime tokenizes it through Tokenizer Contract.

---

### Requirement: E2E Validates Tokenizer Failure

E2E conformance SHALL include tokenizer failure cases.

#### Scenario: Incompatible tokenizer

Given tokenizer vocabulary is incompatible with fixture model

When compatibility validation runs

Then Runtime reports structured tokenizer incompatibility.

### Requirement: Tokenizer Formats Normalize Into Tokenizer Artifact

tokenizer.json, tokenizer_config, SentencePiece, and embedded tokenizer metadata SHALL normalize into Tokenizer Artifact metadata.

#### Scenario: tokenizer.json normalized

Given tokenizer.json is parsed

When normalization completes

Then Tokenizer Contract receives normalized Tokenizer Artifact metadata.

---

### Requirement: Generation Config Does Not Override Tokenizer Policy Silently

Tokenizer-related metadata from generation_config or tokenizer_config SHALL not
silently override Tokenizer or Runtime policy.

#### Scenario: PAD token mismatch

Given tokenizer_config and generation_config disagree on PAD token

When compatibility validation runs

Then Runtime resolves according to policy or reports conflict.

### Requirement: Tokenizer Artifacts May Use Source Cache

Tokenizer Artifacts MAY be resolved through source/cache workflow, and Runtime SHALL validate the tokenizer cache entry before use.

#### Scenario: Cached tokenizer

Given tokenizer artifact is cached

When model loading validates tokenizer compatibility

Then cached tokenizer metadata is validated before use.

---

### Requirement: Tokenizer Cache Does Not Override Compatibility

Tokenizer cache hit SHALL not bypass tokenizer/model compatibility validation.

#### Scenario: Wrong tokenizer cached

Given cached tokenizer is incompatible with model

When loading runs

Then Runtime rejects compatibility.

