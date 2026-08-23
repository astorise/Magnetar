# Tasks

## 1. Tokenizer Scope

- [x] Define tokenizer as inference Runtime contract.
- [x] Document tokenizer versus Model Artifact.
- [x] Document tokenizer versus Component Artifact.
- [x] Document tokenizer versus chat template.
- [x] Document tokenizer versus generation.
- [x] Document tokenizer versus Provider.
- [x] Document tokenizer versus client workspace tools.

## 2. Tokenizer Module

- [x] Create first-class `tokenizer` module or equivalent.
- [x] Export canonical tokenizer types from crate root.
- [x] Keep tokenizer contract platform-neutral.
- [x] Keep tokenizer contract independent from Wasmtime.
- [x] Keep tokenizer contract independent from Provider selection.
- [x] Add module-level documentation.

## 3. Tokenizer Artifact Relationship

- [x] Reuse Model Artifact tokenizer kinds.
- [x] Support tokenizer artifact reference.
- [x] Support tokenizer config reference.
- [x] Support vocabulary artifact reference.
- [x] Support special tokens artifact reference.
- [x] Validate tokenizer artifacts are Runtime-registered.
- [x] Prevent tokenizer from reading arbitrary filesystem paths directly.

## 4. Tokenizer Identity

- [x] Define TokenizerId.
- [x] Define TokenizerArtifactId.
- [x] Define tokenizer digest reference.
- [x] Define tokenizer family.
- [x] Define tokenizer revision.
- [x] Define tokenizer compatibility metadata.
- [x] Add tokenizer identity tests.

## 5. Tokenizer Metadata

- [x] Define vocabulary size.
- [x] Define added token count.
- [x] Define token ID range.
- [x] Define model max length where known.
- [x] Define unknown token metadata.
- [x] Define BOS token metadata.
- [x] Define EOS token metadata.
- [x] Define PAD token metadata.
- [x] Define SEP token metadata.
- [x] Define CLS token metadata.
- [x] Define MASK token metadata.
- [x] Define additional special tokens.
- [x] Define byte fallback metadata.
- [x] Define normalization metadata.
- [x] Define pre-tokenizer metadata.

## 6. Special Tokens

- [x] Define special token identity.
- [x] Define special token string.
- [x] Define special token ID.
- [x] Define whether token is added during encode.
- [x] Define whether token is skipped during decode.
- [x] Define additional special token list.
- [x] Detect special token conflicts.
- [x] Detect missing required special tokens.
- [x] Add special token tests.

## 7. Encode Contract

- [x] Define encode input.
- [x] Include text input.
- [x] Include add-special-tokens option.
- [x] Include truncation policy.
- [x] Include max token length option.
- [x] Include return-offsets option.
- [x] Include padding option where batch encoding uses it.
- [x] Include special token policy.
- [x] Define encode output.
- [x] Include token IDs.
- [x] Include token count.
- [x] Include optional offsets.
- [x] Include optional attention mask.
- [x] Include optional token type IDs.
- [x] Include diagnostics.
- [x] Add encode tests.

## 8. Decode Contract

- [x] Define decode input.
- [x] Include token IDs.
- [x] Include skip-special-tokens option.
- [x] Include cleanup option where supported.
- [x] Include streaming state reference where applicable.
- [x] Define decode output.
- [x] Include decoded text.
- [x] Include consumed token count.
- [x] Include pending partial state where applicable.
- [x] Include diagnostics.
- [x] Add decode tests.

## 9. Streaming Decode

- [x] Define streaming decode state.
- [x] Define incremental token input.
- [x] Define emitted text chunk.
- [x] Define pending partial output.
- [x] Handle partial UTF-8.
- [x] Handle byte fallback tokens.
- [x] Handle special token suppression.
- [x] Handle whitespace normalization where supported.
- [x] Define flush behavior.
- [x] Define invalid state behavior.
- [x] Add streaming decode tests.

## 10. Token ID Validation

- [x] Define token ID type.
- [x] Define valid token ID range.
- [x] Reject invalid token IDs.
- [x] Reject out-of-vocabulary token IDs unless fallback policy exists.
- [x] Add token ID validation tests.

## 11. Model Compatibility

- [x] Validate tokenizer digest against model manifest.
- [x] Validate vocabulary size compatibility.
- [x] Validate special token ID compatibility.
- [x] Validate model max length compatibility.
- [x] Validate tokenizer family compatibility.
- [x] Validate added token compatibility.
- [x] Validate normalization compatibility where declared.
- [x] Reject incompatible tokenizer/model combinations.
- [x] Add compatibility tests.

## 12. Chat Template Boundary

- [x] Document chat template before tokenizer flow.
- [x] Ensure tokenizer does not own structured message rendering.
- [x] Ensure tokenizer accepts rendered text or explicit tokenizer input.
- [x] Keep template rendering behavior for later change.
- [x] Add boundary tests where template stubs exist.

## 13. Prompt Length Accounting

- [x] Count tokens after special tokens are applied.
- [x] Count tokens after template rendering where applicable.
- [x] Expose prompt token count.
- [x] Validate context length.
- [x] Detect prompt too long.
- [x] Add prompt length tests.

## 14. Truncation

- [x] Define truncation policy enum.
- [x] Include none.
- [x] Include left.
- [x] Include right.
- [x] Include middle where supported.
- [x] Include model-default.
- [x] Include client-policy.
- [x] Reject overlong prompt when truncation forbidden.
- [x] Emit truncation diagnostics when truncation applied.
- [x] Add truncation tests.

## 15. Offsets

- [x] Define token offset type.
- [x] Define byte offset.
- [x] Define character offset where supported.
- [x] Define unsupported offsets behavior.
- [x] Validate offsets align with token output.
- [x] Add offset tests where supported.
- [x] Add unsupported offset tests.

## 16. Batch Encoding

- [x] Define batch encode input.
- [x] Preserve input ordering.
- [x] Define per-input output.
- [x] Define batch padding behavior.
- [x] Define per-input truncation behavior.
- [x] Define partial failure behavior.
- [x] Define attention mask behavior.
- [x] Add batch encode tests.

## 17. Padding

- [x] Define padding policy.
- [x] Include none.
- [x] Include longest.
- [x] Include max-length.
- [x] Include model-default.
- [x] Require PAD token when padding is requested.
- [x] Reject padding if PAD token missing.
- [x] Add padding tests.

## 18. Attention Mask

- [x] Define attention mask output.
- [x] Align attention mask length with token IDs.
- [x] Align mask with padding behavior.
- [x] Add attention mask tests.

## 19. Token Type IDs

- [x] Define token type ID output.
- [x] Support when tokenizer/model family requires it.
- [x] Return unsupported error when requested but unavailable.
- [x] Add token type ID tests.

## 20. Stop Token Preparation

- [x] Define textual stop sequence resolution where feasible.
- [x] Define token stop pattern representation.
- [x] Define unsupported stop sequence resolution behavior.
- [x] Preserve full stop behavior for generation change.
- [x] Add stop token preparation tests.

## 21. Memory Manager Integration

- [x] Register tokenizer artifact residency with Memory Manager.
- [x] Account for vocabulary memory.
- [x] Account for added token memory.
- [x] Account for temporary encode buffers.
- [x] Account for batch token buffers.
- [x] Account for streaming decode state buffers.
- [x] Reject unbounded allocation.
- [x] Add memory integration tests.

## 22. Component Implementation Path

- [x] Allow tokenizer implementation as Component where supported.
- [x] Keep Tokenizer Component Artifact separate from tokenizer data.
- [x] Validate Component trust independently.
- [x] Validate tokenizer artifact trust independently.
- [x] Link only inference-scoped tokenizer authority.
- [x] Add Component tokenizer tests where supported.

## 23. Native Implementation Path

- [x] Allow tokenizer implementation as native Runtime code.
- [x] Keep native implementation behind tokenizer contract.
- [x] Avoid Provider/Device selection exposure.
- [x] Add native tokenizer fixture tests where implemented.

## 24. Browser Implementation Path

- [x] Allow browser-compatible tokenizer implementation.
- [x] Do not require Wasmtime.
- [x] Do not require native Provider loading.
- [x] Respect browser memory constraints.
- [x] Add wasm32 compile checks where feasible.

## 25. Error Model

- [x] Define tokenizer artifact missing error.
- [x] Define tokenizer artifact invalid error.
- [x] Define tokenizer incompatible with model error.
- [x] Define unsupported tokenizer family error.
- [x] Define invalid token ID error.
- [x] Define unknown token error.
- [x] Define invalid UTF-8 error.
- [x] Define decode pending partial status.
- [x] Define offsets unsupported error.
- [x] Define padding token missing error.
- [x] Define truncation required error.
- [x] Define truncation forbidden error.
- [x] Define prompt too long error.
- [x] Define batch input invalid error.
- [x] Define special token missing error.
- [x] Define special token conflict error.
- [x] Define vocabulary mismatch error.
- [x] Define added token mismatch error.
- [x] Define memory allocation failed error.
- [x] Define streaming state invalid error.
- [x] Define implementation unavailable error.

## 26. Observability

- [x] Emit tokenizer loaded observation.
- [x] Emit tokenizer compatibility checked observation.
- [x] Emit encode requested observation.
- [x] Emit encode completed observation.
- [x] Emit encode failed observation.
- [x] Emit decode requested observation.
- [x] Emit decode completed observation.
- [x] Emit decode failed observation.
- [x] Emit streaming decode chunk observation.
- [x] Emit streaming decode pending partial observation.
- [x] Emit prompt too long observation.
- [x] Emit truncation applied observation.
- [x] Emit tokenizer memory pressure observation.
- [x] Emit tokenizer implementation unavailable observation.
- [x] Avoid raw prompt logging by default.

## 27. Tests

- [x] Test valid tokenizer metadata.
- [x] Test missing tokenizer artifact.
- [x] Test invalid tokenizer artifact.
- [x] Test tokenizer/model compatibility success.
- [x] Test tokenizer/model compatibility failure.
- [x] Test encode simple text.
- [x] Test encode with special tokens.
- [x] Test decode simple tokens.
- [x] Test decode skipping special tokens.
- [x] Test invalid token ID.
- [x] Test streaming decode partial UTF-8 or equivalent fixture.
- [x] Test prompt too long.
- [x] Test truncation forbidden.
- [x] Test truncation applied.
- [x] Test padding without PAD token fails.
- [x] Test batch encode preserves order.
- [x] Test offsets unsupported.
- [x] Test stop sequence token preparation where implemented.

## 28. Documentation

- [x] Document Tokenizer Contract.
- [x] Document Tokenizer Artifact relationship.
- [x] Document Tokenizer Component relationship.
- [x] Document encode.
- [x] Document decode.
- [x] Document streaming decode.
- [x] Document special tokens.
- [x] Document tokenizer/model compatibility.
- [x] Document chat template boundary.
- [x] Document prompt length accounting.
- [x] Document truncation.
- [x] Document memory relationship.
- [x] Document browser relationship.
- [x] Document non-goals.

## 29. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run wasm32 check where feasible.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run Tokenizer tests.
- [x] Run Model Artifact tests.
- [x] Run Memory Manager tests where impacted.
- [x] Run Component Runtime tests where impacted.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify tokenizer does not read arbitrary filesystem paths.
- [x] Verify tokenizer does not select Provider or Device.
- [x] Verify raw prompt logging is not default.