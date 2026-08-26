# Tasks

## 1. Roadmap Scope

- [x] Define post-baseline model format roadmap.
- [x] Document format support as ingestion/validation/normalization.
- [x] Document format support versus Provider support.
- [x] Document format support versus Model Component support.
- [x] Document local file boundary.
- [x] Document network/download boundary.
- [x] Document non-goals.

## 2. Model Format Roadmap Module

- [x] Create model format roadmap documentation or module.
- [x] Define format support phases.
- [x] Define normalized manifest target.
- [x] Define parser-to-artifact normalization path.
- [x] Define conformance fixture strategy.
- [x] Add roadmap validation tests where applicable.

## 3. Normalized Manifest

- [x] Define artifact identity metadata.
- [x] Define digest metadata.
- [x] Define architecture family metadata.
- [x] Define model type metadata.
- [x] Define config metadata.
- [x] Define weight file metadata.
- [x] Define tensor inventory metadata.
- [x] Define tokenizer file metadata.
- [x] Define chat template metadata.
- [x] Define generation defaults metadata.
- [x] Define quantization metadata.
- [x] Define adapter metadata.
- [x] Define license metadata.
- [x] Define provenance metadata.
- [x] Define trust metadata.
- [x] Define integrity metadata.
- [x] Define source metadata.
- [x] Define annotations.
- [x] Add manifest tests.

## 4. Safetensors Support

- [x] Define safetensors parser boundary.
- [x] Parse metadata.
- [x] Parse tensor name inventory.
- [x] Parse tensor shape metadata.
- [x] Parse tensor dtype metadata.
- [x] Parse tensor byte range metadata.
- [x] Validate integrity where available.
- [x] Add sharding placeholder.
- [x] Add memory mapping placeholder.
- [x] Add streaming read placeholder.
- [x] Prevent raw file handle exposure.
- [x] Prevent raw memory pointer exposure.
- [x] Add safetensors tests.

## 5. Sharded Weights

- [x] Define shard index metadata.
- [x] Define shard file list.
- [x] Define tensor-to-shard mapping.
- [x] Define per-shard digest.
- [x] Define total size estimate.
- [x] Detect missing shards.
- [x] Detect duplicate tensors.
- [x] Validate tensor shape consistency.
- [x] Define loading order policy.
- [x] Define partial loading placeholder.
- [x] Add sharded artifact tests.

## 6. Hugging Face-style Config

- [x] Parse architecture metadata.
- [x] Parse model_type.
- [x] Parse hidden_size.
- [x] Parse num_hidden_layers.
- [x] Parse num_attention_heads.
- [x] Parse num_key_value_heads.
- [x] Parse head_dim.
- [x] Parse intermediate_size.
- [x] Parse vocab_size.
- [x] Parse max_position_embeddings.
- [x] Parse rms_norm_eps.
- [x] Parse hidden_act.
- [x] Parse RoPE metadata.
- [x] Parse tie_word_embeddings.
- [x] Preserve torch_dtype as source metadata.
- [x] Preserve unsupported fields as annotations or reject by policy.
- [x] Add config tests.

## 7. tokenizer.json Support

- [x] Define tokenizer.json parser boundary.
- [x] Parse tokenizer identity.
- [x] Parse vocabulary metadata.
- [x] Parse model metadata where available.
- [x] Parse normalizer metadata.
- [x] Parse pre-tokenizer metadata.
- [x] Parse decoder metadata.
- [x] Parse added tokens.
- [x] Parse special tokens.
- [x] Validate encode/decode compatibility metadata.
- [x] Parse offset support metadata where available.
- [x] Add tokenizer.json tests.

## 8. tokenizer_config Support

- [x] Parse tokenizer class metadata.
- [x] Parse model max length.
- [x] Parse padding side.
- [x] Parse truncation side.
- [x] Parse chat template reference or inline template.
- [x] Parse BOS token metadata.
- [x] Parse EOS token metadata.
- [x] Parse PAD token metadata.
- [x] Parse added special token metadata.
- [x] Parse clean-up tokenization spaces metadata.
- [x] Prevent silent Runtime policy override.
- [x] Add tokenizer_config tests.

## 9. generation_config Support

- [x] Parse max length metadata.
- [x] Parse max new tokens metadata.
- [x] Parse temperature.
- [x] Parse top-k.
- [x] Parse top-p.
- [x] Parse repetition penalty.
- [x] Parse EOS token ID.
- [x] Parse BOS token ID.
- [x] Parse PAD token ID.
- [x] Parse do_sample.
- [x] Parse stop strings where present.
- [x] Treat parsed values as defaults.
- [x] Add generation_config tests.

## 10. Chat Template Support

- [x] Define chat template metadata.
- [x] Define template identity.
- [x] Define source metadata.
- [x] Define tokenizer compatibility.
- [x] Define model family compatibility.
- [x] Define variable requirements.
- [x] Define special token interaction.
- [x] Define rendering diagnostics.
- [x] Enforce raw prompt redaction.
- [x] Prevent arbitrary filesystem fetch during inference.
- [x] Prevent arbitrary network fetch during inference.
- [x] Add chat template tests.

## 11. SentencePiece Support

- [x] Define SentencePiece artifact support.
- [x] Parse model identity.
- [x] Parse vocabulary size.
- [x] Parse special token metadata.
- [x] Parse normalization metadata where available.
- [x] Validate encode/decode behavior through Tokenizer Contract.
- [x] Define browser support status.
- [x] Parse license/provenance metadata where available.
- [x] Reject unsupported SentencePiece features explicitly.
- [x] Add SentencePiece tests.

## 12. GGUF Support

- [x] Define GGUF parser boundary.
- [x] Extract GGUF metadata.
- [x] Extract tensor inventory.
- [x] Extract tensor shape metadata.
- [x] Extract tensor dtype metadata.
- [x] Extract quantization metadata.
- [x] Extract tokenizer metadata where embedded.
- [x] Extract architecture metadata.
- [x] Extract alignment/storage metadata.
- [x] Validate integrity where available.
- [x] Define memory mapping policy.
- [x] Normalize into Model Artifact metadata.
- [x] Prevent GGUFProvider.
- [x] Add GGUF tests.

## 13. Adapter Format Support

- [x] Define LoRA safetensors support.
- [x] Parse adapter_config metadata.
- [x] Parse target module metadata.
- [x] Parse rank.
- [x] Parse alpha.
- [x] Parse scaling.
- [x] Preserve dropout as training/source metadata.
- [x] Parse base model compatibility.
- [x] Parse tensor inventory.
- [x] Parse dtype metadata.
- [x] Parse quantization metadata.
- [x] Parse license/provenance metadata.
- [x] Normalize into Adapter Artifact.
- [x] Add adapter format tests.

## 14. Quantized Artifact Metadata

- [x] Define quantization method metadata.
- [x] Define bits per value.
- [x] Define group size.
- [x] Define storage dtype.
- [x] Define compute dtype expectation.
- [x] Define scale dtype.
- [x] Define zero point dtype.
- [x] Define packing layout.
- [x] Define tensor-specific quantization metadata.
- [x] Define dequantization requirements.
- [x] Define Provider/Kernel compatibility metadata.
- [x] Prevent hidden quantization/dequantization.
- [x] Add quantized artifact tests.

## 15. Source And Distribution Boundary

- [x] Use existing source validation contracts.
- [x] Prevent arbitrary Runtime downloads during inference.
- [x] Preserve CLI download UX as future work.
- [x] Preserve Runtime normalized artifact validation.
- [x] Add source boundary tests.

## 16. Local File Boundary

- [x] Require local path resolution outside Runtime or through authorized source.
- [x] Prevent Runtime arbitrary directory scanning.
- [x] Convert local paths to client-provided artifact references.
- [x] Validate references before loading.
- [x] Add local file boundary tests.

## 17. Trust And Integrity

- [x] Validate digests.
- [x] Validate parts.
- [x] Validate shards.
- [x] Validate manifest consistency.
- [x] Validate metadata consistency.
- [x] Validate tensor inventory consistency.
- [x] Validate tokenizer compatibility.
- [x] Validate license/provenance policy.
- [x] Validate signature status where available.
- [x] Validate revocation status where available.
- [x] Prevent format-alone trust.
- [x] Add trust/integrity tests.

## 18. Format Normalization

- [x] Normalize external metadata into stable Magnetar metadata.
- [x] Preserve source annotations.
- [x] Avoid making source annotations authoritative without validation.
- [x] Treat torch_dtype as source metadata unless policy validates it.
- [x] Add normalization tests.

## 19. Format Conformance

- [x] Add valid minimal artifact fixture.
- [x] Add missing required metadata fixture.
- [x] Add invalid tensor shape fixture.
- [x] Add invalid dtype fixture.
- [x] Add invalid shard index fixture.
- [x] Add missing shard fixture.
- [x] Add duplicate tensor fixture.
- [x] Add tokenizer mismatch fixture.
- [x] Add unsupported quantization fixture.
- [x] Add malformed file metadata fixture.
- [x] Add untrusted artifact fixture.
- [x] Add redaction checks.

## 20. Error Model

- [x] Define model-format-unsupported error.
- [x] Define model-format-invalid error.
- [x] Define model-format-parser-failed error.
- [x] Define model-manifest-invalid error.
- [x] Define model-manifest-missing error.
- [x] Define model-config-invalid error.
- [x] Define safetensors-invalid error.
- [x] Define safetensors-tensor-missing error.
- [x] Define safetensors-dtype-unsupported error.
- [x] Define shard-index-invalid error.
- [x] Define shard-missing error.
- [x] Define shard-digest-mismatch error.
- [x] Define tokenizer-json-invalid error.
- [x] Define tokenizer-config-invalid error.
- [x] Define generation-config-invalid error.
- [x] Define chat-template-invalid error.
- [x] Define sentencepiece-unsupported error.
- [x] Define gguf-invalid error.
- [x] Define gguf-quantization-unsupported error.
- [x] Define adapter-format-invalid error.
- [x] Define quantization-metadata-invalid error.
- [x] Define model-format-trust-denied error.
- [x] Define model-format-integrity-failed error.
- [x] Define model-format-local-file-denied error.
- [x] Define model-format-network-denied error.
- [x] Define internal-model-format error.

## 21. Observability

- [x] Emit model format detected observation.
- [x] Emit manifest normalized observation.
- [x] Emit manifest validation failed observation.
- [x] Emit config parsed observation.
- [x] Emit config validation failed observation.
- [x] Emit tensor inventory parsed observation.
- [x] Emit tensor inventory mismatch observation.
- [x] Emit tokenizer metadata parsed observation.
- [x] Emit tokenizer compatibility failed observation.
- [x] Emit generation config parsed observation.
- [x] Emit chat template parsed observation.
- [x] Emit safetensors parsed observation.
- [x] Emit shard index parsed observation.
- [x] Emit shard missing observation.
- [x] Emit GGUF metadata parsed observation.
- [x] Emit quantization metadata parsed observation.
- [x] Emit adapter metadata parsed observation.
- [x] Emit integrity validation failed observation.
- [x] Emit trust validation failed observation.
- [x] Verify default redaction.

## 22. Tests

- [x] Test safetensors normalization.
- [x] Test sharded safetensors normalization.
- [x] Test config normalization.
- [x] Test tokenizer.json normalization.
- [x] Test tokenizer_config normalization.
- [x] Test generation_config normalization.
- [x] Test chat template normalization.
- [x] Test SentencePiece unsupported or supported behavior.
- [x] Test GGUF normalization.
- [x] Test adapter format normalization.
- [x] Test quantized metadata validation.
- [x] Test local file boundary.
- [x] Test network boundary.
- [x] Test format-alone does not grant trust.
- [x] Test no raw file/mmap pointer exposure.
- [x] Test Model Loading still validates normalized artifacts.

## 23. Documentation

- [x] Document post-baseline model format roadmap.
- [x] Document normalized manifest.
- [x] Document safetensors support.
- [x] Document sharded weights.
- [x] Document Hugging Face-style config.
- [x] Document tokenizer.json support.
- [x] Document tokenizer_config support.
- [x] Document generation_config support.
- [x] Document chat template support.
- [x] Document SentencePiece support.
- [x] Document GGUF support.
- [x] Document adapter format support.
- [x] Document quantized artifact metadata.
- [x] Document source/download boundary.
- [x] Document local file boundary.
- [x] Document trust/integrity.
- [x] Document conformance.
- [x] Document non-goals.

## 24. Final Validation

- [x] Run OpenSpec validation.
- [x] Verify roadmap does not bypass Model Artifact contract.
- [x] Verify roadmap does not introduce format Providers.
- [x] Verify roadmap does not introduce arbitrary Runtime filesystem access.
- [x] Verify roadmap does not introduce arbitrary Runtime network access.
- [x] Verify normalized artifacts still pass Model Loading.