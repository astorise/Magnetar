# Tasks

## 1. Model Artifact Scope

- [x] Define Model Artifact as inference data.
- [x] Document distinction from Component Artifact.
- [x] Document distinction from Provider binary.
- [x] Document distinction from Device metadata.
- [x] Document distinction from Runtime configuration.
- [x] Document distinction from Model Instance.
- [x] Document that model architecture is not Provider.

## 2. Model Module

- [x] Create first-class `model` module or equivalent.
- [x] Export canonical model artifact types from crate root.
- [x] Keep model artifact identity separate from Component identity.
- [x] Keep model artifact identity separate from Provider identity.
- [x] Keep model artifact identity separate from Device identity.
- [x] Add module-level documentation.

## 3. Artifact Kinds

- [x] Define `model-bundle` artifact kind.
- [x] Define `model-weights` artifact kind.
- [x] Define `model-config` artifact kind.
- [x] Define `tokenizer` artifact kind.
- [x] Define `tokenizer-config` artifact kind.
- [x] Define `chat-template` artifact kind.
- [x] Define `prompt-template` artifact kind.
- [x] Define `generation-config` artifact kind.
- [x] Define `quantization-config` artifact kind.
- [x] Define `adapter` artifact kind.
- [x] Define `vocabulary` artifact kind.
- [x] Define `special-tokens` artifact kind.

## 4. Artifact Identity

- [x] Define ModelArtifactId.
- [x] Define logical model name.
- [x] Define model revision.
- [x] Define optional model variant.
- [x] Define artifact digest.
- [x] Define digest algorithm.
- [x] Define source identity.
- [x] Define shard identity.
- [x] Ensure digest is authoritative content identity.
- [x] Reject identity based only on path, tag, or alias.

## 5. Digest Validation

- [x] Support SHA-256 digest.
- [x] Compute digest from artifact bytes.
- [x] Compare computed digest with declared digest.
- [x] Reject digest mismatch.
- [x] Normalize digest format.
- [x] Add digest tests.
- [x] Add digest mismatch tests.

## 6. Manifest Schema

- [x] Define Model Artifact manifest schema.
- [x] Include schema version.
- [x] Include artifact kind.
- [x] Include model identity.
- [x] Include revision.
- [x] Include architecture metadata.
- [x] Include artifact parts.
- [x] Include digests.
- [x] Include dtype metadata.
- [x] Include tensor metadata where available.
- [x] Include tokenizer references.
- [x] Include template references.
- [x] Include generation defaults.
- [x] Include quantization metadata.
- [x] Include sharding metadata.
- [x] Include Runtime feature requirements.
- [x] Include Memory Manager feature requirements.
- [x] Include Provider Capability requirements.
- [x] Include optional Component requirements.
- [x] Include optional license metadata.
- [x] Include optional provenance metadata.
- [x] Include optional signature metadata.

## 7. Manifest Validation

- [x] Validate required fields.
- [x] Validate schema version.
- [x] Validate artifact kind.
- [x] Validate logical model name.
- [x] Validate revision.
- [x] Validate architecture metadata.
- [x] Validate artifact part references.
- [x] Validate digest syntax.
- [x] Validate dtype metadata.
- [x] Validate tokenizer references.
- [x] Validate template references.
- [x] Validate quantization metadata.
- [x] Validate sharding metadata.
- [x] Reject invalid manifest before loading.

## 8. Bundle Validation

- [x] Validate all required bundle parts.
- [x] Validate model weights reference.
- [x] Validate model config reference.
- [x] Validate tokenizer reference when required.
- [x] Validate tokenizer config reference when required.
- [x] Validate chat template reference when required.
- [x] Validate generation config reference when present.
- [x] Validate quantization config reference when present.
- [x] Reject incomplete bundle.

## 9. Architecture Metadata

- [x] Define architecture family.
- [x] Define architecture identifier.
- [x] Define architecture version where useful.
- [x] Define variant.
- [x] Define required architecture implementation.
- [x] Reject unsupported architecture.
- [x] Ensure architecture does not select Provider.
- [x] Add architecture validation tests.

## 10. Component Relationship

- [x] Allow optional Model Component requirement.
- [x] Keep Model Component Artifact separate from Model Artifact.
- [x] Validate Component requirement by Capability or architecture role.
- [x] Do not merge Component trust with Model Artifact trust.
- [x] Add tests proving Component Artifact and Model Artifact identities differ.

## 11. Provider Relationship

- [x] Prevent Model Artifact from selecting Provider directly.
- [x] Prevent manifest field from acting as authoritative Provider selector.
- [x] Allow declared required Capabilities.
- [x] Allow declared compute constraints.
- [x] Use Runtime Resolution later for Provider selection.
- [x] Add tests rejecting direct Provider pinning.

## 12. Device Relationship

- [x] Prevent Model Artifact from selecting Device directly.
- [x] Allow memory requirement metadata.
- [x] Allow dtype/layout requirement metadata.
- [x] Allow sharding requirement metadata.
- [x] Use Runtime planning and Memory Manager for placement.
- [x] Add tests rejecting direct Device pinning.

## 13. DType Metadata

- [x] Define storage dtype.
- [x] Define compute dtype requirement or preference.
- [x] Define supported compute dtype list.
- [x] Define quantized dtype identifiers.
- [x] Validate unsupported storage dtype.
- [x] Validate unsupported compute dtype.
- [x] Integrate with Memory Manager feasibility.
- [x] Add dtype tests.

## 14. Quantization Metadata

- [x] Define quantization format.
- [x] Define group size.
- [x] Define block size.
- [x] Define scale dtype.
- [x] Define zero-point dtype.
- [x] Define per-channel/per-tensor metadata.
- [x] Define required dequantization workspace metadata.
- [x] Define Provider Capability requirements for quantized execution.
- [x] Reject unsupported quantization format.
- [x] Add quantization tests.

## 15. Sharding Metadata

- [x] Define shard identity.
- [x] Define shard digest.
- [x] Define shard size.
- [x] Define shard ordering.
- [x] Define tensor-to-shard mapping where available.
- [x] Validate all required shards.
- [x] Reject missing shard.
- [x] Reject shard digest mismatch.
- [x] Add sharding tests.

## 16. Tensor Metadata

- [x] Define tensor name.
- [x] Define tensor shape.
- [x] Define tensor storage dtype.
- [x] Define tensor layout.
- [x] Define tensor shard reference.
- [x] Define tensor offset.
- [x] Define tensor size.
- [x] Define tensor quantization metadata.
- [x] Define expected compute dtype where useful.
- [x] Validate tensor metadata.
- [x] Add tensor metadata tests.

## 17. Tokenizer Association

- [x] Allow model bundle to reference tokenizer artifact.
- [x] Allow model bundle to reference tokenizer config.
- [x] Allow vocabulary artifact reference.
- [x] Allow special tokens reference.
- [x] Validate required tokenizer association for text generation models.
- [x] Do not define tokenizer execution contract in this change.

## 18. Template Association

- [x] Allow chat template artifact reference.
- [x] Allow prompt template artifact reference.
- [x] Validate template reference syntax.
- [x] Validate missing template when required by architecture metadata.
- [x] Do not define full template rendering contract in this change.

## 19. Generation Defaults

- [x] Define optional generation config artifact.
- [x] Define default temperature.
- [x] Define default top-p.
- [x] Define default top-k.
- [x] Define default max tokens.
- [x] Define default stop tokens.
- [x] Define default repetition penalty.
- [x] Treat defaults as overridable policy, not hard Runtime law.

## 20. Adapter Compatibility

- [x] Define adapter artifact kind.
- [x] Define adapter target architecture.
- [x] Define adapter base model compatibility.
- [x] Define adapter dtype metadata.
- [x] Define adapter rank metadata where useful.
- [x] Define placeholder for LoRA/adapter behavior.
- [x] Defer full adapter loading semantics.

## 21. Trust Policy

- [x] Define Model Artifact trust status.
- [x] Reuse or align digest trust semantics with Component Artifact trust.
- [x] Ensure Model Artifact cannot declare itself trusted.
- [x] Support trusted digest.
- [x] Support rejected digest.
- [x] Support revoked digest.
- [x] Support source policy.
- [x] Support publisher policy.
- [x] Add trust tests.

## 22. License Metadata

- [x] Define license metadata field.
- [x] Define license identifier.
- [x] Define license URL where provided.
- [x] Define usage restrictions metadata where provided.
- [x] Record license metadata.
- [x] Do not implement enforcement in this change.

## 23. Provenance Metadata

- [x] Define source repository metadata.
- [x] Define registry/source model identifier.
- [x] Define conversion tool metadata.
- [x] Define conversion timestamp metadata.
- [x] Define builder identity metadata.
- [x] Define commit digest metadata.
- [x] Define publisher metadata.
- [x] Keep provenance separate from trust.
- [x] Add provenance tests.

## 24. Signature Metadata

- [x] Define optional signature metadata.
- [x] Bind signature metadata to digest.
- [x] Treat unsupported signatures as unverified or policy denied.
- [x] Do not trust signature presence by default.
- [x] Add signature metadata tests.

## 25. Source Model

- [x] Define local source metadata.
- [x] Define local cache source metadata.
- [x] Define client-provided source metadata.
- [x] Reserve registry source metadata.
- [x] Reserve Hugging Face style source metadata.
- [x] Reserve OCI source metadata.
- [x] Reserve Tachyon source metadata.
- [x] Treat source identity as metadata, not trust.

## 26. Memory Manager Integration

- [x] Request memory feasibility for model loading.
- [x] Pass storage dtype to Memory Manager.
- [x] Pass compute dtype requirements to Memory Manager.
- [x] Pass quantization workspace requirements to Memory Manager.
- [x] Pass sharding metadata to Memory Manager.
- [x] Pass adapter residency placeholder to Memory Manager.
- [x] Reject model loading when memory feasibility fails.
- [x] Add memory feasibility tests.

## 27. Model Residency Planning

- [x] Define ModelResidencyPlan placeholder.
- [x] Distinguish artifact bytes from resident model memory.
- [x] Distinguish compressed storage from compute-ready memory.
- [x] Distinguish host residency from device residency.
- [x] Distinguish provider-owned residency from Runtime-owned residency.
- [x] Do not fully define Model Instance lifecycle in this change.

## 28. Error Model

- [x] Define manifest missing error.
- [x] Define manifest invalid error.
- [x] Define unsupported manifest version error.
- [x] Define artifact digest mismatch error.
- [x] Define missing required part error.
- [x] Define shard digest mismatch error.
- [x] Define incomplete shard set error.
- [x] Define unsupported model architecture error.
- [x] Define unsupported artifact format error.
- [x] Define unsupported storage dtype error.
- [x] Define unsupported compute dtype error.
- [x] Define unsupported quantization format error.
- [x] Define invalid tensor metadata error.
- [x] Define tokenizer reference missing error.
- [x] Define template reference missing error.
- [x] Define trust rejected error.
- [x] Define revoked artifact error.
- [x] Define license policy denied error.
- [x] Define memory feasibility failed error.
- [x] Define model source unavailable error.

## 29. Observability

- [x] Emit model artifact discovered observation.
- [x] Emit manifest loaded observation.
- [x] Emit manifest validation failed observation.
- [x] Emit digest computed observation.
- [x] Emit digest mismatch observation.
- [x] Emit shard validated observation.
- [x] Emit artifact trusted observation.
- [x] Emit artifact rejected observation.
- [x] Emit memory feasibility checked observation.
- [x] Emit model residency planned observation.
- [x] Emit model artifact cached observation.
- [x] Emit model artifact evicted observation.
- [x] Emit model source failure observation.

## 30. Tests

- [x] Test valid model bundle manifest.
- [x] Test missing manifest.
- [x] Test invalid manifest.
- [x] Test unsupported manifest version.
- [x] Test digest mismatch.
- [x] Test missing required model part.
- [x] Test missing shard.
- [x] Test shard digest mismatch.
- [x] Test unsupported architecture.
- [x] Test unsupported storage dtype.
- [x] Test unsupported compute dtype.
- [x] Test unsupported quantization format.
- [x] Test tokenizer reference missing.
- [x] Test template reference missing.
- [x] Test direct Provider selection rejected.
- [x] Test direct Device selection rejected.
- [x] Test Model Artifact and Component Artifact distinction.
- [x] Test trust rejected.
- [x] Test revoked artifact.
- [x] Test memory feasibility failure.

## 31. Documentation

- [x] Document Model Artifact.
- [x] Document Model Artifact versus Component Artifact.
- [x] Document Model Artifact versus Provider.
- [x] Document Model Artifact versus Model Instance.
- [x] Document model bundle structure.
- [x] Document digest identity.
- [x] Document manifest schema.
- [x] Document architecture metadata.
- [x] Document dtype metadata.
- [x] Document quantization metadata.
- [x] Document sharding metadata.
- [x] Document tokenizer/template association.
- [x] Document Memory Manager relationship.
- [x] Document trust and provenance.
- [x] Document non-goals.

## 32. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run Model Artifact tests.
- [x] Run Memory Manager integration tests.
- [x] Run Component Artifact tests where impacted.
- [x] Run Provider conformance tests where impacted.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify Model Artifact does not select Provider.
- [x] Verify Model Artifact does not select Device.
- [x] Verify Model Artifact and Component Artifact remain distinct.
