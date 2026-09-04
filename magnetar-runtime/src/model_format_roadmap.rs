//! Post-baseline model format roadmap contract (see
//! `openspec/changes/define-post-baseline-model-format-roadmap`).
//!
//! The first Magnetar baseline uses fixture model artifacts only -- small,
//! deterministic, local, CPU-only, and conformance-friendly. This module does
//! not implement byte-level safetensors/GGUF/SentencePiece parsers, does not
//! implement model downloads, and does not define model hub or CLI pull UX --
//! the proposal's "Non-Goals" section rules all of that out explicitly.
//! Instead it defines, as executable Rust types and validation functions, the
//! roadmap **contract** that any future model format ingestion work must
//! satisfy:
//!
//! - [`ModelFormatRoadmapPhase`]: the twelve post-baseline format phases
//!   (normalized manifest -> safetensors -> sharded weights -> Hugging
//!   Face-style config -> tokenizer.json -> tokenizer_config ->
//!   generation_config -> chat template -> SentencePiece -> GGUF -> adapter
//!   formats -> quantized artifact metadata).
//! - [`reject_model_format_provider_name`]: an executable, regression-proof
//!   rejection of format-shaped Provider names (`GGUFProvider`,
//!   `SafetensorsProvider`, `QwenSafetensorsProvider`-shaped), implementing
//!   "Format Support Is Not Provider Support".
//! - [`reject_format_execution_graph`]: "Format Parsers Do Not Produce
//!   Execution Graphs" -- a format parser is never an authoritative graph
//!   source.
//! - [`NormalizedManifestCoverage`]: proves the *existing*
//!   [`crate::ModelManifest`] contract already carries every field the
//!   roadmap's "Normalized Manifest" phase names (identity, digest,
//!   architecture, config, weights, tensor inventory, tokenizer, chat
//!   template, generation defaults, quantization, license, provenance,
//!   source), rather than introducing a parallel manifest type.
//! - Per-format metadata contracts (all normalizing into existing types --
//!   [`crate::ModelTensorMetadata`], [`crate::ModelArchitecture`],
//!   [`crate::TokenizerMetadata`], [`crate::AdapterArtifact`] -- no format
//!   introduces a parallel artifact type):
//!   [`SafetensorsManifest`] (Phase 2), [`ShardIndex`] with
//!   [`detect_missing_shards`] / [`detect_duplicate_tensor_names`] /
//!   [`validate_shard_tensor_shape_consistency`] (Phase 3),
//!   [`HfConfigMetadata`] (Phase 4, `torch_dtype` preserved as source
//!   metadata only), [`TokenizerJsonMetadata`] /
//!   [`normalize_tokenizer_json`] (Phase 5), [`TokenizerConfigMetadata`]
//!   (Phase 6), [`GenerationConfigMetadata`] (Phase 7, defaults only),
//!   [`ChatTemplateMetadata`] (Phase 8, source kind structurally excludes
//!   arbitrary filesystem/network fetch), [`SentencePieceMetadata`] /
//!   [`reject_unsupported_sentencepiece_feature`] (Phase 9),
//!   [`GgufMetadata`] (Phase 10), [`LoraAdapterFormatMetadata`] /
//!   [`normalize_lora_adapter`] (Phase 11, never auto-activates or
//!   auto-trusts), and [`ModelFormatQuantizationDeclaration`] /
//!   [`validate_model_format_quantization`] (Phase 12, composing
//!   [`crate::provider_roadmap::reject_hidden_dequantization`]).
//! - [`reject_arbitrary_model_download`] / [`validate_local_file_boundary`]:
//!   the source and local-file boundaries -- every [`crate::ModelArtifactSource`]
//!   variant is closed and authorized; nothing else is accepted.
//! - [`model_format_grants_no_trust`]: "a model format SHALL not be trusted
//!   merely because it is recognized" -- trust is always evaluated through
//!   [`crate::ModelTrustStore`], never inferred from a format parser.
//! - [`torch_dtype_does_not_force_compute_dtype`]: "Source Metadata Is Not
//!   Automatically Runtime Policy" made structurally checkable.
//! - [`ModelFormatRoadmapError`]: the 25 structured error categories from the
//!   proposal's "Error Model" section.
//! - [`ModelFormatRoadmapObservationKind`] / [`ModelFormatRoadmapObservation`]:
//!   the 19 observation categories, with redacted metadata only.
//! - [`ModelFormatConformanceFixtureKind`]: the twelve fixture categories from
//!   the proposal's "Format Conformance" section.
//! - [`ModelFormatRoadmapConformanceReport`] / [`run_model_format_roadmap_conformance`]:
//!   a small conformance report, in the shape of
//!   [`crate::ProviderRoadmapConformanceReport`], asserting the roadmap
//!   guarantees above hold.

use crate::{
    AdapterArtifact, AdapterArtifactId, AdapterBaseModelCompatibility, AdapterMethod,
    AdapterTargetModule, AdapterTargetModuleRole, AdapterTrustStatus, CapabilityBinding,
    ComputeDType, KernelQuantizationMetadata, ModelArchitecture, ModelArtifactId,
    ModelArtifactKind, ModelArtifactSource, ModelDType, ModelDigest, ModelGenerationDefaults,
    ModelLicenseMetadata, ModelManifest, ModelName, ModelProvenance, ModelQuantization,
    ModelRevision, ModelShard, ModelShardId, ModelTensorMetadata, ModelTrustDecision,
    ModelTrustStore, SpecialToken, TokenIdRange, TokenizerArtifactId, TokenizerFamily, TokenizerId,
    TokenizerMetadata, TokenizerRevision,
};
use crate::{
    compute::redact_backend_diagnostic,
    provider_roadmap::{reject_hidden_dequantization, validate_quantization_declaration},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

pub const MODEL_FORMAT_ROADMAP_VERSION: &str = "0.1.0";

// ---------------------------------------------------------------------
// Roadmap phases
// ---------------------------------------------------------------------

/// Post-baseline model format phases from the proposal, in `SHOULD`-order
/// (exact implementation order may vary).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ModelFormatRoadmapPhase {
    NormalizedManifest,
    Safetensors,
    ShardedWeights,
    HfConfig,
    TokenizerJson,
    TokenizerConfig,
    GenerationConfig,
    ChatTemplate,
    SentencePiece,
    Gguf,
    AdapterFormats,
    QuantizedMetadata,
}

/// The twelve roadmap phases in the proposal's order.
pub const MODEL_FORMAT_ROADMAP_PHASES: &[ModelFormatRoadmapPhase] = &[
    ModelFormatRoadmapPhase::NormalizedManifest,
    ModelFormatRoadmapPhase::Safetensors,
    ModelFormatRoadmapPhase::ShardedWeights,
    ModelFormatRoadmapPhase::HfConfig,
    ModelFormatRoadmapPhase::TokenizerJson,
    ModelFormatRoadmapPhase::TokenizerConfig,
    ModelFormatRoadmapPhase::GenerationConfig,
    ModelFormatRoadmapPhase::ChatTemplate,
    ModelFormatRoadmapPhase::SentencePiece,
    ModelFormatRoadmapPhase::Gguf,
    ModelFormatRoadmapPhase::AdapterFormats,
    ModelFormatRoadmapPhase::QuantizedMetadata,
];

impl ModelFormatRoadmapPhase {
    pub const fn id(self) -> &'static str {
        match self {
            Self::NormalizedManifest => "normalized-manifest",
            Self::Safetensors => "safetensors",
            Self::ShardedWeights => "sharded-weights",
            Self::HfConfig => "hf-config",
            Self::TokenizerJson => "tokenizer-json",
            Self::TokenizerConfig => "tokenizer-config",
            Self::GenerationConfig => "generation-config",
            Self::ChatTemplate => "chat-template",
            Self::SentencePiece => "sentencepiece",
            Self::Gguf => "gguf",
            Self::AdapterFormats => "adapter-formats",
            Self::QuantizedMetadata => "quantized-metadata",
        }
    }

    /// This phase's `SHOULD`-order position, 1-indexed as in the proposal.
    pub const fn ordinal(self) -> u8 {
        match self {
            Self::NormalizedManifest => 1,
            Self::Safetensors => 2,
            Self::ShardedWeights => 3,
            Self::HfConfig => 4,
            Self::TokenizerJson => 5,
            Self::TokenizerConfig => 6,
            Self::GenerationConfig => 7,
            Self::ChatTemplate => 8,
            Self::SentencePiece => 9,
            Self::Gguf => 10,
            Self::AdapterFormats => 11,
            Self::QuantizedMetadata => 12,
        }
    }

    /// Whether every artifact this phase produces SHALL normalize into an
    /// existing Runtime contract rather than a new parallel artifact type.
    /// Always `true`: this is the roadmap's central invariant.
    pub const fn normalizes_into_existing_contract(self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------
// No format Providers / no parser-owned execution graphs
// ---------------------------------------------------------------------

/// Format-shaped name fragments the roadmap's "Format Support Is Not
/// Provider Support" requirement forbids as a Provider identity, matching
/// the proposal's `GGUFProvider` / `SafetensorsProvider` /
/// `QwenSafetensorsProvider` examples. Non-exhaustive by design, as in
/// [`crate::provider_roadmap::reject_model_family_provider_name`].
const FORBIDDEN_MODEL_FORMAT_NAME_FRAGMENTS: &[&str] = &[
    "gguf",
    "safetensors",
    "sentencepiece",
    "tokenizerjson",
    "tokenizerconfig",
    "generationconfig",
    "chattemplate",
    "huggingfaceconfig",
    "loraadapter",
    "adapterformat",
    "quantizedartifact",
];

/// Rejects a Provider identity that names a model format instead of a
/// capability or hardware target, implementing "Format Support Is Not
/// Provider Support" (`specs/model-format-roadmap/spec.md`). Names such as
/// `ReferenceCpuProvider` or `CudaProvider` are not rejected.
pub fn reject_model_format_provider_name(name: &str) -> Result<(), ModelFormatRoadmapError> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err(ModelFormatRoadmapError::InternalModelFormatError {
            reason: "provider name must not be empty".into(),
        });
    }
    let lower = trimmed.to_ascii_lowercase().replace(['-', '_'], "");
    if let Some(format_part) = lower.strip_suffix("provider")
        && FORBIDDEN_MODEL_FORMAT_NAME_FRAGMENTS
            .iter()
            .any(|fragment| format_part.contains(fragment))
    {
        return Err(ModelFormatRoadmapError::ModelFormatUnsupported {
            reason: format!(
                "'{trimmed}' names a model-format Provider; model formats normalize into \
                 Model Artifact, Tokenizer Artifact, or Adapter Artifact, they do not become \
                 Providers"
            ),
        });
    }
    Ok(())
}

/// Rejects a format parser attempting to supply an authoritative execution
/// graph, implementing "Format Parsers Do Not Produce Execution Graphs"
/// (`specs/model-component/spec.md`). A parser may only ever produce
/// normalized metadata; execution graphs remain owned by Model Component.
pub fn reject_format_execution_graph(
    parser_supplied_graph: bool,
) -> Result<(), ModelFormatRoadmapError> {
    if parser_supplied_graph {
        Err(ModelFormatRoadmapError::ModelFormatUnsupported {
            reason: "format parsers do not produce authoritative execution graphs; \
                     architecture behavior is owned by Model Component"
                .into(),
        })
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Phase 1: Normalized manifest coverage
// ---------------------------------------------------------------------

/// Whether an already-validated [`ModelManifest`] carries each field the
/// roadmap's "Normalized Manifest" phase names. Implemented against the
/// *existing* Model Artifact contract rather than a parallel manifest type:
/// [`ModelManifest`] is already the canonical input to Model Loading.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NormalizedManifestCoverage {
    pub identity: bool,
    pub digest: bool,
    pub architecture_family: bool,
    pub weight_files: bool,
    pub tensor_inventory: bool,
    pub tokenizer: bool,
    pub chat_template: bool,
    pub generation_defaults: bool,
    pub quantization: bool,
    pub license: bool,
    pub provenance: bool,
    pub source: bool,
}

impl NormalizedManifestCoverage {
    /// Reports which roadmap-named fields `manifest` actually carries. Every
    /// field is independently observable -- a manifest is not "complete"
    /// just because it parses.
    pub fn from_manifest(manifest: &ModelManifest) -> Self {
        Self {
            identity: !manifest.id.name.as_str().is_empty(),
            digest: !manifest.id.digest.value.is_empty(),
            architecture_family: !manifest.architecture.family.is_empty(),
            weight_files: manifest
                .parts
                .values()
                .any(|part| part.kind == crate::ModelArtifactKind::ModelWeights),
            tensor_inventory: !manifest.tensors.is_empty(),
            tokenizer: manifest.tokenizer.is_some(),
            chat_template: manifest.chat_template.is_some(),
            generation_defaults: manifest.generation.is_some(),
            quantization: manifest.quantization.is_some(),
            license: manifest.license.is_some(),
            provenance: manifest.provenance.is_some(),
            source: manifest.source.is_some(),
        }
    }

    /// Whether every roadmap-named field is present. `identity`,
    /// `digest`, `architecture_family`, and `weight_files` are the only
    /// fields required for baseline Model Loading; the rest are optional
    /// per-format extensions.
    pub const fn covers_required_fields(&self) -> bool {
        self.identity && self.digest && self.architecture_family && self.weight_files
    }
}

// ---------------------------------------------------------------------
// Phase 2: safetensors
// ---------------------------------------------------------------------

/// A single safetensors tensor entry: name, shape, dtype, and byte range
/// only. Deliberately has no field through which a raw file handle or memory
/// pointer could be represented -- "Safetensors parsing SHALL not expose raw
/// file handles or memory pointers through public APIs" holds structurally.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafetensorsTensorEntry {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: ModelDType,
    pub byte_offset: u64,
    pub byte_length: u64,
}

/// Parsed safetensors metadata: tensor inventory plus free-form header
/// metadata. Sharding, memory mapping, and streaming reads are policy
/// placeholders (see [`MemoryMappingPolicy`]) -- this type never carries a
/// live file handle or mapped-memory pointer.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SafetensorsManifest {
    pub tensors: Vec<SafetensorsTensorEntry>,
    pub header_metadata: BTreeMap<String, String>,
}

impl SafetensorsManifest {
    /// Validates tensor names are non-empty and unique and shapes are
    /// non-degenerate, implementing part of "Safetensors Support".
    pub fn validate(&self) -> Result<(), ModelFormatRoadmapError> {
        let mut seen = BTreeSet::new();
        for tensor in &self.tensors {
            if tensor.name.trim().is_empty() {
                return Err(ModelFormatRoadmapError::SafetensorsInvalid {
                    reason: "tensor name must not be empty".into(),
                });
            }
            if !seen.insert(tensor.name.clone()) {
                return Err(ModelFormatRoadmapError::SafetensorsInvalid {
                    reason: format!("duplicate tensor name '{}'", tensor.name),
                });
            }
            if tensor.shape.contains(&0) {
                return Err(ModelFormatRoadmapError::SafetensorsInvalid {
                    reason: format!("tensor '{}' has a degenerate shape", tensor.name),
                });
            }
            if tensor.byte_length == 0 {
                return Err(ModelFormatRoadmapError::SafetensorsInvalid {
                    reason: format!("tensor '{}' has zero byte length", tensor.name),
                });
            }
        }
        Ok(())
    }

    /// Normalizes into [`ModelTensorMetadata`], the existing tensor
    /// inventory contract Model Loading already consumes.
    pub fn into_tensor_metadata(&self) -> Vec<ModelTensorMetadata> {
        self.tensors
            .iter()
            .map(|tensor| ModelTensorMetadata {
                name: tensor.name.clone(),
                shape: tensor.shape.clone(),
                storage_dtype: tensor.dtype,
                layout: None,
                shard: None,
                offset_bytes: Some(tensor.byte_offset),
                size_bytes: Some(tensor.byte_length),
                quantization: None,
                expected_compute_dtype: None,
                digest: None,
            })
            .collect()
    }
}

/// Whether memory mapping is permitted for a safetensors load, and whether
/// the mapped region is ever exposed. `exposes_raw_pointer` is asserted
/// `false` by [`MemoryMappingPolicy::validate`] -- no policy this roadmap
/// accepts may expose a raw pointer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryMappingPolicy {
    pub mapping_allowed: bool,
    pub streaming_read_allowed: bool,
    pub exposes_raw_pointer: bool,
}

impl MemoryMappingPolicy {
    pub fn validate(&self) -> Result<(), ModelFormatRoadmapError> {
        if self.exposes_raw_pointer {
            return Err(ModelFormatRoadmapError::ModelFormatUnsupported {
                reason: "memory mapping policy must not expose a raw pointer".into(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Phase 3: sharded weights
// ---------------------------------------------------------------------

/// Sharded weight index: the shard list plus the tensor-to-shard mapping,
/// reusing the existing [`ModelShard`] / [`ModelShardId`] contract instead
/// of a parallel sharding type.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ShardIndex {
    pub shards: Vec<ModelShard>,
    pub tensor_shard_map: BTreeMap<String, ModelShardId>,
}

impl ShardIndex {
    pub fn total_size_bytes(&self) -> u64 {
        self.shards.iter().map(|shard| shard.size_bytes).sum()
    }

    fn shard_ids(&self) -> BTreeSet<ModelShardId> {
        self.shards.iter().map(|shard| shard.id.clone()).collect()
    }
}

/// "Detect missing shards": every shard a tensor references must be present
/// in the index.
pub fn detect_missing_shards(index: &ShardIndex) -> Result<(), ModelFormatRoadmapError> {
    let known = index.shard_ids();
    for (tensor, shard) in &index.tensor_shard_map {
        if !known.contains(shard) {
            return Err(ModelFormatRoadmapError::ShardMissing {
                shard: format!("{} (referenced by tensor '{tensor}')", shard.as_str()),
            });
        }
    }
    Ok(())
}

/// "Detect duplicate tensors": a tensor name SHALL not appear in more than
/// one shard mapping.
pub fn detect_duplicate_tensor_names(
    tensors: &[ModelTensorMetadata],
) -> Result<(), ModelFormatRoadmapError> {
    let mut seen = BTreeSet::new();
    for tensor in tensors {
        if !seen.insert(tensor.name.clone()) {
            return Err(ModelFormatRoadmapError::ShardIndexInvalid {
                reason: format!("duplicate tensor '{}' across shards", tensor.name),
            });
        }
    }
    Ok(())
}

/// "Validate tensor shape consistency": the same tensor name SHALL declare
/// the same shape everywhere it appears (for example, once in a manifest
/// preview and once in its owning shard).
pub fn validate_shard_tensor_shape_consistency(
    tensors: &[ModelTensorMetadata],
) -> Result<(), ModelFormatRoadmapError> {
    let mut known: BTreeMap<&str, &[u64]> = BTreeMap::new();
    for tensor in tensors {
        match known.get(tensor.name.as_str()) {
            Some(shape) if *shape != tensor.shape.as_slice() => {
                return Err(ModelFormatRoadmapError::ShardIndexInvalid {
                    reason: format!("tensor '{}' has inconsistent shapes", tensor.name),
                });
            }
            _ => {
                known.insert(&tensor.name, &tensor.shape);
            }
        }
    }
    Ok(())
}

/// Ordered shard loading policy: shards SHALL load in ascending
/// [`ModelShard::order`], never an unspecified order.
pub fn validate_shard_loading_order(shards: &[ModelShard]) -> Result<(), ModelFormatRoadmapError> {
    let mut last: Option<u32> = None;
    for shard in shards {
        if let Some(previous) = last
            && shard.order <= previous
        {
            return Err(ModelFormatRoadmapError::ShardIndexInvalid {
                reason: format!(
                    "shard '{}' order {} does not strictly increase",
                    shard.id.as_str(),
                    shard.order
                ),
            });
        }
        last = Some(shard.order);
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Phase 4: Hugging Face-style config
// ---------------------------------------------------------------------

/// RoPE metadata as commonly found in Hugging Face-style `config.json`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HfRopeMetadata {
    pub theta: Option<u64>,
    pub scaling_type: Option<String>,
    pub scaling_factor: Option<u32>,
}

/// Normalized Hugging Face-style config fields from the proposal's "Phase 4"
/// list. `torch_dtype` and any field not otherwise modeled are preserved in
/// `annotations` -- never promoted to authoritative Runtime policy.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HfConfigMetadata {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub hidden_size: Option<u64>,
    pub num_hidden_layers: Option<u32>,
    pub num_attention_heads: Option<u32>,
    pub num_key_value_heads: Option<u32>,
    pub head_dim: Option<u32>,
    pub intermediate_size: Option<u64>,
    pub vocab_size: Option<u64>,
    pub max_position_embeddings: Option<u64>,
    pub hidden_act: Option<String>,
    pub rope: Option<HfRopeMetadata>,
    pub tie_word_embeddings: Option<bool>,
    /// Source metadata only -- see [`torch_dtype_does_not_force_compute_dtype`].
    pub torch_dtype: Option<String>,
    pub annotations: BTreeMap<String, String>,
}

impl HfConfigMetadata {
    /// Normalizes into [`ModelArchitecture`], the existing architecture
    /// contract Model Component validates against.
    pub fn normalize_architecture(&self, family: impl Into<String>) -> ModelArchitecture {
        ModelArchitecture::new(family, self.model_type.clone())
    }
}

/// "Source Metadata Is Not Automatically Runtime Policy": `torch_dtype` is
/// accepted but structurally ignored -- the return value is always
/// `requested_compute_dtype`, proving Runtime compute dtype policy alone
/// decides the outcome.
pub fn torch_dtype_does_not_force_compute_dtype(
    torch_dtype: Option<&str>,
    requested_compute_dtype: ModelDType,
) -> ModelDType {
    let _ = torch_dtype;
    requested_compute_dtype
}

// ---------------------------------------------------------------------
// Phase 5: tokenizer.json
// ---------------------------------------------------------------------

/// Parsed `tokenizer.json` metadata from the proposal's "Phase 5" list.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TokenizerJsonMetadata {
    pub vocabulary_size: u32,
    pub added_tokens: Vec<SpecialToken>,
    pub special_tokens: Vec<SpecialToken>,
    pub normalizer: Option<String>,
    pub pre_tokenizer: Option<String>,
    pub decoder: Option<String>,
    pub supports_offsets: bool,
}

/// Normalizes `tokenizer.json` metadata into [`TokenizerMetadata`], the
/// existing Tokenizer Artifact contract, implementing "tokenizer.json
/// Support".
#[allow(clippy::too_many_arguments)]
pub fn normalize_tokenizer_json(
    id: TokenizerId,
    artifact: TokenizerArtifactId,
    digest: ModelDigest,
    family: TokenizerFamily,
    revision: TokenizerRevision,
    parsed: &TokenizerJsonMetadata,
) -> Result<TokenizerMetadata, ModelFormatRoadmapError> {
    if parsed.vocabulary_size == 0 {
        return Err(ModelFormatRoadmapError::TokenizerJsonInvalid {
            reason: "vocabulary must not be empty".into(),
        });
    }
    let max_id = parsed
        .special_tokens
        .iter()
        .chain(parsed.added_tokens.iter())
        .map(|token| token.id)
        .max()
        .unwrap_or(0)
        .max(parsed.vocabulary_size.saturating_sub(1));
    Ok(TokenizerMetadata {
        id,
        artifact,
        digest,
        family,
        revision,
        vocabulary_size: parsed.vocabulary_size,
        added_token_count: parsed.added_tokens.len() as u32,
        token_id_range: TokenIdRange::new(0, max_id),
        model_max_length: None,
        special_tokens: parsed.special_tokens.clone(),
        additional_special_tokens: parsed.added_tokens.clone(),
        byte_fallback: false,
        normalization: parsed.normalizer.clone(),
        pre_tokenizer: parsed.pre_tokenizer.clone(),
        supports_offsets: parsed.supports_offsets,
        supports_token_type_ids: false,
        supports_browser: false,
    })
}

// ---------------------------------------------------------------------
// Phase 6: tokenizer_config
// ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PaddingSide {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TruncationSide {
    Left,
    Right,
}

/// Parsed `tokenizer_config.json` metadata from the proposal's "Phase 6"
/// list.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TokenizerConfigMetadata {
    pub tokenizer_class: Option<String>,
    pub model_max_length: Option<u32>,
    pub padding_side: Option<PaddingSide>,
    pub truncation_side: Option<TruncationSide>,
    pub chat_template_reference: Option<String>,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub pad_token: Option<String>,
    pub added_special_tokens: Vec<String>,
    pub clean_up_tokenization_spaces: Option<bool>,
}

/// "tokenizer_config SHALL not override Runtime policy silently": a parsed
/// `tokenizer_config` value only becomes effective policy when
/// `runtime_policy_validated` attests Runtime explicitly validated it;
/// otherwise it is source annotation only.
pub fn reject_silent_tokenizer_config_override(
    runtime_policy_validated: bool,
) -> Result<(), ModelFormatRoadmapError> {
    if runtime_policy_validated {
        Ok(())
    } else {
        Err(ModelFormatRoadmapError::TokenizerConfigInvalid {
            reason: "tokenizer_config value requires explicit Runtime policy validation \
                     before it can override Runtime behavior"
                .into(),
        })
    }
}

// ---------------------------------------------------------------------
// Phase 7: generation_config
// ---------------------------------------------------------------------

/// Parsed `generation_config.json` metadata from the proposal's "Phase 7"
/// list.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GenerationConfigMetadata {
    pub max_length: Option<u32>,
    pub max_new_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub eos_token_id: Option<u32>,
    pub bos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub do_sample: Option<bool>,
    pub stop_strings: Vec<String>,
}

impl GenerationConfigMetadata {
    /// Normalizes into [`ModelGenerationDefaults`], the existing generation
    /// defaults contract. "Generation config values SHALL be defaults, not
    /// mandatory Runtime policy" -- callers apply
    /// [`apply_generation_override`] before use.
    pub fn as_defaults(&self) -> ModelGenerationDefaults {
        ModelGenerationDefaults {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            max_tokens: self.max_new_tokens.or(self.max_length),
            stop_tokens: self.stop_strings.clone(),
            repetition_penalty: self.repetition_penalty,
        }
    }
}

/// "Runtime Generation API may override [generation_config defaults]
/// according to policy": an explicit `requested` value always wins over the
/// parsed `default`.
pub fn apply_generation_override<T>(default: Option<T>, requested: Option<T>) -> Option<T> {
    requested.or(default)
}

// ---------------------------------------------------------------------
// Phase 8: chat templates
// ---------------------------------------------------------------------

/// Chat template source kinds this roadmap accepts. Deliberately has no
/// `Http` / `RemoteUrl` / `ArbitraryFilesystem` variant -- "Templates SHALL
/// not be fetched from arbitrary filesystem or network during inference"
/// holds because no such source can be constructed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatTemplateSourceKind {
    EmbeddedInManifest,
    AuthorizedLocalArtifact,
    ClientProvidedInline,
}

/// Chat template metadata from the proposal's "Phase 8" list.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatTemplateMetadata {
    pub identity: String,
    pub source: ChatTemplateSourceKind,
    pub tokenizer_compatible: bool,
    pub model_family_compatible: bool,
    pub required_variables: BTreeSet<String>,
    pub special_token_interaction: BTreeSet<String>,
}

/// Validates a chat template against the variables a render call actually
/// supplies, implementing "Chat Template Support": tokenizer/model-family
/// compatibility must hold and every required variable must be present.
pub fn validate_chat_template(
    metadata: &ChatTemplateMetadata,
    provided_variables: &BTreeSet<String>,
) -> Result<(), ModelFormatRoadmapError> {
    if !metadata.tokenizer_compatible || !metadata.model_family_compatible {
        return Err(ModelFormatRoadmapError::ChatTemplateInvalid {
            reason: format!(
                "chat template '{}' is not compatible with the active tokenizer or model family",
                metadata.identity
            ),
        });
    }
    if let Some(missing) = metadata
        .required_variables
        .iter()
        .find(|variable| !provided_variables.contains(*variable))
    {
        return Err(ModelFormatRoadmapError::ChatTemplateInvalid {
            reason: format!("chat template is missing required variable '{missing}'"),
        });
    }
    Ok(())
}

/// "Enforce raw prompt redaction": redacts a chat-template rendering
/// diagnostic before it reaches observability, reusing
/// `redact_backend_diagnostic` rather than a parallel redaction path.
pub fn redact_chat_template_diagnostic(message: &str) -> String {
    redact_backend_diagnostic(message)
}

// ---------------------------------------------------------------------
// Phase 9: SentencePiece
// ---------------------------------------------------------------------

/// Parsed SentencePiece metadata from the proposal's "Phase 9" list.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SentencePieceMetadata {
    pub model_identity: String,
    pub vocabulary_size: u32,
    pub special_tokens: Vec<SpecialToken>,
    pub normalization: Option<String>,
    pub browser_supported: bool,
    pub license: Option<ModelLicenseMetadata>,
    pub supported_features: BTreeSet<String>,
}

/// "Unsupported SentencePiece features SHALL fail explicitly."
pub fn reject_unsupported_sentencepiece_feature(
    metadata: &SentencePieceMetadata,
    requested_feature: &str,
) -> Result<(), ModelFormatRoadmapError> {
    if metadata.supported_features.contains(requested_feature) {
        Ok(())
    } else {
        Err(ModelFormatRoadmapError::SentencePieceUnsupported {
            feature: requested_feature.into(),
        })
    }
}

// ---------------------------------------------------------------------
// Phase 10: GGUF
// ---------------------------------------------------------------------

/// A single GGUF tensor entry, mirroring [`SafetensorsTensorEntry`] but
/// additionally carrying quantization metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GgufTensorEntry {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: ModelDType,
    pub quantization: Option<ModelQuantization>,
}

/// Parsed GGUF metadata from the proposal's "Phase 10" list. "GGUF support
/// SHALL not create `GGUFProvider`" is enforced by
/// [`reject_model_format_provider_name`], not by anything in this type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GgufMetadata {
    pub architecture: String,
    pub alignment: u32,
    pub tensors: Vec<GgufTensorEntry>,
    pub tokenizer_embedded: Option<TokenizerJsonMetadata>,
    pub key_values: BTreeMap<String, String>,
}

impl GgufMetadata {
    /// Normalizes GGUF tensors into [`ModelTensorMetadata`], the same
    /// contract safetensors normalizes into -- "GGUF quantized tensors
    /// SHALL use Tensor Layout and Quantization metadata", not a
    /// GGUF-specific tensor type.
    pub fn into_tensor_metadata(&self) -> Vec<ModelTensorMetadata> {
        self.tensors
            .iter()
            .map(|tensor| ModelTensorMetadata {
                name: tensor.name.clone(),
                shape: tensor.shape.clone(),
                storage_dtype: tensor.dtype,
                layout: tensor
                    .quantization
                    .is_some()
                    .then(|| "quantized-packed".to_string()),
                shard: None,
                offset_bytes: None,
                size_bytes: None,
                quantization: tensor.quantization.clone(),
                expected_compute_dtype: None,
                digest: None,
            })
            .collect()
    }

    pub fn validate(&self) -> Result<(), ModelFormatRoadmapError> {
        if self.architecture.trim().is_empty() {
            return Err(ModelFormatRoadmapError::GgufInvalid {
                reason: "GGUF metadata must declare an architecture".into(),
            });
        }
        if self.tensors.is_empty() {
            return Err(ModelFormatRoadmapError::GgufInvalid {
                reason: "GGUF metadata must declare at least one tensor".into(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Phase 11: adapter formats
// ---------------------------------------------------------------------

/// Parsed LoRA adapter format metadata (`adapter_config.json` plus LoRA
/// safetensors tensors) from the proposal's "Phase 11" list.
#[derive(Clone, Debug, PartialEq)]
pub struct LoraAdapterFormatMetadata {
    pub target_modules: Vec<String>,
    pub rank: u32,
    pub alpha: u32,
    pub scaling: Option<f32>,
    /// Preserved as training/source metadata only; never Runtime policy.
    pub dropout: Option<f32>,
    pub base_model: AdapterBaseModelCompatibility,
    pub tensors: Vec<ModelTensorMetadata>,
    pub storage_dtype: ModelDType,
    pub compute_dtype: Option<ComputeDType>,
    pub quantization: Option<ModelQuantization>,
    pub required_capabilities: Vec<CapabilityBinding>,
    pub license: Option<ModelLicenseMetadata>,
    pub provenance: Option<ModelProvenance>,
}

/// Normalizes LoRA adapter format metadata into [`AdapterArtifact`], the
/// existing Adapter Artifact contract, implementing "Adapter Format
/// Support". Trust is always [`AdapterTrustStatus::Unknown`] -- "Parsing an
/// adapter format SHALL not activate the adapter" and, per
/// [`model_format_grants_no_trust`], parsing SHALL not grant trust either;
/// both require a separate, explicit policy decision.
pub fn normalize_lora_adapter(
    id: AdapterArtifactId,
    metadata: &LoraAdapterFormatMetadata,
) -> AdapterArtifact {
    AdapterArtifact {
        id,
        method: AdapterMethod::Lora,
        base_model: metadata.base_model.clone(),
        targets: metadata
            .target_modules
            .iter()
            .map(|name| AdapterTargetModule {
                name: name.clone(),
                role: AdapterTargetModuleRole::Other,
                layer_selector: None,
                expected_shape: Vec::new(),
            })
            .collect(),
        storage_dtype: metadata.storage_dtype,
        compute_dtype: metadata.compute_dtype,
        rank: Some(metadata.rank),
        alpha: Some(metadata.alpha),
        tensors: metadata.tensors.clone(),
        quantization: metadata.quantization.clone(),
        required_capabilities: metadata.required_capabilities.clone(),
        license: metadata.license.clone(),
        provenance: metadata.provenance.clone(),
        trust: AdapterTrustStatus::Unknown,
    }
}

// ---------------------------------------------------------------------
// Phase 12: quantized artifact metadata
// ---------------------------------------------------------------------

/// A quantized artifact's declared metadata: the existing
/// [`ModelQuantization`] weight-storage contract, plus optional
/// Provider/Kernel compatibility metadata reusing the existing
/// [`KernelQuantizationMetadata`] execution-side contract rather than a
/// parallel type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelFormatQuantizationDeclaration {
    pub model_quantization: ModelQuantization,
    pub kernel_compatibility: Option<KernelQuantizationMetadata>,
}

/// "No hidden quantization or dequantization SHALL occur": requires
/// [`ModelQuantization`] to be present with a declared scale dtype (storage
/// dtype and group/block sizing are already mandatory-shaped by the
/// existing type's constructor sites), and, when Provider/Kernel
/// compatibility is declared, delegates to the existing
/// [`validate_quantization_declaration`] /
/// [`reject_hidden_dequantization`] execution-side checks so quantization
/// metadata and dequantization behavior are validated by one rule set.
pub fn validate_model_format_quantization(
    declaration: &ModelFormatQuantizationDeclaration,
    dequantization_declared: bool,
) -> Result<(), ModelFormatRoadmapError> {
    if declaration.model_quantization.scale_dtype.is_none() {
        return Err(ModelFormatRoadmapError::QuantizationMetadataInvalid {
            reason: "quantized artifact metadata must declare a scale dtype".into(),
        });
    }
    if let Some(kernel_metadata) = &declaration.kernel_compatibility {
        validate_quantization_declaration(kernel_metadata).map_err(|error| {
            ModelFormatRoadmapError::QuantizationMetadataInvalid {
                reason: error.to_string(),
            }
        })?;
        reject_hidden_dequantization(dequantization_declared).map_err(|error| {
            ModelFormatRoadmapError::QuantizationMetadataInvalid {
                reason: error.to_string(),
            }
        })?;
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Source, local file, trust, and integrity boundaries
// ---------------------------------------------------------------------

/// "Model format support SHALL not imply arbitrary download behavior":
/// [`ModelArtifactSource`] is a closed enum of authorized source kinds
/// (local path, local cache, client-provided, registry, Hugging Face, OCI,
/// Tachyon); this function exists so callers have one explicit place that
/// states the source was checked, rather than assuming any string is a
/// valid model reference.
pub fn reject_arbitrary_model_download(
    source: &ModelArtifactSource,
) -> Result<(), ModelFormatRoadmapError> {
    match source {
        ModelArtifactSource::LocalPath(_)
        | ModelArtifactSource::LocalCache(_)
        | ModelArtifactSource::ClientProvided(_)
        | ModelArtifactSource::Registry(_)
        | ModelArtifactSource::HuggingFace(_)
        | ModelArtifactSource::Oci(_)
        | ModelArtifactSource::Tachyon(_) => Ok(()),
    }
}

/// "Runtime SHALL not scan arbitrary local directories during inference": a
/// [`ModelArtifactSource::LocalPath`] is only accepted when `authorized`
/// attests it was supplied through an explicit client-provided artifact
/// source, never discovered by directory scanning.
pub fn validate_local_file_boundary(
    source: &ModelArtifactSource,
    authorized: bool,
) -> Result<(), ModelFormatRoadmapError> {
    if let ModelArtifactSource::LocalPath(path) = source
        && !authorized
    {
        return Err(ModelFormatRoadmapError::ModelFormatLocalFileDenied {
            path: path.display().to_string(),
        });
    }
    Ok(())
}

/// A raw model reference string naming a network scheme is denied outright:
/// "Runtime SHALL not perform arbitrary network downloads during
/// inference". A caller that has a real remote source uses
/// [`ModelArtifactSource::HuggingFace`] / `Oci` / `Registry` / `Tachyon`
/// instead of a raw URL.
pub fn reject_raw_network_model_reference(reference: &str) -> Result<(), ModelFormatRoadmapError> {
    let lower = reference.trim().to_ascii_lowercase();
    if lower.starts_with("http://") || lower.starts_with("https://") || lower.starts_with("ftp://")
    {
        return Err(ModelFormatRoadmapError::ModelFormatNetworkDenied {
            reference: reference.into(),
        });
    }
    Ok(())
}

/// "A model format SHALL not be trusted merely because it is recognized":
/// trust always comes from evaluating [`ModelTrustStore`] policy against the
/// manifest, never from which parser produced the manifest.
pub fn model_format_grants_no_trust(
    store: &ModelTrustStore,
    manifest: &ModelManifest,
) -> ModelTrustDecision {
    store.evaluate(manifest)
}

// ---------------------------------------------------------------------
// Format conformance
// ---------------------------------------------------------------------

/// Conformance fixture categories from the proposal's "Format Conformance"
/// section.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ModelFormatConformanceFixtureKind {
    ValidMinimalArtifact,
    MissingRequiredMetadata,
    InvalidTensorShape,
    InvalidDType,
    InvalidShardIndex,
    MissingShard,
    DuplicateTensor,
    TokenizerMismatch,
    UnsupportedQuantization,
    MalformedFileMetadata,
    UntrustedArtifact,
    RedactionCheck,
}

/// The twelve fixture categories, in the proposal's order.
pub const MODEL_FORMAT_CONFORMANCE_FIXTURES: &[ModelFormatConformanceFixtureKind] = &[
    ModelFormatConformanceFixtureKind::ValidMinimalArtifact,
    ModelFormatConformanceFixtureKind::MissingRequiredMetadata,
    ModelFormatConformanceFixtureKind::InvalidTensorShape,
    ModelFormatConformanceFixtureKind::InvalidDType,
    ModelFormatConformanceFixtureKind::InvalidShardIndex,
    ModelFormatConformanceFixtureKind::MissingShard,
    ModelFormatConformanceFixtureKind::DuplicateTensor,
    ModelFormatConformanceFixtureKind::TokenizerMismatch,
    ModelFormatConformanceFixtureKind::UnsupportedQuantization,
    ModelFormatConformanceFixtureKind::MalformedFileMetadata,
    ModelFormatConformanceFixtureKind::UntrustedArtifact,
    ModelFormatConformanceFixtureKind::RedactionCheck,
];

impl ModelFormatConformanceFixtureKind {
    pub const fn id(self) -> &'static str {
        match self {
            Self::ValidMinimalArtifact => "valid-minimal-artifact",
            Self::MissingRequiredMetadata => "missing-required-metadata",
            Self::InvalidTensorShape => "invalid-tensor-shape",
            Self::InvalidDType => "invalid-dtype",
            Self::InvalidShardIndex => "invalid-shard-index",
            Self::MissingShard => "missing-shard",
            Self::DuplicateTensor => "duplicate-tensor",
            Self::TokenizerMismatch => "tokenizer-mismatch",
            Self::UnsupportedQuantization => "unsupported-quantization",
            Self::MalformedFileMetadata => "malformed-file-metadata",
            Self::UntrustedArtifact => "untrusted-artifact",
            Self::RedactionCheck => "redaction-check",
        }
    }
}

// ---------------------------------------------------------------------
// Error Model
// ---------------------------------------------------------------------

/// Structured model format roadmap error, covering every error category
/// from the proposal's "Error Model" section.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelFormatRoadmapError {
    ModelFormatUnsupported { reason: String },
    ModelFormatInvalid { reason: String },
    ModelFormatParserFailed { reason: String },
    ModelManifestInvalid { reason: String },
    ModelManifestMissing { reason: String },
    ModelConfigInvalid { reason: String },
    SafetensorsInvalid { reason: String },
    SafetensorsTensorMissing { tensor: String },
    SafetensorsDTypeUnsupported { dtype: String },
    ShardIndexInvalid { reason: String },
    ShardMissing { shard: String },
    ShardDigestMismatch { shard: String },
    TokenizerJsonInvalid { reason: String },
    TokenizerConfigInvalid { reason: String },
    GenerationConfigInvalid { reason: String },
    ChatTemplateInvalid { reason: String },
    SentencePieceUnsupported { feature: String },
    GgufInvalid { reason: String },
    GgufQuantizationUnsupported { method: String },
    AdapterFormatInvalid { reason: String },
    QuantizationMetadataInvalid { reason: String },
    ModelFormatTrustDenied { reason: String },
    ModelFormatIntegrityFailed { reason: String },
    ModelFormatLocalFileDenied { path: String },
    ModelFormatNetworkDenied { reference: String },
    InternalModelFormatError { reason: String },
}

impl ModelFormatRoadmapError {
    pub const fn id(&self) -> &'static str {
        match self {
            Self::ModelFormatUnsupported { .. } => "model-format-unsupported",
            Self::ModelFormatInvalid { .. } => "model-format-invalid",
            Self::ModelFormatParserFailed { .. } => "model-format-parser-failed",
            Self::ModelManifestInvalid { .. } => "model-manifest-invalid",
            Self::ModelManifestMissing { .. } => "model-manifest-missing",
            Self::ModelConfigInvalid { .. } => "model-config-invalid",
            Self::SafetensorsInvalid { .. } => "safetensors-invalid",
            Self::SafetensorsTensorMissing { .. } => "safetensors-tensor-missing",
            Self::SafetensorsDTypeUnsupported { .. } => "safetensors-dtype-unsupported",
            Self::ShardIndexInvalid { .. } => "shard-index-invalid",
            Self::ShardMissing { .. } => "shard-missing",
            Self::ShardDigestMismatch { .. } => "shard-digest-mismatch",
            Self::TokenizerJsonInvalid { .. } => "tokenizer-json-invalid",
            Self::TokenizerConfigInvalid { .. } => "tokenizer-config-invalid",
            Self::GenerationConfigInvalid { .. } => "generation-config-invalid",
            Self::ChatTemplateInvalid { .. } => "chat-template-invalid",
            Self::SentencePieceUnsupported { .. } => "sentencepiece-unsupported",
            Self::GgufInvalid { .. } => "gguf-invalid",
            Self::GgufQuantizationUnsupported { .. } => "gguf-quantization-unsupported",
            Self::AdapterFormatInvalid { .. } => "adapter-format-invalid",
            Self::QuantizationMetadataInvalid { .. } => "quantization-metadata-invalid",
            Self::ModelFormatTrustDenied { .. } => "model-format-trust-denied",
            Self::ModelFormatIntegrityFailed { .. } => "model-format-integrity-failed",
            Self::ModelFormatLocalFileDenied { .. } => "model-format-local-file-denied",
            Self::ModelFormatNetworkDenied { .. } => "model-format-network-denied",
            Self::InternalModelFormatError { .. } => "internal-model-format-error",
        }
    }
}

impl fmt::Display for ModelFormatRoadmapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelFormatUnsupported { reason }
            | Self::ModelFormatInvalid { reason }
            | Self::ModelFormatParserFailed { reason }
            | Self::ModelManifestInvalid { reason }
            | Self::ModelManifestMissing { reason }
            | Self::ModelConfigInvalid { reason }
            | Self::SafetensorsInvalid { reason }
            | Self::ShardIndexInvalid { reason }
            | Self::TokenizerJsonInvalid { reason }
            | Self::TokenizerConfigInvalid { reason }
            | Self::GenerationConfigInvalid { reason }
            | Self::ChatTemplateInvalid { reason }
            | Self::GgufInvalid { reason }
            | Self::AdapterFormatInvalid { reason }
            | Self::QuantizationMetadataInvalid { reason }
            | Self::ModelFormatTrustDenied { reason }
            | Self::ModelFormatIntegrityFailed { reason }
            | Self::InternalModelFormatError { reason } => {
                write!(f, "{}: {reason}", self.id())
            }
            Self::SafetensorsTensorMissing { tensor } => write!(f, "{}: {tensor}", self.id()),
            Self::SafetensorsDTypeUnsupported { dtype } => write!(f, "{}: {dtype}", self.id()),
            Self::ShardMissing { shard } | Self::ShardDigestMismatch { shard } => {
                write!(f, "{}: {shard}", self.id())
            }
            Self::SentencePieceUnsupported { feature } => write!(f, "{}: {feature}", self.id()),
            Self::GgufQuantizationUnsupported { method } => write!(f, "{}: {method}", self.id()),
            Self::ModelFormatLocalFileDenied { path } => write!(f, "{}: {path}", self.id()),
            Self::ModelFormatNetworkDenied { reference } => {
                write!(f, "{}: {reference}", self.id())
            }
        }
    }
}

impl Error for ModelFormatRoadmapError {}

// ---------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------

/// Observation categories from the proposal's "Observability" section.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ModelFormatRoadmapObservationKind {
    FormatDetected,
    ManifestNormalized,
    ManifestValidationFailed,
    ConfigParsed,
    ConfigValidationFailed,
    TensorInventoryParsed,
    TensorInventoryMismatch,
    TokenizerMetadataParsed,
    TokenizerCompatibilityFailed,
    GenerationConfigParsed,
    ChatTemplateParsed,
    SafetensorsParsed,
    ShardIndexParsed,
    ShardMissing,
    GgufMetadataParsed,
    QuantizationMetadataParsed,
    AdapterMetadataParsed,
    IntegrityValidationFailed,
    TrustValidationFailed,
}

/// A single model format roadmap observation. Structurally guaranteed to
/// never carry raw model weights, raw tokenizer data, raw prompts, raw file
/// contents, secrets, filesystem authority, raw tensor values, memory
/// pointers, or Provider/Device/Kernel handles by default: the only fields
/// are an enum `kind`, an optional artifact identity, and a
/// `redacted_metadata` string map whose values are always passed through
/// `redact_backend_diagnostic`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelFormatRoadmapObservation {
    pub kind: ModelFormatRoadmapObservationKind,
    pub artifact: Option<String>,
    pub redacted_metadata: BTreeMap<String, String>,
}

impl ModelFormatRoadmapObservation {
    pub fn new(kind: ModelFormatRoadmapObservationKind) -> Self {
        Self {
            kind,
            artifact: None,
            redacted_metadata: BTreeMap::new(),
        }
    }

    pub fn with_artifact(mut self, artifact: impl Into<String>) -> Self {
        self.artifact = Some(artifact.into());
        self
    }

    pub fn with_redacted_metadata(
        mut self,
        key: impl Into<String>,
        value: impl AsRef<str>,
    ) -> Self {
        self.redacted_metadata
            .insert(key.into(), redact_backend_diagnostic(value.as_ref()));
        self
    }
}

// ---------------------------------------------------------------------
// Conformance report
// ---------------------------------------------------------------------

/// A single model format roadmap conformance check result, mirroring
/// [`crate::ProviderRoadmapConformanceResult`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelFormatRoadmapConformanceResult {
    pub requirement: String,
    pub passed: bool,
    pub diagnostic: Option<String>,
}

/// A collected set of [`ModelFormatRoadmapConformanceResult`]s.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelFormatRoadmapConformanceReport {
    pub results: Vec<ModelFormatRoadmapConformanceResult>,
}

impl ModelFormatRoadmapConformanceReport {
    pub fn is_conformant(&self) -> bool {
        self.results.iter().all(|result| result.passed)
    }
}

fn record(
    results: &mut Vec<ModelFormatRoadmapConformanceResult>,
    requirement: impl Into<String>,
    passed: bool,
    diagnostic: impl Into<String>,
) {
    let diagnostic = diagnostic.into();
    results.push(ModelFormatRoadmapConformanceResult {
        requirement: requirement.into(),
        passed,
        diagnostic: (!passed).then_some(diagnostic),
    });
}

/// Runs the model format roadmap conformance checks described in this
/// module's doc comment: format-shaped Provider names rejected; format
/// parsers cannot supply execution graphs; missing/duplicate/inconsistent
/// shard tensors are detected; `torch_dtype` never forces compute dtype;
/// generation_config defaults are always overridable; chat templates cannot
/// declare a network/filesystem source; unsupported SentencePiece features
/// fail explicitly; hidden dequantization is rejected; arbitrary local file
/// and network access are denied; and format alone never grants trust.
pub fn run_model_format_roadmap_conformance() -> ModelFormatRoadmapConformanceReport {
    let mut results = Vec::new();

    for name in [
        "GGUFProvider",
        "SafetensorsProvider",
        "QwenSafetensorsProvider",
    ] {
        let outcome = reject_model_format_provider_name(name);
        record(
            &mut results,
            format!("model-format Provider name '{name}' is rejected"),
            outcome.is_err(),
            format!("unexpected outcome: {outcome:?}"),
        );
    }
    for name in ["ReferenceCpuProvider", "CudaProvider"] {
        let outcome = reject_model_format_provider_name(name);
        record(
            &mut results,
            format!("hardware/optimized Provider name '{name}' is allowed"),
            outcome.is_ok(),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        let outcome = reject_format_execution_graph(true);
        record(
            &mut results,
            "format parser supplying an execution graph is rejected",
            outcome.is_err(),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        let mut index = ShardIndex::default();
        index.tensor_shard_map.insert(
            "layer.0.weight".into(),
            ModelShardId::new("shard-missing").unwrap(),
        );
        let outcome = detect_missing_shards(&index);
        record(
            &mut results,
            "a tensor referencing an absent shard is detected",
            matches!(outcome, Err(ModelFormatRoadmapError::ShardMissing { .. })),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        let duplicate = vec![
            ModelTensorMetadata {
                name: "layer.0.weight".into(),
                shape: vec![4, 4],
                storage_dtype: ModelDType::F32,
                layout: None,
                shard: None,
                offset_bytes: None,
                size_bytes: None,
                quantization: None,
                expected_compute_dtype: None,
                digest: None,
            },
            ModelTensorMetadata {
                name: "layer.0.weight".into(),
                shape: vec![4, 4],
                storage_dtype: ModelDType::F32,
                layout: None,
                shard: None,
                offset_bytes: None,
                size_bytes: None,
                quantization: None,
                expected_compute_dtype: None,
                digest: None,
            },
        ];
        let outcome = detect_duplicate_tensor_names(&duplicate);
        record(
            &mut results,
            "a duplicate tensor name across shards is detected",
            outcome.is_err(),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        let dtype = torch_dtype_does_not_force_compute_dtype(Some("bfloat16"), ModelDType::F32);
        record(
            &mut results,
            "torch_dtype does not force Runtime compute dtype",
            dtype == ModelDType::F32,
            format!("unexpected compute dtype: {dtype:?}"),
        );
    }

    {
        let overridden = apply_generation_override(Some(0.7_f32), Some(1.2_f32));
        let default_only = apply_generation_override(Some(0.7_f32), None);
        record(
            &mut results,
            "generation_config defaults are overridable by an explicit request",
            overridden == Some(1.2) && default_only == Some(0.7),
            format!("unexpected outcome: overridden={overridden:?} default_only={default_only:?}"),
        );
    }

    {
        let metadata = ChatTemplateMetadata {
            identity: "qwen-chat".into(),
            source: ChatTemplateSourceKind::EmbeddedInManifest,
            tokenizer_compatible: true,
            model_family_compatible: true,
            required_variables: BTreeSet::from(["messages".to_string()]),
            special_token_interaction: BTreeSet::new(),
        };
        let missing = validate_chat_template(&metadata, &BTreeSet::new());
        record(
            &mut results,
            "chat template rendering without a required variable fails",
            missing.is_err(),
            format!("unexpected outcome: {missing:?}"),
        );
        let present = validate_chat_template(&metadata, &BTreeSet::from(["messages".to_string()]));
        record(
            &mut results,
            "chat template rendering with every required variable succeeds",
            present.is_ok(),
            format!("unexpected outcome: {present:?}"),
        );
    }

    {
        let metadata = SentencePieceMetadata {
            model_identity: "spm-1".into(),
            vocabulary_size: 32000,
            special_tokens: Vec::new(),
            normalization: None,
            browser_supported: false,
            license: None,
            supported_features: BTreeSet::from(["bpe".to_string()]),
        };
        let outcome = reject_unsupported_sentencepiece_feature(&metadata, "byte-fallback");
        record(
            &mut results,
            "unsupported SentencePiece feature fails explicitly",
            matches!(
                outcome,
                Err(ModelFormatRoadmapError::SentencePieceUnsupported { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        let hidden = reject_hidden_dequantization(false);
        record(
            &mut results,
            "hidden dequantization is rejected",
            hidden.is_err(),
            format!("unexpected outcome: {hidden:?}"),
        );
    }

    {
        let outcome = validate_local_file_boundary(
            &ModelArtifactSource::LocalPath(std::path::PathBuf::from("/models/qwen")),
            false,
        );
        record(
            &mut results,
            "unauthorized local file access is denied",
            matches!(
                outcome,
                Err(ModelFormatRoadmapError::ModelFormatLocalFileDenied { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        let outcome = reject_raw_network_model_reference("https://example.com/model.safetensors");
        record(
            &mut results,
            "a raw network URL model reference is denied",
            matches!(
                outcome,
                Err(ModelFormatRoadmapError::ModelFormatNetworkDenied { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        let store = ModelTrustStore::default();
        let manifest = minimal_trust_probe_manifest();
        let decision = model_format_grants_no_trust(&store, &manifest);
        record(
            &mut results,
            "an unrecognized digest is not trusted merely because the format parsed",
            decision.status == crate::ModelTrustStatus::Unknown,
            format!("unexpected trust decision: {decision:?}"),
        );
    }

    ModelFormatRoadmapConformanceReport { results }
}

/// A minimal, otherwise-unremarkable manifest used only to probe trust
/// evaluation in [`run_model_format_roadmap_conformance`] -- it carries no
/// digest the fixture `ModelTrustStore` has been told to trust, reject, or
/// revoke.
fn minimal_trust_probe_manifest() -> ModelManifest {
    let digest = ModelDigest::parse(format!("sha256:{}", "1".repeat(64))).unwrap();
    let id = ModelArtifactId::new(
        ModelArtifactKind::ModelBundle,
        ModelName::new("format-roadmap-probe").unwrap(),
        ModelRevision::new("v1").unwrap(),
        digest,
    );
    ModelManifest {
        schema_version: crate::MODEL_ARTIFACT_SCHEMA_VERSION,
        id,
        architecture: ModelArchitecture::new("probe", "probe"),
        parts: BTreeMap::new(),
        storage_dtype: None,
        compute_dtype: None,
        supported_compute_dtypes: BTreeSet::new(),
        tensors: Vec::new(),
        tokenizer: None,
        tokenizer_config: None,
        chat_template: None,
        prompt_template: None,
        generation: None,
        quantization: None,
        shards: Vec::new(),
        runtime_features: BTreeSet::new(),
        memory_features: BTreeSet::new(),
        provider_capabilities: Vec::new(),
        component: None,
        license: None,
        provenance: None,
        signatures: Vec::new(),
        source: None,
    }
}
