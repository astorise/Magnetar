//! Model Artifact contracts.
//!
//! A Model Artifact is inference data: weights, configuration, tokenizer data,
//! templates, quantization metadata, adapters, and related manifests. It is not
//! executable Component code, not a Provider, not Device metadata, and not a
//! loaded Model Instance.

use crate::{
    CapabilityBinding, CapabilityId, CapabilityVersion, ComponentArtifactReference, ComputeDType,
    DTypeDescriptor, MemoryAllocationClass, MemoryAllocationOwner, MemoryAllocationRequest,
    MemoryDTypeRelation, MemoryFeasibility, MemoryManager, MemoryPlacement,
};
use serde::Deserialize;
use sha2::{Digest as ShaDigest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt, fs,
    path::{Path, PathBuf},
};

pub const MODEL_ARTIFACT_SCHEMA: &str = "magnetar-model-artifact";
pub const MODEL_ARTIFACT_SCHEMA_VERSION: u64 = 1;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ModelArtifactKind {
    ModelBundle,
    ModelWeights,
    ModelConfig,
    Tokenizer,
    TokenizerConfig,
    ChatTemplate,
    PromptTemplate,
    GenerationConfig,
    QuantizationConfig,
    Adapter,
    Vocabulary,
    SpecialTokens,
}

impl ModelArtifactKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ModelBundle => "model-bundle",
            Self::ModelWeights => "model-weights",
            Self::ModelConfig => "model-config",
            Self::Tokenizer => "tokenizer",
            Self::TokenizerConfig => "tokenizer-config",
            Self::ChatTemplate => "chat-template",
            Self::PromptTemplate => "prompt-template",
            Self::GenerationConfig => "generation-config",
            Self::QuantizationConfig => "quantization-config",
            Self::Adapter => "adapter",
            Self::Vocabulary => "vocabulary",
            Self::SpecialTokens => "special-tokens",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "model-bundle" => Some(Self::ModelBundle),
            "model-weights" => Some(Self::ModelWeights),
            "model-config" => Some(Self::ModelConfig),
            "tokenizer" => Some(Self::Tokenizer),
            "tokenizer-config" => Some(Self::TokenizerConfig),
            "chat-template" => Some(Self::ChatTemplate),
            "prompt-template" => Some(Self::PromptTemplate),
            "generation-config" => Some(Self::GenerationConfig),
            "quantization-config" => Some(Self::QuantizationConfig),
            "adapter" => Some(Self::Adapter),
            "vocabulary" => Some(Self::Vocabulary),
            "special-tokens" => Some(Self::SpecialTokens),
            _ => None,
        }
    }
}

impl fmt::Display for ModelArtifactKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModelName(String);

impl ModelName {
    pub fn new(value: impl Into<String>) -> Result<Self, ModelArtifactError> {
        let value = value.into();
        validate_identity_segment(&value, "model name")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModelRevision(String);

impl ModelRevision {
    pub fn new(value: impl Into<String>) -> Result<Self, ModelArtifactError> {
        let value = value.into();
        validate_identity_segment(&value, "model revision")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModelVariant(String);

impl ModelVariant {
    pub fn new(value: impl Into<String>) -> Result<Self, ModelArtifactError> {
        let value = value.into();
        validate_identity_segment(&value, "model variant")?;
        Ok(Self(value))
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModelSourceIdentity(String);

impl ModelSourceIdentity {
    pub fn new(value: impl Into<String>) -> Result<Self, ModelArtifactError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(ModelArtifactError::InvalidManifest {
                message: "model source identity must not be empty".into(),
            });
        }
        Ok(Self(value))
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModelShardId(String);

impl ModelShardId {
    pub fn new(value: impl Into<String>) -> Result<Self, ModelArtifactError> {
        let value = value.into();
        validate_identity_segment(&value, "model shard id")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModelDigest {
    pub algorithm: String,
    pub value: String,
}

impl ModelDigest {
    pub fn sha256(bytes: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        let digest = hasher.finalize();
        Self {
            algorithm: "sha256".into(),
            value: format!("sha256:{}", lower_hex(&digest)),
        }
    }

    pub fn parse(value: impl Into<String>) -> Result<Self, ModelArtifactError> {
        let value = value.into().to_ascii_lowercase();
        let Some(hex) = value.strip_prefix("sha256:") else {
            return Err(ModelArtifactError::UnsupportedDigestAlgorithm {
                algorithm: value.split(':').next().unwrap_or_default().into(),
            });
        };
        if hex.len() != 64
            || !hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(ModelArtifactError::InvalidDigest { digest: value });
        }
        Ok(Self {
            algorithm: "sha256".into(),
            value,
        })
    }

    pub fn verify_bytes(&self, bytes: &[u8]) -> Result<(), ModelArtifactError> {
        let computed = Self::sha256(bytes);
        if &computed == self {
            Ok(())
        } else {
            Err(ModelArtifactError::DigestMismatch {
                declared: self.value.clone(),
                computed: computed.value,
            })
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModelArtifactId {
    pub kind: ModelArtifactKind,
    pub name: ModelName,
    pub revision: ModelRevision,
    pub variant: Option<ModelVariant>,
    pub digest: ModelDigest,
    pub source: Option<ModelSourceIdentity>,
    pub shard: Option<ModelShardId>,
}

impl ModelArtifactId {
    pub fn new(
        kind: ModelArtifactKind,
        name: ModelName,
        revision: ModelRevision,
        digest: ModelDigest,
    ) -> Self {
        Self {
            kind,
            name,
            revision,
            variant: None,
            digest,
            source: None,
            shard: None,
        }
    }

    pub fn with_variant(mut self, variant: ModelVariant) -> Self {
        self.variant = Some(variant);
        self
    }

    pub fn with_source(mut self, source: ModelSourceIdentity) -> Self {
        self.source = Some(source);
        self
    }

    pub fn with_shard(mut self, shard: ModelShardId) -> Self {
        self.shard = Some(shard);
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelArchitecture {
    pub family: String,
    pub identifier: String,
    pub version: Option<String>,
    pub variant: Option<String>,
    pub required_component_role: Option<String>,
}

impl ModelArchitecture {
    pub fn new(family: impl Into<String>, identifier: impl Into<String>) -> Self {
        Self {
            family: family.into(),
            identifier: identifier.into(),
            version: None,
            variant: None,
            required_component_role: None,
        }
    }

    pub fn validate(&self) -> Result<(), ModelArtifactError> {
        validate_identity_segment(&self.family, "model architecture family")?;
        validate_identity_segment(&self.identifier, "model architecture identifier")?;
        if self.family.ends_with("provider") || self.identifier.ends_with("provider") {
            return Err(ModelArtifactError::ProviderSelectionNotAllowed {
                field: "architecture".into(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelArtifactPart {
    pub name: String,
    pub kind: ModelArtifactKind,
    pub digest: ModelDigest,
    pub size_bytes: Option<u64>,
    pub required: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelShard {
    pub id: ModelShardId,
    pub digest: ModelDigest,
    pub size_bytes: u64,
    pub order: u32,
}

impl ModelShard {
    pub fn verify_bytes(&self, bytes: &[u8]) -> Result<(), ModelArtifactError> {
        self.digest
            .verify_bytes(bytes)
            .map_err(|_| ModelArtifactError::ShardDigestMismatch {
                shard: self.id.as_str().into(),
            })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelTensorMetadata {
    pub name: String,
    pub shape: Vec<u64>,
    pub storage_dtype: ModelDType,
    pub layout: Option<String>,
    pub shard: Option<ModelShardId>,
    pub offset_bytes: Option<u64>,
    pub size_bytes: Option<u64>,
    pub quantization: Option<ModelQuantization>,
    pub expected_compute_dtype: Option<ModelDType>,
    /// The specific bytes that count as this tensor's content for the
    /// artifact it belongs to, when the artifact declares one
    /// (`bind-materialized-weight-content-to-model-artifact-digests`'s
    /// "Tensor Content Digest Binding" requirement). `None` means no
    /// digest was declared for this tensor -- permissive, not "no content
    /// required" -- mirroring `required_weight_names`'s existing
    /// "empty/absent means unknown" precedent rather than making every
    /// pre-existing manifest that predates this field suddenly fail.
    pub digest: Option<ModelDigest>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ModelDType {
    Bool,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F16,
    Bf16,
    F32,
    F64,
    Q4K,
    Q5K,
    Q8,
}

impl ModelDType {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "bool" | "boolean" => Some(Self::Bool),
            "u8" | "uint8" => Some(Self::U8),
            "i8" | "int8" | "sint8" => Some(Self::I8),
            "u16" | "uint16" => Some(Self::U16),
            "i16" | "int16" | "sint16" => Some(Self::I16),
            "u32" | "uint32" => Some(Self::U32),
            "i32" | "int32" | "sint32" => Some(Self::I32),
            "u64" | "uint64" => Some(Self::U64),
            "i64" | "int64" | "sint64" => Some(Self::I64),
            "f16" | "fp16" | "float16" => Some(Self::F16),
            "bf16" | "bfloat16" => Some(Self::Bf16),
            "f32" | "fp32" | "float32" => Some(Self::F32),
            "f64" | "fp64" | "float64" => Some(Self::F64),
            "q4_k" | "q4k" => Some(Self::Q4K),
            "q5_k" | "q5k" => Some(Self::Q5K),
            "q8" | "q8_0" => Some(Self::Q8),
            _ => None,
        }
    }

    pub const fn descriptor(self) -> DTypeDescriptor {
        match self {
            Self::Bool => DTypeDescriptor::portable(ComputeDType::Boolean),
            Self::U8 => DTypeDescriptor::portable(ComputeDType::UInt8),
            Self::I8 => DTypeDescriptor::portable(ComputeDType::SInt8),
            Self::U16 => DTypeDescriptor::portable(ComputeDType::UInt16),
            Self::I16 => DTypeDescriptor::portable(ComputeDType::SInt16),
            Self::U32 => DTypeDescriptor::portable(ComputeDType::UInt32),
            Self::I32 => DTypeDescriptor::portable(ComputeDType::SInt32),
            Self::U64 => DTypeDescriptor::portable(ComputeDType::UInt64),
            Self::I64 => DTypeDescriptor::portable(ComputeDType::SInt64),
            Self::F16 => DTypeDescriptor::portable(ComputeDType::Float16),
            Self::Bf16 => DTypeDescriptor::portable(ComputeDType::BrainFloat16),
            Self::F32 => DTypeDescriptor::portable(ComputeDType::Float32),
            Self::F64 => DTypeDescriptor::portable(ComputeDType::Float64),
            Self::Q4K => DTypeDescriptor::ProviderSpecific {
                id: String::new(),
                size_bytes: 1,
            },
            Self::Q5K => DTypeDescriptor::ProviderSpecific {
                id: String::new(),
                size_bytes: 1,
            },
            Self::Q8 => DTypeDescriptor::ProviderSpecific {
                id: String::new(),
                size_bytes: 1,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ModelQuantizationFormat {
    GgufQ4K,
    GgufQ5K,
    /// GGUF's `Q8_0` block quantization (`ggml_type` 8): 32 elements per
    /// 34-byte block (2-byte `f16` scale + 32 `i8` quants). Paired with
    /// [`ModelDType::Q8`], which `ModelDType::parse` already accepts under
    /// `"q8"`/`"q8_0"`; this variant was missing even though that dtype
    /// existed, found while implementing `formats/gguf`'s real parser
    /// (`implement-model-format-parsers`).
    GgufQ8,
    Gptq,
    Awq,
    BitsAndBytes,
}

impl ModelQuantizationFormat {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "gguf-q4-k" | "q4_k" => Some(Self::GgufQ4K),
            "gguf-q5-k" | "q5_k" => Some(Self::GgufQ5K),
            "gguf-q8-0" | "q8_0" => Some(Self::GgufQ8),
            "gptq" => Some(Self::Gptq),
            "awq" => Some(Self::Awq),
            "bitsandbytes" | "bnb" => Some(Self::BitsAndBytes),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelQuantization {
    pub format: ModelQuantizationFormat,
    pub group_size: Option<u32>,
    pub block_size: Option<u32>,
    pub scale_dtype: Option<ModelDType>,
    pub zero_point_dtype: Option<ModelDType>,
    pub per_channel: bool,
    pub workspace_bytes: Option<u64>,
    pub required_capabilities: Vec<CapabilityBinding>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelGenerationDefaults {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub max_tokens: Option<u32>,
    pub stop_tokens: Vec<String>,
    pub repetition_penalty: Option<f32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelAdapterCompatibility {
    pub target_architecture: String,
    pub base_model: ModelName,
    pub dtype: Option<ModelDType>,
    pub rank: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelLicenseMetadata {
    pub identifier: String,
    pub url: Option<String>,
    pub usage_restrictions: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct ModelProvenance {
    pub source_repository: Option<String>,
    pub source_model_id: Option<String>,
    pub conversion_tool: Option<String>,
    pub conversion_timestamp: Option<String>,
    pub builder: Option<String>,
    pub commit_digest: Option<String>,
    pub publisher: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelSignature {
    pub kind: String,
    pub key_id: Option<String>,
    pub digest: ModelDigest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelArtifactSource {
    LocalPath(PathBuf),
    LocalCache(String),
    ClientProvided(String),
    Registry(String),
    HuggingFace(String),
    Oci(String),
    Tachyon(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelComponentRequirement {
    pub role: String,
    pub capability: Option<CapabilityBinding>,
    pub artifact: Option<ComponentArtifactReference<'static>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelManifest {
    pub schema_version: u64,
    pub id: ModelArtifactId,
    pub architecture: ModelArchitecture,
    pub parts: BTreeMap<String, ModelArtifactPart>,
    pub storage_dtype: Option<ModelDType>,
    pub compute_dtype: Option<ModelDType>,
    pub supported_compute_dtypes: BTreeSet<ModelDType>,
    pub tensors: Vec<ModelTensorMetadata>,
    pub tokenizer: Option<String>,
    pub tokenizer_config: Option<String>,
    pub chat_template: Option<String>,
    pub prompt_template: Option<String>,
    pub generation: Option<ModelGenerationDefaults>,
    pub quantization: Option<ModelQuantization>,
    pub shards: Vec<ModelShard>,
    pub runtime_features: BTreeSet<String>,
    pub memory_features: BTreeSet<String>,
    pub provider_capabilities: Vec<CapabilityBinding>,
    pub component: Option<ModelComponentRequirement>,
    pub license: Option<ModelLicenseMetadata>,
    pub provenance: Option<ModelProvenance>,
    pub signatures: Vec<ModelSignature>,
    pub source: Option<ModelArtifactSource>,
}

impl ModelManifest {
    pub fn load_yaml(path: impl AsRef<Path>) -> Result<Self, ModelArtifactError> {
        let path = path.as_ref();
        let text =
            fs::read_to_string(path).map_err(|source| ModelArtifactError::SourceUnavailable {
                path: path.into(),
                source,
            })?;
        Self::from_yaml_str(&text)
    }

    pub fn from_yaml_str(value: &str) -> Result<Self, ModelArtifactError> {
        RawModelManifest::from_yaml_str(value)?.try_into()
    }

    pub fn validate(&self) -> Result<ModelArtifactRecord, ModelArtifactError> {
        if self.schema_version != MODEL_ARTIFACT_SCHEMA_VERSION {
            return Err(ModelArtifactError::UnsupportedManifestVersion {
                found: self.schema_version,
            });
        }
        self.architecture.validate()?;
        if self.parts.values().any(|part| part.name.trim().is_empty()) {
            return Err(ModelArtifactError::InvalidManifest {
                message: "artifact parts must have names".into(),
            });
        }
        if self
            .parts
            .values()
            .any(|part| part.required && part.kind == ModelArtifactKind::ModelWeights)
            || self.id.kind != ModelArtifactKind::ModelBundle
        {
            // valid
        } else {
            return Err(ModelArtifactError::MissingRequiredPart {
                part: "model-weights".into(),
            });
        }
        if self.id.kind == ModelArtifactKind::ModelBundle
            && !self
                .parts
                .values()
                .any(|part| part.required && part.kind == ModelArtifactKind::ModelConfig)
        {
            return Err(ModelArtifactError::MissingRequiredPart {
                part: "model-config".into(),
            });
        }
        validate_reference("tokenizer", self.tokenizer.as_deref(), &self.parts)?;
        validate_reference(
            "tokenizer-config",
            self.tokenizer_config.as_deref(),
            &self.parts,
        )?;
        validate_reference("chat-template", self.chat_template.as_deref(), &self.parts)?;
        validate_reference(
            "prompt-template",
            self.prompt_template.as_deref(),
            &self.parts,
        )?;
        let mut shard_ids = BTreeSet::new();
        for shard in &self.shards {
            if !shard_ids.insert(shard.id.clone()) {
                return Err(ModelArtifactError::InvalidShard {
                    shard: shard.id.0.clone(),
                    message: "duplicate shard id".into(),
                });
            }
        }
        for tensor in &self.tensors {
            if tensor.name.trim().is_empty() || tensor.shape.contains(&0) {
                return Err(ModelArtifactError::InvalidTensorMetadata {
                    tensor: tensor.name.clone(),
                });
            }
            if let Some(shard) = &tensor.shard
                && !shard_ids.contains(shard)
            {
                return Err(ModelArtifactError::MissingShard {
                    shard: shard.0.clone(),
                });
            }
        }
        Ok(ModelArtifactRecord {
            id: self.id.clone(),
            trust: ModelTrustDecision::new(ModelTrustStatus::Unknown, "trust not evaluated"),
            provenance: self.provenance.clone(),
            license: self.license.clone(),
        })
    }

    pub fn reject_provider_selector(field: &str) -> ModelArtifactError {
        ModelArtifactError::ProviderSelectionNotAllowed {
            field: field.into(),
        }
    }

    pub fn reject_device_selector(field: &str) -> ModelArtifactError {
        ModelArtifactError::DeviceSelectionNotAllowed {
            field: field.into(),
        }
    }

    pub fn residency_plan(&self) -> Result<ModelResidencyPlan, ModelArtifactError> {
        let artifact_bytes = self
            .parts
            .values()
            .filter_map(|part| part.size_bytes)
            .try_fold(0u64, |left, right| {
                left.checked_add(right)
                    .ok_or(ModelArtifactError::SizeOverflow)
            })?;
        let compute_ready_bytes = if let (Some(storage), Some(compute)) =
            (self.storage_dtype, self.compute_dtype)
        {
            self.tensors.iter().try_fold(0u64, |acc, tensor| {
                let elements = tensor.shape.iter().try_fold(1u64, |left, right| {
                    left.checked_mul(*right)
                        .ok_or(ModelArtifactError::SizeOverflow)
                })?;
                let dtype = MemoryDTypeRelation::new(storage.descriptor(), compute.descriptor());
                let storage_bytes = dtype.storage_size_bytes(elements)?;
                let workspace = dtype.compute_workspace_bytes(elements)?;
                acc.checked_add(storage_bytes)
                    .and_then(|value| value.checked_add(workspace))
                    .ok_or(ModelArtifactError::SizeOverflow)
            })?
        } else {
            artifact_bytes
        };
        let quantization_workspace_bytes = self
            .quantization
            .as_ref()
            .and_then(|quantization| quantization.workspace_bytes)
            .unwrap_or(0);
        Ok(ModelResidencyPlan {
            artifact: self.id.clone(),
            artifact_bytes,
            compute_ready_bytes,
            quantization_workspace_bytes,
            host_residency: true,
            device_residency: false,
            provider_owned: false,
        })
    }

    pub fn memory_request(
        &self,
        placement: MemoryPlacement,
    ) -> Result<MemoryAllocationRequest, ModelArtifactError> {
        let plan = self.residency_plan()?;
        let size_bytes = plan
            .compute_ready_bytes
            .checked_add(plan.quantization_workspace_bytes)
            .ok_or(ModelArtifactError::SizeOverflow)?;
        let mut request = MemoryAllocationRequest::new(
            MemoryAllocationClass::ModelArtifact,
            size_bytes,
            placement,
            MemoryAllocationOwner::InferenceArtifact(self.id.name.as_str().into()),
        )
        .with_alignment(64);
        if let (Some(storage), Some(compute)) = (self.storage_dtype, self.compute_dtype) {
            request = request.with_dtype_relation(MemoryDTypeRelation::new(
                storage.descriptor(),
                compute.descriptor(),
            ));
        }
        Ok(request)
    }

    pub fn memory_feasibility(
        &self,
        manager: &MemoryManager,
        placement: MemoryPlacement,
    ) -> Result<MemoryFeasibility, ModelArtifactError> {
        Ok(manager.feasibility(&self.memory_request(placement)?))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelResidencyPlan {
    pub artifact: ModelArtifactId,
    pub artifact_bytes: u64,
    pub compute_ready_bytes: u64,
    pub quantization_workspace_bytes: u64,
    pub host_residency: bool,
    pub device_residency: bool,
    pub provider_owned: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelTrustStatus {
    Unknown,
    Trusted,
    Rejected,
    Revoked,
    PolicyDenied,
}

/// `pub(crate)` fields and constructor, not `pub`: this is the authority
/// `ModelLoadingCoordinator::validate_preconditions` trusts outright
/// (`Trusted` skips straight past trust validation) -- an external caller
/// SHALL NOT be able to construct one claiming `Trusted` directly and pass
/// it to the public `load_model`/`load_model_observed` as if a
/// `ModelTrustStore` had actually evaluated it (a further audit of PR #36
/// found every field and the constructor here were previously `pub`, with
/// `ModelTrustStore::evaluate` -- the one fail-closed, Runtime-owned
/// mechanism that actually exists for this -- entirely optional to go
/// through). Public read-only accessors below; the one legitimate way to
/// obtain a `Trusted` decision is `ModelTrustStore::evaluate`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelTrustDecision {
    pub(crate) status: ModelTrustStatus,
    pub(crate) reason: String,
}

impl ModelTrustDecision {
    pub(crate) fn new(status: ModelTrustStatus, reason: impl Into<String>) -> Self {
        Self {
            status,
            reason: reason.into(),
        }
    }

    /// This decision's trust status. Read-only: see the struct-level doc
    /// comment for why `status` is not a public field.
    pub const fn status(&self) -> ModelTrustStatus {
        self.status
    }

    /// This decision's human-readable reason. Read-only: see the
    /// struct-level doc comment for why `reason` is not a public field.
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelTrustStore {
    pub trusted_digests: BTreeSet<String>,
    pub rejected_digests: BTreeSet<String>,
    pub revoked_digests: BTreeSet<String>,
    pub trusted_sources: BTreeSet<String>,
    pub trusted_publishers: BTreeSet<String>,
}

impl ModelTrustStore {
    pub fn trust_digest(mut self, digest: impl Into<String>) -> Self {
        self.trusted_digests
            .insert(digest.into().to_ascii_lowercase());
        self
    }

    pub fn reject_digest(mut self, digest: impl Into<String>) -> Self {
        self.rejected_digests
            .insert(digest.into().to_ascii_lowercase());
        self
    }

    pub fn revoke_digest(mut self, digest: impl Into<String>) -> Self {
        self.revoked_digests
            .insert(digest.into().to_ascii_lowercase());
        self
    }

    pub fn evaluate(&self, manifest: &ModelManifest) -> ModelTrustDecision {
        let digest = &manifest.id.digest.value;
        if self.revoked_digests.contains(digest) {
            return ModelTrustDecision::new(ModelTrustStatus::Revoked, "digest revoked");
        }
        if self.rejected_digests.contains(digest) {
            return ModelTrustDecision::new(ModelTrustStatus::Rejected, "digest rejected");
        }
        if self.trusted_digests.contains(digest) {
            return ModelTrustDecision::new(ModelTrustStatus::Trusted, "digest trusted by policy");
        }
        if let Some(provenance) = &manifest.provenance
            && let Some(publisher) = &provenance.publisher
            && self.trusted_publishers.contains(publisher)
        {
            return ModelTrustDecision::new(
                ModelTrustStatus::Unknown,
                "publisher identity is metadata only; no authenticated trust mechanism matched",
            );
        }
        ModelTrustDecision::new(ModelTrustStatus::Unknown, "no model trust policy matched")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelArtifactRecord {
    pub id: ModelArtifactId,
    pub trust: ModelTrustDecision,
    pub provenance: Option<ModelProvenance>,
    pub license: Option<ModelLicenseMetadata>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelObservationKind {
    ArtifactDiscovered,
    ManifestLoaded,
    ManifestValidationFailed,
    DigestComputed,
    DigestMismatch,
    ShardValidated,
    ArtifactTrusted,
    ArtifactRejected,
    MemoryFeasibilityChecked,
    ResidencyPlanned,
    ArtifactCached,
    ArtifactEvicted,
    SourceFailure,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelObservation {
    pub kind: ModelObservationKind,
    pub artifact: Option<ModelArtifactId>,
    pub message: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelArtifactObserver {
    observations: Vec<ModelObservation>,
}

impl ModelArtifactObserver {
    pub fn observations(&self) -> &[ModelObservation] {
        &self.observations
    }

    pub fn artifact_discovered(
        &mut self,
        artifact: Option<ModelArtifactId>,
        message: impl Into<String>,
    ) {
        self.emit(ModelObservationKind::ArtifactDiscovered, artifact, message);
    }

    pub fn manifest_loaded(&mut self, artifact: ModelArtifactId) {
        self.emit(
            ModelObservationKind::ManifestLoaded,
            Some(artifact),
            "model artifact manifest loaded",
        );
    }

    pub fn manifest_validation_failed(
        &mut self,
        artifact: Option<ModelArtifactId>,
        message: impl Into<String>,
    ) {
        self.emit(
            ModelObservationKind::ManifestValidationFailed,
            artifact,
            message,
        );
    }

    pub fn digest_computed(&mut self, artifact: ModelArtifactId) {
        self.emit(
            ModelObservationKind::DigestComputed,
            Some(artifact),
            "model artifact digest computed",
        );
    }

    pub fn digest_mismatch(&mut self, artifact: ModelArtifactId) {
        self.emit(
            ModelObservationKind::DigestMismatch,
            Some(artifact),
            "model artifact digest mismatch",
        );
    }

    pub fn shard_validated(&mut self, artifact: ModelArtifactId, shard: &ModelShardId) {
        self.emit(
            ModelObservationKind::ShardValidated,
            Some(artifact),
            format!("model artifact shard '{}' validated", shard.as_str()),
        );
    }

    pub fn artifact_trusted(&mut self, artifact: ModelArtifactId) {
        self.emit(
            ModelObservationKind::ArtifactTrusted,
            Some(artifact),
            "model artifact trusted by policy",
        );
    }

    pub fn artifact_rejected(
        &mut self,
        artifact: Option<ModelArtifactId>,
        message: impl Into<String>,
    ) {
        self.emit(ModelObservationKind::ArtifactRejected, artifact, message);
    }

    pub fn memory_feasibility_checked(&mut self, artifact: ModelArtifactId, feasible: bool) {
        self.emit(
            ModelObservationKind::MemoryFeasibilityChecked,
            Some(artifact),
            if feasible {
                "model artifact memory feasibility accepted"
            } else {
                "model artifact memory feasibility rejected"
            },
        );
    }

    pub fn residency_planned(&mut self, artifact: ModelArtifactId) {
        self.emit(
            ModelObservationKind::ResidencyPlanned,
            Some(artifact),
            "model artifact residency planned",
        );
    }

    pub fn artifact_cached(&mut self, artifact: ModelArtifactId) {
        self.emit(
            ModelObservationKind::ArtifactCached,
            Some(artifact),
            "model artifact cached",
        );
    }

    pub fn artifact_evicted(&mut self, artifact: ModelArtifactId) {
        self.emit(
            ModelObservationKind::ArtifactEvicted,
            Some(artifact),
            "model artifact evicted",
        );
    }

    pub fn source_failure(&mut self, message: impl Into<String>) {
        self.emit(ModelObservationKind::SourceFailure, None, message);
    }

    fn emit(
        &mut self,
        kind: ModelObservationKind,
        artifact: Option<ModelArtifactId>,
        message: impl Into<String>,
    ) {
        self.observations.push(ModelObservation {
            kind,
            artifact,
            message: message.into(),
        });
    }
}

#[derive(Debug)]
pub enum ModelArtifactError {
    ManifestMissing {
        path: PathBuf,
    },
    InvalidManifest {
        message: String,
    },
    UnsupportedManifestVersion {
        found: u64,
    },
    UnsupportedArtifactKind {
        kind: String,
    },
    UnsupportedArchitecture {
        architecture: String,
    },
    UnsupportedArtifactFormat {
        format: String,
    },
    UnsupportedDigestAlgorithm {
        algorithm: String,
    },
    InvalidDigest {
        digest: String,
    },
    DigestMismatch {
        declared: String,
        computed: String,
    },
    MissingRequiredPart {
        part: String,
    },
    MissingShard {
        shard: String,
    },
    ShardDigestMismatch {
        shard: String,
    },
    InvalidShard {
        shard: String,
        message: String,
    },
    UnsupportedStorageDType {
        dtype: String,
    },
    UnsupportedComputeDType {
        dtype: String,
    },
    UnsupportedQuantizationFormat {
        format: String,
    },
    InvalidTensorMetadata {
        tensor: String,
    },
    TokenizerReferenceMissing {
        reference: String,
    },
    TemplateReferenceMissing {
        reference: String,
    },
    ProviderSelectionNotAllowed {
        field: String,
    },
    DeviceSelectionNotAllowed {
        field: String,
    },
    TrustRejected {
        reason: String,
    },
    RevokedArtifact {
        reason: String,
    },
    LicensePolicyDenied {
        reason: String,
    },
    MemoryFeasibilityFailed {
        reason: String,
    },
    SourceUnavailable {
        path: PathBuf,
        source: std::io::Error,
    },
    SizeOverflow,
}

impl fmt::Display for ModelArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ManifestMissing { path } => {
                write!(f, "model artifact manifest '{}' is missing", path.display())
            }
            Self::InvalidManifest { message } => {
                write!(f, "model artifact manifest is invalid: {message}")
            }
            Self::UnsupportedManifestVersion { found } => {
                write!(f, "unsupported model artifact manifest version {found}")
            }
            Self::UnsupportedArtifactKind { kind } => {
                write!(f, "unsupported model artifact kind '{kind}'")
            }
            Self::UnsupportedArchitecture { architecture } => {
                write!(f, "unsupported model architecture '{architecture}'")
            }
            Self::UnsupportedArtifactFormat { format } => {
                write!(f, "unsupported model artifact format '{format}'")
            }
            Self::UnsupportedDigestAlgorithm { algorithm } => {
                write!(f, "unsupported model digest algorithm '{algorithm}'")
            }
            Self::InvalidDigest { digest } => write!(f, "invalid model digest '{digest}'"),
            Self::DigestMismatch { declared, computed } => write!(
                f,
                "model artifact digest mismatch: declared {declared}, computed {computed}"
            ),
            Self::MissingRequiredPart { part } => {
                write!(f, "model artifact is missing required part '{part}'")
            }
            Self::MissingShard { shard } => write!(f, "model artifact is missing shard '{shard}'"),
            Self::ShardDigestMismatch { shard } => {
                write!(f, "model artifact shard '{shard}' digest mismatch")
            }
            Self::InvalidShard { shard, message } => {
                write!(f, "model artifact shard '{shard}' is invalid: {message}")
            }
            Self::UnsupportedStorageDType { dtype } => {
                write!(f, "unsupported model storage dtype '{dtype}'")
            }
            Self::UnsupportedComputeDType { dtype } => {
                write!(f, "unsupported model compute dtype '{dtype}'")
            }
            Self::UnsupportedQuantizationFormat { format } => {
                write!(f, "unsupported quantization format '{format}'")
            }
            Self::InvalidTensorMetadata { tensor } => {
                write!(f, "invalid tensor metadata for '{tensor}'")
            }
            Self::TokenizerReferenceMissing { reference } => {
                write!(f, "tokenizer reference '{reference}' is missing")
            }
            Self::TemplateReferenceMissing { reference } => {
                write!(f, "template reference '{reference}' is missing")
            }
            Self::ProviderSelectionNotAllowed { field } => {
                write!(f, "model artifact field '{field}' may not select Provider")
            }
            Self::DeviceSelectionNotAllowed { field } => {
                write!(f, "model artifact field '{field}' may not select Device")
            }
            Self::TrustRejected { reason } => write!(f, "model artifact trust rejected: {reason}"),
            Self::RevokedArtifact { reason } => write!(f, "model artifact revoked: {reason}"),
            Self::LicensePolicyDenied { reason } => {
                write!(f, "model artifact license policy denied: {reason}")
            }
            Self::MemoryFeasibilityFailed { reason } => {
                write!(f, "model artifact memory feasibility failed: {reason}")
            }
            Self::SourceUnavailable { path, source } => {
                write!(
                    f,
                    "model artifact source '{}' unavailable: {source}",
                    path.display()
                )
            }
            Self::SizeOverflow => write!(f, "model artifact size overflows u64"),
        }
    }
}

impl Error for ModelArtifactError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::SourceUnavailable { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<crate::MemoryError> for ModelArtifactError {
    fn from(error: crate::MemoryError) -> Self {
        Self::MemoryFeasibilityFailed {
            reason: error.to_string(),
        }
    }
}

#[derive(Deserialize)]
struct RawModelManifest {
    schema: String,
    schema_version: u64,
    kind: String,
    digest: String,
    model: RawModelIdentity,
    architecture: RawModelArchitecture,
    #[serde(default)]
    artifacts: BTreeMap<String, RawModelPart>,
    #[serde(default)]
    storage_dtype: Option<String>,
    #[serde(default)]
    compute_dtype: Option<String>,
    #[serde(default)]
    supported_compute_dtypes: Vec<String>,
    #[serde(default)]
    tokenizer: Option<String>,
    #[serde(default)]
    tokenizer_config: Option<String>,
    #[serde(default)]
    chat_template: Option<String>,
    #[serde(default)]
    prompt_template: Option<String>,
    #[serde(default)]
    generation: Option<RawGenerationDefaults>,
    #[serde(default)]
    quantization: Option<RawQuantization>,
    #[serde(default)]
    shards: Vec<RawShard>,
    #[serde(default)]
    tensors: Vec<RawTensor>,
    #[serde(default)]
    runtime_features: BTreeSet<String>,
    #[serde(default)]
    memory_features: BTreeSet<String>,
    #[serde(default)]
    provider_capabilities: Vec<String>,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    device: Option<String>,
    #[serde(default)]
    license: Option<RawLicense>,
    #[serde(default)]
    provenance: Option<ModelProvenance>,
}

impl RawModelManifest {
    fn from_yaml_str(value: &str) -> Result<Self, ModelArtifactError> {
        serde_norway::from_str(value).map_err(|source| ModelArtifactError::InvalidManifest {
            message: source.to_string(),
        })
    }
}

impl TryFrom<RawModelManifest> for ModelManifest {
    type Error = ModelArtifactError;

    fn try_from(raw: RawModelManifest) -> Result<Self, Self::Error> {
        if raw.schema != MODEL_ARTIFACT_SCHEMA {
            return Err(ModelArtifactError::InvalidManifest {
                message: format!("schema must be {MODEL_ARTIFACT_SCHEMA}"),
            });
        }
        if raw.provider.is_some() {
            return Err(ModelManifest::reject_provider_selector("provider"));
        }
        if raw.device.is_some() {
            return Err(ModelManifest::reject_device_selector("device"));
        }
        let kind = ModelArtifactKind::parse(&raw.kind)
            .ok_or(ModelArtifactError::UnsupportedArtifactKind { kind: raw.kind })?;
        let name = ModelName::new(raw.model.name)?;
        let revision = ModelRevision::new(raw.model.revision)?;
        let digest = ModelDigest::parse(raw.digest)?;
        let mut id = ModelArtifactId::new(kind, name, revision, digest);
        if let Some(variant) = raw.model.variant {
            id = id.with_variant(ModelVariant::new(variant)?);
        }
        if let Some(source) = raw.model.source {
            id = id.with_source(ModelSourceIdentity::new(source)?);
        }
        let storage_dtype = parse_optional_dtype(raw.storage_dtype, true)?;
        let compute_dtype = parse_optional_dtype(raw.compute_dtype, false)?;
        let supported_compute_dtypes = raw
            .supported_compute_dtypes
            .into_iter()
            .map(|value| parse_dtype(value, false))
            .collect::<Result<BTreeSet<_>, _>>()?;
        Ok(Self {
            schema_version: raw.schema_version,
            id,
            architecture: raw.architecture.try_into()?,
            parts: raw
                .artifacts
                .into_iter()
                .map(|(name, part)| {
                    let part = part.try_into_part(name.clone())?;
                    Ok((name, part))
                })
                .collect::<Result<_, ModelArtifactError>>()?,
            storage_dtype,
            compute_dtype,
            supported_compute_dtypes,
            tensors: raw
                .tensors
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<_, _>>()?,
            tokenizer: raw.tokenizer,
            tokenizer_config: raw.tokenizer_config,
            chat_template: raw.chat_template,
            prompt_template: raw.prompt_template,
            generation: raw.generation.map(Into::into),
            quantization: raw.quantization.map(TryInto::try_into).transpose()?,
            shards: raw
                .shards
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<_, _>>()?,
            runtime_features: raw.runtime_features,
            memory_features: raw.memory_features,
            provider_capabilities: raw
                .provider_capabilities
                .into_iter()
                .map(parse_capability_binding)
                .collect::<Result<_, _>>()?,
            component: None,
            license: raw.license.map(Into::into),
            provenance: raw.provenance,
            signatures: Vec::new(),
            source: None,
        })
    }
}

#[derive(Deserialize)]
struct RawModelIdentity {
    name: String,
    revision: String,
    #[serde(default)]
    variant: Option<String>,
    #[serde(default)]
    source: Option<String>,
}

#[derive(Deserialize)]
struct RawModelArchitecture {
    family: String,
    identifier: String,
    #[serde(default)]
    version: Option<String>,
    #[serde(default)]
    variant: Option<String>,
    #[serde(default)]
    required_component_role: Option<String>,
}

impl TryFrom<RawModelArchitecture> for ModelArchitecture {
    type Error = ModelArtifactError;

    fn try_from(raw: RawModelArchitecture) -> Result<Self, Self::Error> {
        let architecture = Self {
            family: raw.family,
            identifier: raw.identifier,
            version: raw.version,
            variant: raw.variant,
            required_component_role: raw.required_component_role,
        };
        architecture.validate()?;
        Ok(architecture)
    }
}

#[derive(Deserialize)]
struct RawModelPart {
    kind: String,
    digest: String,
    #[serde(default)]
    size_bytes: Option<u64>,
    #[serde(default)]
    required: Option<bool>,
}

impl RawModelPart {
    fn try_into_part(self, name: String) -> Result<ModelArtifactPart, ModelArtifactError> {
        Ok(ModelArtifactPart {
            name,
            kind: ModelArtifactKind::parse(&self.kind)
                .ok_or(ModelArtifactError::UnsupportedArtifactKind { kind: self.kind })?,
            digest: ModelDigest::parse(self.digest)?,
            size_bytes: self.size_bytes,
            required: self.required.unwrap_or(true),
        })
    }
}

#[derive(Deserialize)]
struct RawShard {
    id: String,
    digest: String,
    size_bytes: u64,
    order: u32,
}

impl TryFrom<RawShard> for ModelShard {
    type Error = ModelArtifactError;

    fn try_from(raw: RawShard) -> Result<Self, Self::Error> {
        Ok(Self {
            id: ModelShardId::new(raw.id)?,
            digest: ModelDigest::parse(raw.digest)?,
            size_bytes: raw.size_bytes,
            order: raw.order,
        })
    }
}

#[derive(Deserialize)]
struct RawTensor {
    name: String,
    shape: Vec<u64>,
    storage_dtype: String,
    #[serde(default)]
    layout: Option<String>,
    #[serde(default)]
    shard: Option<String>,
    #[serde(default)]
    offset_bytes: Option<u64>,
    #[serde(default)]
    size_bytes: Option<u64>,
    #[serde(default)]
    expected_compute_dtype: Option<String>,
    #[serde(default)]
    digest: Option<String>,
}

impl TryFrom<RawTensor> for ModelTensorMetadata {
    type Error = ModelArtifactError;

    fn try_from(raw: RawTensor) -> Result<Self, Self::Error> {
        Ok(Self {
            name: raw.name,
            shape: raw.shape,
            storage_dtype: parse_dtype(raw.storage_dtype, true)?,
            layout: raw.layout,
            shard: raw.shard.map(ModelShardId::new).transpose()?,
            offset_bytes: raw.offset_bytes,
            size_bytes: raw.size_bytes,
            quantization: None,
            expected_compute_dtype: raw
                .expected_compute_dtype
                .map(|value| parse_dtype(value, false))
                .transpose()?,
            digest: raw.digest.map(ModelDigest::parse).transpose()?,
        })
    }
}

#[derive(Deserialize)]
struct RawQuantization {
    format: String,
    #[serde(default)]
    group_size: Option<u32>,
    #[serde(default)]
    block_size: Option<u32>,
    #[serde(default)]
    scale_dtype: Option<String>,
    #[serde(default)]
    zero_point_dtype: Option<String>,
    #[serde(default)]
    per_channel: bool,
    #[serde(default)]
    workspace_bytes: Option<u64>,
    #[serde(default)]
    required_capabilities: Vec<String>,
}

impl TryFrom<RawQuantization> for ModelQuantization {
    type Error = ModelArtifactError;

    fn try_from(raw: RawQuantization) -> Result<Self, Self::Error> {
        Ok(Self {
            format: ModelQuantizationFormat::parse(&raw.format).ok_or({
                ModelArtifactError::UnsupportedQuantizationFormat { format: raw.format }
            })?,
            group_size: raw.group_size,
            block_size: raw.block_size,
            scale_dtype: raw
                .scale_dtype
                .map(|value| parse_dtype(value, false))
                .transpose()?,
            zero_point_dtype: raw
                .zero_point_dtype
                .map(|value| parse_dtype(value, false))
                .transpose()?,
            per_channel: raw.per_channel,
            workspace_bytes: raw.workspace_bytes,
            required_capabilities: raw
                .required_capabilities
                .into_iter()
                .map(parse_capability_binding)
                .collect::<Result<_, _>>()?,
        })
    }
}

#[derive(Deserialize)]
struct RawGenerationDefaults {
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    stop_tokens: Vec<String>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
}

impl From<RawGenerationDefaults> for ModelGenerationDefaults {
    fn from(raw: RawGenerationDefaults) -> Self {
        Self {
            temperature: raw.temperature,
            top_p: raw.top_p,
            top_k: raw.top_k,
            max_tokens: raw.max_tokens,
            stop_tokens: raw.stop_tokens,
            repetition_penalty: raw.repetition_penalty,
        }
    }
}

#[derive(Deserialize)]
struct RawLicense {
    identifier: String,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    usage_restrictions: Vec<String>,
}

impl From<RawLicense> for ModelLicenseMetadata {
    fn from(raw: RawLicense) -> Self {
        Self {
            identifier: raw.identifier,
            url: raw.url,
            usage_restrictions: raw.usage_restrictions,
        }
    }
}

fn parse_optional_dtype(
    value: Option<String>,
    storage: bool,
) -> Result<Option<ModelDType>, ModelArtifactError> {
    value.map(|value| parse_dtype(value, storage)).transpose()
}

fn parse_dtype(value: String, storage: bool) -> Result<ModelDType, ModelArtifactError> {
    ModelDType::parse(&value).ok_or({
        if storage {
            ModelArtifactError::UnsupportedStorageDType { dtype: value }
        } else {
            ModelArtifactError::UnsupportedComputeDType { dtype: value }
        }
    })
}

fn parse_capability_binding(value: String) -> Result<CapabilityBinding, ModelArtifactError> {
    let Some((id, version)) = value.split_once('@') else {
        return Err(ModelArtifactError::InvalidManifest {
            message: format!("capability binding '{value}' must use id@major.minor.patch"),
        });
    };
    let mut parts = version.split('.');
    let parse_part = |part: Option<&str>| -> Result<u64, ModelArtifactError> {
        part.ok_or_else(|| ModelArtifactError::InvalidManifest {
            message: format!("capability binding '{value}' must use semantic version"),
        })?
        .parse()
        .map_err(|_| ModelArtifactError::InvalidManifest {
            message: format!("capability binding '{value}' has invalid semantic version"),
        })
    };
    let version = CapabilityVersion::new(
        parse_part(parts.next())?,
        parse_part(parts.next())?,
        parse_part(parts.next())?,
    );
    if parts.next().is_some() || id.trim().is_empty() {
        return Err(ModelArtifactError::InvalidManifest {
            message: format!("capability binding '{value}' must use id@major.minor.patch"),
        });
    }
    Ok(CapabilityBinding::new(CapabilityId::new(id), version))
}

fn validate_reference(
    kind: &str,
    reference: Option<&str>,
    parts: &BTreeMap<String, ModelArtifactPart>,
) -> Result<(), ModelArtifactError> {
    let Some(reference) = reference else {
        return Ok(());
    };
    if parts.contains_key(reference) {
        Ok(())
    } else if kind.contains("tokenizer") {
        Err(ModelArtifactError::TokenizerReferenceMissing {
            reference: reference.into(),
        })
    } else {
        Err(ModelArtifactError::TemplateReferenceMissing {
            reference: reference.into(),
        })
    }
}

fn validate_identity_segment(value: &str, label: &str) -> Result<(), ModelArtifactError> {
    if value.trim().is_empty() {
        return Err(ModelArtifactError::InvalidManifest {
            message: format!("{label} must not be empty"),
        });
    }
    if value.contains('/') || value.contains('\\') || value.contains(':') {
        return Err(ModelArtifactError::InvalidManifest {
            message: format!("{label} must not be a path, URI, Provider, or Device selector"),
        });
    }
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return Err(ModelArtifactError::InvalidManifest {
            message: format!("{label} must use portable ASCII characters"),
        });
    }
    Ok(())
}

fn lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}
