//! Pinned Laya decision inference through a resident ModernBERT Component.
//!
//! The checkpoint, configurations and tokenizer are individually bound to the
//! selected public revision. The caller separately trusts the model digest and
//! registers a trusted ModernBERT Component before loading this profile.

use anyhow::{Context, Result, ensure};
use magnetar_component_laya::{
    Answer, Calibrator, SpecialTokens,
    head::{CheckpointTensor, DecisionHead, checkpoint_tensor_shapes},
    prepare_sequence,
};
pub use magnetar_component_laya::{ChoiceOption, Question, QuestionType};
use magnetar_component_modernbert::checkpoint::{
    CheckpointNamespace, CheckpointProfile, DecodedTensor, StorageDType, TensorDescriptor,
    WeightSpec,
};
use magnetar_loader_huggingface::{
    HuggingFaceTokenizer, SafetensorsPayloadSource, inspect_authenticated_safetensors_file,
};
use magnetar_runtime::{
    ArtifactFormat, ComponentDigest, DecisionPrediction, DecisionResult, DecisionScore,
    EncoderGraphConfig, EncoderLayerConfig, ModelArchitecture, ModelArchitectureConfig,
    ModelArtifactId, ModelArtifactKind, ModelArtifactPart, ModelArtifactSource, ModelDType,
    ModelDigest, ModelManifest, ModelName, ModelRevision, ModelTensorMetadata, ModelTrustStore,
    ProductionArtifactPayloadSource, ProductionIngestionError, ProductionLoadedEncoder,
    ProductionModelSource, ProductionPayloadRange, Provider, tensor_from_raw_bytes,
    tokenizer::{EncodeInput, Tokenizer},
};
use std::{collections::BTreeMap, path::Path, sync::Arc};

const CHECKPOINT_DIGEST: &str =
    "sha256:891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c";
const ENCODER_CONFIG_DIGEST: &str =
    "sha256:bf3ab80598fdccf414855a2ce80f22859e4492d06ca8a62ddd1cfb63972f8979";
const DECISION_CONFIG_DIGEST: &str =
    "sha256:ae287b56bbcf5f8c4f4541ae9dfd00c914c4c48b940b8398c3058af37ba92bbd";
const TOKENIZER_DIGEST: &str =
    "sha256:6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30";
const CLS: u32 = 50_281;
const SEP: u32 = 50_282;
const MASK: u32 = 50_284;
const MAX_TOKENS: usize = 512;
const MAX_OPTIONS: usize = 255;
const MAX_TEXT_BYTES: usize = 1_048_576;

#[cfg(test)]
mod tests;

struct GraphTensor {
    source: ModelTensorMetadata,
    spec: WeightSpec,
    length: u64,
}

struct GraphPayloads {
    source: SafetensorsPayloadSource,
    tensors: BTreeMap<String, GraphTensor>,
}

impl ProductionArtifactPayloadSource for GraphPayloads {
    fn read_payload(
        &self,
        range: &ProductionPayloadRange,
    ) -> std::result::Result<Vec<u8>, ProductionIngestionError> {
        let bad_range = || ProductionIngestionError::PayloadOutOfBounds {
            identity: range.identity.clone(),
        };
        let entry = self.tensors.get(&range.identity).ok_or_else(bad_range)?;
        if range.offset != 0 || range.length != entry.length {
            return Err(bad_range());
        }
        let invalid = |reason: String| ProductionIngestionError::MalformedMetadata { reason };
        let raw = read_source_tensor(&self.source, &entry.source)?;
        let decoded = tensor_from_raw_bytes(&entry.source, &raw)
            .map_err(|error| invalid(error.to_string()))?;
        let shape = decoded
            .shape
            .iter()
            .map(|&size| usize::try_from(size).map_err(|error| invalid(error.to_string())))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let values = entry
            .spec
            .graph_values(&DecodedTensor {
                shape,
                values: decoded.data,
            })
            .map_err(|error| invalid(error.to_string()))?;
        let bytes: Vec<u8> = values.into_iter().flat_map(f32::to_le_bytes).collect();
        if let Some(expected) = &range.digest {
            expected.verify_bytes(&bytes).map_err(|error| {
                ProductionIngestionError::IntegrityMismatch {
                    identity: format!("{}: {error}", range.identity),
                }
            })?;
        }
        Ok(bytes)
    }
}

fn read_source_tensor(
    source: &SafetensorsPayloadSource,
    tensor: &ModelTensorMetadata,
) -> std::result::Result<Vec<u8>, ProductionIngestionError> {
    let missing = || ProductionIngestionError::MalformedMetadata {
        reason: format!("tensor '{}' has no complete byte range", tensor.name),
    };
    source.read_payload(&ProductionPayloadRange {
        identity: tensor.name.clone(),
        offset: tensor.offset_bytes.ok_or_else(missing)?,
        length: tensor.size_bytes.ok_or_else(missing)?,
        digest: tensor.digest.clone(),
    })
}

fn pinned_json(bytes: &[u8], expected: &str) -> Result<Vec<u8>> {
    ensure!(bytes.len() <= 64 * 1024, "Laya configuration is too large");
    let text = std::str::from_utf8(bytes).context("Laya configuration is not UTF-8")?;
    // Git may check text fixtures out with CRLF on Windows. This normalization
    // changes line endings only; every other byte remains pinned.
    let normalized = text.replace("\r\n", "\n").into_bytes();
    ensure!(
        ModelDigest::sha256(&normalized).value == expected,
        "Laya configuration digest mismatch"
    );
    Ok(normalized)
}

/// Resident encoder plus the pinned checkpoint's learned decision head.
pub struct LoadedLaya {
    encoder: ProductionLoadedEncoder,
    head: DecisionHead,
    calibrator: Calibrator,
}

impl LoadedLaya {
    /// Load the selected Laya revision and its normalized encoder/decision
    /// configurations. The supplied Component digest must already be registered
    /// and trusted; `trust` must independently trust the checkpoint digest.
    pub fn load_pinned(
        checkpoint: &Path,
        encoder_config: &[u8],
        decision_config: &[u8],
        component: ComponentDigest,
        trust: ModelTrustStore,
        provider: Arc<dyn Provider>,
    ) -> Result<Self> {
        let encoder_config = pinned_json(encoder_config, ENCODER_CONFIG_DIGEST)?;
        let decision_config = pinned_json(decision_config, DECISION_CONFIG_DIGEST)?;
        let profile = CheckpointProfile::from_huggingface_config(
            &encoder_config,
            CheckpointNamespace::Encoder,
        )?;
        let calibrator = Calibrator::from_config(&decision_config)?;
        let settings = profile.graph_settings();
        let parent = checkpoint
            .parent()
            .context("checkpoint has no parent directory")?;
        let filename = checkpoint
            .file_name()
            .and_then(|name| name.to_str())
            .context("checkpoint filename is not valid UTF-8")?;
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(checkpoint.to_path_buf()),
            parent.to_path_buf(),
        );
        let digest = ModelDigest::parse(CHECKPOINT_DIGEST)?;
        let raw = inspect_authenticated_safetensors_file(&source, filename, &digest)
            .context("could not authenticate the Laya checkpoint")?;
        let inventory = raw
            .tensors
            .iter()
            .map(|tensor| {
                let shape = tensor
                    .shape
                    .iter()
                    .map(|&size| usize::try_from(size))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let dtype = match tensor.storage_dtype {
                    ModelDType::F16 => StorageDType::F16,
                    ModelDType::Bf16 => StorageDType::BF16,
                    ModelDType::F32 => StorageDType::F32,
                    _ => StorageDType::Unsupported,
                };
                Ok((tensor.name.clone(), TensorDescriptor { shape, dtype }))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        profile.validate_inventory(&inventory)?;
        let head_shapes = checkpoint_tensor_shapes(settings.hidden_size, 2, 2)?;
        ensure!(
            raw.tensors.len() == profile.weights().len() + head_shapes.len() + 1,
            "unexpected Laya checkpoint tensor inventory"
        );
        let source_tensors: BTreeMap<_, _> = raw
            .tensors
            .into_iter()
            .map(|tensor| (tensor.name.clone(), tensor))
            .collect();
        let temperature = source_tensors
            .get("temperature")
            .context("missing Laya temperature")?;
        ensure!(
            temperature.shape == [3] && temperature.storage_dtype == ModelDType::F32,
            "invalid Laya temperature tensor"
        );
        let mut head_weights = BTreeMap::new();
        for (name, expected_shape) in &head_shapes {
            let metadata = source_tensors
                .get(name)
                .with_context(|| format!("missing Laya head tensor {name}"))?;
            let bytes = read_source_tensor(&raw.payloads, metadata)?;
            let tensor = tensor_from_raw_bytes(metadata, &bytes)?;
            let shape = tensor
                .shape
                .iter()
                .map(|&size| usize::try_from(size))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            ensure!(
                &shape == expected_shape,
                "invalid shape for Laya head tensor {name}"
            );
            head_weights.insert(
                name.clone(),
                CheckpointTensor {
                    shape,
                    values: tensor.data,
                },
            );
        }
        let head = DecisionHead::from_checkpoint_tensors(settings.hidden_size, 2, 2, head_weights)?;
        let mut tensors = Vec::with_capacity(profile.weights().len());
        let mut graph_tensors = BTreeMap::new();
        for spec in profile.weights() {
            let source = source_tensors
                .get(&spec.checkpoint_name)
                .with_context(|| format!("missing encoder tensor {}", spec.checkpoint_name))?
                .clone();
            let shape = spec
                .graph_shape
                .iter()
                .map(|&size| u64::try_from(size))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let length = shape
                .iter()
                .try_fold(4_u64, |bytes, &size| bytes.checked_mul(size))
                .context("encoder tensor size overflow")?;
            tensors.push(ModelTensorMetadata {
                name: spec.graph_name.clone(),
                shape,
                storage_dtype: ModelDType::F32,
                layout: None,
                shard: None,
                offset_bytes: Some(0),
                size_bytes: Some(length),
                quantization: None,
                expected_compute_dtype: None,
                digest: None,
            });
            graph_tensors.insert(
                spec.graph_name.clone(),
                GraphTensor {
                    source,
                    spec: spec.clone(),
                    length,
                },
            );
        }
        let architecture = ModelArchitectureConfig {
            hidden_size: settings.hidden_size as u64,
            intermediate_size: settings.intermediate_size as u64,
            num_hidden_layers: u32::try_from(settings.layers.len())?,
            num_attention_heads: u32::try_from(settings.attention_heads)?,
            num_key_value_heads: u32::try_from(settings.attention_heads)?,
            head_dim: (settings.hidden_size / settings.attention_heads) as u64,
            vocab_size: settings.vocabulary_size as u64,
            rms_norm_eps: settings.norm_epsilon as f32,
            rope_theta: settings.layers[0].rope_theta,
            hidden_activation: Some("gelu".into()),
            attention_bias: settings.attention_bias,
            ..Default::default()
        };
        let encoder = EncoderGraphConfig {
            norm_epsilon: settings.norm_epsilon,
            max_positions: settings.max_positions as u64,
            norm_bias: settings.norm_bias,
            attention_bias: settings.attention_bias,
            mlp_bias: settings.mlp_bias,
            layers: settings
                .layers
                .iter()
                .map(|layer| {
                    Ok(EncoderLayerConfig {
                        local_radius: layer.local_radius.map(u32::try_from).transpose()?,
                        rope_theta: layer.rope_theta,
                        attention_normalization: layer.attention_normalization,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        };
        let parts = BTreeMap::from([
            (
                "weights".into(),
                ModelArtifactPart {
                    name: "weights".into(),
                    kind: ModelArtifactKind::ModelWeights,
                    digest: digest.clone(),
                    size_bytes: Some(842_609_210),
                    required: true,
                },
            ),
            (
                "config".into(),
                ModelArtifactPart {
                    name: "config".into(),
                    kind: ModelArtifactKind::ModelConfig,
                    digest: ModelDigest::sha256(&encoder_config),
                    size_bytes: Some(encoder_config.len() as u64),
                    required: true,
                },
            ),
        ]);
        let manifest = ModelManifest {
            schema_version: magnetar_runtime::model::MODEL_ARTIFACT_SCHEMA_VERSION,
            id: ModelArtifactId::new(
                ModelArtifactKind::ModelBundle,
                ModelName::new("laya")?,
                ModelRevision::new("fd4f9dbc")?,
                digest,
            ),
            architecture: ModelArchitecture::new("modernbert", "laya-modernbert"),
            parts,
            storage_dtype: None,
            compute_dtype: None,
            supported_compute_dtypes: Default::default(),
            tensors,
            tokenizer: None,
            tokenizer_config: None,
            chat_template: None,
            prompt_template: None,
            generation: None,
            quantization: None,
            shards: Vec::new(),
            runtime_features: Default::default(),
            memory_features: Default::default(),
            provider_capabilities: Vec::new(),
            component: None,
            license: None,
            provenance: None,
            signatures: Vec::new(),
            source: Some(ModelArtifactSource::LocalPath(checkpoint.to_path_buf())),
            architecture_config: Some(architecture),
            artifact_format: ArtifactFormat::HuggingFace,
        };
        let payloads = GraphPayloads {
            source: raw.payloads,
            tensors: graph_tensors,
        };
        let encoder =
            ProductionLoadedEncoder::load(manifest, &payloads, trust, provider, component, encoder)
                .context("could not load the resident Laya encoder")?;
        Ok(Self {
            encoder,
            head,
            calibrator,
        })
    }

    /// Evaluate one already-tokenized, optionally padded question. Marker
    /// positions must be distinct valid `[MASK]` tokens in presentation order.
    pub fn decide_tokenized(
        &mut self,
        ids: &[u32],
        valid: &[bool],
        markers: &[usize],
        kind: QuestionType,
    ) -> Result<DecisionResult> {
        if ids.len() != valid.len()
            || markers.len() < 2
            || markers.len() > MAX_OPTIONS
            || (kind == QuestionType::Noul && markers.len() != 2)
            || markers.windows(2).any(|window| window[0] >= window[1])
            || markers
                .iter()
                .any(|&index| ids.get(index) != Some(&MASK) || valid.get(index) != Some(&true))
        {
            return Err(super::invalid_request(
                "invalid Laya tokenized question or markers",
            ));
        }
        let hidden = self
            .encoder
            .encode(ids, valid)
            .context("Laya encoder failed")?;
        let result = self
            .head
            .forward(&hidden.data, ids.len(), valid, markers, kind)?
            .decide(&self.calibrator, kind)?;
        let output = DecisionResult {
            options: result
                .option_scores
                .iter()
                .zip(&result.decision.probabilities)
                .map(|(&score, &probability)| DecisionScore {
                    score: score.into(),
                    probability,
                })
                .collect(),
            prediction: match result.decision.answer {
                Answer::Choice(index) => DecisionPrediction::SelectedOption(u32::try_from(index)?),
                Answer::Score(value) | Answer::Noul(value) => {
                    DecisionPrediction::NumericValue(value)
                }
            },
            confidence: result.decision.confidence,
            actions: Some(
                result
                    .action_scores
                    .iter()
                    .zip(&result.action_probabilities)
                    .map(|(&score, &probability)| DecisionScore {
                        score: score.into(),
                        probability,
                    })
                    .collect(),
            ),
        };
        output.validate()?;
        Ok(output)
    }

    /// Render and evaluate one text question using the pinned tokenizer.
    /// State is already serialized text. Upstream's right truncation applies.
    pub fn decide_text(
        &mut self,
        tokenizer: &HuggingFaceTokenizer,
        state: &str,
        question: &Question<'_>,
    ) -> Result<DecisionResult> {
        ensure!(
            tokenizer
                .metadata()
                .digest
                .value
                .eq_ignore_ascii_case(TOKENIZER_DIGEST),
            "Laya tokenizer digest mismatch"
        );
        let count = match question {
            Question::Choice { options, .. } => options.len(),
            Question::Score { levels, .. } => levels.len(),
            Question::Noul { .. } => 2,
        };
        if !(2..=MAX_OPTIONS).contains(&count) || state.len() > MAX_TEXT_BYTES {
            return Err(super::invalid_request("invalid Laya question size"));
        }
        let mut bytes = state.len();
        let mut add = |text: &str| -> Result<()> {
            bytes = bytes
                .checked_add(text.len())
                .ok_or_else(|| super::invalid_request("Laya input size overflow"))?;
            if bytes > MAX_TEXT_BYTES {
                return Err(super::invalid_request("Laya text input is too large"));
            }
            Ok(())
        };
        match question {
            Question::Choice {
                instructions,
                options,
            } => {
                add(instructions)?;
                for option in *options {
                    add(option.label)?;
                    if let Some(description) = option.description {
                        add(description)?;
                    }
                }
            }
            Question::Score {
                instructions,
                levels,
            } => {
                add(instructions)?;
                for level in *levels {
                    add(level)?;
                }
            }
            Question::Noul {
                instructions,
                when_false,
                when_true,
            } => {
                add(instructions)?;
                if let Some(text) = when_false {
                    add(text)?;
                }
                if let Some(text) = when_true {
                    add(text)?;
                }
            }
        }
        for (text, expected) in [("[CLS]", CLS), ("[SEP]", SEP), ("[MASK]", MASK)] {
            ensure!(
                tokenizer.encode(EncodeInput::new(text))?.token_ids == [expected],
                "invalid Laya tokenizer control token"
            );
        }
        let prepared = prepare_sequence(
            question,
            state,
            SpecialTokens {
                cls: CLS,
                sep: SEP,
                mask: MASK,
                mask_text: "[MASK]",
            },
            MAX_TOKENS,
            192,
            false,
            |text| {
                tokenizer
                    .encode(EncodeInput::new(text))
                    .map(|result| result.token_ids)
            },
        )
        .map_err(|error| super::invalid_request(format!("invalid Laya question: {error}")))?;
        let valid = vec![true; prepared.input_ids.len()];
        self.decide_tokenized(
            &prepared.input_ids,
            &valid,
            &prepared.marker_positions,
            prepared.question_type,
        )
    }

    /// Release encoder weights held by the Provider.
    pub fn unload(self) -> Result<()> {
        self.encoder.unload().map_err(Into::into)
    }
}
