//! Tokenizer contracts for inference Runtime.
//!
//! A tokenizer is the Runtime boundary between user-visible text and
//! model-visible token IDs. It is not a Model Artifact, Component Artifact,
//! chat template, generation engine, Provider selector, Device selector, or
//! client workspace tool. Tokenizer data is resolved through registered
//! inference artifacts; implementations stay hidden behind this contract.

use crate::{
    ComponentDigest, ComponentError, InferenceArtifactKind, InferenceArtifactReference,
    InferenceArtifactRegistry, MemoryAllocationClass, MemoryAllocationOwner,
    MemoryAllocationRequest, MemoryManager, MemoryPlacement, ModelArtifactKind, ModelDigest,
    ModelManifest,
};
use std::{collections::BTreeSet, error::Error, fmt};

pub type TokenId = u32;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TokenizerId(String);

impl TokenizerId {
    pub fn new(value: impl Into<String>) -> Result<Self, TokenizerError> {
        let value = value.into();
        validate_identity(&value, "tokenizer id")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TokenizerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TokenizerArtifactId(String);

impl TokenizerArtifactId {
    pub fn new(value: impl Into<String>) -> Result<Self, TokenizerError> {
        let value = value.into();
        validate_identity(&value, "tokenizer artifact id")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TokenizerFamily(String);

impl TokenizerFamily {
    pub fn new(value: impl Into<String>) -> Result<Self, TokenizerError> {
        let value = value.into();
        validate_identity(&value, "tokenizer family")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TokenizerRevision(String);

impl TokenizerRevision {
    pub fn new(value: impl Into<String>) -> Result<Self, TokenizerError> {
        let value = value.into();
        validate_identity(&value, "tokenizer revision")?;
        Ok(Self(value))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizerArtifactReference {
    pub id: TokenizerArtifactId,
    pub kind: ModelArtifactKind,
    pub digest: ModelDigest,
}

impl TokenizerArtifactReference {
    pub fn new(
        id: TokenizerArtifactId,
        kind: ModelArtifactKind,
        digest: ModelDigest,
    ) -> Result<Self, TokenizerError> {
        if !matches!(
            kind,
            ModelArtifactKind::Tokenizer
                | ModelArtifactKind::TokenizerConfig
                | ModelArtifactKind::Vocabulary
                | ModelArtifactKind::SpecialTokens
        ) {
            return Err(TokenizerError::InvalidTokenizerArtifact {
                artifact: id.as_str().into(),
            });
        }
        Ok(Self { id, kind, digest })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizerArtifactSet {
    pub tokenizer: TokenizerArtifactReference,
    pub tokenizer_config: Option<TokenizerArtifactReference>,
    pub vocabulary: Option<TokenizerArtifactReference>,
    pub special_tokens: Option<TokenizerArtifactReference>,
}

impl TokenizerArtifactSet {
    pub fn validate_registered(
        &self,
        registry: &InferenceArtifactRegistry,
    ) -> Result<(), TokenizerError> {
        for artifact in self.references() {
            registry
                .resolve(InferenceArtifactKind::Tokenizer, artifact.id.as_str(), None)
                .map_err(|_| TokenizerError::TokenizerArtifactMissing {
                    artifact: artifact.id.as_str().into(),
                })?;
        }
        Ok(())
    }

    pub fn references(&self) -> Vec<&TokenizerArtifactReference> {
        [
            Some(&self.tokenizer),
            self.tokenizer_config.as_ref(),
            self.vocabulary.as_ref(),
            self.special_tokens.as_ref(),
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    pub fn memory_requests(&self) -> Vec<MemoryAllocationRequest> {
        self.references()
            .into_iter()
            .map(|reference| {
                MemoryAllocationRequest::new(
                    MemoryAllocationClass::TokenizerArtifact,
                    1,
                    MemoryPlacement::HostOrdinary,
                    MemoryAllocationOwner::InferenceArtifact(reference.id.as_str().into()),
                )
            })
            .collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizerCompatibility {
    pub expected_digest: Option<ModelDigest>,
    pub expected_vocabulary_size: Option<u32>,
    pub expected_family: Option<TokenizerFamily>,
    pub expected_model_max_length: Option<u32>,
    pub expected_added_tokens: Option<u32>,
    pub expected_special_tokens: Vec<SpecialTokenKind>,
    pub expected_normalization: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizerMetadata {
    pub id: TokenizerId,
    pub artifact: TokenizerArtifactId,
    pub digest: ModelDigest,
    pub family: TokenizerFamily,
    pub revision: TokenizerRevision,
    pub vocabulary_size: u32,
    pub added_token_count: u32,
    pub token_id_range: TokenIdRange,
    pub model_max_length: Option<u32>,
    pub special_tokens: Vec<SpecialToken>,
    pub additional_special_tokens: Vec<SpecialToken>,
    pub byte_fallback: bool,
    pub normalization: Option<String>,
    pub pre_tokenizer: Option<String>,
    pub supports_offsets: bool,
    pub supports_token_type_ids: bool,
    pub supports_browser: bool,
}

impl TokenizerMetadata {
    pub fn validate(&self) -> Result<(), TokenizerError> {
        if self.vocabulary_size == 0 {
            return Err(TokenizerError::VocabularyMismatch {
                expected: 1,
                actual: 0,
            });
        }
        self.token_id_range.validate()?;
        let mut kinds = BTreeSet::new();
        let mut ids = BTreeSet::new();
        for token in self
            .special_tokens
            .iter()
            .chain(self.additional_special_tokens.iter())
        {
            token.validate(&self.token_id_range)?;
            if !kinds.insert((token.kind, token.text.clone())) || !ids.insert(token.id) {
                return Err(TokenizerError::SpecialTokenConflict);
            }
        }
        Ok(())
    }

    pub fn special_token(&self, kind: SpecialTokenKind) -> Option<&SpecialToken> {
        self.special_tokens
            .iter()
            .chain(self.additional_special_tokens.iter())
            .find(|token| token.kind == kind)
    }

    pub fn validate_compatibility(
        &self,
        compatibility: &TokenizerCompatibility,
    ) -> Result<(), TokenizerError> {
        if let Some(expected) = &compatibility.expected_digest
            && expected != &self.digest
        {
            return Err(TokenizerError::TokenizerIncompatibleWithModel);
        }
        if let Some(expected) = compatibility.expected_vocabulary_size
            && expected != self.vocabulary_size
        {
            return Err(TokenizerError::VocabularyMismatch {
                expected,
                actual: self.vocabulary_size,
            });
        }
        if let Some(expected) = &compatibility.expected_family
            && expected != &self.family
        {
            return Err(TokenizerError::UnsupportedTokenizerFamily {
                family: self.family.as_str().into(),
            });
        }
        if let Some(expected) = compatibility.expected_model_max_length
            && self
                .model_max_length
                .is_some_and(|actual| actual < expected)
        {
            return Err(TokenizerError::TokenizerIncompatibleWithModel);
        }
        if let Some(expected) = compatibility.expected_added_tokens
            && expected != self.added_token_count
        {
            return Err(TokenizerError::AddedTokenMismatch {
                expected,
                actual: self.added_token_count,
            });
        }
        if compatibility
            .expected_special_tokens
            .iter()
            .any(|kind| self.special_token(*kind).is_none())
        {
            return Err(TokenizerError::SpecialTokenMissing);
        }
        if let Some(expected) = &compatibility.expected_normalization
            && self.normalization.as_ref() != Some(expected)
        {
            return Err(TokenizerError::TokenizerIncompatibleWithModel);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenIdRange {
    pub start: TokenId,
    pub end_inclusive: TokenId,
}

impl TokenIdRange {
    pub const fn new(start: TokenId, end_inclusive: TokenId) -> Self {
        Self {
            start,
            end_inclusive,
        }
    }

    pub const fn contains(self, id: TokenId) -> bool {
        id >= self.start && id <= self.end_inclusive
    }

    pub fn validate(self) -> Result<(), TokenizerError> {
        if self.start > self.end_inclusive {
            return Err(TokenizerError::InvalidTokenId {
                token_id: self.start,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SpecialTokenKind {
    Unknown,
    Bos,
    Eos,
    Pad,
    Sep,
    Cls,
    Mask,
    Additional,
    Stop,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpecialToken {
    pub kind: SpecialTokenKind,
    pub text: String,
    pub id: TokenId,
    pub added_during_encode: bool,
    pub skipped_during_decode: bool,
}

impl SpecialToken {
    pub fn new(kind: SpecialTokenKind, text: impl Into<String>, id: TokenId) -> Self {
        Self {
            kind,
            text: text.into(),
            id,
            added_during_encode: false,
            skipped_during_decode: true,
        }
    }

    pub fn validate(&self, range: &TokenIdRange) -> Result<(), TokenizerError> {
        if self.text.is_empty() {
            return Err(TokenizerError::SpecialTokenMissing);
        }
        if !range.contains(self.id) {
            return Err(TokenizerError::InvalidTokenId { token_id: self.id });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TruncationPolicy {
    None,
    Left,
    Right,
    Middle,
    ModelDefault,
    ClientPolicy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PaddingPolicy {
    None,
    Longest,
    MaxLength,
    ModelDefault,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpecialTokenPolicy {
    Preserve,
    Add,
    Skip,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenOffset {
    pub byte_start: u32,
    pub byte_end: u32,
    pub char_start: Option<u32>,
    pub char_end: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodeInput {
    pub text: String,
    pub add_special_tokens: bool,
    pub truncation: TruncationPolicy,
    pub max_tokens: Option<usize>,
    pub return_offsets: bool,
    pub padding: PaddingPolicy,
    pub special_token_policy: SpecialTokenPolicy,
}

impl EncodeInput {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            add_special_tokens: false,
            truncation: TruncationPolicy::None,
            max_tokens: None,
            return_offsets: false,
            padding: PaddingPolicy::None,
            special_token_policy: SpecialTokenPolicy::Preserve,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodeOutput {
    pub token_ids: Vec<TokenId>,
    pub token_count: usize,
    pub offsets: Option<Vec<TokenOffset>>,
    pub attention_mask: Option<Vec<u8>>,
    pub token_type_ids: Option<Vec<u32>>,
    pub diagnostics: Vec<TokenizerDiagnostic>,
}

impl EncodeOutput {
    pub fn validate(&self) -> Result<(), TokenizerError> {
        if self.token_count != self.token_ids.len() {
            return Err(TokenizerError::BatchInputInvalid {
                message: "token count must match token id length".into(),
            });
        }
        if let Some(offsets) = &self.offsets
            && offsets.len() != self.token_ids.len()
        {
            return Err(TokenizerError::BatchInputInvalid {
                message: "offset count must match token id length".into(),
            });
        }
        if let Some(mask) = &self.attention_mask
            && mask.len() != self.token_ids.len()
        {
            return Err(TokenizerError::BatchInputInvalid {
                message: "attention mask length must match token id length".into(),
            });
        }
        if let Some(token_type_ids) = &self.token_type_ids
            && token_type_ids.len() != self.token_ids.len()
        {
            return Err(TokenizerError::BatchInputInvalid {
                message: "token type id length must match token id length".into(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodeInput {
    pub token_ids: Vec<TokenId>,
    pub skip_special_tokens: bool,
    pub clean_up_tokenization_spaces: bool,
    pub streaming_state: Option<StreamingDecodeState>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodeOutput {
    pub text: String,
    pub consumed_token_count: usize,
    pub pending_partial_state: Option<StreamingDecodeState>,
    pub diagnostics: Vec<TokenizerDiagnostic>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StreamingDecodeState {
    pub pending_bytes: Vec<u8>,
    pub emitted_token_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchEncodeInput {
    pub inputs: Vec<EncodeInput>,
    pub padding: PaddingPolicy,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchEncodeOutput {
    pub outputs: Vec<Result<EncodeOutput, TokenizerError>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenStopPattern {
    pub text: String,
    pub token_ids: Vec<TokenId>,
    pub exact: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TokenizerDiagnosticKind {
    TruncationApplied,
    PromptTooLong,
    PendingPartial,
    Unsupported,
    MemoryPressure,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizerDiagnostic {
    pub kind: TokenizerDiagnosticKind,
    pub message: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TokenizerObservationKind {
    Loaded,
    CompatibilityChecked,
    EncodeRequested,
    EncodeCompleted,
    EncodeFailed,
    DecodeRequested,
    DecodeCompleted,
    DecodeFailed,
    StreamingDecodeChunk,
    StreamingDecodePendingPartial,
    PromptTooLong,
    TruncationApplied,
    MemoryPressure,
    ImplementationUnavailable,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizerObservation {
    pub kind: TokenizerObservationKind,
    pub tokenizer: Option<TokenizerId>,
    pub token_count: Option<usize>,
    pub message: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TokenizerObserver {
    observations: Vec<TokenizerObservation>,
}

impl TokenizerObserver {
    pub fn observe(
        &mut self,
        kind: TokenizerObservationKind,
        tokenizer: Option<TokenizerId>,
        token_count: Option<usize>,
        message: impl Into<String>,
    ) {
        self.observations.push(TokenizerObservation {
            kind,
            tokenizer,
            token_count,
            message: message.into(),
        });
    }

    pub fn observations(&self) -> &[TokenizerObservation] {
        &self.observations
    }
}

pub trait Tokenizer {
    fn metadata(&self) -> &TokenizerMetadata;
    fn encode(&self, input: EncodeInput) -> Result<EncodeOutput, TokenizerError>;
    fn decode(&self, input: DecodeInput) -> Result<DecodeOutput, TokenizerError>;

    fn batch_encode(&self, input: BatchEncodeInput) -> BatchEncodeOutput {
        BatchEncodeOutput {
            outputs: input
                .inputs
                .into_iter()
                .map(|mut item| {
                    item.padding = input.padding;
                    self.encode(item)
                })
                .collect(),
        }
    }

    fn streaming_decode(
        &self,
        state: StreamingDecodeState,
        token_ids: Vec<TokenId>,
        flush: bool,
    ) -> Result<DecodeOutput, TokenizerError> {
        let mut output = self.decode(DecodeInput {
            token_ids,
            skip_special_tokens: true,
            clean_up_tokenization_spaces: false,
            streaming_state: Some(state),
        })?;
        if flush {
            output.pending_partial_state = None;
        }
        Ok(output)
    }

    fn resolve_stop_sequence(&self, text: &str) -> Result<TokenStopPattern, TokenizerError> {
        let output = self.encode(EncodeInput::new(text))?;
        Ok(TokenStopPattern {
            text: text.into(),
            token_ids: output.token_ids,
            exact: true,
        })
    }
}

#[derive(Clone, Debug)]
pub struct RuntimeTokenizer<T> {
    implementation: T,
    artifacts: TokenizerArtifactSet,
}

impl<T: Tokenizer> RuntimeTokenizer<T> {
    pub fn new(implementation: T, artifacts: TokenizerArtifactSet) -> Self {
        Self {
            implementation,
            artifacts,
        }
    }

    pub fn load(
        &self,
        registry: &InferenceArtifactRegistry,
        observer: &mut TokenizerObserver,
    ) -> Result<(), TokenizerError> {
        self.artifacts.validate_registered(registry)?;
        self.implementation.metadata().validate()?;
        observer.observe(
            TokenizerObservationKind::Loaded,
            Some(self.implementation.metadata().id.clone()),
            None,
            "tokenizer loaded",
        );
        Ok(())
    }

    pub fn validate_for_model(
        &self,
        model: &ModelManifest,
        compatibility: &TokenizerCompatibility,
        observer: &mut TokenizerObserver,
    ) -> Result<(), TokenizerError> {
        validate_model_references_tokenizer(model, &self.artifacts)?;
        self.implementation
            .metadata()
            .validate_compatibility(compatibility)?;
        observer.observe(
            TokenizerObservationKind::CompatibilityChecked,
            Some(self.implementation.metadata().id.clone()),
            None,
            "tokenizer compatibility checked",
        );
        Ok(())
    }

    pub fn encode(
        &self,
        input: EncodeInput,
        observer: &mut TokenizerObserver,
    ) -> Result<EncodeOutput, TokenizerError> {
        observer.observe(
            TokenizerObservationKind::EncodeRequested,
            Some(self.implementation.metadata().id.clone()),
            None,
            "encode requested",
        );
        let result = self.implementation.encode(input).and_then(|output| {
            output.validate()?;
            Ok(output)
        });
        match &result {
            Ok(output) => observer.observe(
                TokenizerObservationKind::EncodeCompleted,
                Some(self.implementation.metadata().id.clone()),
                Some(output.token_count),
                "encode completed",
            ),
            Err(TokenizerError::PromptTooLong { .. }) => observer.observe(
                TokenizerObservationKind::PromptTooLong,
                Some(self.implementation.metadata().id.clone()),
                None,
                "prompt too long",
            ),
            Err(_) => observer.observe(
                TokenizerObservationKind::EncodeFailed,
                Some(self.implementation.metadata().id.clone()),
                None,
                "encode failed",
            ),
        }
        result
    }

    pub fn decode(
        &self,
        input: DecodeInput,
        observer: &mut TokenizerObserver,
    ) -> Result<DecodeOutput, TokenizerError> {
        observer.observe(
            TokenizerObservationKind::DecodeRequested,
            Some(self.implementation.metadata().id.clone()),
            None,
            "decode requested",
        );
        let result = self.implementation.decode(input);
        match &result {
            Ok(output) => observer.observe(
                TokenizerObservationKind::DecodeCompleted,
                Some(self.implementation.metadata().id.clone()),
                Some(output.consumed_token_count),
                "decode completed",
            ),
            Err(_) => observer.observe(
                TokenizerObservationKind::DecodeFailed,
                Some(self.implementation.metadata().id.clone()),
                None,
                "decode failed",
            ),
        }
        result
    }

    pub fn memory_requests(&self) -> Vec<MemoryAllocationRequest> {
        self.artifacts.memory_requests()
    }
}

#[derive(Clone, Debug)]
pub struct FixtureTokenizer {
    metadata: TokenizerMetadata,
}

impl FixtureTokenizer {
    pub fn new(metadata: TokenizerMetadata) -> Self {
        Self { metadata }
    }
}

impl Tokenizer for FixtureTokenizer {
    fn metadata(&self) -> &TokenizerMetadata {
        &self.metadata
    }

    fn encode(&self, input: EncodeInput) -> Result<EncodeOutput, TokenizerError> {
        if input.return_offsets && !self.metadata.supports_offsets {
            return Err(TokenizerError::OffsetsUnsupported);
        }
        if !matches!(input.padding, PaddingPolicy::None)
            && self.metadata.special_token(SpecialTokenKind::Pad).is_none()
        {
            return Err(TokenizerError::PaddingTokenMissing);
        }
        let mut token_ids = input
            .text
            .bytes()
            .map(|byte| TokenId::from(byte) + 1)
            .collect::<Vec<_>>();
        if input.add_special_tokens {
            if let Some(bos) = self.metadata.special_token(SpecialTokenKind::Bos) {
                token_ids.insert(0, bos.id);
            }
            if let Some(eos) = self.metadata.special_token(SpecialTokenKind::Eos) {
                token_ids.push(eos.id);
            }
        }
        let limit = input
            .max_tokens
            .or(self.metadata.model_max_length.map(|value| value as usize));
        let mut diagnostics = Vec::new();
        if let Some(limit) = limit
            && token_ids.len() > limit
        {
            match input.truncation {
                TruncationPolicy::None => {
                    return Err(TokenizerError::PromptTooLong {
                        token_count: token_ids.len(),
                        limit,
                    });
                }
                TruncationPolicy::Left | TruncationPolicy::ClientPolicy => {
                    let offset = token_ids.len() - limit;
                    token_ids = token_ids[offset..].to_vec();
                    diagnostics.push(TokenizerDiagnostic {
                        kind: TokenizerDiagnosticKind::TruncationApplied,
                        message: "left truncation applied".into(),
                    });
                }
                TruncationPolicy::Right | TruncationPolicy::ModelDefault => {
                    token_ids.truncate(limit);
                    diagnostics.push(TokenizerDiagnostic {
                        kind: TokenizerDiagnosticKind::TruncationApplied,
                        message: "right truncation applied".into(),
                    });
                }
                TruncationPolicy::Middle => {
                    let left = limit / 2;
                    let right = limit - left;
                    let mut truncated = token_ids[..left].to_vec();
                    truncated.extend_from_slice(&token_ids[token_ids.len() - right..]);
                    token_ids = truncated;
                    diagnostics.push(TokenizerDiagnostic {
                        kind: TokenizerDiagnosticKind::TruncationApplied,
                        message: "middle truncation applied".into(),
                    });
                }
            }
        }
        let offsets = input.return_offsets.then(|| {
            (0..token_ids.len())
                .map(|index| TokenOffset {
                    byte_start: index as u32,
                    byte_end: index as u32 + 1,
                    char_start: Some(index as u32),
                    char_end: Some(index as u32 + 1),
                })
                .collect()
        });
        Ok(EncodeOutput {
            token_count: token_ids.len(),
            attention_mask: Some(vec![1; token_ids.len()]),
            token_type_ids: self
                .metadata
                .supports_token_type_ids
                .then(|| vec![0; token_ids.len()]),
            token_ids,
            offsets,
            diagnostics,
        })
    }

    fn decode(&self, input: DecodeInput) -> Result<DecodeOutput, TokenizerError> {
        for id in &input.token_ids {
            if !self.metadata.token_id_range.contains(*id) {
                return Err(TokenizerError::InvalidTokenId { token_id: *id });
            }
        }
        let special_ids = self
            .metadata
            .special_tokens
            .iter()
            .filter(|token| input.skip_special_tokens && token.skipped_during_decode)
            .map(|token| token.id)
            .collect::<BTreeSet<_>>();
        let mut bytes = input.streaming_state.unwrap_or_default().pending_bytes;
        bytes.extend(
            input
                .token_ids
                .iter()
                .filter(|id| !special_ids.contains(id))
                .filter_map(|id| id.checked_sub(1))
                .filter_map(|id| u8::try_from(id).ok()),
        );
        let text = String::from_utf8(bytes).map_err(|error| {
            if self.metadata.byte_fallback {
                TokenizerError::DecodePendingPartial
            } else {
                TokenizerError::InvalidUtf8 {
                    message: error.to_string(),
                }
            }
        })?;
        Ok(DecodeOutput {
            text,
            consumed_token_count: input.token_ids.len(),
            pending_partial_state: None,
            diagnostics: Vec::new(),
        })
    }
}

pub fn tokenizer_memory_feasibility(
    metadata: &TokenizerMetadata,
    manager: &MemoryManager,
    placement: MemoryPlacement,
) -> Result<crate::MemoryFeasibility, TokenizerError> {
    let vocabulary_bytes = u64::from(metadata.vocabulary_size)
        .checked_mul(8)
        .and_then(|value| value.checked_add(u64::from(metadata.added_token_count) * 32))
        .ok_or(TokenizerError::MemoryAllocationFailed)?;
    Ok(manager.feasibility(&MemoryAllocationRequest::new(
        MemoryAllocationClass::TokenizerArtifact,
        vocabulary_bytes,
        placement,
        MemoryAllocationOwner::InferenceArtifact(metadata.artifact.as_str().into()),
    )))
}

pub fn tokenizer_component_artifact_reference(
    component_id: impl Into<String>,
    digest: ComponentDigest,
) -> Result<InferenceArtifactReference, ComponentError> {
    InferenceArtifactReference::new(InferenceArtifactKind::Tokenizer, component_id, digest)
}

fn validate_model_references_tokenizer(
    model: &ModelManifest,
    artifacts: &TokenizerArtifactSet,
) -> Result<(), TokenizerError> {
    let Some(reference) = &model.tokenizer else {
        return Err(TokenizerError::TokenizerArtifactMissing {
            artifact: artifacts.tokenizer.id.as_str().into(),
        });
    };
    let Some(part) = model.parts.get(reference) else {
        return Err(TokenizerError::TokenizerArtifactMissing {
            artifact: reference.clone(),
        });
    };
    if part.kind != ModelArtifactKind::Tokenizer {
        return Err(TokenizerError::InvalidTokenizerArtifact {
            artifact: reference.clone(),
        });
    }
    if part.digest != artifacts.tokenizer.digest {
        return Err(TokenizerError::TokenizerIncompatibleWithModel);
    }
    Ok(())
}

fn validate_identity(value: &str, label: &str) -> Result<(), TokenizerError> {
    if value.trim().is_empty()
        || value.contains('/')
        || value.contains('\\')
        || value.contains(':')
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return Err(TokenizerError::InvalidTokenizerArtifact {
            artifact: format!("{label}:{value}"),
        });
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TokenizerError {
    TokenizerArtifactMissing { artifact: String },
    InvalidTokenizerArtifact { artifact: String },
    TokenizerIncompatibleWithModel,
    UnsupportedTokenizerFamily { family: String },
    InvalidTokenId { token_id: TokenId },
    UnknownToken { token: String },
    InvalidUtf8 { message: String },
    DecodePendingPartial,
    OffsetsUnsupported,
    PaddingTokenMissing,
    TruncationRequired { token_count: usize, limit: usize },
    TruncationForbidden,
    PromptTooLong { token_count: usize, limit: usize },
    BatchInputInvalid { message: String },
    SpecialTokenMissing,
    SpecialTokenConflict,
    VocabularyMismatch { expected: u32, actual: u32 },
    AddedTokenMismatch { expected: u32, actual: u32 },
    MemoryAllocationFailed,
    StreamingStateInvalid,
    ImplementationUnavailable,
    UnsupportedTokenTypeIds,
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TokenizerArtifactMissing { artifact } => {
                write!(f, "tokenizer artifact '{artifact}' is missing")
            }
            Self::InvalidTokenizerArtifact { artifact } => {
                write!(f, "tokenizer artifact '{artifact}' is invalid")
            }
            Self::TokenizerIncompatibleWithModel => {
                f.write_str("tokenizer is incompatible with model artifact")
            }
            Self::UnsupportedTokenizerFamily { family } => {
                write!(f, "unsupported tokenizer family '{family}'")
            }
            Self::InvalidTokenId { token_id } => write!(f, "invalid token id {token_id}"),
            Self::UnknownToken { token } => write!(f, "unknown token '{token}'"),
            Self::InvalidUtf8 { message } => write!(f, "invalid utf-8 during decode: {message}"),
            Self::DecodePendingPartial => f.write_str("decode has pending partial output"),
            Self::OffsetsUnsupported => f.write_str("token offsets are unsupported"),
            Self::PaddingTokenMissing => f.write_str("padding token is missing"),
            Self::TruncationRequired { token_count, limit } => write!(
                f,
                "truncation is required for {token_count} tokens with limit {limit}"
            ),
            Self::TruncationForbidden => f.write_str("truncation is forbidden"),
            Self::PromptTooLong { token_count, limit } => {
                write!(f, "prompt has {token_count} tokens, limit is {limit}")
            }
            Self::BatchInputInvalid { message } => write!(f, "batch input invalid: {message}"),
            Self::SpecialTokenMissing => f.write_str("required special token is missing"),
            Self::SpecialTokenConflict => f.write_str("special token conflict"),
            Self::VocabularyMismatch { expected, actual } => {
                write!(
                    f,
                    "vocabulary mismatch: expected {expected}, actual {actual}"
                )
            }
            Self::AddedTokenMismatch { expected, actual } => {
                write!(
                    f,
                    "added token mismatch: expected {expected}, actual {actual}"
                )
            }
            Self::MemoryAllocationFailed => f.write_str("tokenizer memory allocation failed"),
            Self::StreamingStateInvalid => f.write_str("streaming decode state is invalid"),
            Self::ImplementationUnavailable => f.write_str("tokenizer implementation unavailable"),
            Self::UnsupportedTokenTypeIds => f.write_str("token type ids are unsupported"),
        }
    }
}

impl Error for TokenizerError {}
