//! One-shot and chat-turn inference pipelines built entirely from the
//! Runtime Inference API (`magnetar_runtime::inference_api` and friends).
//!
//! Honesty note (see the module doc comment in `main.rs` for the full
//! boundary statement): `magnetar-runtime` today is a contracts/validation
//! layer, not an end-to-end inference engine. There is no Provider-backed
//! forward pass wired in yet, so [`placeholder_logits`] below supplies a
//! constant all-zero logits vector for every decode step, and
//! [`fixture_tokenizer_metadata`] backs a [`FixtureTokenizer`] -- a
//! placeholder byte-based tokenizer (`token_id = byte + 1`), not a real
//! BPE/SentencePiece tokenizer. This module calls the real Runtime
//! Session/Generation/Tokenizer API surface end-to-end (session creation,
//! tokenization, generation admission, the Generation Contract decode loop,
//! decoding) -- it does not fabricate success or bypass Runtime validation
//! -- but the *text* it prints is not meaningful model output.

use magnetar_runtime::{
    AdmissionState, BatchingPolicy, CacheUsageSummary, ChatMessage, ChatTemplateFormatter,
    CliBoundaryError, DecodeInput, FixtureTokenizer, GenerationModelReference,
    GenerationParameters, GenerationRequestId, GenerationTokenizerReference, InferenceApiError,
    InferenceApiObserver, InferenceSessionId, MODEL_ARTIFACT_SCHEMA_VERSION, ModelArchitecture,
    ModelArtifactId, ModelArtifactKind, ModelManifest, ModelName, ModelRef, ModelRevision,
    PromptInput, ReferenceCpuProvider, Runtime, RuntimeGenerationExecutionEvidence,
    RuntimeGenerationExecutor, RuntimeGenerationStep, SamplingPolicy, SessionCreationRequest,
    SessionMemoryBudget, SessionPolicy, SpecialToken, SpecialTokenKind, StopConditions,
    StreamingMode, TokenId, TokenIdRange, TokenizationRequest, TokenizerArtifactId,
    TokenizerFamily, TokenizerId, TokenizerMetadata, TokenizerRevision, cancel_inference_session,
    close_inference_session, create_inference_session, create_one_shot_session, decode_tokens,
    prepare_generation, submit_generation, tokenize_prompt_input,
};
use magnetar_runtime::{ModelDigest, build_generation_request, run_generation_loop};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

/// Maximum number of decode steps a one-shot `magnetar run`/`chat` turn
/// requests. Kept small: with placeholder all-zero logits every decode step
/// is deterministic and uninteresting past a handful of tokens.
pub const DEFAULT_MAX_NEW_TOKENS: usize = 8;

/// Builds the fixture [`TokenizerMetadata`] this CLI uses until a real
/// Tokenizer artifact pipeline exists. Mirrors
/// `magnetar-runtime`'s own internal test fixture
/// (`generation_tokenizer_metadata` in `magnetar-runtime/src/tests.rs`) but
/// is written independently here since that helper is `#[cfg(test)]`-gated
/// and not part of the public API.
pub fn fixture_tokenizer_metadata() -> TokenizerMetadata {
    TokenizerMetadata {
        id: TokenizerId::new("magnetar-cli-fixture").expect("valid fixture tokenizer id"),
        artifact: TokenizerArtifactId::new("magnetar-cli-fixture-tokenizer")
            .expect("valid fixture tokenizer artifact id"),
        digest: ModelDigest::sha256(b"magnetar-cli-fixture-tokenizer"),
        family: TokenizerFamily::new("fixture").expect("valid fixture tokenizer family"),
        revision: TokenizerRevision::new("1.0.0").expect("valid fixture tokenizer revision"),
        vocabulary_size: 256,
        added_token_count: 2,
        token_id_range: TokenIdRange::new(1, 300),
        model_max_length: Some(64),
        special_tokens: vec![SpecialToken::new(SpecialTokenKind::Eos, "<eos>", 299)],
        additional_special_tokens: vec![SpecialToken::new(SpecialTokenKind::Stop, "<stop>", 298)],
        byte_fallback: false,
        normalization: None,
        pre_tokenizer: None,
        supports_offsets: true,
        supports_token_type_ids: false,
        supports_browser: true,
    }
}

/// Builds a minimal, honestly-labeled fixture [`ModelManifest`] for
/// `label` (typically a `ModelRef`'s string form, or a canonicalized local
/// model file path -- see `commands::cmd_model_load`), used only to
/// exercise the real `magnetar_runtime::load_model` call end to end
/// (§6/§42 "Call Runtime model loading"). This CLI increment has no
/// persistent Model Artifact manifest storage -- see this module's top doc
/// comment for the same honesty note about the fixture tokenizer -- so
/// every input maps to the same fixture shape, distinguished only by its
/// digest (derived from `label` itself). Real Model Artifact trust
/// evaluation (`ModelTrustStore::evaluate`, called by
/// `commands::cmd_model_load`) still runs against this manifest -- nothing
/// here fabricates a trusted result.
pub fn fixture_model_manifest(label: &str) -> ModelManifest {
    ModelManifest {
        schema_version: MODEL_ARTIFACT_SCHEMA_VERSION,
        id: ModelArtifactId::new(
            ModelArtifactKind::ModelWeights,
            ModelName::new("magnetar-cli-fixture").expect("valid fixture model name"),
            ModelRevision::new("1").expect("valid fixture model revision"),
            ModelDigest::sha256(format!("magnetar-cli-fixture:{label}").as_bytes()),
        ),
        architecture: ModelArchitecture::new("magnetar-cli-fixture", "fixture"),
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

/// Placeholder byte the fixture pipeline decodes to for every generated
/// token (`'.'`), chosen only because it is ASCII-printable and therefore
/// always valid UTF-8 -- the fixture tokenizer's decode step maps
/// `token_id - 1` straight to a byte with no further validation, so an
/// arbitrary favored logits index could otherwise decode to an invalid
/// lone byte.
const PLACEHOLDER_BYTE_INDEX: usize = b'.' as usize;

/// TODO(real-model-execution): replace with a Provider-backed forward pass
/// once model execution graph wiring lands. Until then, every decode step
/// receives a placeholder logits vector that deterministically favors
/// [`PLACEHOLDER_BYTE_INDEX`] -- generated tokens are deterministic
/// placeholders, not real model output.
fn placeholder_logits(vocabulary_size: usize) -> Vec<f32> {
    let mut logits = vec![0.0f32; vocabulary_size];
    if let Some(favored) = logits.get_mut(PLACEHOLDER_BYTE_INDEX) {
        *favored = 1.0;
    }
    logits
}

#[derive(Clone)]
struct CliPlaceholderGenerationExecutor {
    vocabulary_size: usize,
}

impl RuntimeGenerationExecutor for CliPlaceholderGenerationExecutor {
    fn execute_generation_step(
        &self,
        _runtime: &Runtime,
        _request: &magnetar_runtime::GenerationRequest,
        _generated_tokens: &[TokenId],
    ) -> Result<RuntimeGenerationStep, InferenceApiError> {
        Ok(RuntimeGenerationStep::new(
            placeholder_logits(self.vocabulary_size),
            RuntimeGenerationExecutionEvidence::complete(),
        ))
    }
}

fn build_cli_runtime() -> Result<Runtime, CliBoundaryError> {
    Runtime::builder()
        .register_provider(Arc::new(ReferenceCpuProvider::new()))
        .generation_executor(Arc::new(CliPlaceholderGenerationExecutor {
            vocabulary_size: fixture_tokenizer_metadata().vocabulary_size as usize,
        }))
        .build()
        .map_err(|error| CliBoundaryError::CliRuntimeUnavailable {
            reason: error.to_string(),
        })
}

/// Builds a [`SessionCreationRequest`] against the fixture tokenizer and a
/// `LoadedModelContext` reference naming the caller-supplied [`ModelRef`].
/// No real model is loaded: `magnetar-runtime` has no Provider-backed model
/// execution graph wired in yet (see the module doc comment above), so this
/// reference exists only to carry the CLI-resolved model name through the
/// Runtime Inference API's session/generation request shape.
fn session_creation_request(model_ref: &ModelRef) -> SessionCreationRequest {
    let metadata = fixture_tokenizer_metadata();
    let mut allowed_capabilities = BTreeSet::new();
    allowed_capabilities.insert("generation".to_string());
    SessionCreationRequest {
        model: GenerationModelReference::LoadedModelContext(format!(
            "magnetar-cli-fixture:{model_ref}"
        )),
        tokenizer: GenerationTokenizerReference {
            tokenizer_id: metadata.id.clone(),
            metadata,
        },
        generation_defaults: GenerationParameters::greedy(),
        policy: SessionPolicy::default(),
        memory: SessionMemoryBudget::default(),
        allowed_capabilities,
        correlation_id: None,
        created_at_millis: 0,
    }
}

/// Minimal CLI-owned chat template formatter, satisfying the Runtime's
/// authorized [`ChatTemplateFormatter`] contract for `magnetar chat`'s
/// multi-turn `PromptInput::ChatMessages` path (see
/// [`ChatSession::turn`]). A simple, pure, deterministic `"{role}: {content}"`
/// join per message -- no filesystem or network access of any kind, which
/// is what makes "Runtime does not fetch templates from arbitrary
/// files/network during inference" (§16) true by construction for this
/// formatter: it has no code path that could reach either. This is the
/// concrete difference between "CLI pre-renders" (the existing
/// `PromptInput::PlainText` path used by `run` and by a chat session's
/// first turn) and "Runtime applies an authorized template" (this
/// formatter, invoked from inside `tokenize_prompt_input`, for chat turns
/// after the first).
pub struct CliChatTemplateFormatter;

impl ChatTemplateFormatter for CliChatTemplateFormatter {
    fn format(&self, messages: &[ChatMessage]) -> Result<String, InferenceApiError> {
        let mut rendered = String::new();
        for message in messages {
            rendered.push_str(&message.role);
            rendered.push_str(": ");
            rendered.push_str(&message.content);
            rendered.push('\n');
        }
        Ok(rendered)
    }
}

/// Runs one full pipeline turn: tokenize `prompt_input` (rendering it
/// through `chat_formatter` first when it is [`PromptInput::ChatMessages`]),
/// build and prepare a generation request against `session`, admit it into
/// a fresh Continuous Batch, drive [`run_generation_loop`] to completion
/// through the Runtime-registered placeholder executor while recording every
/// observation into `observer`, and decode the resulting token IDs back to
/// text. Every step calls the real Runtime Inference API function of the same
/// name; nothing here re-implements tokenization or generation.
fn run_pipeline_turn(
    runtime: &mut Runtime,
    session: &InferenceSessionId,
    request_id: &str,
    prompt_input: PromptInput,
    chat_formatter: Option<&dyn ChatTemplateFormatter>,
    observer: &mut InferenceApiObserver,
) -> Result<String, CliBoundaryError> {
    let metadata = fixture_tokenizer_metadata();
    let tokenizer = FixtureTokenizer::new(metadata.clone());

    let tokenized = tokenize_prompt_input(
        &tokenizer,
        TokenizationRequest::new(prompt_input),
        chat_formatter,
    )?;

    let generation_request = build_generation_request(
        GenerationRequestId::new(request_id).map_err(InferenceApiError::from)?,
        Some(session.clone()),
        GenerationModelReference::LoadedModelContext(format!("magnetar-cli-fixture:{request_id}")),
        GenerationTokenizerReference {
            tokenizer_id: metadata.id.clone(),
            metadata: metadata.clone(),
        },
        tokenized,
        DEFAULT_MAX_NEW_TOKENS,
        GenerationParameters::greedy(),
        StopConditions::default(),
        StreamingMode::TokenIds,
    );

    let prepared = prepare_generation(runtime, generation_request)?;

    // `allow_queueing: false` mirrors magnetar-runtime's own
    // `inference_api_one_shot_pipeline_uses_session_tokenizer_and_generation_contracts`
    // test fixture: with the default policy a single admitted operation is
    // still routed through `Queued` (continuous batch admission policy
    // enqueues admitted operations) rather than `Accepted`, which is not a
    // rejection -- but this CLI wants a deterministic, immediate admission
    // for a one-shot/interactive turn rather than caller-managed queueing.
    let batch = runtime.create_continuous_batch(BatchingPolicy {
        allow_queueing: false,
        ..BatchingPolicy::default()
    });
    let (admission, _slot) = submit_generation(runtime, &batch, &prepared)?;
    if !matches!(admission, AdmissionState::Accepted) {
        return Err(CliBoundaryError::CliRuntimeRequestFailed(
            InferenceApiError::GenerationRejected {
                reason: format!("generation was not accepted: {admission:?}"),
            },
        ));
    }

    let result = run_generation_loop(
        runtime,
        &prepared,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated: &[TokenId]| false,
        observer,
    )?;

    let decoded = decode_tokens(
        &tokenizer,
        DecodeInput {
            token_ids: result.output.generated_token_ids,
            skip_special_tokens: true,
            clean_up_tokenization_spaces: true,
            streaming_state: None,
        },
    )?;

    Ok(decoded.text)
}

/// `magnetar run`'s full pipeline: build a Runtime, create a one-shot
/// session, run a single pipeline turn, and close the session. Returns the
/// decoded (placeholder) text and the [`InferenceApiObserver`] recorded
/// during that turn (see `render::print_generation_observations`, which
/// renders it -- a post-hoc replay of the observation trail, not true
/// incremental streaming; see that function's doc comment) on success.
pub fn one_shot(
    model_ref: &ModelRef,
    prompt: &str,
) -> Result<(String, InferenceApiObserver), CliBoundaryError> {
    let mut runtime = build_cli_runtime()?;

    let session = create_one_shot_session(&mut runtime, session_creation_request(model_ref))?;

    let mut observer = InferenceApiObserver::new();
    let outcome = run_pipeline_turn(
        &mut runtime,
        &session,
        "cli-run-1",
        PromptInput::PlainText(prompt.to_string()),
        None,
        &mut observer,
    );

    // Always attempt to close the session, but surface the pipeline error
    // (if any) rather than a close failure that only occurs because the
    // pipeline already failed.
    let close_result = close_inference_session(&mut runtime, &session);
    let text = outcome?;
    close_result?;
    Ok((text, observer))
}

/// A `magnetar chat` session: one persistent Runtime Inference Session
/// reused across turns, plus a CLI-owned transcript kept entirely outside
/// the Runtime session. This is the concrete implementation of "CLI Session
/// Metadata Is Separate From Runtime Session State"
/// (`specs/cli-boundary/spec.md`): the Runtime session never sees the
/// transcript, only per-turn prompt/generation requests.
pub struct ChatSession {
    runtime: Runtime,
    session: InferenceSessionId,
    /// CLI-owned transcript: (role, text) pairs. Never sent to Runtime as a
    /// whole -- only the current turn's prompt text is.
    transcript: Vec<(String, String)>,
    next_turn: u64,
}

impl ChatSession {
    pub fn open(model_ref: &ModelRef) -> Result<Self, CliBoundaryError> {
        let mut runtime = build_cli_runtime()?;
        let session = create_inference_session(&mut runtime, session_creation_request(model_ref))?;
        Ok(Self {
            runtime,
            session,
            transcript: Vec::new(),
            next_turn: 0,
        })
    }

    /// Runs one chat turn and returns the assistant's (placeholder) reply
    /// plus the [`InferenceApiObserver`] recorded during it.
    ///
    /// Chat Template Boundary (§16): the first turn sends
    /// [`PromptInput::PlainText`] directly -- the same "CLI pre-renders"
    /// path `pipeline::one_shot` uses for `run`. Every turn after the
    /// first instead sends the CLI-owned transcript (plus this turn's
    /// line) as [`PromptInput::ChatMessages`] through
    /// [`CliChatTemplateFormatter`], so Runtime applies the authorized
    /// chat template via `tokenize_prompt_input` rather than the CLI
    /// joining strings itself -- the concrete "Runtime applies authorized
    /// chat template" half of the boundary, made explicit by this branch.
    pub fn turn(
        &mut self,
        user_line: &str,
    ) -> Result<(String, InferenceApiObserver), CliBoundaryError> {
        self.next_turn += 1;
        let request_id = format!("cli-chat-{}", self.next_turn);
        let mut observer = InferenceApiObserver::new();

        let reply = if self.transcript.is_empty() {
            run_pipeline_turn(
                &mut self.runtime,
                &self.session,
                &request_id,
                PromptInput::PlainText(user_line.to_string()),
                None,
                &mut observer,
            )?
        } else {
            let mut messages: Vec<ChatMessage> = self
                .transcript
                .iter()
                .map(|(role, text)| ChatMessage::new(role.clone(), text.clone()))
                .collect();
            messages.push(ChatMessage::new("user", user_line));
            let formatter = CliChatTemplateFormatter;
            run_pipeline_turn(
                &mut self.runtime,
                &self.session,
                &request_id,
                PromptInput::ChatMessages(messages),
                Some(&formatter),
                &mut observer,
            )?
        };

        self.transcript
            .push(("user".to_string(), user_line.to_string()));
        self.transcript
            .push(("assistant".to_string(), reply.clone()));
        Ok((reply, observer))
    }

    /// CLI-owned transcript, never sent to Runtime as a whole.
    pub fn transcript(&self) -> &[(String, String)] {
        &self.transcript
    }

    /// Cancellation (§19 "CLI Cancellation Calls Runtime Cancellation"):
    /// calls the real `magnetar_runtime::cancel_inference_session`. After
    /// this returns `Ok(())` the underlying Runtime session is in
    /// `SessionLifecycleState::Cancelled` and rejects further
    /// [`Self::turn`] calls -- callers should stop the interactive loop
    /// afterward rather than calling [`Self::close`] (a cancelled session
    /// does not need a separate close). This is the CLI's cancellation call
    /// path for Runtime-owned (inference) work; CLI-owned file/Git/network/
    /// tool work is not tracked by `ChatSession` and has nothing to cancel
    /// here since every such call in this synchronous CLI already runs to
    /// completion or structured failure before control returns to the
    /// caller (see `commands::cmd_chat`'s "cancel" REPL command for the
    /// user-facing entry point).
    pub fn cancel(&mut self) -> Result<(), CliBoundaryError> {
        cancel_inference_session(&mut self.runtime, &self.session)?;
        Ok(())
    }

    pub fn close(mut self) -> Result<(), CliBoundaryError> {
        close_inference_session(&mut self.runtime, &self.session)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnetar_runtime::{ModelInstanceUnloadPolicy, unload_model_instance};

    #[test]
    fn one_shot_pipeline_succeeds_end_to_end_for_a_sample_prompt() {
        let model_ref = ModelRef::new("qwen-test").unwrap();
        let (text, _observer) = one_shot(&model_ref, "hello there").unwrap();
        // Placeholder logits are all-zero, so decoded text is not
        // meaningful -- this test only proves the pipeline runs end to end
        // through the real Runtime Session/Generation/Tokenizer API without
        // bypassing any of it.
        let _ = text;
    }

    /// Streaming (§18): asserts the observation trail's order and count
    /// for a known, small `DEFAULT_MAX_NEW_TOKENS`. Placeholder logits
    /// deterministically favor a non-EOS/non-stop byte every step (see
    /// `placeholder_logits`), so generation always runs to exactly
    /// `DEFAULT_MAX_NEW_TOKENS` `TokenGenerated` observations rather than
    /// stopping early.
    #[test]
    fn one_shot_pipeline_emits_observation_trail_in_runtime_order() {
        use magnetar_runtime::InferenceApiObservationKind;

        let model_ref = ModelRef::new("qwen-test").unwrap();
        let (_text, observer) = one_shot(&model_ref, "hi").unwrap();
        let kinds: Vec<_> = observer.observations().iter().map(|o| o.kind).collect();

        assert_eq!(
            kinds.first(),
            Some(&InferenceApiObservationKind::GenerationStarted)
        );
        assert_eq!(
            kinds.last(),
            Some(&InferenceApiObservationKind::StreamClosed)
        );
        let token_generated_count = kinds
            .iter()
            .filter(|kind| **kind == InferenceApiObservationKind::TokenGenerated)
            .count();
        assert_eq!(token_generated_count, DEFAULT_MAX_NEW_TOKENS);

        // Preserve Runtime event order: GenerationCompleted must follow
        // the final TokenGenerated, and StreamClosed must follow that.
        let last_token_generated = kinds
            .iter()
            .rposition(|kind| *kind == InferenceApiObservationKind::TokenGenerated)
            .unwrap();
        let generation_completed = kinds
            .iter()
            .position(|kind| *kind == InferenceApiObservationKind::GenerationCompleted)
            .unwrap();
        assert!(generation_completed > last_token_generated);
    }

    /// Chat Template Boundary (§16): turn 1 sends `PromptInput::PlainText`
    /// directly (`self.transcript` is empty at that point); turn 2 already
    /// takes the `PromptInput::ChatMessages` + `CliChatTemplateFormatter`
    /// path (`self.transcript` is non-empty by then). If that wiring
    /// dropped the formatter, `tokenize_prompt_input` would surface
    /// `InferenceApiError::PolicyDenied` (see the direct-call test below)
    /// and turn 2 would fail here instead of succeeding. A third+ turn is
    /// deliberately not exercised in this test: the fixture tokenizer's
    /// shared `model_max_length` (64) plus the fixed 8-token placeholder
    /// reply per turn means the rendered transcript approaches that limit
    /// by the third turn regardless of how short the user's lines are --
    /// a pipeline-wide fixture constant, not something specific to chat
    /// template rendering.
    #[test]
    fn chat_session_keeps_transcript_cli_side_and_reuses_one_runtime_session() {
        let model_ref = ModelRef::new("qwen-test").unwrap();
        let mut chat = ChatSession::open(&model_ref).unwrap();
        chat.turn("first line").unwrap();
        chat.turn("second line").unwrap();
        assert_eq!(chat.transcript().len(), 4);
        assert_eq!(chat.transcript()[0].0, "user");
        assert_eq!(chat.transcript()[1].0, "assistant");
        chat.close().unwrap();
    }

    #[test]
    fn cli_chat_template_formatter_renders_a_known_transcript_deterministically() {
        let messages = vec![
            ChatMessage::new("user", "hello"),
            ChatMessage::new("assistant", "hi there"),
            ChatMessage::new("user", "how are you"),
        ];
        let formatter = CliChatTemplateFormatter;
        let expected = "user: hello\nassistant: hi there\nuser: how are you\n";
        assert_eq!(formatter.format(&messages).unwrap(), expected);
        // Deterministic: formatting the same input twice yields the same
        // output.
        assert_eq!(formatter.format(&messages).unwrap(), expected);
    }

    /// Proves `tokenize_prompt_input` (the real Runtime function) rejects
    /// `PromptInput::ChatMessages` with no chat formatter -- the Runtime
    /// boundary this CLI relies on, verified directly rather than assumed.
    #[test]
    fn tokenize_chat_messages_without_formatter_is_policy_denied() {
        let metadata = fixture_tokenizer_metadata();
        let tokenizer = FixtureTokenizer::new(metadata);
        let messages = vec![ChatMessage::new("user", "hello")];
        let error = tokenize_prompt_input(
            &tokenizer,
            TokenizationRequest::new(PromptInput::ChatMessages(messages)),
            None,
        )
        .unwrap_err();
        assert!(matches!(error, InferenceApiError::PolicyDenied { .. }));
    }

    #[test]
    fn model_unload_against_unregistered_ref_surfaces_structured_runtime_error() {
        use magnetar_runtime::ModelInstanceId;

        let mut runtime = Runtime::builder().build().unwrap();
        let instance = ModelInstanceId::new("cli-unregistered-instance").unwrap();
        let error = unload_model_instance(
            &mut runtime,
            &instance,
            ModelInstanceUnloadPolicy::RejectActiveUse,
        )
        .unwrap_err();
        let boundary_error = CliBoundaryError::from(error);
        assert!(boundary_error.runtime_category().is_some());
        assert!(matches!(
            boundary_error,
            CliBoundaryError::CliRuntimeRequestFailed(_)
        ));
    }

    /// §19/§29 "Test CLI cancellation calls Runtime cancellation":
    /// `ChatSession::cancel` calls the real Runtime session cancellation,
    /// and the underlying Runtime session becomes unusable afterward (a
    /// further turn is rejected by Runtime, not silently accepted).
    #[test]
    fn chat_session_cancel_calls_runtime_cancellation_and_session_becomes_unusable() {
        let model_ref = ModelRef::new("qwen-test").unwrap();
        let mut chat = ChatSession::open(&model_ref).unwrap();
        chat.turn("first line").unwrap();
        chat.cancel().unwrap();
        let error = chat.turn("after cancel").unwrap_err();
        assert!(error.runtime_category().is_some());
    }

    #[test]
    fn fixture_model_manifest_is_deterministic_for_the_same_label() {
        let a = fixture_model_manifest("qwen-test");
        let b = fixture_model_manifest("qwen-test");
        assert_eq!(a.id, b.id);
        let c = fixture_model_manifest("a-different-label");
        assert_ne!(a.id, c.id);
    }

    #[test]
    fn model_ref_rejects_path_like_input() {
        let error = ModelRef::new("../etc/passwd").unwrap_err();
        let boundary_error = CliBoundaryError::from(error);
        assert!(matches!(
            boundary_error,
            CliBoundaryError::CliRuntimeRequestFailed(_)
        ));
    }
}
