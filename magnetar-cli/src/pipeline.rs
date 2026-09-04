//! One-shot and chat-turn inference pipelines built entirely from the
//! Runtime Inference API (`magnetar_runtime::inference_api` and friends).
//!
//! `magnetar run` and `magnetar chat` now use the runtime's first native
//! fixture path. The CLI still owns prompt/context assembly, but logits come
//! from Runtime model execution evidence rather than a CLI placeholder.

use magnetar_runtime::{
    ChatMessage, ChatTemplateFormatter, CliBoundaryError, FirstNativeChatSession,
    FirstNativeRuntimeError, InferenceApiError, InferenceApiObserver, InferenceSessionId,
    MODEL_ARTIFACT_SCHEMA_VERSION, ModelArchitecture, ModelArtifactId, ModelArtifactKind,
    ModelDigest, ModelManifest, ModelName, ModelRef, ModelRevision,
    register_qwen_component_artifact, run_first_native_generation,
};

/// The real Qwen Component binary this CLI bundles for its `"qwen-test"`
/// self-test/demo alias (`reach-architecture-freeze-1` task 12.4). This is
/// deliberately the only `include_bytes!` of the Qwen Component anywhere in
/// production code now: `magnetar-runtime` itself has none, and requires an
/// embedder like this crate -- the "deployment / CLI / Component source
/// adapter" boundary the task's design calls for -- to push one explicitly
/// via `register_qwen_component_artifact` before requesting first-native
/// generation. There is currently exactly one real caller-facing "model"
/// (`run_first_native_generation`/`FirstNativeChatSession::open` both
/// hard-require `model_ref == "qwen-test"`), so bundling its Component here
/// is not a stand-in for real model loading -- it genuinely is the whole
/// feature this alias offers.
const QWEN_COMPONENT_BYTES: &[u8] = include_bytes!("../fixtures/qwen-real.component.wasm");
const QWEN_COMPONENT_MANIFEST_BYTES: &[u8] =
    include_bytes!("../fixtures/qwen-real.component.wasm.magnetar-component.yaml");

/// Pushes this CLI's bundled Qwen Component artifact to the Runtime, if it
/// has not been already. Idempotent by construction
/// (`register_qwen_component_artifact` itself no-ops past the first real
/// registration), so every call site that might need first-native
/// generation can call this unconditionally rather than tracking its own
/// "have I registered yet" state.
fn ensure_qwen_component_registered() {
    register_qwen_component_artifact(
        QWEN_COMPONENT_BYTES.to_vec(),
        QWEN_COMPONENT_MANIFEST_BYTES.to_vec(),
    );
}
#[cfg(test)]
use magnetar_runtime::{
    SpecialToken, SpecialTokenKind, TokenIdRange, TokenizerArtifactId, TokenizerFamily,
    TokenizerId, TokenizerMetadata, TokenizerRevision,
};
use std::collections::{BTreeMap, BTreeSet};

/// Maximum number of decode steps used by the native fixture path.
pub const DEFAULT_MAX_NEW_TOKENS: usize = 8;

fn first_native_runtime_error_to_api_error(error: FirstNativeRuntimeError) -> InferenceApiError {
    let reason = error.reason().to_string();
    match error.code() {
        "model-not-found" => InferenceApiError::ModelResolutionFailed { reason },
        "artifact-invalid" => InferenceApiError::ModelLoadingFailed { reason },
        "trust-rejected" => InferenceApiError::PolicyDenied { reason },
        "load-failed" => InferenceApiError::ModelLoadingFailed { reason },
        "component-load-failed" => InferenceApiError::ModelComponentUnavailable { reason },
        "plan-unavailable" => InferenceApiError::GraphPlanningFailed { reason },
        "provider-unavailable" => InferenceApiError::ProviderUnavailable { reason },
        "generation-failed" => InferenceApiError::GenerationFailed { reason },
        "generation-cancelled" => InferenceApiError::GenerationCancelled,
        _ => InferenceApiError::GenerationFailed {
            reason: format!("{}: {reason}", error.code()),
        },
    }
}

/// Builds the fixture [`TokenizerMetadata`] this CLI uses until a real
/// Tokenizer artifact pipeline exists. Mirrors
/// `magnetar-runtime`'s own internal test fixture
/// (`generation_tokenizer_metadata` in `magnetar-runtime/src/tests.rs`) but
/// is written independently here since that helper is `#[cfg(test)]`-gated
/// and not part of the public API.
///
/// Test-only: `FirstNativeChatSession::open` (task 8.3) now builds its own
/// session against the `E2eFixture`'s own tokenizer metadata directly,
/// so this CLI-local fixture no longer has a production caller -- only the
/// `tokenize_chat_messages_without_formatter_is_policy_denied` test below
/// still exercises it.
#[cfg(test)]
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

/// `magnetar run`'s full pipeline: delegate prompt execution to Runtime's
/// first-native fixture path and return the rendered output plus redacted
/// observations.
pub fn one_shot(
    model_ref: &ModelRef,
    prompt: &str,
) -> Result<(String, InferenceApiObserver), CliBoundaryError> {
    ensure_qwen_component_registered();
    let generated = run_first_native_generation(model_ref, prompt).map_err(|error| {
        CliBoundaryError::CliRuntimeRequestFailed(first_native_runtime_error_to_api_error(error))
    })?;
    Ok((generated.text, generated.observer))
}

/// A `magnetar chat` session: one persistent Runtime Inference Session
/// reused across turns (task 8.3: every turn executes through this same
/// persistent [`FirstNativeChatSession`], not a fresh one-shot Runtime per
/// turn), plus a CLI-owned transcript kept entirely outside the Runtime
/// session. This is the concrete implementation of "CLI Session Metadata Is
/// Separate From Runtime Session State" (`specs/cli-boundary/spec.md`): the
/// Runtime session never sees the transcript, only per-turn prompt/
/// generation requests.
pub struct ChatSession {
    chat: FirstNativeChatSession,
    /// CLI-owned transcript: (role, text) pairs. Never sent to Runtime as a
    /// whole -- only the current turn's prompt text is.
    transcript: Vec<(String, String)>,
    next_turn: u64,
    cancelled: bool,
}

impl ChatSession {
    pub fn open(model_ref: &ModelRef) -> Result<Self, CliBoundaryError> {
        ensure_qwen_component_registered();
        let chat = FirstNativeChatSession::open(model_ref).map_err(|error| {
            CliBoundaryError::CliRuntimeRequestFailed(first_native_runtime_error_to_api_error(
                error,
            ))
        })?;
        Ok(Self {
            chat,
            transcript: Vec::new(),
            next_turn: 0,
            cancelled: false,
        })
    }

    /// The persistent Runtime `InferenceSessionId` every turn of this chat
    /// session executes through -- stable across the whole session.
    pub fn session_id(&self) -> &InferenceSessionId {
        self.chat.session_id()
    }

    /// Runs one chat turn and returns the fixture-generated reply plus the
    /// [`InferenceApiObserver`] recorded during it.
    ///
    /// Chat Template Boundary (§16): the first turn sends
    /// `PromptInput::PlainText` directly -- the same "CLI pre-renders"
    /// path `pipeline::one_shot` uses for `run`. Every turn after the
    /// first also sends `PromptInput::PlainText`, but rendered from the
    /// CLI-owned transcript (plus this turn's line) via
    /// [`CliChatTemplateFormatter`] first -- the template is applied
    /// CLI-side, matching this boundary's "MAY" (not "SHALL") Runtime
    /// templating language; it does not itself send `PromptInput::
    /// ChatMessages` or invoke Runtime's `tokenize_prompt_input` chat
    /// templating path.
    ///
    /// Every branch executes through `self.chat` -- this chat session's one
    /// persistent Runtime, Model Instance, and `InferenceSessionId` -- so
    /// two turns of the same `ChatSession` share Runtime session identity
    /// (task 8.3), rather than each turn building and tearing down its own.
    pub fn turn(
        &mut self,
        user_line: &str,
    ) -> Result<(String, InferenceApiObserver), CliBoundaryError> {
        if self.cancelled {
            return Err(CliBoundaryError::CliRuntimeRequestFailed(
                InferenceApiError::GenerationCancelled,
            ));
        }
        self.next_turn += 1;

        let (reply, observer) = if self.transcript.is_empty() {
            let generated = self
                .chat
                .turn(user_line, DEFAULT_MAX_NEW_TOKENS)
                .map_err(|error| {
                    CliBoundaryError::CliRuntimeRequestFailed(
                        first_native_runtime_error_to_api_error(error),
                    )
                })?;
            (generated.text, generated.observer)
        } else {
            let mut messages: Vec<ChatMessage> = self
                .transcript
                .iter()
                .map(|(role, text)| ChatMessage::new(role.clone(), text.clone()))
                .collect();
            messages.push(ChatMessage::new("user", user_line));
            let formatter = CliChatTemplateFormatter;
            let rendered = formatter.format(&messages)?;
            let generated = self
                .chat
                .turn(&rendered, DEFAULT_MAX_NEW_TOKENS)
                .map_err(|error| {
                    CliBoundaryError::CliRuntimeRequestFailed(
                        first_native_runtime_error_to_api_error(error),
                    )
                })?;
            (generated.text, generated.observer)
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
    /// calls the real `magnetar_runtime::cancel_inference_session` against
    /// this chat session's persistent `InferenceSessionId` -- the same
    /// session [`Self::turn`] executes every generation through (task 8.3),
    /// not an orphan session no turn ever touches. After this returns
    /// `Ok(())` the underlying Runtime session is in
    /// `SessionLifecycleState::Cancelled` and `self.cancelled` rejects
    /// further [`Self::turn`] calls -- callers should stop the interactive
    /// loop afterward rather than calling [`Self::close`] (a cancelled
    /// session does not need a separate close). This is the CLI's
    /// cancellation call path for Runtime-owned (inference) work; CLI-owned
    /// file/Git/network/tool work is not tracked by `ChatSession` and has
    /// nothing to cancel here since every such call in this synchronous CLI
    /// already runs to completion or structured failure before control
    /// returns to the caller (see `commands::cmd_chat`'s "cancel" REPL
    /// command for the user-facing entry point).
    pub fn cancel(&mut self) -> Result<(), CliBoundaryError> {
        self.chat.cancel().map_err(|error| {
            CliBoundaryError::CliRuntimeRequestFailed(first_native_runtime_error_to_api_error(
                error,
            ))
        })?;
        self.cancelled = true;
        Ok(())
    }

    pub fn close(self) -> Result<(), CliBoundaryError> {
        self.chat.close().map_err(|error| {
            CliBoundaryError::CliRuntimeRequestFailed(first_native_runtime_error_to_api_error(
                error,
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnetar_runtime::{
        FixtureTokenizer, ModelInstanceUnloadPolicy, PromptInput, Runtime, TokenizationRequest,
        tokenize_prompt_input, unload_model_instance,
    };

    #[test]
    fn one_shot_pipeline_uses_native_fixture_generation() {
        let model_ref = ModelRef::new("qwen-test").unwrap();
        let (text, observer) = one_shot(&model_ref, "hello there").unwrap();
        assert!(!text.is_empty());
        assert!(observer.observations().iter().any(|observation| {
            observation.kind == magnetar_runtime::InferenceApiObservationKind::ProviderExecuted
        }));
    }

    #[test]
    fn one_shot_rejects_unknown_model_ref_instead_of_ignoring_it() {
        let model_ref = ModelRef::new("different-model").unwrap();
        let error = one_shot(&model_ref, "hello").unwrap_err();
        assert!(matches!(
            error.runtime_category(),
            Some(InferenceApiError::ModelResolutionFailed { reason })
                if reason.contains("different-model")
        ));
    }

    /// Streaming (§18): asserts the observation trail's order and count
    /// for a known, small `DEFAULT_MAX_NEW_TOKENS`. Placeholder logits
    /// deterministically favor a non-EOS/non-stop byte every step (see
    /// token generation comes from the runtime-owned fixture path rather
    /// than a CLI-provided logits source.
    #[test]
    fn first_native_errors_map_to_structured_runtime_categories() {
        let cases = [
            (
                "model-not-found",
                "missing model",
                InferenceApiError::ModelResolutionFailed {
                    reason: "missing model".into(),
                },
            ),
            (
                "artifact-invalid",
                "bad artifact",
                InferenceApiError::ModelLoadingFailed {
                    reason: "bad artifact".into(),
                },
            ),
            (
                "trust-rejected",
                "untrusted",
                InferenceApiError::PolicyDenied {
                    reason: "untrusted".into(),
                },
            ),
            (
                "load-failed",
                "load failed",
                InferenceApiError::ModelLoadingFailed {
                    reason: "load failed".into(),
                },
            ),
            (
                "component-load-failed",
                "component failed",
                InferenceApiError::ModelComponentUnavailable {
                    reason: "component failed".into(),
                },
            ),
            (
                "provider-unavailable",
                "no provider",
                InferenceApiError::ProviderUnavailable {
                    reason: "no provider".into(),
                },
            ),
            (
                "plan-unavailable",
                "no plan",
                InferenceApiError::GraphPlanningFailed {
                    reason: "no plan".into(),
                },
            ),
            (
                "generation-failed",
                "bad generation",
                InferenceApiError::GenerationFailed {
                    reason: "bad generation".into(),
                },
            ),
            (
                "generation-cancelled",
                "cancelled",
                InferenceApiError::GenerationCancelled,
            ),
        ];

        for (code, reason, expected) in cases {
            assert_eq!(
                first_native_runtime_error_to_api_error(FirstNativeRuntimeError::new(code, reason)),
                expected
            );
        }
    }

    #[test]
    fn one_shot_pipeline_certifies_native_observation_trail() {
        let model_ref = ModelRef::new("qwen-test").unwrap();
        let (_text, observer) = one_shot(&model_ref, "hi").unwrap();
        assert!(observer.observations().iter().any(|observation| {
            observation.kind == magnetar_runtime::InferenceApiObservationKind::TokenGenerated
        }));
    }

    /// Chat Template Boundary (§16): turn 1 sends `PromptInput::PlainText`
    /// directly (`self.transcript` is empty at that point); turn 2 already
    /// takes the `PromptInput::ChatMessages` + `CliChatTemplateFormatter`
    /// path (`self.transcript` is non-empty by then). If that wiring
    /// dropped the formatter, `tokenize_prompt_input` would surface
    /// `InferenceApiError::PolicyDenied` (see the direct-call test below)
    /// and turn 2 would fail here instead of succeeding. A third+ turn is
    /// deliberately not exercised in this test: the fixture tokenizer's
    /// shared `model_max_length` is deliberately small, so this test stays
    /// on the first turn and focuses on transcript ownership.
    #[test]
    fn chat_session_keeps_transcript_cli_side_and_reuses_one_runtime_session() {
        let model_ref = ModelRef::new("qwen-test").unwrap();
        let mut chat = ChatSession::open(&model_ref).unwrap();
        let (reply, observer) = chat.turn("first line").unwrap();
        assert!(!reply.is_empty());
        assert!(!observer.observations().is_empty());
        assert_eq!(chat.transcript().len(), 2);
        chat.close().unwrap();
    }

    /// Task 8.3's literal spec scenario: "two chat turns execute" through
    /// one `ChatSession` "use the same Runtime InferenceSession
    /// identifier." Before the fix this covers, `ChatSession::turn` called
    /// `run_first_native_generation`, which built and closed an entirely
    /// separate throwaway Runtime and session on every call -- the session
    /// id this test captures before a turn would have had no relationship
    /// at all to what that turn actually executed through.
    #[test]
    fn chat_session_turns_share_the_same_runtime_session_identifier() {
        let model_ref = ModelRef::new("qwen-test").unwrap();
        let mut chat = ChatSession::open(&model_ref).unwrap();
        let session_id_before = chat.session_id().clone();
        chat.turn("first line").unwrap();
        assert_eq!(chat.session_id(), &session_id_before);
        chat.close().unwrap();
    }

    #[test]
    fn chat_session_rejects_unknown_model_ref_instead_of_ignoring_it() {
        // `open` now itself loads the Model Instance into this chat
        // session's persistent Runtime (task 8.3) rather than deferring
        // that to the first `turn` call, so an unknown model ref fails
        // closed here already instead of producing a session whose first
        // turn is guaranteed to fail.
        let model_ref = ModelRef::new("different-model").unwrap();
        let Err(error) = ChatSession::open(&model_ref) else {
            panic!("expected an unknown model ref to fail ChatSession::open");
        };
        assert!(matches!(
            error.runtime_category(),
            Some(InferenceApiError::ModelResolutionFailed { reason })
                if reason.contains("different-model")
        ));
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
