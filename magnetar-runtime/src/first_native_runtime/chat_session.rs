//! Fixture generation runs and the first-native chat session.

use super::*;

/// Builds the graphs `execute_generation_step` will dispatch against for
/// `fixture` at `prompt_token_count` -- the same graph *source* as that
/// method, not just a semantically-equivalent one, since callers (many
/// conformance checks and tests) use this to build a
/// [`PreparedExecutionPlan`] (via [`first_native_plans_for_prompt`]) that
/// must fingerprint-match whatever graph actually gets dispatched. Task
/// 11.5: under the strict, default build this is the real Qwen Component
/// (matching `execute_generation_step`'s own strict branch).
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub(super) fn first_native_component_graphs_for_prompt(
    fixture: &E2eFixture,
    prompt_token_count: u64,
) -> Result<FirstNativeComponentGraphs, E2eConformanceError> {
    build_first_native_graphs_from_real_qwen_component(fixture, prompt_token_count)
        .map(|(graphs, _definition, _instance)| graphs)
}

/// Test-oracle branch (`reach-architecture-freeze-1` task 12.6): when no
/// strict Component engine is available, a test build still needs *some*
/// graph to drive downstream (KV, sampling, tokenization, ...) coverage
/// with, so it falls back to the Rust-synthesized recipe -- never reachable
/// outside `#[cfg(test)]`, and never the thing first-native generation
/// treats as authoritative in a real build.
#[cfg(all(
    test,
    not(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))
))]
pub(super) fn first_native_component_graphs_for_prompt(
    fixture: &E2eFixture,
    prompt_token_count: u64,
) -> Result<FirstNativeComponentGraphs, E2eConformanceError> {
    let semantics = qwen_component_graph_semantics_for_prompt(
        &fixture.config,
        &fixture.identity,
        prompt_token_count,
    )?;
    build_first_native_graphs_from_component_output(fixture, prompt_token_count, semantics)
}

/// Production, non-test branch when no strict Component engine is
/// available (no `wasmtime-component-engine`, or `wasm32`, which has none
/// today -- see task group 10's investigation note). Correctif 7 / task
/// group 10's fail-closed requirement, and the explicit decision behind
/// task 12.6: production first-native generation has exactly one semantic
/// source for Model Component graphs, the real Component itself. There is
/// no second, Rust-synthesized production path to fall back to, silently
/// or otherwise -- this fails structurally instead.
#[cfg(all(
    not(test),
    not(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))
))]
pub(super) fn first_native_component_graphs_for_prompt(
    _fixture: &E2eFixture,
    _prompt_token_count: u64,
) -> Result<FirstNativeComponentGraphs, E2eConformanceError> {
    Err(E2eConformanceError::ModelComponentFailed {
        reason: "no Component engine is available on this build target; first-native generation \
                 requires a real Model Component and has no unattested fallback graph source"
            .into(),
    })
}

/// Builds real published prepared execution plans for `fixture` at
/// `prompt_token_count`, so callers exercising the first-native hot-path
/// dispatch functions directly (conformance checks, tests) supply a genuine
/// `PreparedExecutionPlan` instead of skipping plan-bound execution (those
/// functions no longer accept `None`; see
/// `execute_qwen_prefill_hidden_states_through_dispatch`'s doc comment).
pub(super) fn first_native_plans_for_prompt(
    runtime: &Runtime,
    fixture: &E2eFixture,
    instance: &ModelInstanceId,
    prompt_token_count: u64,
) -> Result<FirstNativePreparedPlans, E2eConformanceError> {
    let component_graphs = first_native_component_graphs_for_prompt(fixture, prompt_token_count)?;
    prepare_first_native_execution_plans(runtime, instance, component_graphs, prompt_token_count)
}

/// Runs the generation loop with real published plans built for `request`'s
/// prompt length, so callers exercise the same plan-bound hot path
/// production uses (via `run_generation_loop_with_execution_plans`) instead
/// of the no-plan `run_generation_loop` entry point, which the first-native
/// executor rejects with a structured `GraphPlanningFailed` error.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_first_native_generation_loop_with_plans(
    runtime: &mut Runtime,
    fixture: &E2eFixture,
    instance: &ModelInstanceId,
    request: &GenerationRequest,
    sampling_policy: SamplingPolicy,
    cache_usage: CacheUsageSummary,
    should_cancel: impl FnMut(&[TokenId]) -> bool,
    observer: &mut InferenceApiObserver,
) -> Result<GenerationResult, InferenceApiError> {
    let prompt_token_count = request.input_token_ids.len() as u64;
    let mut prepared_plans =
        first_native_plans_for_prompt(runtime, fixture, instance, prompt_token_count).map_err(
            |error| InferenceApiError::GraphPlanningFailed {
                reason: error.to_string(),
            },
        )?;
    let mut execution_plans = RuntimeGenerationExecutionPlans {
        prefill: &mut prepared_plans.prefill,
        decode: &mut prepared_plans.decode,
    };
    run_generation_loop_with_execution_plans(
        runtime,
        request,
        sampling_policy,
        cache_usage,
        should_cancel,
        observer,
        &mut execution_plans,
    )
}

pub(super) fn generation_tokenizer_reference(fixture: &E2eFixture) -> GenerationTokenizerReference {
    GenerationTokenizerReference {
        tokenizer_id: fixture.tokenizer.metadata().id.clone(),
        metadata: fixture.tokenizer.metadata().clone(),
    }
}

pub(super) fn run_success_path_with_prompt(
    fixture: &E2eFixture,
    model_ref: &ModelRef,
    prompt: &str,
) -> Result<E2eRunOutcome, E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);

    // Model resolution.
    let mut registry = ModelRegistry::new();
    registry.register(model_ref.clone(), fixture.manifest.id.clone());
    let resolution = registry.resolve(&ModelResolutionRequest::new(model_ref.clone()))?;
    if resolution.artifact != fixture.manifest.id {
        return Err(E2eConformanceError::ModelResolutionFailed {
            reason: "resolved artifact does not match fixture manifest".into(),
        });
    }

    // Model Loading + Model Instance.
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    require_ready_first_native_instance(&runtime, &instance)?;

    // Session creation.
    let session_request = SessionCreationRequest {
        model: GenerationModelReference::ModelInstance(instance.clone()),
        tokenizer: generation_tokenizer_reference(fixture),
        generation_defaults: GenerationParameters::greedy(),
        policy: SessionPolicy::default(),
        memory: SessionMemoryBudget::default(),
        allowed_capabilities: BTreeSet::new(),
        correlation_id: None,
        created_at_millis: 0,
    };
    let session = create_inference_session(&mut runtime, session_request)?;

    // Tokenization (plain-text path).
    let tokenized = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::PlainText(prompt.into())),
        None,
    )?;
    let mut observer = InferenceApiObserver::new();
    // Task 11.5: the strict, default path now builds its graphs by calling
    // the real Qwen Component's `graph-builder` host imports, not
    // `qwen_prefill_graph`/`qwen_decode_graph`.
    #[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
    let component_graphs = {
        let (graphs, definition, instance) = build_first_native_graphs_from_real_qwen_component(
            fixture,
            tokenized.token_ids.len() as u64,
        )?;
        observer.observe(
            InferenceApiObservationKind::ComponentValidated,
            format!("component_definition={definition:?}"),
            None,
        );
        observer.observe(
            InferenceApiObservationKind::ComponentInstantiated,
            format!("component_instance={instance:?}"),
            None,
        );
        graphs
    };
    // Correctif 7 / task group 10, task 12.6: without a real Component
    // engine, production has no second, Rust-synthesized graph source to
    // fall back to -- it fails closed, structurally. A test build still
    // needs *some* graph to drive downstream coverage with when built
    // without a strict engine, so it keeps the Rust-synthesized recipe as a
    // test-oracle-only fallback.
    #[cfg(all(
        test,
        not(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))
    ))]
    let component_graphs = {
        let semantics = qwen_component_graph_semantics_for_prompt(
            &fixture.config,
            &fixture.identity,
            tokenized.token_ids.len() as u64,
        )?;
        build_first_native_graphs_from_component_output(
            fixture,
            tokenized.token_ids.len() as u64,
            semantics,
        )?
    };
    #[cfg(all(
        not(test),
        not(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))
    ))]
    let component_graphs: FirstNativeComponentGraphs =
        return Err(E2eConformanceError::ModelComponentFailed {
            reason: "no Component engine is available on this build target; first-native \
                     generation requires a real Model Component and has no unattested \
                     fallback graph source"
                .into(),
        });
    let mut prepared_plans = prepare_first_native_execution_plans(
        &runtime,
        &instance,
        component_graphs,
        tokenized.token_ids.len() as u64,
    )?;
    let _plan_generations = (
        prepared_plans.prefill.generation,
        prepared_plans.prefill_node_count,
        prepared_plans.decode.generation,
        prepared_plans.decode_node_count,
    );

    // Generation request build + validate.
    let request = build_generation_request(
        GenerationRequestId::new("e2e-success-path")?,
        Some(session.clone()),
        GenerationModelReference::ModelInstance(instance.clone()),
        generation_tokenizer_reference(fixture),
        tokenized,
        4,
        GenerationParameters::greedy(),
        StopConditions {
            eos: EosPolicy {
                eos_token_ids: vec![E2E_FIXTURE_EOS_TOKEN],
                ..EosPolicy::default()
            },
            ..StopConditions::default()
        },
        StreamingMode::TokenIds,
    );
    let request = prepare_generation(&runtime, request)?;

    // Prefill/decode/sample/stream through the real forward pass.
    let mut execution_plans = RuntimeGenerationExecutionPlans {
        prefill: &mut prepared_plans.prefill,
        decode: &mut prepared_plans.decode,
    };
    let generation_result = run_generation_loop_with_execution_plans(
        &mut runtime,
        &request,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated_so_far| false,
        &mut observer,
        &mut execution_plans,
    )?;

    // Streaming decode of generated tokens. The byte fixture can produce token
    // sequences that are structurally valid but not a complete UTF-8 text for
    // every arbitrary prompt, so the user-facing helper keeps generation
    // successful and renders token IDs when text decoding cannot complete.
    let decoded_text = decode_tokens_streaming(
        &fixture.tokenizer,
        StreamingDecodeRequest::new(generation_result.output.generated_token_ids.clone()),
    )
    .map(|decoded| decoded.text)
    .unwrap_or_else(|_| {
        let tokens = generation_result
            .output
            .generated_token_ids
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(" ");
        format!("[generated token ids: {tokens}]")
    });
    let generation_result = generation_result.with_decoded_text(decoded_text);

    // Session close + Model Instance cleanup.
    let kv_observations = runtime.kv_caches().observations().to_vec();
    close_inference_session(&mut runtime, &session)?;
    unload_model_instance(
        &mut runtime,
        &instance,
        ModelInstanceUnloadPolicy::DrainActiveUse,
    )
    .map_err(E2eConformanceError::from)?;

    Ok(E2eRunOutcome {
        generation_result,
        observer,
        kv_observations,
    })
}

/// Runs the full required success path: resolve, load, instantiate, create
/// session, tokenize (plain text), generate through a real Reference CPU
/// forward pass with greedy Sampling, stream, close session, cleanup.
pub(super) fn run_success_path(fixture: &E2eFixture) -> Result<E2eRunOutcome, E2eConformanceError> {
    run_success_path_with_prompt(fixture, &ModelRef::new("qwen-test")?, "hi")
}

pub fn run_first_native_fixture_generation(
    prompt: &str,
) -> Result<FirstNativeFixtureGeneration, E2eConformanceError> {
    let fixture = e2e_fixture()?;
    let outcome = run_success_path_with_prompt(&fixture, &ModelRef::new("qwen-test")?, prompt)?;
    if !outcome
        .kv_observations
        .iter()
        .any(|observation| observation.kind == KvCacheObservationKind::PrefillCompleted)
    {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "first native fixture generation produced no KV prefill commit".into(),
        });
    }
    validate_e2e_no_shortcuts(
        outcome.observer.observations(),
        &reference_cpu_kernel_advertisements(),
    )?;
    let text = outcome
        .generation_result
        .decoded_text
        .clone()
        .ok_or_else(|| E2eConformanceError::StreamingFailed {
            reason: "first native fixture generation produced no decoded text".into(),
        })?;
    Ok(FirstNativeFixtureGeneration {
        text,
        result: outcome.generation_result,
        observer: outcome.observer,
    })
}

pub fn run_first_native_generation(
    model_ref: &ModelRef,
    prompt: &str,
) -> Result<FirstNativeFixtureGeneration, FirstNativeRuntimeError> {
    if model_ref.as_str() != "qwen-test" {
        return Err(FirstNativeRuntimeError::model_not_found(model_ref));
    }
    let fixture = e2e_fixture().map_err(FirstNativeRuntimeError::from_conformance)?;
    let outcome = run_success_path_with_prompt(&fixture, model_ref, prompt)
        .map_err(FirstNativeRuntimeError::from_conformance)?;
    validate_e2e_no_shortcuts(
        outcome.observer.observations(),
        &reference_cpu_kernel_advertisements(),
    )
    .map_err(FirstNativeRuntimeError::from_conformance)?;
    let text = outcome
        .generation_result
        .decoded_text
        .clone()
        .ok_or_else(|| {
            FirstNativeRuntimeError::from_conformance(E2eConformanceError::StreamingFailed {
                reason: "first native generation produced no decoded text".into(),
            })
        })?;
    Ok(FirstNativeFixtureGeneration {
        text,
        result: outcome.generation_result,
        observer: outcome.observer,
    })
}

/// A persistent first-native chat session (task 8.3): one `Runtime`, one
/// `ModelInstance`, and one `InferenceSessionId` created once by [`Self::open`]
/// and reused by every [`Self::turn`] -- unlike [`run_first_native_generation`],
/// which builds and tears down a fresh `Runtime` and `InferenceSessionId` for
/// every call. This is what makes `magnetar chat`'s cancellation and
/// session-identity guarantees genuine: cancelling targets the very session
/// turns execute through, not an orphan nothing else touches.
pub struct FirstNativeChatSession {
    pub(super) fixture: E2eFixture,
    pub(super) runtime: Runtime,
    pub(super) instance: ModelInstanceId,
    pub(super) session: InferenceSessionId,
    pub(super) next_request: u64,
}

impl FirstNativeChatSession {
    pub fn open(model_ref: &ModelRef) -> Result<Self, FirstNativeRuntimeError> {
        if model_ref.as_str() != "qwen-test" {
            return Err(FirstNativeRuntimeError::model_not_found(model_ref));
        }
        let fixture = e2e_fixture().map_err(FirstNativeRuntimeError::from_conformance)?;
        let mut runtime = build_runtime_with_model_execution_engine(&fixture);

        let mut registry = ModelRegistry::new();
        registry.register(model_ref.clone(), fixture.manifest.id.clone());
        let resolution = registry
            .resolve(&ModelResolutionRequest::new(model_ref.clone()))
            .map_err(|error| FirstNativeRuntimeError::from_conformance(error.into()))?;
        if resolution.artifact != fixture.manifest.id {
            return Err(FirstNativeRuntimeError::from_conformance(
                E2eConformanceError::ModelResolutionFailed {
                    reason: "resolved artifact does not match fixture manifest".into(),
                },
            ));
        }

        let (instance, _memory) = load_fixture_instance(&fixture, &mut runtime)
            .map_err(FirstNativeRuntimeError::from_conformance)?;
        require_ready_first_native_instance(&runtime, &instance)
            .map_err(|error| FirstNativeRuntimeError::from_conformance(error.into()))?;

        let session_request = SessionCreationRequest {
            model: GenerationModelReference::ModelInstance(instance.clone()),
            tokenizer: generation_tokenizer_reference(&fixture),
            generation_defaults: GenerationParameters::greedy(),
            policy: SessionPolicy::default(),
            memory: SessionMemoryBudget::default(),
            allowed_capabilities: BTreeSet::new(),
            correlation_id: None,
            created_at_millis: 0,
        };
        let session = create_inference_session(&mut runtime, session_request)
            .map_err(|error| FirstNativeRuntimeError::from_conformance(error.into()))?;

        Ok(Self {
            fixture,
            runtime,
            instance,
            session,
            next_request: 0,
        })
    }

    /// The persistent Runtime `InferenceSessionId` every turn executes
    /// through -- stable across the life of this chat session.
    pub fn session_id(&self) -> &InferenceSessionId {
        &self.session
    }

    /// Runs one chat turn's prefill/decode generation through this
    /// session's persistent `Runtime`, `ModelInstance`, and
    /// `InferenceSessionId`. `prompt` is caller-rendered text (the CLI owns
    /// transcript assembly and chat template rendering); this only
    /// tokenizes, plans, and executes it -- it never re-creates the Runtime
    /// session or Model Instance `open` already established.
    pub fn turn(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<FirstNativeFixtureGeneration, FirstNativeRuntimeError> {
        self.next_request += 1;
        let request_id =
            GenerationRequestId::new(format!("first-native-chat-{}", self.next_request))
                .map_err(|error| FirstNativeRuntimeError::from_conformance(error.into()))?;

        let tokenized = tokenize_prompt_input(
            &self.fixture.tokenizer,
            TokenizationRequest::new(PromptInput::PlainText(prompt.into())),
            None,
        )
        .map_err(|error| FirstNativeRuntimeError::from_conformance(error.into()))?;

        let mut observer = InferenceApiObserver::new();
        // Task 11.5: the strict, default path now builds its graphs by
        // calling the real Qwen Component's `graph-builder` host imports,
        // not `qwen_prefill_graph`/`qwen_decode_graph`.
        #[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
        let component_graphs = {
            let (graphs, definition, instance) =
                build_first_native_graphs_from_real_qwen_component(
                    &self.fixture,
                    tokenized.token_ids.len() as u64,
                )
                .map_err(FirstNativeRuntimeError::from_conformance)?;
            observer.observe(
                InferenceApiObservationKind::ComponentValidated,
                format!("component_definition={definition:?}"),
                None,
            );
            observer.observe(
                InferenceApiObservationKind::ComponentInstantiated,
                format!("component_instance={instance:?}"),
                None,
            );
            graphs
        };
        // Correctif 7 / task group 10, task 12.6: see the sibling fallback
        // in `run_success_path_with_prompt` -- production has no second,
        // Rust-synthesized graph source without a strict Component engine;
        // it fails closed. The Rust-synthesized recipe survives only as a
        // test-oracle fallback so a test build without a strict engine
        // still has some graph to drive downstream coverage with.
        #[cfg(all(
            test,
            not(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))
        ))]
        let component_graphs = {
            let semantics = qwen_component_graph_semantics_for_prompt(
                &self.fixture.config,
                &self.fixture.identity,
                tokenized.token_ids.len() as u64,
            )
            .map_err(FirstNativeRuntimeError::from_conformance)?;
            build_first_native_graphs_from_component_output(
                &self.fixture,
                tokenized.token_ids.len() as u64,
                semantics,
            )
            .map_err(FirstNativeRuntimeError::from_conformance)?
        };
        #[cfg(all(
            not(test),
            not(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))
        ))]
        let component_graphs: FirstNativeComponentGraphs = return Err(
            FirstNativeRuntimeError::from_conformance(E2eConformanceError::ModelComponentFailed {
                reason: "no Component engine is available on this build target; \
                             first-native generation requires a real Model Component and has \
                             no unattested fallback graph source"
                    .into(),
            }),
        );

        let mut prepared_plans = prepare_first_native_execution_plans(
            &self.runtime,
            &self.instance,
            component_graphs,
            tokenized.token_ids.len() as u64,
        )
        .map_err(FirstNativeRuntimeError::from_conformance)?;

        let request = build_generation_request(
            request_id,
            Some(self.session.clone()),
            GenerationModelReference::ModelInstance(self.instance.clone()),
            generation_tokenizer_reference(&self.fixture),
            tokenized,
            max_new_tokens,
            GenerationParameters::greedy(),
            StopConditions {
                eos: EosPolicy {
                    eos_token_ids: vec![E2E_FIXTURE_EOS_TOKEN],
                    ..EosPolicy::default()
                },
                ..StopConditions::default()
            },
            StreamingMode::TokenIds,
        );
        let request = prepare_generation(&self.runtime, request)
            .map_err(|error| FirstNativeRuntimeError::from_conformance(error.into()))?;

        let mut execution_plans = RuntimeGenerationExecutionPlans {
            prefill: &mut prepared_plans.prefill,
            decode: &mut prepared_plans.decode,
        };
        let generation_result = run_generation_loop_with_execution_plans(
            &mut self.runtime,
            &request,
            SamplingPolicy::default(),
            CacheUsageSummary::default(),
            |_generated_so_far| false,
            &mut observer,
            &mut execution_plans,
        )
        .map_err(|error| FirstNativeRuntimeError::from_conformance(error.into()))?;

        let decoded_text = decode_tokens_streaming(
            &self.fixture.tokenizer,
            StreamingDecodeRequest::new(generation_result.output.generated_token_ids.clone()),
        )
        .map(|decoded| decoded.text)
        .unwrap_or_else(|_| {
            let tokens = generation_result
                .output
                .generated_token_ids
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(" ");
            format!("[generated token ids: {tokens}]")
        });
        let generation_result = generation_result.with_decoded_text(decoded_text);

        validate_e2e_no_shortcuts(
            observer.observations(),
            &reference_cpu_kernel_advertisements(),
        )
        .map_err(FirstNativeRuntimeError::from_conformance)?;

        let text = generation_result.decoded_text.clone().ok_or_else(|| {
            FirstNativeRuntimeError::from_conformance(E2eConformanceError::StreamingFailed {
                reason: "first native chat turn produced no decoded text".into(),
            })
        })?;

        Ok(FirstNativeFixtureGeneration {
            text,
            result: generation_result,
            observer,
        })
    }

    /// Cancels this session's persistent Runtime `InferenceSessionId` --
    /// the same session [`Self::turn`] executes every generation through.
    pub fn cancel(&mut self) -> Result<(), FirstNativeRuntimeError> {
        cancel_inference_session(&mut self.runtime, &self.session)
            .map_err(|error| FirstNativeRuntimeError::from_conformance(error.into()))
    }

    pub fn close(mut self) -> Result<(), FirstNativeRuntimeError> {
        close_inference_session(&mut self.runtime, &self.session)
            .map_err(|error| FirstNativeRuntimeError::from_conformance(error.into()))?;
        unload_model_instance(
            &mut self.runtime,
            &self.instance,
            ModelInstanceUnloadPolicy::DrainActiveUse,
        )
        .map_err(|error| FirstNativeRuntimeError::from_conformance(error.into()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
