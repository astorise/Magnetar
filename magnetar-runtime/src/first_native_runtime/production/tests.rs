//! Unit tests for the parent module.
//!
//! Shared mock executors and fixtures live in `first_native_runtime/tests.rs`.

use super::super::tests::*;
use super::*;

/// `register_qwen_component_artifact` must be safe to call unconditionally,
/// every time a caller might need first-native generation, without the
/// caller tracking its own "have I registered yet" state (task 12.4's
/// design: `magnetar-cli` calls this before every `one_shot`/`ChatSession::
/// open`). Proves the second call is a genuine no-op, not a panic or an
/// error, regardless of whether the bytes differ from the first call's.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn register_qwen_component_artifact_is_idempotent() {
    register_qwen_component_artifact(
        b"first-call-bytes".to_vec(),
        b"first-call-manifest".to_vec(),
    );
    register_qwen_component_artifact(
        b"second-call-different-bytes".to_vec(),
        b"second-call-different-manifest".to_vec(),
    );
}

/// Task groups 10-11 end to end: a `FirstNativeModelConfig` deliberately different
/// from the canonical E2E fixture (hidden_size=8, attention_head_count=4,
/// kv_head_count=2 -- genuinely GQA-shaped, unlike the fixture's 2/2) is
/// wrapped as a production-shaped `ModelManifest` (no fixture manifest
/// constructor, real `architecture_config`, a real payload source keyed
/// only by canonical tensor name/bytes) and driven through
/// `load_production_model_instance` -> `production_model_fixture` ->
/// the same real Qwen Component graph production, generic Runtime
/// Inference API, and generation loop every other first-native caller
/// uses -- with no `qwen-test` identity or fixture manifest anywhere in
/// this path.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn production_loading_generates_end_to_end_with_a_non_canonical_qwen_config() {
    // vocab_size=258 (not 64): the delegate tokenizer below is the real
    // byte-level FixtureTokenizer (each byte encodes to `byte + 1`, up to
    // 256, plus BOS/EOS), so the vocabulary must cover that range
    // regardless of this test's otherwise-deliberately-non-canonical
    // dimensions.
    let architecture = first_native_architecture_metadata(8, 1, 4, 2, 2, 16, 258, 1_000_000);
    let config = FirstNativeModelConfig {
        architecture,
        rope: FirstNativeRopeConfig {
            base: 10_000.0,
            scale: None,
            dimension: 2,
            position_mode: FirstNativeRopePositionMode::Sequential,
            dynamic_scaling_supported: false,
            frequency_scaling: None,
        },
        rmsnorm_epsilon: 1e-6,
        tied_embeddings: false,
        attention_bias: false,
        mixture_of_experts: None,
        require_bos: false,
        require_pad: false,
        expected_added_tokens: None,
        chat_template_required: false,
    };

    let weights = e2e_fixture_weights(&config).expect("synthetic weights build");
    let tensors = e2e_fixture_weight_inventory(&config).expect("tensor inventory builds");
    let mut bytes_by_name = BTreeMap::new();
    for tensor in &tensors {
        let host_tensor = weights
            .get(&tensor.name)
            .expect("a weight exists for every inventory tensor");
        let mut bytes = Vec::with_capacity(host_tensor.data.len() * 4);
        for value in &host_tensor.data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes_by_name.insert(tensor.name.clone(), bytes);
    }
    let payload_source = ProductionIntegrationPayloadSource { bytes_by_name };

    let digest = ModelDigest::parse(format!("sha256:{}", "7".repeat(64))).unwrap();
    let id = ModelArtifactId::new(
        ModelArtifactKind::ModelBundle,
        ModelName::new("production-integration-test").unwrap(),
        ModelRevision::new("r1").unwrap(),
        digest,
    );
    let mut parts = BTreeMap::new();
    parts.insert(
        "weights".to_string(),
        ModelArtifactPart {
            name: "weights".to_string(),
            kind: ModelArtifactKind::ModelWeights,
            digest: ModelDigest::parse(format!("sha256:{}", "8".repeat(64))).unwrap(),
            size_bytes: None,
            required: true,
        },
    );
    parts.insert(
        "config".to_string(),
        ModelArtifactPart {
            name: "config".to_string(),
            kind: ModelArtifactKind::ModelConfig,
            digest: ModelDigest::parse(format!("sha256:{}", "9".repeat(64))).unwrap(),
            size_bytes: None,
            required: true,
        },
    );
    let manifest = ModelManifest {
        schema_version: crate::MODEL_ARTIFACT_SCHEMA_VERSION,
        id,
        architecture: ModelArchitecture::new("qwen", "production-integration-test"),
        parts,
        storage_dtype: Some(ModelDType::F32),
        compute_dtype: None,
        supported_compute_dtypes: BTreeSet::from([ModelDType::F32]),
        tensors,
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
        architecture_config: Some(architecture_config_from_first_native_model_config(&config)),
        artifact_format: ArtifactFormat::HuggingFace,
    };

    let tokenizer_metadata = e2e_fixture_tokenizer().unwrap().metadata().clone();
    let delegate_tokenizer: std::sync::Arc<dyn crate::tokenizer::Tokenizer + Send + Sync> =
        std::sync::Arc::new(e2e_fixture_tokenizer().unwrap());
    let fixture = production_model_fixture(
        manifest.clone(),
        tokenizer_metadata,
        delegate_tokenizer,
    )
    .expect(
        "production fixture builds against a genuinely different config than the canonical fixture",
    );
    assert_eq!(fixture.config.architecture.attention_head_count, 4);
    assert_eq!(fixture.config.architecture.kv_head_count, 2);

    let mut runtime = build_runtime_with_model_execution_engine(&fixture);
    let instance = load_production_model_instance(&mut runtime, &manifest, &payload_source).expect(
        "production loading succeeds through the generic Inference API, not a fixture loader",
    );
    require_ready_first_native_instance(&runtime, &instance)
        .expect("a production-loaded instance is genuinely ready");

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
    let session = create_inference_session(&mut runtime, session_request).expect("session creates");

    let tokenized = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::PlainText("hi".into())),
        None,
    )
    .expect("tokenization succeeds");

    let (component_graphs, _definition, _component_instance) =
        build_first_native_graphs_from_real_qwen_component(
            &fixture,
            tokenized.token_ids.len() as u64,
        )
        .expect("the real Qwen Component produces graphs for this non-canonical config");

    let mut prepared_plans = prepare_first_native_execution_plans(
        &runtime,
        &instance,
        component_graphs,
        tokenized.token_ids.len() as u64,
    )
    .expect("execution plans prepare");

    let mut observer = InferenceApiObserver::new();
    let request = build_generation_request(
        GenerationRequestId::new("production-integration-test").unwrap(),
        Some(session.clone()),
        GenerationModelReference::ModelInstance(instance.clone()),
        generation_tokenizer_reference(&fixture),
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
    let request = prepare_generation(&runtime, request).expect("generation request validates");

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
    )
    .expect("generation runs end to end through the real Component graph and Provider dispatch");

    assert!(
        !generation_result.output.generated_token_ids.is_empty(),
        "production loading + generation produced at least one token"
    );

    close_inference_session(&mut runtime, &session).expect("session closes");
    unload_model_instance(
        &mut runtime,
        &instance,
        ModelInstanceUnloadPolicy::DrainActiveUse,
    )
    .expect("model instance unloads cleanly, no leaked resources");
}

/// Steps 3-4 of the auditor's astorise/Magnetar#83 remediation plan,
/// proven directly: `production_model_fixture` no longer hardcodes the
/// reconstructed `FirstNativeModelConfig`'s architecture family to
/// `QWEN_ARCHITECTURE_FAMILY` -- it threads through whatever family the
/// real ingested `ModelManifest.architecture.family` actually carries
/// (`first_native_model_config_from_architecture_config`'s new `family`
/// parameter), and `production_model_component_identity`'s deliberately
/// unrestricted `supported_architecture_families` (see its own doc
/// comment) means a non-"qwen" family is not rejected by
/// `ModelComponentDescriptor::validate_model_artifact` either.
///
/// This is the honestly-achievable half of the auditor's steps 6-7 (a
/// real Component-vs-family compatibility test): today's
/// `.magnetar-component.yaml` schema has no field for a *specific*
/// Component to declare its own supported families, so a genuine "Llama
/// Artifact + Qwen Component -> reject" test cannot yet be written --
/// that remains an open follow-up tracked on astorise/Magnetar#83. What
/// *can* be proven today, and is proven here, is that the generic path
/// stops silently discarding/overwriting a real non-"qwen" family --
/// exactly what steps 3-4 fix, and the regression this test guards
/// against (before this fix, every family below would have been
/// silently rewritten to `"qwen"`, and the loop's `assert_eq!` would
/// have caught it).
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn production_model_fixture_threads_the_real_ingested_architecture_family() {
    let config = e2e_fixture_config();
    let weights = e2e_fixture_weights(&config).expect("synthetic weights build");

    for family in ["llama", "mistral", "gemma"] {
        let architecture = ModelArchitecture::new(family, "architecture-family-test");
        let mut manifest = e2e_fixture_manifest_from_weights(&config, &architecture, &weights)
            .unwrap_or_else(|e| panic!("{family}: manifest builds: {e}"));
        manifest.architecture_config =
            Some(architecture_config_from_first_native_model_config(&config));

        let tokenizer_metadata = e2e_fixture_tokenizer().unwrap().metadata().clone();
        let delegate_tokenizer: std::sync::Arc<dyn crate::tokenizer::Tokenizer + Send + Sync> =
            std::sync::Arc::new(e2e_fixture_tokenizer().unwrap());

        let fixture = production_model_fixture(manifest, tokenizer_metadata, delegate_tokenizer)
            .unwrap_or_else(|e| {
                panic!("family '{family}' must not be rejected by the generic production path: {e}")
            });

        assert_eq!(
            fixture.config.architecture.family, family,
            "the real ingested family must be threaded through unchanged, not \
             silently rewritten to '{QWEN_ARCHITECTURE_FAMILY}'"
        );
    }
}

/// The load-bearing correctness proof for `ProductionLoadedModel::
/// load_with_component` (`wire-generic-inference-component-runtime`'s
/// follow-up phase, closing the Tachyon integration audit's MAG-02): a
/// model loaded against an *explicitly registered* Component digest
/// generates *exactly* the same tokens as the same model loaded through
/// [`ProductionLoadedModel::load`]'s pre-existing hardcoded-singleton
/// path, for the identical underlying Component bytes. This proves the
/// registered Component genuinely drives generation end to end (plan
/// production in `prepare_generation` and dispatch-time graph production in
/// `E2eRuntimeModelExecutionEngine::execute_generation_step` both honor the
/// same digest), not merely that graph construction alone matches (already
/// proven by `register_inference_component_artifact_enforces_trust_is_
/// idempotent_and_matches_the_singleton_path`).
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn production_loaded_model_load_with_component_matches_the_singleton_path() {
    // `e2e_fixture()`'s own manifest is shaped for the in-memory,
    // fixture-only `E2eRuntimeModelExecutionEngine` path (no declared
    // per-tensor byte offset/size) -- `load_production_qwen_instance_for_
    // provider` (the real production-loading path `ProductionLoadedModel::
    // load` drives) requires those, so this test builds its own
    // production-shaped manifest around the same canonical config, exactly
    // like `production_loading_generates_end_to_end_with_a_non_canonical_
    // qwen_config` does for its own deliberately-different config.
    // `tied_embeddings: false` (unlike the canonical fixture's own `true`):
    // this test drives loading through the real production path
    // (`load_production_model_instance_for_provider`), which expects a
    // literal `lm_head` weight resource bound -- tied-embedding derivation
    // is an ingestion-layer concern (`loaders/huggingface`'s
    // `append_synthetic_lm_head_if_tied`) this hand-built manifest
    // deliberately bypasses, exactly like `production_loading_generates_
    // end_to_end_with_a_non_canonical_qwen_config`'s own config already
    // does. Otherwise identical to the canonical fixture the real
    // `qwen-real.component.wasm` Component already exercises elsewhere.
    let (fixture, payload_source) = production_shaped_canonical_fixture();

    let component_trust = ComponentTrustStore::default()
        .trust_digest(&ComponentDigest::sha256(QWEN_REAL_COMPONENT_BYTES).value);
    let digest = register_inference_component_artifact(
        QWEN_REAL_COMPONENT_BYTES.to_vec(),
        QWEN_REAL_COMPONENT_MANIFEST_BYTES.to_vec(),
        &component_trust,
    )
    .expect("registration succeeds")
    .digest;

    let model_trust =
        || ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone());
    let request = || ProductionGenerationRequest {
        prompt: PromptInput::PlainText("hi".into()),
        parameters: GenerationParameters::greedy(),
        stop_conditions: StopConditions::default(),
        max_new_tokens: Some(2),
        max_generation_millis: None,
    };

    let mut via_singleton = ProductionLoadedModel::load(
        fixture.clone(),
        &payload_source,
        model_trust(),
        Arc::new(ReferenceCpuProvider::new()),
    )
    .expect("singleton-path loading succeeds");
    let singleton_result = via_singleton
        .generate(request(), None)
        .expect("singleton-path generation succeeds");

    let mut via_named = ProductionLoadedModel::load_with_component(
        fixture.clone(),
        &payload_source,
        model_trust(),
        Arc::new(ReferenceCpuProvider::new()),
        Some(digest),
    )
    .expect("named-component loading succeeds");
    let named_result = via_named
        .generate(request(), None)
        .expect("named-component generation succeeds");

    assert_eq!(
        singleton_result.result.output.generated_token_ids,
        named_result.result.output.generated_token_ids,
        "loading via an explicitly registered Component digest must produce identical \
         generation to the hardcoded singleton path, for the identical underlying Component \
         bytes"
    );
}

/// astorise/Magnetar#83 steps 6-7's required backward-compatibility proof:
/// a Component manifest with no `compatibility` block at all (every
/// manifest written before this field existed, e.g. `synthetic-minimal.
/// component.wasm.magnetar-component.yaml`, deliberately left untouched by
/// this change) must keep behaving exactly as permissively as before --
/// loading must not start rejecting real callers just because they have
/// not yet adopted the new declaration. `family: "gemma"` is deliberately
/// nothing the synthetic Component could plausibly declare support for
/// (it declares nothing at all): if the absent-block-means-permissive
/// semantics were ever accidentally inverted, this load would start
/// failing with `ArchitectureUnsupported` even though nothing about this
/// specific Component's own manifest changed.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn production_loaded_model_load_with_component_permits_a_component_with_no_declared_compatibility()
{
    let mut config = e2e_fixture_config();
    config.tied_embeddings = false;
    let weights = e2e_fixture_weights(&config).expect("synthetic weights build");
    let tensors = e2e_fixture_weight_inventory(&config).expect("tensor inventory builds");
    let mut bytes_by_name = BTreeMap::new();
    for tensor in &tensors {
        let host_tensor = weights
            .get(&tensor.name)
            .expect("a weight exists for every inventory tensor");
        let mut bytes = Vec::with_capacity(host_tensor.data.len() * 4);
        for value in &host_tensor.data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes_by_name.insert(tensor.name.clone(), bytes);
    }
    let payload_source = ProductionIntegrationPayloadSource { bytes_by_name };

    let digest_seed = ModelDigest::parse(format!("sha256:{}", "d".repeat(64))).unwrap();
    let id = ModelArtifactId::new(
        ModelArtifactKind::ModelBundle,
        ModelName::new("no-declared-compatibility-test").unwrap(),
        ModelRevision::new("r1").unwrap(),
        digest_seed,
    );
    let mut parts = BTreeMap::new();
    parts.insert(
        "weights".to_string(),
        ModelArtifactPart {
            name: "weights".to_string(),
            kind: ModelArtifactKind::ModelWeights,
            digest: ModelDigest::parse(format!("sha256:{}", "e".repeat(64))).unwrap(),
            size_bytes: None,
            required: true,
        },
    );
    parts.insert(
        "config".to_string(),
        ModelArtifactPart {
            name: "config".to_string(),
            kind: ModelArtifactKind::ModelConfig,
            digest: ModelDigest::parse(format!("sha256:{}", "f".repeat(64))).unwrap(),
            size_bytes: None,
            required: true,
        },
    );
    let manifest = ModelManifest {
        schema_version: crate::MODEL_ARTIFACT_SCHEMA_VERSION,
        id,
        architecture: ModelArchitecture::new("gemma", "no-declared-compatibility-test"),
        parts,
        storage_dtype: Some(ModelDType::F32),
        compute_dtype: None,
        supported_compute_dtypes: BTreeSet::from([ModelDType::F32]),
        tensors,
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
        architecture_config: Some(architecture_config_from_first_native_model_config(&config)),
        artifact_format: ArtifactFormat::HuggingFace,
    };

    let tokenizer_metadata = e2e_fixture_tokenizer().unwrap().metadata().clone();
    let delegate_tokenizer: std::sync::Arc<dyn crate::tokenizer::Tokenizer + Send + Sync> =
        std::sync::Arc::new(e2e_fixture_tokenizer().unwrap());
    let fixture =
        production_model_fixture(manifest.clone(), tokenizer_metadata, delegate_tokenizer)
            .expect("production fixture builds for a family the generic path never rejects");

    let component_trust = ComponentTrustStore::default()
        .trust_digest(&ComponentDigest::sha256(SYNTHETIC_MINIMAL_COMPONENT_BYTES).value);
    let digest = register_inference_component_artifact(
        SYNTHETIC_MINIMAL_COMPONENT_BYTES.to_vec(),
        SYNTHETIC_MINIMAL_COMPONENT_MANIFEST_BYTES.to_vec(),
        &component_trust,
    )
    .expect("registration succeeds")
    .digest;

    let model_trust =
        ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone());

    let via_named = ProductionLoadedModel::load_with_component(
        fixture,
        &payload_source,
        model_trust,
        Arc::new(ReferenceCpuProvider::new()),
        Some(digest),
    )
    .expect(
        "a Component with no declared `compatibility.architecture_families` must remain \
         permissive, exactly like every manifest written before this field existed",
    );
    via_named
        .unload()
        .expect("model instance unloads cleanly, no leaked resources");
}

/// astorise/Magnetar#75's required negative proof, at the same
/// `ProductionLoadedModel::load_with_component` layer
/// `production_loaded_model_load_with_component_matches_the_singleton_path`
/// exercises for the family gate: the real checked-in Llama Component,
/// registered with its manifest narrowed to
/// `compatibility.artifact_formats: [huggingface]` (the checked-in one
/// declares both formats). Pairing it with a hand-built Model Artifact
/// manifest whose `architecture.family` is `"llama"` (so the *family* gate
/// passes) but
/// whose `artifact_format` is [`ArtifactFormat::Gguf`] must be rejected by
/// the *format* gate alone -- proving `load_with_component`'s new
/// `artifact_formats` check is real and independent of the pre-existing
/// family check, before this fix no such rejection existed anywhere in
/// this call chain and this exact pairing would have loaded successfully.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn production_loaded_model_load_with_component_rejects_a_format_mismatched_component() {
    let mut config = e2e_fixture_config();
    config.tied_embeddings = false;
    let weights = e2e_fixture_weights(&config).expect("synthetic weights build");
    let tensors = e2e_fixture_weight_inventory(&config).expect("tensor inventory builds");
    let mut bytes_by_name = BTreeMap::new();
    for tensor in &tensors {
        let host_tensor = weights
            .get(&tensor.name)
            .expect("a weight exists for every inventory tensor");
        let mut bytes = Vec::with_capacity(host_tensor.data.len() * 4);
        for value in &host_tensor.data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes_by_name.insert(tensor.name.clone(), bytes);
    }
    let payload_source = ProductionIntegrationPayloadSource { bytes_by_name };

    let digest_seed = ModelDigest::parse(format!("sha256:{}", "1".repeat(64))).unwrap();
    let id = ModelArtifactId::new(
        ModelArtifactKind::ModelBundle,
        ModelName::new("format-mismatch-test").unwrap(),
        ModelRevision::new("r1").unwrap(),
        digest_seed,
    );
    let mut parts = BTreeMap::new();
    parts.insert(
        "weights".to_string(),
        ModelArtifactPart {
            name: "weights".to_string(),
            kind: ModelArtifactKind::ModelWeights,
            digest: ModelDigest::parse(format!("sha256:{}", "2".repeat(64))).unwrap(),
            size_bytes: None,
            required: true,
        },
    );
    parts.insert(
        "config".to_string(),
        ModelArtifactPart {
            name: "config".to_string(),
            kind: ModelArtifactKind::ModelConfig,
            digest: ModelDigest::parse(format!("sha256:{}", "3".repeat(64))).unwrap(),
            size_bytes: None,
            required: true,
        },
    );
    let manifest = ModelManifest {
        schema_version: crate::MODEL_ARTIFACT_SCHEMA_VERSION,
        id,
        architecture: ModelArchitecture::new("llama", "format-mismatch-test"),
        parts,
        storage_dtype: Some(ModelDType::F32),
        compute_dtype: None,
        supported_compute_dtypes: BTreeSet::from([ModelDType::F32]),
        tensors,
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
        architecture_config: Some(architecture_config_from_first_native_model_config(&config)),
        // The real Llama Component, registered below with a manifest that
        // declares `compatibility.artifact_formats: [huggingface]` only --
        // `Gguf` here is the one and only thing that must trigger the
        // rejection below.
        artifact_format: ArtifactFormat::Gguf,
    };

    let tokenizer_metadata = e2e_fixture_tokenizer().unwrap().metadata().clone();
    let delegate_tokenizer: std::sync::Arc<dyn crate::tokenizer::Tokenizer + Send + Sync> =
        std::sync::Arc::new(e2e_fixture_tokenizer().unwrap());
    let fixture = production_model_fixture(manifest, tokenizer_metadata, delegate_tokenizer)
        .expect("production fixture builds against the canonical config");

    let huggingface_only = String::from_utf8(LLAMA_REAL_COMPONENT_MANIFEST_BYTES.to_vec())
        .unwrap()
        .replace("    - gguf\n", "");
    assert!(!huggingface_only.contains("- gguf"));
    // The registry caches a manifest per Component digest, so the narrowed
    // manifest goes with a byte-distinct copy of the same Component: one
    // trailing custom section (id 0, name "magnetar-test") changes its
    // digest and nothing it does.
    let mut component_bytes = LLAMA_REAL_COMPONENT_BYTES.to_vec();
    let name = b"magnetar-test";
    component_bytes.extend([0, (name.len() + 1) as u8, name.len() as u8]);
    component_bytes.extend(name);
    let component_trust = ComponentTrustStore::default()
        .trust_digest(&ComponentDigest::sha256(&component_bytes).value);
    let huggingface_only = huggingface_only.replace(
        &ComponentDigest::sha256(LLAMA_REAL_COMPONENT_BYTES).value,
        &ComponentDigest::sha256(&component_bytes).value,
    );
    let digest = register_inference_component_artifact(
        component_bytes,
        huggingface_only.into_bytes(),
        &component_trust,
    )
    .expect("registration succeeds")
    .digest;

    let model_trust =
        ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone());

    // `ProductionLoadedModel` does not implement `Debug`, so this checks
    // the `Result` by hand rather than via `expect_err` (which requires
    // `T: Debug` to format the Ok value in its panic message).
    match ProductionLoadedModel::load_with_component(
        fixture,
        &payload_source,
        model_trust,
        Arc::new(ReferenceCpuProvider::new()),
        Some(digest),
    ) {
        Ok(_) => panic!(
            "a GGUF-declared Model Artifact paired with a real Component that declares \
             `compatibility.artifact_formats: [huggingface]` must be rejected, even though the \
             architecture family (`llama`) matches"
        ),
        Err(error) => {
            assert!(
                matches!(
                    error,
                    InferenceApiError::ModelComponentUnavailable { ref reason }
                        if reason.contains("artifact format unsupported")
                ),
                "expected the rejection to be the artifact-format compatibility gate \
                 (`ModelComponentError::ArtifactFormatUnsupported`), got: {error:#?}"
            );
        }
    }
}

/// `bound-memory-manager-bookkeeping`: a resident [`ProductionLoadedModel`]
/// serving many requests must keep every Runtime/Provider bookkeeping table
/// proportional to live state, not to every request it ever served. Real
/// CUDA campaigns slowed down by ~20 ms per previously generated token:
/// `MemoryManager` kept every released allocation in the map its admission
/// scans walk, edges both pre-admitted and explicitly written orphaned one
/// allocation per step, superseded KV generations left Provider tracking
/// entries behind, and each finished request left its final KV tensors,
/// session and cache records in place. This records each table after every
/// request: the two bounded logs must stay within their caps, everything
/// else must not grow after the second request.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn resident_production_model_bookkeeping_stays_bounded_across_requests() {
    let (fixture, payload_source) = production_shaped_canonical_fixture();
    let provider = Arc::new(ReferenceCpuProvider::new());
    let mut model = ProductionLoadedModel::load(
        fixture.clone(),
        &payload_source,
        ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
        provider.clone(),
    )
    .expect("production loading succeeds");
    let request = || ProductionGenerationRequest {
        prompt: PromptInput::PlainText("hi".into()),
        parameters: GenerationParameters::greedy(),
        stop_conditions: StopConditions::default(),
        max_new_tokens: Some(8),
        max_generation_millis: None,
    };
    let bounded_logs = BTreeMap::from([
        (
            "memory.retired_allocations",
            crate::memory::RETIRED_ALLOCATION_HISTORY_CAPACITY,
        ),
        (
            "memory.observations",
            2 * crate::memory::MEMORY_OBSERVATION_LOG_CAPACITY,
        ),
        (
            "memory.residency_observations",
            2 * crate::memory::MEMORY_OBSERVATION_LOG_CAPACITY,
        ),
    ]);
    let mut history = Vec::new();
    for _ in 0..6 {
        model
            .generate(request(), None)
            .expect("generation succeeds");
        let mut sizes: BTreeMap<String, usize> = model
            .runtime
            .memory()
            .bookkeeping_sizes()
            .into_iter()
            .map(|(name, size)| (format!("memory.{name}"), size))
            .collect();
        model.runtime.memory().assert_allocation_totals_consistent();
        sizes.insert("sessions".into(), model.runtime.sessions().count());
        sizes.insert(
            "kv_caches".into(),
            model.runtime.kv_caches().caches().count(),
        );
        sizes.insert(
            "provider.stored_tensors".into(),
            provider.executor().stored_tensor_count(),
        );
        sizes.insert(
            "provider.tracked_allocations".into(),
            provider.executor().tracked_allocations().len(),
        );
        history.push(sizes);
    }

    let orphaned = {
        let memory = model.runtime.memory();
        let tracked: BTreeSet<_> = provider
            .executor()
            .tracked_allocations()
            .into_values()
            .collect();
        let resident = memory.residency_allocations();
        memory
            .allocations()
            .filter(|allocation| memory.is_live_allocation(allocation.id))
            .filter(|allocation| {
                !tracked.contains(&allocation.id) && !resident.contains(&allocation.id)
            })
            .count()
    };
    assert_eq!(
        orphaned, 0,
        "every live allocation is referenced by a residency record or its Provider"
    );
    let (second, last) = (&history[1], &history[history.len() - 1]);
    for (name, size) in last {
        match bounded_logs.get(name.as_str()) {
            Some(cap) => assert!(size <= cap, "{name} = {size} exceeds its cap {cap}"),
            None => assert!(
                *size <= second[name],
                "{name} grows with every served request: {} after request 1, {size} after \
                 request {}",
                second[name],
                history.len() - 1
            ),
        }
    }
}

/// `attribute-redacted-inference-costs` A06 (tasks 2.2/2.3): a request that
/// crosses graph build, input upload, dispatch, materialization, sampling
/// and commit is attributed per step with leaves that never exceed the
/// step's wall time, device time and overlap are reported unavailable
/// rather than guessed, and nothing request-specific -- here a sentinel in
/// the prompt -- reaches the exported report.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[test]
fn production_generation_steps_are_attributed_by_phase_without_leaking_the_prompt() {
    use crate::phase_attribution::{CounterAvailability, StepKind, StepPhase};

    let (fixture, payload_source) = production_shaped_canonical_fixture();
    let mut model = ProductionLoadedModel::load(
        fixture.clone(),
        &payload_source,
        ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
        Arc::new(ReferenceCpuProvider::new()),
    )
    .expect("production loading succeeds");
    let sentinel = "zq-sentinel-4b1d";
    let request = || ProductionGenerationRequest {
        prompt: PromptInput::PlainText(sentinel.into()),
        parameters: GenerationParameters::greedy(),
        stop_conditions: StopConditions::default(),
        max_new_tokens: Some(4),
        max_generation_millis: None,
    };

    model
        .generate(request(), None)
        .expect("generation succeeds");
    assert!(
        model.take_phase_attribution().steps.is_empty(),
        "attribution is off by default"
    );

    model.enable_phase_attribution();
    let outcome = model
        .generate(request(), None)
        .expect("generation succeeds");
    let report = model.take_phase_attribution();
    let generated = outcome.result.output.usage.generated_tokens as u64;
    assert_eq!(report.device_time, CounterAvailability::Unavailable);
    assert_eq!(report.host_device_overlap, CounterAvailability::Unavailable);
    assert_eq!(report.host_time, CounterAvailability::Measured);
    assert_eq!(report.summary(StepKind::Prefill).unwrap().steps, 1);
    assert_eq!(
        report.summary(StepKind::Decode).unwrap().steps,
        generated - 1
    );
    let preparation = report.summary(StepKind::RequestPreparation).unwrap();
    assert_eq!(preparation.steps, 1);
    let preparation_phases: Vec<StepPhase> = preparation
        .phases
        .iter()
        .map(|(key, _)| key.phase)
        .collect();
    assert_eq!(
        preparation_phases,
        vec![
            StepPhase::SessionCreate,
            StepPhase::Tokenize,
            StepPhase::PlanPrepare,
            StepPhase::RequestValidate,
            StepPhase::GraphBuild,
        ]
    );
    for step in &report.steps {
        assert!(step.attributed_micros() <= step.wall_micros, "{step:?}");
        if step.kind == StepKind::RequestPreparation {
            continue;
        }
        let phases: Vec<StepPhase> = step.phases.iter().map(|(key, _)| key.phase).collect();
        for expected in [
            StepPhase::GraphBuild,
            StepPhase::InstanceCheck,
            StepPhase::KvStatePrepare,
            StepPhase::InputUpload,
            StepPhase::NodeInputResolve,
            StepPhase::NodeDispatch,
            StepPhase::OutputMaterialize,
            StepPhase::LogitsExtract,
            StepPhase::KvStateStore,
            StepPhase::Sampling,
            StepPhase::Commit,
        ] {
            assert!(phases.contains(&expected), "{expected:?} missing: {step:?}");
        }
        // A step's KV update: a pending write (copy-based history) or an
        // in-place store write (`append-kv-history-in-place`).
        assert!(
            phases.contains(&StepPhase::KvPendingWrite) || phases.contains(&StepPhase::KvAppend),
            "no KV update phase: {step:?}"
        );
        assert!(
            step.phases
                .iter()
                .filter(|(key, _)| key.phase == StepPhase::NodeDispatch)
                .all(|(key, _)| key.operator.is_some())
        );
    }
    let decode = report
        .steps
        .iter()
        .find(|step| step.kind == StepKind::Decode)
        .unwrap();
    assert!(
        decode
            .phases
            .iter()
            .any(|(key, _)| key.phase == StepPhase::KvAppend),
        "decode appends to the KV history: {decode:?}"
    );

    let exported = serde_json::to_string(&report).unwrap();
    assert!(!exported.contains(sentinel));
    assert!(!exported.contains("kv."));
    assert!(!exported.contains("edge."));
    assert!(!exported.contains("session-"));
    assert!(!exported.contains("request="));

    model.disable_phase_attribution();
    model
        .generate(request(), None)
        .expect("generation succeeds");
    assert!(model.take_phase_attribution().steps.is_empty());
}
