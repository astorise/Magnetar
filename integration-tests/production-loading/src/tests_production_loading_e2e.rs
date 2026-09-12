use crate::test_support::{VOCAB_SIZE, register_real_qwen_component, write_tiny_production_bundle};
use magnetar_loader_huggingface::HuggingFaceIngestor;
use magnetar_runtime::model::{ModelDType, ModelTrustStore};
use magnetar_runtime::production_model_ingestion::{
    ProductionModelArtifactIngestor, ProductionModelSource,
};
use magnetar_runtime::tokenizer::Tokenizer;
use magnetar_runtime::{
    ModelArtifactSource, load_production_qwen_instance, production_qwen_fixture,
};
use std::{fs, sync::Arc};

#[test]
fn tachyon_shaped_real_production_ingestion_loads_through_the_real_qwen_component() {
    register_real_qwen_component();
    let dir = tempfile::tempdir().unwrap();
    write_tiny_production_bundle(dir.path());

    // Real external ingestion: config.json, tokenizer.json,
    // tokenizer_config.json, and real Safetensors bytes, all parsed by
    // the real loaders/huggingface crate -- no fixture manifest, no hand-
    // built tensor inventory constructor. Tachyon-shaped source identity
    // (task 11.3): Tachyon distributes bundle bytes to an authorized local
    // root and hands Magnetar that Tachyon-sourced identity, not a raw
    // path string -- `root` is what actually authorizes file access;
    // `ModelArtifactSource::Tachyon` here is provenance metadata only
    // (Decision 2/2.4: source kind never grants trust).
    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::Tachyon("tachyon-node-7:integration-test-bundle".into()),
        dir.path().to_path_buf(),
    );
    let ingested = HuggingFaceIngestor::new()
        .ingest(&source)
        .expect("real production ingestion succeeds");
    assert_eq!(
        ingested.manifest.storage_dtype,
        Some(ModelDType::F32),
        "storage dtype normalized from the real Safetensors header"
    );

    // Real tokenizer.json-backed tokenizer, loaded by the real external
    // ingestor's own tokenizer implementation (not FixtureTokenizer's
    // byte-level default).
    let tokenizer_bytes = fs::read(dir.path().join("tokenizer.json")).unwrap();
    let real_tokenizer = magnetar_loader_huggingface::HuggingFaceTokenizer::from_bytes(
        &tokenizer_bytes,
        None,
        "integration-test-tokenizer",
        Some(VOCAB_SIZE),
    )
    .expect("real tokenizer.json loads");
    let tokenizer_metadata = real_tokenizer.metadata().clone();
    let real_tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(real_tokenizer);

    // Runtime trust: an embedder decides what it trusts; this test trusts
    // exactly the digest the real ingestor computed for this bundle.
    let trust_store =
        ModelTrustStore::default().trust_digest(ingested.manifest.id.digest.value.clone());

    let fixture = production_qwen_fixture(
        ingested.manifest.clone(),
        tokenizer_metadata,
        real_tokenizer,
    )
    .expect("production fixture builds from real ingested data");

    // Real generation (task 12.2), through the real compiled Qwen
    // Component: run_production_qwen_generation builds its own Runtime
    // internally (Runtime's execution-engine wiring stays
    // magnetar-runtime-internal by design -- a guarded invariant, not an
    // oversight), then drives loading, session creation, tokenization,
    // graph production, execution-plan preparation, and the generation
    // loop through the same primitives every other first-native caller
    // uses. No Rust-synthesized fallback graph, no fixture manifest, no
    // qwen-test identity anywhere in this path.
    let outcome = magnetar_runtime::run_production_qwen_generation(
        fixture,
        ingested.payload_source.as_ref(),
        trust_store,
        "hi",
    )
    .expect(
        "generation runs end to end: real config.json -> real tokenizer.json -> real \
         Safetensors -> the real compiled Qwen Component -> real Provider dispatch",
    );

    assert!(
        !outcome.result.output.generated_token_ids.is_empty(),
        "real production ingestion + loading + generation produced at least one token"
    );
    // Not asserting specific text (weights are synthetic/random) -- only
    // that the full real-artifact pipeline produced decoded text, not a
    // raw-token-id fallback string.
    assert!(
        !outcome.text.starts_with("[generated token ids:"),
        "the real tokenizer decoded the generated tokens as real text: {}",
        outcome.text
    );

    // `close-tachyon-scope-audit-gaps` task 5.3: real measured throughput,
    // not the always-`None` prior default. Asserted here (not a synthetic
    // fixture-executor unit test) because a trivial fake generation step
    // completes in microseconds -- too fast to reliably register a
    // nonzero measured duration -- while this real Qwen Component +
    // Reference CPU dispatch genuinely takes measurable time.
    let usage = &outcome.result.output.usage;
    assert!(
        usage.tokens_per_second.is_some(),
        "real generation that produced at least one token must report real measured \
         tokens_per_second, not None"
    );
    assert!(
        usage.prefill_duration_millis.is_some(),
        "real generation must report real measured prefill duration"
    );
    let total_millis = usage.prefill_duration_millis.unwrap_or(0)
        + usage.decode_duration_millis.unwrap_or(0);
    assert!(
        total_millis > 0,
        "real generation work must take measurable wall-clock time"
    );
    // Consistent with generated token count and measured duration: within
    // integer-division rounding of generated_tokens * 1000 / total_millis.
    let expected_tokens_per_second =
        (usage.generated_tokens as u64 * 1000) / total_millis;
    assert_eq!(
        usage.tokens_per_second,
        Some(expected_tokens_per_second),
        "tokens_per_second must be derived from this same real generated_tokens/total_millis, \
         not an independent or estimated value"
    );
}

#[test]
fn real_production_ingestion_rejects_when_untrusted() {
    let dir = tempfile::tempdir().unwrap();
    write_tiny_production_bundle(dir.path());
    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
        dir.path().to_path_buf(),
    );
    let ingested = HuggingFaceIngestor::new().ingest(&source).unwrap();

    // Decision 2: parsing/normalizing alone never grants trust. An
    // untrusted Runtime (default trust store) rejects this real ingested
    // manifest before any materialization.
    let mut runtime = magnetar_runtime::Runtime::builder()
        .register_provider(Arc::new(magnetar_runtime::ReferenceCpuProvider::new()))
        .build()
        .unwrap();
    let error = match load_production_qwen_instance(
        &mut runtime,
        &ingested.manifest,
        ingested.payload_source.as_ref(),
    ) {
        Err(error) => error,
        Ok(_) => panic!("expected trust rejection for an untrusted real ingested manifest"),
    };
    assert!(
        format!("{error}").to_lowercase().contains("trust"),
        "expected a trust-shaped rejection, got: {error}"
    );
}

/// A test-only `Provider` that behaves exactly like `ReferenceCpuProvider`
/// except for declaring `supports_multi_step_decode() == false` --
/// `close-tachyon-scope-audit-gaps` task 4.5 needs a Provider with this
/// property to prove the new fail-fast check works, and there is no need
/// to involve real CUDA hardware (a separate, `#[ignore]`d, real-hardware
/// test in `tests_production_loading_cuda_e2e.rs` covers the actual
/// `CudaProvider` override, task 4.6) to prove the generic check itself.
struct MultiStepDecodeUnsupportedProvider(magnetar_runtime::ReferenceCpuProvider);

impl magnetar_runtime::Provider for MultiStepDecodeUnsupportedProvider {
    fn metadata(&self) -> magnetar_runtime::ProviderMetadata {
        self.0.metadata()
    }

    fn register(
        &self,
        registry: &mut magnetar_runtime::ProviderRegistry,
    ) -> Result<(), magnetar_runtime::ProviderError> {
        self.0.register(registry)
    }

    fn health(&self) -> magnetar_runtime::ProviderHealth {
        self.0.health()
    }

    fn devices(&self) -> Vec<Arc<dyn magnetar_runtime::Device>> {
        self.0.devices()
    }

    fn execution_api(&self) -> Option<Arc<dyn magnetar_runtime::ProviderExecutionApi>> {
        self.0.execution_api()
    }

    fn supports_multi_step_decode(&self) -> bool {
        false
    }
}

/// `close-tachyon-scope-audit-gaps` task 4.5: a generation request needing
/// more than one decode step against a Provider declaring
/// `supports_multi_step_decode() == false` fails fast with the new
/// `InferenceApiError::Unsupported`, before any real execution work runs
/// -- proven here by never registering a real Component/trust store/
/// weight materialization capable of succeeding at all (the ingested
/// manifest is never even loaded into a Runtime), so a success or any
/// *other* error would mean the check ran too late or not at all.
#[test]
fn multi_step_decode_request_against_an_unsupporting_provider_fails_fast() {
    let dir = tempfile::tempdir().unwrap();
    write_tiny_production_bundle(dir.path());
    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
        dir.path().to_path_buf(),
    );
    let mut ingested = HuggingFaceIngestor::new().ingest(&source).unwrap();
    ingested.manifest.generation = Some(magnetar_runtime::model::ModelGenerationDefaults {
        max_tokens: Some(2),
        ..Default::default()
    });

    let tokenizer_bytes = std::fs::read(dir.path().join("tokenizer.json")).unwrap();
    let real_tokenizer = magnetar_loader_huggingface::HuggingFaceTokenizer::from_bytes(
        &tokenizer_bytes,
        None,
        "unsupported-provider-test-tokenizer",
        Some(VOCAB_SIZE),
    )
    .expect("real tokenizer.json loads");
    let tokenizer_metadata = real_tokenizer.metadata().clone();
    let real_tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(real_tokenizer);

    // Deliberately no `register_real_qwen_component()` and an untrusted
    // `ModelTrustStore::default()` -- if the new check did not run before
    // any real execution work, this would fail with a *different* error
    // (component-unavailable or trust-rejected) instead of `Unsupported`.
    let fixture = production_qwen_fixture(
        ingested.manifest.clone(),
        tokenizer_metadata,
        real_tokenizer,
    )
    .expect("production fixture builds from real ingested data");

    let error = magnetar_runtime::run_production_qwen_generation_for_provider(
        fixture,
        ingested.payload_source.as_ref(),
        ModelTrustStore::default(),
        "hi",
        Arc::new(MultiStepDecodeUnsupportedProvider(
            magnetar_runtime::ReferenceCpuProvider::new(),
        )),
    )
    .expect_err(
        "a 2-decode-step request against a Provider declaring \
         supports_multi_step_decode() == false must fail before any real execution work runs",
    );
    assert!(
        matches!(error, magnetar_runtime::InferenceApiError::Unsupported { .. }),
        "expected InferenceApiError::Unsupported, got: {error:?}"
    );
}

/// Counts allocations still genuinely holding memory -- `MemoryManager::
/// release` deliberately leaves a released allocation's ledger entry in
/// place (marked `Released`/`Reusable` for caching/audit purposes,
/// never removed outright: see its own doc comment), so
/// `MemoryManager::allocations().count()` alone conflates "ever
/// allocated" with "currently active" and cannot detect a leak by
/// itself.
fn active_allocation_count(runtime: &magnetar_runtime::Runtime) -> usize {
    runtime
        .memory()
        .allocations()
        .filter(|allocation| allocation.state == magnetar_runtime::MemoryAllocationState::Active)
        .count()
}

/// Task 12.6: unloading a real production-loaded instance leaves no
/// *active* Memory Manager allocation behind. Load through the real
/// external ingestor's output, confirm real active allocations exist
/// (weight materialization genuinely admitted something, not a no-op),
/// unload, and confirm the active count returns to exactly what it was
/// before loading -- not merely "less than during", which a partial leak
/// could still satisfy.
#[test]
fn unloading_a_real_production_instance_leaves_no_memory_manager_allocation() {
    register_real_qwen_component();
    let dir = tempfile::tempdir().unwrap();
    write_tiny_production_bundle(dir.path());
    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
        dir.path().to_path_buf(),
    );
    let ingested = HuggingFaceIngestor::new().ingest(&source).unwrap();

    let mut runtime = magnetar_runtime::Runtime::builder()
        .register_provider(Arc::new(magnetar_runtime::ReferenceCpuProvider::new()))
        .trust_store(
            ModelTrustStore::default().trust_digest(ingested.manifest.id.digest.value.clone()),
        )
        .build()
        .unwrap();

    let baseline_active = active_allocation_count(&runtime);

    let instance = load_production_qwen_instance(
        &mut runtime,
        &ingested.manifest,
        ingested.payload_source.as_ref(),
    )
    .expect("real production loading succeeds");
    let active_while_loaded = active_allocation_count(&runtime);
    assert!(
        active_while_loaded > baseline_active,
        "expected weight materialization to admit at least one real active Memory Manager \
         allocation (baseline={baseline_active}, while loaded={active_while_loaded})"
    );

    magnetar_runtime::unload_model_instance(
        &mut runtime,
        &instance,
        magnetar_runtime::ModelInstanceUnloadPolicy::DrainActiveUse,
    )
    .expect("model instance unloads cleanly");

    let active_after_unload = active_allocation_count(&runtime);
    assert_eq!(
        active_after_unload, baseline_active,
        "unloading a real production-loaded instance must release every active Memory \
         Manager allocation it admitted, not merely some of them"
    );
}
