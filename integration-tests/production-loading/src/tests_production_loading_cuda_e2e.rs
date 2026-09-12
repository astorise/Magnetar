//! Real production Qwen loading end to end on real CUDA hardware (task
//! 12.3): the same real ingested bundle, real tokenizer, and real
//! compiled Qwen Component `tests_production_loading_e2e.rs` proves on
//! Reference CPU, instead dispatched through a real
//! `magnetar_provider_cuda::CudaProvider` via `run_production_qwen_
//! generation_for_provider` -- zero CUDA-specific code in
//! `magnetar-runtime` itself. Skips (returns early, does not fail) when
//! no CUDA-capable device is available, matching `integration-tests/
//! cuda-first-native`'s own pattern.

use crate::test_support::{VOCAB_SIZE, register_real_qwen_component, write_tiny_production_bundle};
use magnetar_loader_huggingface::HuggingFaceIngestor;
use magnetar_provider_cuda::CudaProvider;
use magnetar_runtime::model::ModelTrustStore;
use magnetar_runtime::production_model_ingestion::{
    ProductionModelArtifactIngestor, ProductionModelSource,
};
use magnetar_runtime::tokenizer::Tokenizer;
use magnetar_runtime::{ModelArtifactSource, production_qwen_fixture};
use std::sync::Arc;

#[test]
fn real_production_ingestion_generates_on_real_cuda_hardware() {
    let provider = CudaProvider::new();
    if !provider.is_available() {
        return;
    }
    register_real_qwen_component();

    let dir = tempfile::tempdir().unwrap();
    write_tiny_production_bundle(dir.path());

    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::Tachyon("tachyon-node-7:cuda-integration-test-bundle".into()),
        dir.path().to_path_buf(),
    );
    let mut ingested = HuggingFaceIngestor::new()
        .ingest(&source)
        .expect("real production ingestion succeeds");
    // Prefill only (task 12.3's scope: prove real production ingestion ->
    // loading -> the real compiled Qwen Component -> real CUDA Provider
    // dispatch). Multi-step decode's historical-KV-history concatenation
    // currently requires host-readable tensor bytes
    // (`TensorValue::into_host`), which `CudaProvider::read_tensor_value`
    // deliberately never provides (device-resident by design; only
    // `read_tensor` downloads) -- a real, pre-existing CUDA multi-step
    // decode gap unrelated to production loading itself, out of this
    // change's scope to fix. `max_tokens: 1` means only prefill runs (the
    // first generated token comes directly from it), so this test never
    // reaches a decode step that would need it.
    ingested.manifest.generation = Some(magnetar_runtime::model::ModelGenerationDefaults {
        max_tokens: Some(1),
        ..Default::default()
    });

    let tokenizer_bytes = std::fs::read(dir.path().join("tokenizer.json")).unwrap();
    let real_tokenizer = magnetar_loader_huggingface::HuggingFaceTokenizer::from_bytes(
        &tokenizer_bytes,
        None,
        "cuda-integration-test-tokenizer",
        Some(VOCAB_SIZE),
    )
    .expect("real tokenizer.json loads");
    let tokenizer_metadata = real_tokenizer.metadata().clone();
    let real_tokenizer: Arc<dyn magnetar_runtime::tokenizer::Tokenizer + Send + Sync> =
        Arc::new(real_tokenizer);

    let trust_store =
        ModelTrustStore::default().trust_digest(ingested.manifest.id.digest.value.clone());
    let fixture = production_qwen_fixture(
        ingested.manifest.clone(),
        tokenizer_metadata,
        real_tokenizer,
    )
    .expect("production fixture builds from real ingested data");

    let outcome = magnetar_runtime::run_production_qwen_generation_for_provider(
        fixture,
        ingested.payload_source.as_ref(),
        trust_store,
        "hi",
        Arc::new(CudaProvider::new()),
    )
    .expect(
        "generation runs end to end on real CUDA hardware: real config.json -> real \
         tokenizer.json -> real Safetensors -> the real compiled Qwen Component -> real CUDA \
         Provider dispatch",
    );

    assert!(
        !outcome.result.output.generated_token_ids.is_empty(),
        "real production ingestion + loading + generation on CUDA produced at least one token"
    );
    assert!(
        !outcome.text.starts_with("[generated token ids:"),
        "the real tokenizer decoded the generated tokens as real text: {}",
        outcome.text
    );
    // No silent Reference CPU fallback (task 12.3) is structural here, not
    // merely observed: this Runtime never registers a Reference CPU
    // Provider at all (only the CudaProvider passed to run_production_
    // qwen_generation_for_provider), so a successful dispatch could only
    // have gone through CUDA -- there was nothing else for it to fall
    // back to.

    // `close-tachyon-scope-audit-gaps` task 5.2: the shared generation
    // loop measures real wall-clock time identically for every Provider,
    // with zero CUDA-specific code -- proven here on real CUDA hardware
    // itself, not merely assumed by symmetry with the Reference CPU test.
    assert!(
        outcome.result.output.usage.tokens_per_second.is_some(),
        "real CUDA generation that produced at least one token must report real measured \
         tokens_per_second, not None"
    );
    assert!(
        outcome.result.output.usage.prefill_duration_millis.is_some(),
        "real CUDA generation must report real measured prefill duration"
    );
}

/// `close-tachyon-scope-audit-gaps` task 4.6: on real CUDA hardware, a
/// generation request needing more than one decode step now fails with
/// the new, explicit `InferenceApiError::Unsupported` -- checked before
/// any real prefill work runs -- instead of the internal
/// `TensorError::ResidencyUnavailable` that `real_production_ingestion_
/// generates_on_real_cuda_hardware` above works around today by pinning
/// `max_tokens: 1`. Same real ingested bundle/tokenizer/Component, same
/// real `CudaProvider`; only `max_tokens` differs.
#[test]
#[ignore = "requires real CUDA hardware; run via gpu-runner-smoke.yml (task 4.6)"]
fn multi_step_decode_on_real_cuda_hardware_fails_with_explicit_unsupported() {
    let provider = CudaProvider::new();
    if !provider.is_available() {
        return;
    }
    register_real_qwen_component();

    let dir = tempfile::tempdir().unwrap();
    write_tiny_production_bundle(dir.path());

    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::Tachyon(
            "tachyon-node-7:cuda-unsupported-decode-integration-test-bundle".into(),
        ),
        dir.path().to_path_buf(),
    );
    let mut ingested = HuggingFaceIngestor::new()
        .ingest(&source)
        .expect("real production ingestion succeeds");
    ingested.manifest.generation = Some(magnetar_runtime::model::ModelGenerationDefaults {
        max_tokens: Some(2),
        ..Default::default()
    });

    let tokenizer_bytes = std::fs::read(dir.path().join("tokenizer.json")).unwrap();
    let real_tokenizer = magnetar_loader_huggingface::HuggingFaceTokenizer::from_bytes(
        &tokenizer_bytes,
        None,
        "cuda-unsupported-decode-test-tokenizer",
        Some(VOCAB_SIZE),
    )
    .expect("real tokenizer.json loads");
    let tokenizer_metadata = real_tokenizer.metadata().clone();
    let real_tokenizer: Arc<dyn magnetar_runtime::tokenizer::Tokenizer + Send + Sync> =
        Arc::new(real_tokenizer);

    let trust_store =
        ModelTrustStore::default().trust_digest(ingested.manifest.id.digest.value.clone());
    let fixture = production_qwen_fixture(
        ingested.manifest.clone(),
        tokenizer_metadata,
        real_tokenizer,
    )
    .expect("production fixture builds from real ingested data");

    let error = magnetar_runtime::run_production_qwen_generation_for_provider(
        fixture,
        ingested.payload_source.as_ref(),
        trust_store,
        "hi",
        Arc::new(CudaProvider::new()),
    )
    .expect_err(
        "a 2-decode-step request against the real CudaProvider must fail with an explicit \
         Unsupported error, not succeed and not fail with an internal residency error",
    );
    assert!(
        matches!(error, magnetar_runtime::InferenceApiError::Unsupported { .. }),
        "expected InferenceApiError::Unsupported (not TensorError::ResidencyUnavailable \
         surfacing some other way), got: {error:?}"
    );
}

/// Task 12.7: a GPU CI job that dispatches this crate's tests must not be
/// able to report success while every CUDA-gated test above silently
/// skipped (returned early because no device was available) -- that
/// would make "0 real assertions ran" indistinguishable from "every
/// assertion passed". Mirrors `providers/cuda`'s own
/// `hardware_conformance_actually_ran_not_silently_skipped` exactly:
/// `#[ignore]`d by default so a GPU-less dev machine's plain `cargo test`
/// still passes cleanly, and run explicitly via `--include-ignored` only
/// by a job that guarantees a real GPU, where this test's failure means
/// the runner itself lost GPU access.
#[test]
#[ignore = "run explicitly via `cargo test -- --include-ignored` on a host guaranteed to have a GPU; every other test in this file already covers the GPU-less path"]
fn hardware_conformance_actually_ran_not_silently_skipped() {
    let provider = CudaProvider::new();
    assert!(
        provider.is_available(),
        "this test only runs where a compatible CUDA driver/device is guaranteed present -- \
         if it fails, the runner lost GPU access"
    );
}
