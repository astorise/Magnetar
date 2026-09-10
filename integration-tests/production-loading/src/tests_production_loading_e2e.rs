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
