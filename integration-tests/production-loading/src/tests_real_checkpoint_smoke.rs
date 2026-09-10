//! Task 12.4: a real, public Qwen-compatible checkpoint smoke test,
//! pinned by exact revision and content digest -- not the tiny synthetic
//! bundle `tests_production_loading_e2e.rs`/`tests_production_loading_
//! cuda_e2e.rs` use for per-PR speed. Downloads ~1GB and runs a real
//! 24-layer, 896-hidden, GQA (14 attention / 2 KV heads) model through
//! the full production pipeline, so it is `#[ignore]`d by default and
//! meant for a manual or nightly run, never per-PR (the task's own
//! "manual/nightly/hardware profile acceptable if size prevents per-PR
//! execution").
//!
//! To run:
//! 1. Download these files from `Qwen/Qwen2.5-0.5B-Instruct` at the
//!    exact pinned revision below into one directory: `config.json`,
//!    `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`,
//!    `model.safetensors`.
//!    e.g. for each FILE:
//!    `curl -L "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/<revision>/FILE" -o DIR/FILE`
//! 2. `MAGNETAR_QWEN_REAL_CHECKPOINT_DIR=<that directory>`
//! 3. `cargo test --release --manifest-path integration-tests/production-loading/Cargo.toml \
//!      real_public_checkpoint -- --include-ignored --nocapture`
//!
//! This test verifies the downloaded `model.safetensors`' real sha256
//! against the digest this revision declares (a download/CDN-tamper
//! check, independent of and prior to the Runtime's own trust-store
//! evaluation, which only ever trusts a digest this test computed itself
//! from the *ingested* manifest -- Decision 2 still holds: nothing here
//! grants trust merely because a well-known public repo/revision string
//! was named).

use crate::test_support::register_real_qwen_component;
use magnetar_loader_huggingface::{
    HuggingFaceIngestor, HuggingFaceTokenizer, parse_tokenizer_config,
};
use magnetar_runtime::model::ModelDigest;
use magnetar_runtime::model::ModelTrustStore;
use magnetar_runtime::production_model_ingestion::{
    ProductionModelArtifactIngestor, ProductionModelSource,
};
use magnetar_runtime::tokenizer::Tokenizer;
use magnetar_runtime::{ModelArtifactSource, ModelGenerationDefaults, production_qwen_fixture};
use std::{fs, path::PathBuf, sync::Arc};

/// `Qwen/Qwen2.5-0.5B-Instruct`'s exact pinned commit on Hugging Face:
/// <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/commit/7ae557604adf67be50417f59c2c2f167def9a775>.
pub(crate) const CHECKPOINT_REVISION: &str = "7ae557604adf67be50417f59c2c2f167def9a775";
/// `model.safetensors`' declared LFS sha256 at that exact revision (from
/// the Hugging Face Hub API's own `siblings[].lfs.sha256` for this file),
/// verified against the real downloaded bytes before this test trusts
/// them at all.
pub(crate) const CHECKPOINT_WEIGHTS_SHA256: &str =
    "sha256:fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe";
const REAL_VOCAB_SIZE: u64 = 151_936;

pub(crate) fn checkpoint_dir() -> Option<PathBuf> {
    std::env::var_os("MAGNETAR_QWEN_REAL_CHECKPOINT_DIR").map(PathBuf::from)
}

pub(crate) fn require_checkpoint_dir() -> PathBuf {
    checkpoint_dir().unwrap_or_else(|| {
        panic!(
            "MAGNETAR_QWEN_REAL_CHECKPOINT_DIR is not set. Download Qwen/Qwen2.5-0.5B-Instruct \
             revision {CHECKPOINT_REVISION} (config.json, generation_config.json, \
             tokenizer.json, tokenizer_config.json, model.safetensors) into a directory and set \
             that env var to it before running this test with --include-ignored -- see this \
             file's own module doc comment for the exact recipe."
        )
    })
}

/// Verifies `dir/model.safetensors`' real sha256 against
/// [`CHECKPOINT_WEIGHTS_SHA256`] and returns the checkpoint's real,
/// pinned-revision-authorized [`ProductionModelSource`].
pub(crate) fn verified_checkpoint_source(dir: &std::path::Path) -> ProductionModelSource {
    let weights_bytes = fs::read(dir.join("model.safetensors"))
        .expect("model.safetensors is readable in the checkpoint directory");
    let actual_digest = ModelDigest::sha256(&weights_bytes);
    assert_eq!(
        actual_digest.value, CHECKPOINT_WEIGHTS_SHA256,
        "the downloaded model.safetensors does not match revision {CHECKPOINT_REVISION}'s \
         declared content digest -- re-download it"
    );
    ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::HuggingFace(format!(
            "Qwen/Qwen2.5-0.5B-Instruct@{CHECKPOINT_REVISION}"
        )),
        dir.to_path_buf(),
    )
}

/// Task 12.4: a real public Qwen-compatible checkpoint, pinned by
/// revision/digest, loads through the real ingestor, the real compiled
/// Qwen Component, and real Reference CPU generation -- not a synthetic
/// tiny bundle. `max_tokens` is deliberately bounded (a handful of real
/// decode steps, not this checkpoint's own unset-in-`generation_config.
/// json` 64-token Runtime default) so a manual/nightly run finishes in a
/// reasonable time; the point is proving the real pipeline handles a
/// real checkpoint's actual scale (24 layers, 896 hidden, 14 attention /
/// 2 KV heads), not benchmarking full-length generation.
#[test]
#[ignore = "downloads/holds a ~1GB real checkpoint; manual/nightly profile only (task 12.4)"]
fn real_public_checkpoint_loads_and_generates_on_reference_cpu() {
    let dir = require_checkpoint_dir();
    let source = verified_checkpoint_source(&dir);

    register_real_qwen_component();

    let mut ingested = HuggingFaceIngestor::new()
        .ingest(&source)
        .expect("real Qwen2.5-0.5B-Instruct bundle ingests");
    assert_eq!(
        ingested
            .manifest
            .architecture_config
            .as_ref()
            .unwrap()
            .num_hidden_layers,
        24
    );
    assert!(
        ingested
            .manifest
            .architecture_config
            .as_ref()
            .unwrap()
            .tie_word_embeddings
    );
    assert!(
        ingested
            .manifest
            .tensors
            .iter()
            .any(|tensor| tensor.name == "lm_head"),
        "tied checkpoint: lm_head must have been derived from token_embedding (task 10.5)"
    );
    ingested.manifest.generation = Some(ModelGenerationDefaults {
        max_tokens: Some(8),
        ..Default::default()
    });

    let tokenizer_json = fs::read(dir.join("tokenizer.json")).expect("tokenizer.json readable");
    let tokenizer_config_bytes =
        fs::read(dir.join("tokenizer_config.json")).expect("tokenizer_config.json readable");
    let tokenizer_config =
        parse_tokenizer_config(&tokenizer_config_bytes).expect("real tokenizer_config.json parses");
    let real_tokenizer = HuggingFaceTokenizer::from_bytes(
        &tokenizer_json,
        Some(&tokenizer_config),
        "qwen2.5-0.5b-instruct",
        Some(REAL_VOCAB_SIZE),
    )
    .expect("real tokenizer.json loads and matches the declared vocab_size");
    let tokenizer_metadata = real_tokenizer.metadata().clone();
    let real_tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(real_tokenizer);

    let trust_store =
        ModelTrustStore::default().trust_digest(ingested.manifest.id.digest.value.clone());

    let fixture = production_qwen_fixture(
        ingested.manifest.clone(),
        tokenizer_metadata,
        real_tokenizer,
    )
    .expect("production fixture builds from the real ingested Qwen2.5-0.5B-Instruct data");

    let outcome = magnetar_runtime::run_production_qwen_generation(
        fixture,
        ingested.payload_source.as_ref(),
        trust_store,
        "The capital of France is",
    )
    .expect(
        "generation runs end to end on a real public checkpoint: real config.json -> real \
         tokenizer.json -> real Safetensors weights -> the real compiled Qwen Component -> \
         real Reference CPU dispatch",
    );

    assert!(
        !outcome.result.output.generated_token_ids.is_empty(),
        "real checkpoint generation produced at least one token"
    );
    assert!(
        !outcome.text.starts_with("[generated token ids:"),
        "the real tokenizer decoded the generated tokens as real text: {}",
        outcome.text
    );
    eprintln!("real Qwen2.5-0.5B-Instruct generated: {:?}", outcome.text);
}

/// Task 12.5: compares real Reference CPU and real CUDA deterministic
/// output on the same real checkpoint, same real ingested manifest, and
/// same prompt. Prefill-only (`max_tokens: 1`, matching `tests_
/// production_loading_cuda_e2e.rs`'s own documented reason: multi-step
/// decode's historical-KV-history concatenation needs host-readable
/// tensor bytes, which `CudaProvider::read_tensor_value` deliberately
/// never provides). Greedy sampling on identical input is expected to
/// pick the same argmax token on both Providers even though CPU/GPU
/// floating-point accumulation order differs (their real numeric
/// difference, if any, lands far below what would flip an already
/// well-separated real-word logit distribution's argmax); a documented
/// tolerance of "exactly the same decoded token id and text" is used
/// rather than a numeric logit tolerance because neither Provider's
/// generation output exposes raw logits at this public boundary. Skips
/// (does not fail) when no CUDA-capable device is available, matching
/// this crate's other CUDA test.
#[test]
#[ignore = "downloads/holds a ~1GB real checkpoint; manual/nightly hardware profile only (task 12.5)"]
fn real_public_checkpoint_prefill_output_matches_between_cpu_and_cuda() {
    let provider = magnetar_provider_cuda::CudaProvider::new();
    if !provider.is_available() {
        return;
    }
    let dir = require_checkpoint_dir();
    let source = verified_checkpoint_source(&dir);
    register_real_qwen_component();

    let mut ingested = HuggingFaceIngestor::new()
        .ingest(&source)
        .expect("real Qwen2.5-0.5B-Instruct bundle ingests");
    ingested.manifest.generation = Some(ModelGenerationDefaults {
        max_tokens: Some(1),
        ..Default::default()
    });

    let tokenizer_json = fs::read(dir.join("tokenizer.json")).expect("tokenizer.json readable");
    let tokenizer_config_bytes =
        fs::read(dir.join("tokenizer_config.json")).expect("tokenizer_config.json readable");
    let tokenizer_config =
        parse_tokenizer_config(&tokenizer_config_bytes).expect("real tokenizer_config.json parses");
    let real_tokenizer = HuggingFaceTokenizer::from_bytes(
        &tokenizer_json,
        Some(&tokenizer_config),
        "qwen2.5-0.5b-instruct",
        Some(REAL_VOCAB_SIZE),
    )
    .expect("real tokenizer.json loads and matches the declared vocab_size");
    let tokenizer_metadata = real_tokenizer.metadata().clone();
    let real_tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(real_tokenizer);

    let trust_store =
        ModelTrustStore::default().trust_digest(ingested.manifest.id.digest.value.clone());
    let cpu_fixture = production_qwen_fixture(
        ingested.manifest.clone(),
        tokenizer_metadata.clone(),
        Arc::clone(&real_tokenizer),
    )
    .expect("production fixture builds for the Reference CPU run");
    let cpu_outcome = magnetar_runtime::run_production_qwen_generation(
        cpu_fixture,
        ingested.payload_source.as_ref(),
        trust_store.clone(),
        "The capital of France is",
    )
    .expect("Reference CPU generation runs end to end on the real checkpoint");

    let cuda_fixture = production_qwen_fixture(
        ingested.manifest.clone(),
        tokenizer_metadata,
        real_tokenizer,
    )
    .expect("production fixture builds for the CUDA run");
    let cuda_outcome = magnetar_runtime::run_production_qwen_generation_for_provider(
        cuda_fixture,
        ingested.payload_source.as_ref(),
        trust_store,
        "The capital of France is",
        Arc::new(magnetar_provider_cuda::CudaProvider::new()),
    )
    .expect("CUDA generation runs end to end on the real checkpoint");

    assert_eq!(
        cpu_outcome.result.output.generated_token_ids,
        cuda_outcome.result.output.generated_token_ids,
        "Reference CPU and CUDA must select the same greedy prefill token on identical real \
         checkpoint/config/tokenizer/prompt input"
    );
    assert_eq!(
        cpu_outcome.text, cuda_outcome.text,
        "Reference CPU and CUDA must decode to the same real text"
    );
    eprintln!(
        "real Qwen2.5-0.5B-Instruct prefill token matched on CPU and CUDA: {:?}",
        cpu_outcome.text
    );
}
