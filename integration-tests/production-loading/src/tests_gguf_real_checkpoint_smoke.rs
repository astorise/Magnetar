//! Real, public GGUF checkpoint smoke tests, pinned by exact revision and
//! content digest -- not the tiny synthetic bundle
//! `tests_gguf_loading_e2e.rs` uses for per-PR speed. Two variants of the
//! same real `Qwen2.5-0.5B-Instruct` checkpoint: an unquantized F16
//! export, and a genuinely `Q8_0`-quantized export (the real target
//! `resolve-gguf-quantized-projection-transpose-sequencing` exists for --
//! before that fix, every projection weight in the quantized export would
//! have been rejected structurally). Both run through the full production
//! pipeline, so both are `#[ignore]`d by default and meant for a manual or
//! nightly run, never per-PR -- mirrors `tests_real_checkpoint_smoke.rs`'s
//! own rationale exactly.
//!
//! To run the unquantized export:
//! 1. `curl -L "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/<revision>/qwen2.5-0.5b-instruct-fp16.gguf" -o DIR/model.gguf`
//! 2. `MAGNETAR_QWEN_GGUF_REAL_CHECKPOINT_DIR=<that directory>`
//! 3. `cargo test --release --manifest-path integration-tests/production-loading/Cargo.toml \
//!      real_public_gguf_checkpoint -- --include-ignored --nocapture`
//!
//! To run the `Q8_0`-quantized export instead, substitute
//! `qwen2.5-0.5b-instruct-q8_0.gguf` and
//! `MAGNETAR_QWEN_GGUF_Q8_0_REAL_CHECKPOINT_DIR`, and filter on
//! `real_public_q8_0_gguf_checkpoint` instead.
//!
//! Both tests verify the downloaded file's real sha256 against the digest
//! this revision declares (a download/CDN-tamper check, independent of
//! and prior to the Runtime's own trust-store evaluation, which only ever
//! trusts a digest each test computed itself from the *ingested* manifest
//! -- Decision 2 still holds).

use crate::test_support::register_real_qwen_component;
use magnetar_loader_gguf::{GgufIngestor, GgufTokenizer};
use magnetar_runtime::model::{ModelDigest, ModelTrustStore};
use magnetar_runtime::production_model_ingestion::{
    ProductionModelArtifactIngestor, ProductionModelSource,
};
use magnetar_runtime::tokenizer::Tokenizer;
use magnetar_runtime::{ModelArtifactSource, production_qwen_fixture};
use std::{fs, path::PathBuf, sync::Arc};

/// `Qwen/Qwen2.5-0.5B-Instruct-GGUF`'s exact pinned commit on Hugging
/// Face: <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/commit/9217f5db79a29953eb74d5343926648285ec7e67>.
pub(crate) const CHECKPOINT_REVISION: &str = "9217f5db79a29953eb74d5343926648285ec7e67";
/// `qwen2.5-0.5b-instruct-fp16.gguf`'s real sha256 at that exact
/// revision, computed from the real downloaded bytes and verified before
/// this test trusts them at all.
pub(crate) const CHECKPOINT_SHA256: &str =
    "sha256:8e0ae26000627ed62de0e78e41860af70094558b9d2913385c842a6aa06cf3fc";

fn checkpoint_dir() -> Option<PathBuf> {
    std::env::var_os("MAGNETAR_QWEN_GGUF_REAL_CHECKPOINT_DIR").map(PathBuf::from)
}

fn require_checkpoint_dir() -> PathBuf {
    checkpoint_dir().unwrap_or_else(|| {
        panic!(
            "MAGNETAR_QWEN_GGUF_REAL_CHECKPOINT_DIR is not set. Download Qwen/Qwen2.5-0.5B-\
             Instruct-GGUF revision {CHECKPOINT_REVISION}'s qwen2.5-0.5b-instruct-fp16.gguf into \
             a directory as 'model.gguf' and set that env var to it before running this test \
             with --include-ignored -- see this file's own module doc comment for the exact \
             recipe."
        )
    })
}

fn verified_checkpoint_source(dir: &std::path::Path) -> ProductionModelSource {
    let bytes = fs::read(dir.join("model.gguf"))
        .expect("model.gguf is readable in the checkpoint directory");
    let actual_digest = ModelDigest::sha256(&bytes);
    assert_eq!(
        actual_digest.value, CHECKPOINT_SHA256,
        "the downloaded model.gguf does not match revision {CHECKPOINT_REVISION}'s declared \
         content digest -- re-download it"
    );
    ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::HuggingFace(format!(
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF@{CHECKPOINT_REVISION}"
        )),
        dir.to_path_buf(),
    )
}

/// A real public GGUF-format checkpoint, pinned by revision/digest, loads
/// through the real `GgufIngestor`, the real compiled Qwen Component, and
/// real Reference CPU generation -- not a synthetic tiny bundle.
/// `max_tokens` is deliberately bounded so a manual/nightly run finishes
/// in a reasonable time.
#[test]
#[ignore = "downloads/holds a ~1.2GB real checkpoint; manual/nightly profile only"]
fn real_public_gguf_checkpoint_loads_and_generates_on_reference_cpu() {
    let dir = require_checkpoint_dir();
    let source = verified_checkpoint_source(&dir);

    register_real_qwen_component();

    let mut ingested = GgufIngestor::new()
        .ingest(&source)
        .expect("real Qwen2.5-0.5B-Instruct GGUF file ingests");
    let config = ingested
        .manifest
        .architecture_config
        .as_ref()
        .expect("architecture config was normalized");
    assert_eq!(config.num_hidden_layers, 24);
    assert_eq!(config.hidden_size, 896);
    assert_eq!(config.num_attention_heads, 14);
    assert_eq!(config.num_key_value_heads, 2);
    // `Qwen2.5-0.5B-Instruct`'s own `config.json` declares
    // `tie_word_embeddings: true`, and a Hugging Face/Safetensors export
    // of it genuinely omits `lm_head.weight` (see
    // `tests_real_checkpoint_smoke.rs`) -- but this checkpoint's real
    // llama.cpp-produced GGUF export, discovered running this test
    // against the real downloaded file, includes an explicit
    // `output.weight` tensor regardless (llama.cpp's own conversion
    // convention does not omit it just because the source model is
    // tied). This ingestor's `tie_word_embeddings` is evidence-based
    // (whether a real `lm_head`-canonical tensor was discovered), so it
    // correctly reports `false` here -- the real, present `output.weight`
    // tensor is used directly, which is exactly as correct as deriving a
    // synthetic one would have been.
    assert!(
        ingested
            .manifest
            .tensors
            .iter()
            .any(|tensor| tensor.name == "lm_head"),
        "a real lm_head tensor (whether declared by the file or derived from token_embedding) \
         must be present"
    );
    ingested.manifest.generation = Some(magnetar_runtime::model::ModelGenerationDefaults {
        max_tokens: Some(8),
        ..Default::default()
    });

    let file_bytes = fs::read(dir.join("model.gguf")).unwrap();
    let artifact = magnetar_format_gguf::parse(&file_bytes).expect("real GGUF file parses");
    let real_tokenizer = GgufTokenizer::from_gguf_metadata(
        &artifact.metadata,
        "qwen2.5-0.5b-instruct-gguf",
        Some(config.vocab_size),
    )
    .expect("real tokenizer builds from the checkpoint's own embedded vocabulary");
    let tokenizer_metadata = real_tokenizer.metadata().clone();
    let real_tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(real_tokenizer);

    let trust_store =
        ModelTrustStore::default().trust_digest(ingested.manifest.id.digest.value.clone());
    let fixture = production_qwen_fixture(
        ingested.manifest.clone(),
        tokenizer_metadata,
        real_tokenizer,
    )
    .expect("production fixture builds from the real ingested GGUF checkpoint");

    let outcome = magnetar_runtime::run_production_qwen_generation(
        fixture,
        ingested.payload_source.as_ref(),
        trust_store,
        "The capital of France is",
    )
    .expect(
        "generation runs end to end on a real public GGUF checkpoint: real GGUF file -> real \
         embedded tokenizer -> real Safetensors-equivalent weights -> the real compiled Qwen \
         Component -> real Reference CPU dispatch",
    );

    assert!(
        !outcome.result.output.generated_token_ids.is_empty(),
        "real GGUF checkpoint generation produced at least one token"
    );
    assert!(
        !outcome.text.starts_with("[generated token ids:"),
        "the real embedded tokenizer decoded the generated tokens as real text: {}",
        outcome.text
    );
    eprintln!(
        "real Qwen2.5-0.5B-Instruct GGUF generated: {:?}",
        outcome.text
    );
}

/// `qwen2.5-0.5b-instruct-q8_0.gguf`'s real sha256 at the same pinned
/// revision as the unquantized export above.
pub(crate) const Q8_0_CHECKPOINT_SHA256: &str =
    "sha256:ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e";

fn q8_0_checkpoint_dir() -> Option<PathBuf> {
    std::env::var_os("MAGNETAR_QWEN_GGUF_Q8_0_REAL_CHECKPOINT_DIR").map(PathBuf::from)
}

fn require_q8_0_checkpoint_dir() -> PathBuf {
    q8_0_checkpoint_dir().unwrap_or_else(|| {
        panic!(
            "MAGNETAR_QWEN_GGUF_Q8_0_REAL_CHECKPOINT_DIR is not set. Download Qwen/Qwen2.5-0.5B-\
             Instruct-GGUF revision {CHECKPOINT_REVISION}'s qwen2.5-0.5b-instruct-q8_0.gguf into \
             a directory as 'model.gguf' and set that env var to it before running this test \
             with --include-ignored."
        )
    })
}

fn verified_q8_0_checkpoint_source(dir: &std::path::Path) -> ProductionModelSource {
    let bytes = fs::read(dir.join("model.gguf"))
        .expect("model.gguf is readable in the checkpoint directory");
    let actual_digest = ModelDigest::sha256(&bytes);
    assert_eq!(
        actual_digest.value, Q8_0_CHECKPOINT_SHA256,
        "the downloaded model.gguf does not match revision {CHECKPOINT_REVISION}'s declared \
         content digest -- re-download it"
    );
    ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::HuggingFace(format!(
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF@{CHECKPOINT_REVISION}"
        )),
        dir.to_path_buf(),
    )
}

/// The real target this whole chantier exists for: a genuinely
/// `Q8_0`-quantized real public checkpoint, loading and generating end to
/// end through the same real production pipeline the unquantized export
/// above already proves -- the dequantize-then-transpose sequencing fix
/// (`resolve-gguf-quantized-projection-transpose-sequencing`) is what
/// makes this possible at all; before it, every projection weight in this
/// file would have been rejected structurally.
#[test]
#[ignore = "downloads/holds a ~650MB real quantized checkpoint; manual/nightly profile only"]
fn real_public_q8_0_gguf_checkpoint_loads_and_generates_on_reference_cpu() {
    let dir = require_q8_0_checkpoint_dir();
    let source = verified_q8_0_checkpoint_source(&dir);

    register_real_qwen_component();

    let mut ingested = GgufIngestor::new()
        .ingest(&source)
        .expect("real quantized Qwen2.5-0.5B-Instruct GGUF file ingests");
    let config = ingested
        .manifest
        .architecture_config
        .as_ref()
        .expect("architecture config was normalized");
    assert_eq!(config.num_hidden_layers, 24);
    assert_eq!(config.hidden_size, 896);
    // Every projection weight in a real Q8_0 export is quantized; if
    // dequantize-then-transpose sequencing were wrong, ingestion would
    // either fail structurally or (worse) silently produce a corrupted
    // tensor -- this only reaches generation at all because every such
    // tensor was already resolved to real F32 content by ingestion time.
    assert!(
        ingested
            .manifest
            .tensors
            .iter()
            .all(|tensor| tensor.storage_dtype == magnetar_runtime::model::ModelDType::F32),
        "every tensor must have been dequantized to F32 by the time ingestion returns"
    );
    ingested.manifest.generation = Some(magnetar_runtime::model::ModelGenerationDefaults {
        max_tokens: Some(8),
        ..Default::default()
    });

    let file_bytes = fs::read(dir.join("model.gguf")).unwrap();
    let artifact = magnetar_format_gguf::parse(&file_bytes).expect("real GGUF file parses");
    let real_tokenizer = GgufTokenizer::from_gguf_metadata(
        &artifact.metadata,
        "qwen2.5-0.5b-instruct-q8-0-gguf",
        Some(config.vocab_size),
    )
    .expect("real tokenizer builds from the checkpoint's own embedded vocabulary");
    let tokenizer_metadata = real_tokenizer.metadata().clone();
    let real_tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(real_tokenizer);

    let trust_store =
        ModelTrustStore::default().trust_digest(ingested.manifest.id.digest.value.clone());
    let fixture = production_qwen_fixture(
        ingested.manifest.clone(),
        tokenizer_metadata,
        real_tokenizer,
    )
    .expect("production fixture builds from the real ingested quantized GGUF checkpoint");

    let outcome = magnetar_runtime::run_production_qwen_generation(
        fixture,
        ingested.payload_source.as_ref(),
        trust_store,
        "The capital of France is",
    )
    .expect(
        "generation runs end to end on a real public, genuinely quantized GGUF checkpoint: real \
         Q8_0 blocks -> real dequantization -> real transpose -> the real compiled Qwen \
         Component -> real Reference CPU dispatch",
    );

    assert!(
        !outcome.result.output.generated_token_ids.is_empty(),
        "real quantized GGUF checkpoint generation produced at least one token"
    );
    assert!(
        !outcome.text.starts_with("[generated token ids:"),
        "the real embedded tokenizer decoded the generated tokens as real text: {}",
        outcome.text
    );
    // Not asserted equal to the unquantized export's own output: Q8_0
    // quantization is real, lossy numerical approximation, so a
    // difference here would not by itself indicate a bug -- coherent,
    // non-crashing real generation is this test's actual bar.
    eprintln!(
        "real Qwen2.5-0.5B-Instruct Q8_0 GGUF generated: {:?}",
        outcome.text
    );
}
