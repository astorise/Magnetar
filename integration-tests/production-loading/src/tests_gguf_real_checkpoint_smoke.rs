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
use magnetar_runtime::{ModelArtifactSource, production_model_fixture};
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
    let fixture = production_model_fixture(
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
    // Every projection weight in a real Q8_0 export is quantized, and now
    // reaches Model Loading packed (`execute-direct-quantized-weights`):
    // its storage dtype is kept and every quantized projection declares
    // transposed storage instead of being transposed byte by byte, which
    // would corrupt its blocks. Model Loading dequantizes them for Reference
    // CPU, which declines packed weights.
    let quantized: Vec<_> = ingested
        .manifest
        .tensors
        .iter()
        .filter(|tensor| tensor.storage_dtype == magnetar_runtime::model::ModelDType::Q8)
        .collect();
    assert!(
        !quantized.is_empty(),
        "a Q8_0 export keeps quantized tensors"
    );
    for tensor in &quantized {
        let projection = tensor.name == "lm_head" || tensor.name.contains("proj");
        assert_eq!(
            tensor.layout.as_deref()
                == Some(magnetar_runtime::packed_weight::TRANSPOSED_STORAGE_LAYOUT),
            projection,
            "{} declares transposed storage exactly when it is a projection",
            tensor.name
        );
    }
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
    let fixture = production_model_fixture(
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

/// A real, pinned `Q8_0` GGUF checkpoint generated on CUDA by
/// [`generate_q8_0_gguf_on_cuda`].
struct CudaQ8Checkpoint {
    /// Prefixes every printed line, which the GPU workflow greps.
    label: &'static str,
    /// The environment variable naming the directory holding `model.gguf`.
    dir_env: &'static str,
    /// `repo@revision` and file, for the error and the artifact source.
    repository: &'static str,
    revision: &'static str,
    file: &'static str,
    sha256: &'static str,
}

/// `execute-direct-quantized-weights` (B03): a real `Q8_0` export on CUDA,
/// its quantized matrices stored packed and read packed by the decode
/// matmul -- or, with `MAGNETAR_CUDA_PACKED_WEIGHTS=0` (the disable
/// control), dequantized to f32 at load as before. The Model Component is
/// the one declaring the file's own architecture family. Prints the greedy
/// tokens and the warm TTFT/TPOT so the GPU workflow can compare the two
/// runs. Skips without a CUDA Device unless `MAGNETAR_REQUIRE_CUDA=1`. The
/// file is never read whole: ingestion streams its digest, checked against
/// the pinned one before anything is trusted.
fn generate_q8_0_gguf_on_cuda(checkpoint: &CudaQ8Checkpoint) {
    use magnetar_runtime::{
        GenerationParameters, GenerationStreamEvent, ProductionGenerationRequest,
        ProductionLoadedModel, PromptInput, Provider, StopConditions,
    };
    use std::{ops::ControlFlow, time::Instant};

    let label = checkpoint.label;
    // `MAGNETAR_CUDA_DEVICES=0,1`: one Provider spanning those GPUs, the
    // Runtime placing the layers across them
    // (`enforce-local-multigpu-and-expert-placement` G05).
    let ordinals: Option<Vec<usize>> = std::env::var("MAGNETAR_CUDA_DEVICES").ok().map(|list| {
        list.split(',')
            .map(|ordinal| {
                ordinal
                    .trim()
                    .parse()
                    .expect("MAGNETAR_CUDA_DEVICES lists GPU ordinals")
            })
            .collect()
    });
    let (cuda, device_count): (Arc<dyn Provider>, usize) = match &ordinals {
        Some(ordinals) => {
            let multi = magnetar_provider_cuda::multi::CudaMultiProvider::for_devices(
                ordinals,
                "magnetar:provider/cuda",
            );
            let available = multi.is_available();
            (Arc::new(multi), if available { ordinals.len() } else { 0 })
        }
        None => {
            let single = magnetar_provider_cuda::CudaProvider::new();
            let available = single.is_available();
            (Arc::new(single), usize::from(available))
        }
    };
    if device_count == 0 {
        assert!(
            std::env::var("MAGNETAR_REQUIRE_CUDA").as_deref() != Ok("1"),
            "MAGNETAR_REQUIRE_CUDA=1 but no CUDA-capable Device is available"
        );
        eprintln!("skipped: no CUDA-capable Device");
        return;
    }
    let dir = std::env::var_os(checkpoint.dir_env)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            panic!(
                "{} is not set. Download {} revision {}'s {} into a directory as 'model.gguf' \
                 and set that env var to it before running this test with --include-ignored.",
                checkpoint.dir_env, checkpoint.repository, checkpoint.revision, checkpoint.file
            )
        });
    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::HuggingFace(format!(
            "{}@{}",
            checkpoint.repository, checkpoint.revision
        )),
        dir.clone(),
    );
    let load_started = Instant::now();
    let mut ingested = GgufIngestor::new()
        .ingest(&source)
        .expect("the real quantized GGUF file ingests");
    assert_eq!(
        ingested.manifest.id.digest.value, checkpoint.sha256,
        "the downloaded model.gguf does not match revision {}'s declared content digest -- \
         re-download it",
        checkpoint.revision
    );
    let vocab_size = ingested
        .manifest
        .architecture_config
        .as_ref()
        .expect("architecture config was normalized")
        .vocab_size;
    ingested.manifest.generation = Some(magnetar_runtime::model::ModelGenerationDefaults {
        max_tokens: Some(64),
        ..Default::default()
    });
    let tokenizer = magnetar_loader_gguf::load_gguf_tokenizer(
        &source,
        format!("{}-tokenizer", ingested.manifest.id.name.as_str()),
        Some(vocab_size),
    )
    .expect("real tokenizer builds from the checkpoint's own embedded vocabulary");
    let tokenizer_metadata = tokenizer.metadata().clone();
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(tokenizer);
    let trust_store =
        ModelTrustStore::default().trust_digest(ingested.manifest.id.digest.value.clone());
    let component = crate::test_support::real_component_for_architecture_family(
        &ingested.manifest.architecture.family,
    );
    let fixture = production_model_fixture(
        ingested.manifest.clone(),
        tokenizer_metadata,
        Arc::clone(&tokenizer),
    )
    .expect("production fixture builds from the real ingested quantized GGUF checkpoint");

    let mut model = ProductionLoadedModel::load_with_component(
        fixture,
        ingested.payload_source.as_ref(),
        trust_store,
        cuda,
        Some(component),
    )
    .expect("the real Q8_0 checkpoint loads on CUDA");
    let tensor_parallel = model.tensor_parallel();
    if std::env::var("MAGNETAR_MULTI_GPU").as_deref() == Ok("tensor") && device_count > 1 {
        assert!(
            tensor_parallel,
            "MAGNETAR_MULTI_GPU=tensor over {device_count} GPUs must split the model"
        );
    }
    let load_ms = load_started.elapsed().as_secs_f64() * 1_000.0;
    let request = |max_new_tokens| ProductionGenerationRequest {
        prompt: PromptInput::PlainText("The capital of France is".into()),
        parameters: GenerationParameters::greedy(),
        stop_conditions: StopConditions::default(),
        max_new_tokens: Some(max_new_tokens),
        max_generation_millis: None,
    };
    model.generate(request(4), None).expect("warmup generates");

    let mut token_times = Vec::new();
    let started = Instant::now();
    let outcome = model
        .generate_streaming(request(64), None, &mut |event| {
            if let GenerationStreamEvent::Token { .. } = event {
                token_times.push(started.elapsed().as_secs_f64() * 1_000.0);
            }
            ControlFlow::Continue(())
        })
        .expect("the real Q8_0 checkpoint generates on CUDA");
    let tokens = &outcome.result.output.generated_token_ids;
    assert!(tokens.len() > 1, "generated more than one token");
    let ttft = token_times[0];
    let tpot = (token_times[token_times.len() - 1] - ttft) / (token_times.len() - 1) as f64;
    let packed = std::env::var("MAGNETAR_CUDA_PACKED_WEIGHTS").as_deref() != Ok("0");
    eprintln!(
        "[{label}] family {}, {device_count} GPU(s), placement {:?}, tensor parallel {}: \
         packed weights {}: load \
         {load_ms:.0} ms; warm TTFT {ttft:.1} ms, TPOT {tpot:.2} ms over {} tokens",
        ingested.manifest.architecture.family,
        std::env::var("MAGNETAR_LAYER_PLACEMENT").unwrap_or_else(|_| "fill".into()),
        if tensor_parallel { "on" } else { "off" },
        if packed { "on" } else { "off" },
        tokens.len()
    );
    eprintln!("[{label}] generated token ids {tokens:?}");
    eprintln!("[{label}] generated text {:?}", outcome.text);
    model.unload().expect("the real checkpoint unloads");
}

/// The real `Qwen2.5-0.5B-Instruct` `Q8_0` export on CUDA.
#[test]
#[ignore = "downloads/holds a ~650MB real quantized checkpoint and needs a CUDA Device"]
fn real_public_q8_0_gguf_checkpoint_generates_on_cuda() {
    generate_q8_0_gguf_on_cuda(&CudaQ8Checkpoint {
        label: "cuda q8_0 gguf",
        dir_env: "MAGNETAR_QWEN_GGUF_Q8_0_REAL_CHECKPOINT_DIR",
        repository: "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        revision: CHECKPOINT_REVISION,
        file: "qwen2.5-0.5b-instruct-q8_0.gguf",
        sha256: Q8_0_CHECKPOINT_SHA256,
    });
}

/// `bartowski/Mistral-7B-Instruct-v0.3-GGUF`'s exact pinned commit:
/// <https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/commit/61fd4167fff3ab01ee1cfe0da183fa27a944db48>
/// (Apache-2.0). The file declares `general.architecture = llama`, so the
/// Llama Model Component serves it; its query and key projections arrive
/// in llama.cpp's interleaved row order, which the GGUF loader restores.
const MISTRAL_7B_Q8_0_REVISION: &str = "61fd4167fff3ab01ee1cfe0da183fa27a944db48";
/// `Mistral-7B-Instruct-v0.3-Q8_0.gguf`'s sha256 (7,702,565,088 bytes) at
/// that revision, as Hugging Face declares it.
const MISTRAL_7B_Q8_0_SHA256: &str =
    "sha256:404857e776114baada71a08ebd3bba79d721ec7fca99705e7e7b892ae8bc583f";

/// Mistral 7B Instruct v0.3 `Q8_0` on CUDA: 7.7 GB packed, which fits a
/// 12 GB Device only because the quantized matrices stay packed.
#[test]
#[ignore = "downloads/holds a 7.7GB real quantized checkpoint and needs a CUDA Device"]
fn real_public_mistral_7b_q8_0_gguf_checkpoint_generates_on_cuda() {
    generate_q8_0_gguf_on_cuda(&CudaQ8Checkpoint {
        label: "cuda mistral-7b q8_0 gguf",
        dir_env: "MAGNETAR_MISTRAL_7B_GGUF_Q8_0_REAL_CHECKPOINT_DIR",
        repository: "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        revision: MISTRAL_7B_Q8_0_REVISION,
        file: "Mistral-7B-Instruct-v0.3-Q8_0.gguf",
        sha256: MISTRAL_7B_Q8_0_SHA256,
    });
}

/// `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf`'s sha256 (4,372,812,000 bytes) at
/// [`MISTRAL_7B_Q8_0_REVISION`], as Hugging Face declares it: `Q4_K`
/// matrices, with `Q6_K` for some (llama.cpp's `Q4_K_M` mix).
const MISTRAL_7B_Q4_K_M_SHA256: &str =
    "sha256:1270d22c0fbb3d092fb725d4d96c457b7b687a5f5a715abe1e818da303e562b6";

/// The real `Mistral-7B-Instruct-v0.3` `Q4_K_M` export on CUDA: its `Q4_K`
/// matrices stored packed and read packed by the decode matmul, its `Q6_K`
/// ones dequantized at load.
#[test]
#[ignore = "downloads/holds a 4.4GB real quantized checkpoint and needs a CUDA Device"]
fn real_public_mistral_7b_q4_k_m_gguf_checkpoint_generates_on_cuda() {
    generate_q8_0_gguf_on_cuda(&CudaQ8Checkpoint {
        label: "cuda mistral-7b q4_k_m gguf",
        dir_env: "MAGNETAR_MISTRAL_7B_GGUF_Q4_K_M_REAL_CHECKPOINT_DIR",
        repository: "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        revision: MISTRAL_7B_Q8_0_REVISION,
        file: "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        sha256: MISTRAL_7B_Q4_K_M_SHA256,
    });
}

/// `RichardErkhov/Qwen_-_Qwen1.5-MoE-A2.7B-Chat-gguf`'s exact pinned commit
/// (the `Q8_0` export of `Qwen/Qwen1.5-MoE-A2.7B-Chat`): 24 layers of 60
/// routed experts (top 4) and a shared expert.
const QWEN_MOE_Q8_0_REVISION: &str = "947580e7b3904cd4081810d0efec4520a75609c7";
/// `Qwen1.5-MoE-A2.7B-Chat.Q8_0.gguf`'s sha256 (15,226,010,336 bytes) at
/// that revision, as Hugging Face declares it.
const QWEN_MOE_Q8_0_SHA256: &str =
    "sha256:d20e59b6180c775c648703fd63b8df7d01fc107672441fc3746c41f1dcaf904c";

/// Qwen1.5-MoE-A2.7B-Chat `Q8_0` on CUDA (`execute-and-cache-moe-experts`
/// F02): 15 GB packed, more than one 12 GB GPU holds, so it runs with
/// `MAGNETAR_CUDA_DEVICES` listing two, the Runtime placing its layers
/// across them.
#[test]
#[ignore = "downloads/holds a 15GB real quantized checkpoint and needs two CUDA Devices"]
fn real_public_qwen_moe_a2_7b_q8_0_gguf_checkpoint_generates_on_cuda() {
    generate_q8_0_gguf_on_cuda(&CudaQ8Checkpoint {
        label: "cuda qwen1.5-moe-a2.7b q8_0 gguf",
        dir_env: "MAGNETAR_QWEN_MOE_GGUF_Q8_0_REAL_CHECKPOINT_DIR",
        repository: "RichardErkhov/Qwen_-_Qwen1.5-MoE-A2.7B-Chat-gguf",
        revision: QWEN_MOE_Q8_0_REVISION,
        file: "Qwen1.5-MoE-A2.7B-Chat.Q8_0.gguf",
        sha256: QWEN_MOE_Q8_0_SHA256,
    });
}
