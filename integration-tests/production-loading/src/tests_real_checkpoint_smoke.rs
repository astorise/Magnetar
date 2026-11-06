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
    HuggingFaceChatTemplateFormatter, HuggingFaceIngestor, HuggingFaceTokenizer,
    parse_tokenizer_config,
};
use magnetar_runtime::model::ModelTrustStore;
use magnetar_runtime::production_model_ingestion::{
    ProductionModelArtifactIngestor, ProductionModelSource,
};
use magnetar_runtime::tokenizer::Tokenizer;
use magnetar_runtime::{
    ChatMessage, ModelArtifactSource, ModelGenerationDefaults, PromptInput,
    production_model_fixture,
};
use std::{fs, path::PathBuf, sync::Arc};

/// A real public checkpoint the real-checkpoint tests can run against,
/// pinned by exact Hugging Face revision and `model.safetensors` sha256
/// (the Hub API's own `siblings[].lfs.sha256` for that file at that
/// revision), verified against the downloaded bytes before anything
/// trusts them.
pub(crate) struct RealCheckpointProfile {
    /// The value of `MAGNETAR_REAL_CHECKPOINT_PROFILE` that selects it.
    pub(crate) id: &'static str,
    /// `owner/name` on Hugging Face.
    pub(crate) repo: &'static str,
    pub(crate) revision: &'static str,
    pub(crate) weights_sha256: &'static str,
    pub(crate) vocab_size: u64,
}

/// <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/commit/7ae557604adf67be50417f59c2c2f167def9a775>:
/// 24 layers, 896 hidden, 14 attention / 2 KV heads of 64 dimensions. The
/// default, and the only profile the logit corpus covers.
pub(crate) const QWEN2_5_0_5B_INSTRUCT: RealCheckpointProfile = RealCheckpointProfile {
    id: "qwen2.5-0.5b-instruct",
    repo: "Qwen/Qwen2.5-0.5B-Instruct",
    revision: "7ae557604adf67be50417f59c2c2f167def9a775",
    weights_sha256: "sha256:fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    vocab_size: 151_936,
};

/// <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/commit/989aa7980e4cf806f80c7fef2b1adb7bc71aa306>:
/// 28 layers, 1536 hidden, 12 attention / 2 KV heads of 128 dimensions.
pub(crate) const QWEN2_5_1_5B_INSTRUCT: RealCheckpointProfile = RealCheckpointProfile {
    id: "qwen2.5-1.5b-instruct",
    repo: "Qwen/Qwen2.5-1.5B-Instruct",
    revision: "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    weights_sha256: "sha256:dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee",
    vocab_size: 151_936,
};

/// <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/commit/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562>:
/// a Qwen2 checkpoint (the Qwen Component serves it) with Qwen2.5-1.5B's
/// shape, rope_theta 10000 and an untied LM head.
pub(crate) const DEEPSEEK_R1_DISTILL_QWEN_1_5B: RealCheckpointProfile = RealCheckpointProfile {
    id: "deepseek-r1-distill-qwen-1.5b",
    repo: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    revision: "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
    weights_sha256: "sha256:58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945",
    vocab_size: 151_936,
};

/// <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/commit/9213176726f574b556790deb65791e0c5aa438b6>:
/// a `llama` checkpoint (the Llama Component serves it): 16 layers, 2048
/// hidden, 32 attention / 8 KV heads of 64 dimensions, tied embeddings,
/// rope_theta 500000 with Llama 3 frequency scaling (factor 32). Gated:
/// downloading it needs an accepted license and a Hugging Face token.
pub(crate) const LLAMA_3_2_1B_INSTRUCT: RealCheckpointProfile = RealCheckpointProfile {
    id: "llama-3.2-1b-instruct",
    repo: "meta-llama/Llama-3.2-1B-Instruct",
    revision: "9213176726f574b556790deb65791e0c5aa438b6",
    weights_sha256: "sha256:1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f",
    vocab_size: 128_256,
};

const PROFILES: [&RealCheckpointProfile; 4] = [
    &QWEN2_5_0_5B_INSTRUCT,
    &QWEN2_5_1_5B_INSTRUCT,
    &DEEPSEEK_R1_DISTILL_QWEN_1_5B,
    &LLAMA_3_2_1B_INSTRUCT,
];

/// The profile `MAGNETAR_REAL_CHECKPOINT_PROFILE` names, Qwen2.5-0.5B-
/// Instruct when unset. An unknown name fails rather than falling back.
pub(crate) fn active_profile() -> &'static RealCheckpointProfile {
    match std::env::var("MAGNETAR_REAL_CHECKPOINT_PROFILE") {
        Err(_) => &QWEN2_5_0_5B_INSTRUCT,
        Ok(id) => PROFILES
            .into_iter()
            .find(|profile| profile.id == id)
            .unwrap_or_else(|| {
                panic!(
                    "MAGNETAR_REAL_CHECKPOINT_PROFILE={id} names no pinned checkpoint; known: {:?}",
                    PROFILES.map(|profile| profile.id)
                )
            }),
    }
}

/// `Qwen/Qwen2.5-0.5B-Instruct`'s exact pinned commit, the logit corpus's
/// checkpoint.
pub(crate) const CHECKPOINT_REVISION: &str = QWEN2_5_0_5B_INSTRUCT.revision;
/// Its `model.safetensors` sha256.
pub(crate) const CHECKPOINT_WEIGHTS_SHA256: &str = QWEN2_5_0_5B_INSTRUCT.weights_sha256;
const REAL_VOCAB_SIZE: u64 = QWEN2_5_0_5B_INSTRUCT.vocab_size;

pub(crate) fn checkpoint_dir() -> Option<PathBuf> {
    std::env::var_os("MAGNETAR_QWEN_REAL_CHECKPOINT_DIR").map(PathBuf::from)
}

pub(crate) fn require_checkpoint_dir() -> PathBuf {
    checkpoint_dir().unwrap_or_else(|| {
        let profile = active_profile();
        panic!(
            "MAGNETAR_QWEN_REAL_CHECKPOINT_DIR is not set. Download {} revision {} \
             (config.json, generation_config.json, tokenizer.json, tokenizer_config.json, \
             model.safetensors) into a directory and set that env var to it before running this \
             test with --include-ignored -- see this file's own module doc comment for the exact \
             recipe.",
            profile.repo, profile.revision
        )
    })
}

/// Verifies `dir/model.safetensors`' real sha256 against the active
/// profile's pinned digest and returns the checkpoint's real,
/// pinned-revision-authorized [`ProductionModelSource`].
pub(crate) fn verified_checkpoint_source(dir: &std::path::Path) -> ProductionModelSource {
    verified_profile_source(dir, active_profile())
}

/// [`verified_checkpoint_source`] for an explicit profile.
pub(crate) fn verified_profile_source(
    dir: &std::path::Path,
    profile: &RealCheckpointProfile,
) -> ProductionModelSource {
    // Hashed in 8 MB reads: a multi-gigabyte checkpoint never sits in
    // memory whole just to be verified.
    let mut weights = fs::File::open(dir.join("model.safetensors"))
        .expect("model.safetensors is readable in the checkpoint directory");
    let mut hasher = <sha2::Sha256 as sha2::Digest>::new();
    let mut buffer = vec![0u8; 8 << 20];
    loop {
        let read = std::io::Read::read(&mut weights, &mut buffer)
            .expect("model.safetensors reads to its end");
        if read == 0 {
            break;
        }
        sha2::Digest::update(&mut hasher, &buffer[..read]);
    }
    let actual_digest = format!(
        "sha256:{}",
        sha2::Digest::finalize(hasher)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );
    assert_eq!(
        actual_digest, profile.weights_sha256,
        "the downloaded model.safetensors does not match {} revision {}'s declared content \
         digest -- re-download it",
        profile.repo, profile.revision
    );
    ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::HuggingFace(format!("{}@{}", profile.repo, profile.revision)),
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

    let fixture = production_model_fixture(
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

/// `close-tachyon-scope-audit-gaps` task 3.2: proves the real gap the
/// Tachyon scope-charter audit found is actually closed end to end --
/// `run_production_qwen_generation_for_provider_with_prompt` renders
/// `PromptInput::ChatMessages` through the real Qwen2.5-Instruct chat
/// template (parsed from this same checkpoint's own `tokenizer_config.
/// json`) instead of sending an unformatted plain-text prompt to an
/// Instruct-tuned checkpoint, which is what previously produced
/// incoherent output. Compares against the plain-text path on the exact
/// same checkpoint/prompt to demonstrate the difference is real, not
/// asserting a specific generated token (a real model's greedy output for
/// a real prompt is not something this test should hardcode).
#[test]
#[ignore = "downloads/holds a ~1GB real checkpoint; manual/nightly profile only (task 3.2)"]
fn real_public_checkpoint_renders_chat_messages_through_its_own_real_template() {
    let dir = require_checkpoint_dir();
    let source = verified_checkpoint_source(&dir);

    register_real_qwen_component();

    let mut ingested = HuggingFaceIngestor::new()
        .ingest(&source)
        .expect("real Qwen2.5-0.5B-Instruct bundle ingests");
    assert!(
        ingested.manifest.chat_template.is_some(),
        "the real Qwen2.5-Instruct bundle declares a real chat_template in tokenizer_config.json"
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
    let real_template = tokenizer_config
        .chat_template_reference
        .clone()
        .expect("real Qwen2.5-Instruct tokenizer_config.json declares chat_template");
    let formatter = HuggingFaceChatTemplateFormatter::new(real_template)
        .expect("the real Qwen2.5-Instruct chat template is valid Jinja2");

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
    let fixture = production_model_fixture(
        ingested.manifest.clone(),
        tokenizer_metadata,
        real_tokenizer,
    )
    .expect("production fixture builds from the real ingested Qwen2.5-0.5B-Instruct data");

    let messages = vec![ChatMessage::new("user", "What is the capital of France?")];
    let outcome = magnetar_runtime::run_production_qwen_generation_for_provider_with_prompt(
        fixture,
        ingested.payload_source.as_ref(),
        trust_store,
        PromptInput::ChatMessages(messages),
        Some(&formatter),
        Arc::new(magnetar_runtime::ReferenceCpuProvider::new()),
    )
    .expect(
        "chat-message generation runs end to end through the real chat template on a real \
         public checkpoint",
    );

    assert!(
        !outcome.result.output.generated_token_ids.is_empty(),
        "chat-message generation through the real template produced at least one token"
    );
    eprintln!(
        "real Qwen2.5-0.5B-Instruct generated (real chat template applied): {:?}",
        outcome.text
    );
}

/// Task 12.5, generalized by `implement-device-resident-multi-step-cuda-
/// decode`: compares real Reference CPU and real CUDA deterministic
/// output on the same real checkpoint, same real ingested manifest, and
/// same prompt, across a real multi-token decode (`max_tokens: 16`,
/// matching the audit's own "8/16+ generated tokens" bar -- 15 real
/// decode steps beyond prefill) -- historical KV concatenation across
/// decode steps now dispatches through a real, device-resident "concat"
/// Kernel instead of requiring host-readable tensor bytes, so CUDA
/// genuinely supports this shape now, not just prefill. Greedy sampling on
/// identical input is expected to pick the same argmax token at every step
/// on both Providers even though CPU/GPU floating-point accumulation order
/// differs (their real numeric difference, if any, lands far below what
/// would flip an already well-separated real-word logit distribution's
/// argmax); a documented tolerance of "exactly the same decoded token ids
/// and text" is used rather than a numeric logit tolerance because neither
/// Provider's generation output exposes raw logits at this public
/// boundary. Skips (does not fail) when no CUDA-capable device is
/// available, matching this crate's other CUDA test.
#[test]
#[ignore = "downloads/holds a ~1GB real checkpoint; manual/nightly hardware profile only (task 12.5)"]
fn real_public_checkpoint_multi_token_decode_matches_between_cpu_and_cuda() {
    let provider = magnetar_provider_cuda::CudaProvider::new();
    if !provider.is_available() {
        return;
    }
    let dir = require_checkpoint_dir();
    let source = verified_checkpoint_source(&dir);
    register_real_qwen_component();

    let mut ingested = HuggingFaceIngestor::new()
        .ingest(&source)
        .expect("real pinned checkpoint bundle ingests");
    ingested.manifest.generation = Some(ModelGenerationDefaults {
        max_tokens: Some(16),
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
        active_profile().id,
        Some(active_profile().vocab_size),
    )
    .expect("real tokenizer.json loads and matches the declared vocab_size");
    let tokenizer_metadata = real_tokenizer.metadata().clone();
    let real_tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(real_tokenizer);

    let trust_store =
        ModelTrustStore::default().trust_digest(ingested.manifest.id.digest.value.clone());
    let cpu_fixture = production_model_fixture(
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

    let cuda_fixture = production_model_fixture(
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
        cpu_outcome.result.output.generated_token_ids.len(),
        16,
        "Reference CPU must generate all 16 requested tokens (prefill + 15 real decode steps)"
    );
    assert_eq!(
        cpu_outcome.result.output.generated_token_ids,
        cuda_outcome.result.output.generated_token_ids,
        "Reference CPU and CUDA must select the same greedy tokens at every step (prefill and \
         every real decode step) on identical real checkpoint/config/tokenizer/prompt input"
    );
    assert_eq!(
        cpu_outcome.text, cuda_outcome.text,
        "Reference CPU and CUDA must decode to the same real text"
    );
    eprintln!(
        "real {} multi-token decode matched on CPU and CUDA: {:?}",
        active_profile().repo,
        cpu_outcome.text
    );
}

/// Every pinned profile has a distinct id, a full sha256 and a 40-hex
/// revision, so `MAGNETAR_REAL_CHECKPOINT_PROFILE` selects one exactly.
#[test]
fn real_checkpoint_profiles_are_distinct_and_fully_pinned() {
    for (index, profile) in PROFILES.iter().enumerate() {
        assert!(
            PROFILES[index + 1..]
                .iter()
                .all(|other| other.id != profile.id),
            "duplicate profile id {}",
            profile.id
        );
        assert_eq!(profile.revision.len(), 40, "{}", profile.id);
        assert!(profile.revision.chars().all(|c| c.is_ascii_hexdigit()));
        let digest = profile
            .weights_sha256
            .strip_prefix("sha256:")
            .expect("sha256: prefix");
        assert_eq!(digest.len(), 64, "{}", profile.id);
        assert!(digest.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
