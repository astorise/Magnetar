use magnetar_loader_huggingface::HuggingFaceIngestor;
use magnetar_runtime::model::{ModelDType, ModelTrustStore};
use magnetar_runtime::production_model_ingestion::{
    ProductionModelArtifactIngestor, ProductionModelSource,
};
use magnetar_runtime::tokenizer::{DecodeInput, EncodeInput, Tokenizer};
use magnetar_runtime::{
    ModelArtifactSource, load_production_qwen_instance, production_qwen_fixture,
};
use std::{fs, io::Write, sync::Arc};

const HIDDEN_SIZE: u64 = 4;
const LAYER_COUNT: u64 = 1;
const ATTENTION_HEAD_COUNT: u64 = 2;
const KV_HEAD_COUNT: u64 = 2;
const HEAD_DIMENSION: u64 = 2;
const INTERMEDIATE_SIZE: u64 = 8;
const VOCAB_SIZE: u64 = 16;

/// Deterministic pseudo-random `f32` in `[-0.5, 0.5]`, purely a function of
/// `seed` -- mirrors `magnetar-runtime`'s own `fixture_value` (private
/// there), reimplemented here since this crate cannot reach it: no RNG
/// dependency, fully reproducible.
fn tensor_value(seed: u64) -> f32 {
    let mut x = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x2545_F491_4F6C_DD1D);
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    ((x % 1000) as f32 / 1000.0) - 0.5
}

fn tensor_values(name: &str, element_count: u64) -> Vec<f32> {
    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    for byte in name.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    (0..element_count)
        .map(|index| tensor_value(hash.wrapping_add(index)))
        .collect()
}

/// Writes a real, production-shaped Hugging Face Qwen bundle (real
/// `config.json`, real `tokenizer.json`/`tokenizer_config.json`, real
/// single-file Safetensors bytes) to `dir`, sized for the tiny
/// architecture constants above. `tied_embeddings=false`, so `lm_head`
/// gets its own real weight tensor.
fn write_tiny_production_bundle(dir: &std::path::Path) {
    let config_json = format!(
        r#"{{
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "hidden_size": {HIDDEN_SIZE},
            "intermediate_size": {INTERMEDIATE_SIZE},
            "num_hidden_layers": {LAYER_COUNT},
            "num_attention_heads": {ATTENTION_HEAD_COUNT},
            "num_key_value_heads": {KV_HEAD_COUNT},
            "head_dim": {HEAD_DIMENSION},
            "vocab_size": {VOCAB_SIZE},
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false,
            "torch_dtype": "float32",
            "bos_token_id": 0,
            "eos_token_id": 1
        }}"#
    );
    fs::write(dir.join("config.json"), config_json).unwrap();

    // A tiny real WordLevel tokenizer -- enough vocabulary to encode/
    // decode a short ASCII prompt, sized to exactly VOCAB_SIZE.
    let vocab_entries = [
        "<bos>", "<eos>", "hi", "h", "i", " ", "t", "e", "r", "wo", "ld", "!", "a", "b", "c", "d",
    ];
    let mut vocab_json = String::from("{");
    for (id, token) in vocab_entries.iter().take(VOCAB_SIZE as usize).enumerate() {
        if id > 0 {
            vocab_json.push(',');
        }
        vocab_json.push_str(&format!("\"{token}\":{id}"));
    }
    vocab_json.push('}');
    let tokenizer_json = format!(
        r#"{{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {{"id": 0, "content": "<bos>", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}},
                {{"id": 1, "content": "<eos>", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}}
            ],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {{"type": "WordLevel", "vocab": {vocab_json}, "unk_token": "h"}}
        }}"#
    );
    fs::write(dir.join("tokenizer.json"), tokenizer_json).unwrap();
    fs::write(
        dir.join("tokenizer_config.json"),
        r#"{"bos_token": "<bos>", "eos_token": "<eos>"}"#,
    )
    .unwrap();

    let q_dim = ATTENTION_HEAD_COUNT * HEAD_DIMENSION;
    let kv_dim = KV_HEAD_COUNT * HEAD_DIMENSION;
    let mut tensors: Vec<(String, Vec<u64>)> = vec![
        (
            "model.embed_tokens.weight".into(),
            vec![VOCAB_SIZE, HIDDEN_SIZE],
        ),
        ("model.norm.weight".into(), vec![HIDDEN_SIZE]),
        ("lm_head.weight".into(), vec![VOCAB_SIZE, HIDDEN_SIZE]),
    ];
    for layer in 0..LAYER_COUNT {
        tensors.push((
            format!("model.layers.{layer}.input_layernorm.weight"),
            vec![HIDDEN_SIZE],
        ));
        tensors.push((
            format!("model.layers.{layer}.self_attn.q_proj.weight"),
            vec![q_dim, HIDDEN_SIZE],
        ));
        tensors.push((
            format!("model.layers.{layer}.self_attn.k_proj.weight"),
            vec![kv_dim, HIDDEN_SIZE],
        ));
        tensors.push((
            format!("model.layers.{layer}.self_attn.v_proj.weight"),
            vec![kv_dim, HIDDEN_SIZE],
        ));
        tensors.push((
            format!("model.layers.{layer}.self_attn.o_proj.weight"),
            vec![HIDDEN_SIZE, q_dim],
        ));
        tensors.push((
            format!("model.layers.{layer}.post_attention_layernorm.weight"),
            vec![HIDDEN_SIZE],
        ));
        tensors.push((
            format!("model.layers.{layer}.mlp.gate_proj.weight"),
            vec![INTERMEDIATE_SIZE, HIDDEN_SIZE],
        ));
        tensors.push((
            format!("model.layers.{layer}.mlp.up_proj.weight"),
            vec![INTERMEDIATE_SIZE, HIDDEN_SIZE],
        ));
        tensors.push((
            format!("model.layers.{layer}.mlp.down_proj.weight"),
            vec![HIDDEN_SIZE, INTERMEDIATE_SIZE],
        ));
    }

    let mut header = String::from("{");
    let mut data = Vec::new();
    for (name, shape) in &tensors {
        let element_count: u64 = shape.iter().product();
        let values = tensor_values(name, element_count);
        let start = data.len() as u64;
        for value in &values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let end = data.len() as u64;
        let shape_text = shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{shape_text}],\"data_offsets\":[{start},{end}]}},"
        ));
    }
    header.pop(); // trailing comma
    header.push('}');

    let mut file = fs::File::create(dir.join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    file.write_all(&data).unwrap();
}

#[test]
fn tachyon_shaped_real_production_ingestion_loads_through_the_real_qwen_component() {
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

    let mut runtime = magnetar_runtime::Runtime::builder()
        .register_provider(Arc::new(magnetar_runtime::ReferenceCpuProvider::new()))
        .trust_store(trust_store)
        .build()
        .expect("runtime builds");

    let instance = load_production_qwen_instance(
        &mut runtime,
        &ingested.manifest,
        ingested.payload_source.as_ref(),
    )
    .expect("real production loading succeeds through the generic Inference API");

    // Ready, not merely "loaded": lifecycle observed directly through the
    // same public accessor an embedder would use (the lower-level
    // generation-orchestration primitives -- prepared execution plans,
    // the generation loop itself -- remain Runtime-internal; full
    // generation through a non-canonical config is proven inside
    // magnetar-runtime's own test suite, where those primitives are
    // reachable).
    let state = runtime
        .model_instance(&instance)
        .expect("instance is registered")
        .lifecycle();
    assert_eq!(
        state,
        magnetar_runtime::ModelInstanceLifecycleState::Ready,
        "a production-loaded instance from a real ingested bundle is genuinely ready"
    );

    // The real tokenizer round-trips real text independently of Runtime
    // loading -- proving the ingested tokenizer.json is genuinely usable,
    // not merely structurally present.
    let encoded = fixture
        .tokenizer
        .encode(EncodeInput {
            add_special_tokens: false,
            ..EncodeInput::new("hi")
        })
        .expect("the real tokenizer encodes real text");
    assert!(!encoded.token_ids.is_empty());
    let decoded = fixture
        .tokenizer
        .decode(DecodeInput {
            token_ids: encoded.token_ids,
            skip_special_tokens: true,
            clean_up_tokenization_spaces: false,
            streaming_state: None,
        })
        .expect("the real tokenizer decodes its own encoded output");
    assert_eq!(decoded.text, "hi");

    magnetar_runtime::unload_model_instance(
        &mut runtime,
        &instance,
        magnetar_runtime::ModelInstanceUnloadPolicy::DrainActiveUse,
    )
    .expect("model instance unloads cleanly");
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
