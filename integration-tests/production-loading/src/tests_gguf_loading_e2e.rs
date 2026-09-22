//! Real production GGUF loading end to end, mirroring
//! `tests_production_loading_e2e.rs`'s own rationale for Hugging Face
//! bundles: neither `magnetar-runtime` nor `loaders/gguf` can prove this
//! on their own (the same `production-model-ingestion` externalization
//! boundary applies). Proves the real external `GgufIngestor` reaches
//! `magnetar_runtime::load_production_model_instance` and generation
//! through the real compiled Qwen Component and Reference CPU Provider --
//! the exact same production entry points `tests_production_loading_e2e.rs`
//! uses for Hugging Face bundles, proving the production path is genuinely
//! format-agnostic, not Hugging-Face-shaped by accident.

use crate::test_support::{
    ATTENTION_HEAD_COUNT, HEAD_DIMENSION, HIDDEN_SIZE, INTERMEDIATE_SIZE, KV_HEAD_COUNT,
    LAYER_COUNT, VOCAB_SIZE, register_real_qwen_component, tensor_values,
    write_tiny_production_bundle,
};
use magnetar_loader_gguf::GgufIngestor;
use magnetar_runtime::model::ModelTrustStore;
use magnetar_runtime::production_model_ingestion::{
    ProductionModelArtifactIngestor, ProductionModelSource,
};
use magnetar_runtime::tokenizer::Tokenizer;
use magnetar_runtime::{ModelArtifactSource, PromptInput, production_model_fixture};
use std::{fs, sync::Arc};

fn le_u32(value: u32) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}
fn le_u64(value: u64) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}
fn gguf_string(value: &str) -> Vec<u8> {
    let mut out = le_u64(value.len() as u64);
    out.extend_from_slice(value.as_bytes());
    out
}
fn kv_string(key: &str, value: &str) -> Vec<u8> {
    let mut out = gguf_string(key);
    out.extend(le_u32(8));
    out.extend(gguf_string(value));
    out
}
fn kv_uint32(key: &str, value: u32) -> Vec<u8> {
    let mut out = gguf_string(key);
    out.extend(le_u32(4));
    out.extend(le_u32(value));
    out
}
fn kv_string_array(key: &str, values: &[&str]) -> Vec<u8> {
    let mut out = gguf_string(key);
    out.extend(le_u32(9));
    out.extend(le_u32(8));
    out.extend(le_u64(values.len() as u64));
    for value in values {
        out.extend(gguf_string(value));
    }
    out
}

struct GgufTensor {
    name: String,
    /// GGUF `ne[]` order (reverse of the Hugging Face-equivalent shape).
    ne: Vec<u64>,
    data: Vec<u8>,
}

fn build_gguf(kv_entries: &[Vec<u8>], tensors: &[GgufTensor], alignment: u64) -> Vec<u8> {
    let mut file = le_u32(0x4655_4747);
    file.extend(le_u32(3));
    file.extend(le_u64(tensors.len() as u64));
    file.extend(le_u64(kv_entries.len() as u64));
    for entry in kv_entries {
        file.extend(entry);
    }
    let mut tensor_info_bytes = Vec::new();
    let mut data_section = Vec::new();
    let mut next_offset = 0_u64;
    for tensor in tensors {
        let padding = (alignment - (next_offset % alignment)) % alignment;
        data_section.extend(std::iter::repeat_n(0u8, padding as usize));
        next_offset += padding;
        let offset = next_offset;
        tensor_info_bytes.extend(gguf_string(&tensor.name));
        tensor_info_bytes.extend(le_u32(tensor.ne.len() as u32));
        for dimension in &tensor.ne {
            tensor_info_bytes.extend(le_u64(*dimension));
        }
        tensor_info_bytes.extend(le_u32(0)); // F32
        tensor_info_bytes.extend(le_u64(offset));
        data_section.extend(&tensor.data);
        next_offset += tensor.data.len() as u64;
    }
    file.extend(tensor_info_bytes);
    let start_padding = (alignment - (file.len() as u64 % alignment)) % alignment;
    file.extend(std::iter::repeat_n(0u8, start_padding as usize));
    file.extend(data_section);
    file
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Writes a real, production-shaped GGUF file to `dir`, carrying *the
/// exact same real tensor values* `write_tiny_production_bundle` writes
/// for the equivalent Hugging Face bundle (`tensor_values` is seeded by
/// the Hugging Face-convention tensor name string, called here with that
/// same string purely as the deterministic seed -- the GGUF file's own
/// tensor name/shape follow llama.cpp's real convention). This lets a
/// test load the *same weights* through both ingestors and compare
/// generation output directly: any transpose, shape-reversal, or naming
/// bug in `loaders/gguf` would almost certainly change the result, since
/// real (not all-zero or trivially symmetric) values are used throughout.
/// `tied=false`: declares `output.weight` explicitly, matching
/// `write_tiny_production_bundle`'s own untied `lm_head.weight`.
fn write_tiny_gguf_bundle(dir: &std::path::Path) {
    let q_dim = ATTENTION_HEAD_COUNT * HEAD_DIMENSION;
    let kv_dim = KV_HEAD_COUNT * HEAD_DIMENSION;

    let mut kvs = vec![
        kv_string("general.architecture", "qwen2"),
        kv_uint32("qwen2.embedding_length", HIDDEN_SIZE as u32),
        kv_uint32("qwen2.feed_forward_length", INTERMEDIATE_SIZE as u32),
        kv_uint32("qwen2.block_count", LAYER_COUNT as u32),
        kv_uint32("qwen2.attention.head_count", ATTENTION_HEAD_COUNT as u32),
        kv_uint32("qwen2.attention.head_count_kv", KV_HEAD_COUNT as u32),
        kv_uint32("tokenizer.ggml.bos_token_id", 0),
        kv_uint32("tokenizer.ggml.eos_token_id", 1),
        kv_string("tokenizer.ggml.model", "gpt2"),
    ];
    let vocab_entries: Vec<&str> = [
        "<bos>", "<eos>", "hi", "h", "i", " ", "t", "e", "r", "wo", "ld", "!", "a", "b", "c", "d",
    ]
    .into_iter()
    .take(VOCAB_SIZE as usize)
    .collect();
    kvs.push(kv_string_array("tokenizer.ggml.tokens", &vocab_entries));
    kvs.push(kv_string_array("tokenizer.ggml.merges", &[]));

    // (hf_name_seed, gguf_name, ne) -- ne is the reverse of the
    // equivalent Hugging Face row-major shape; see `weights.rs`'s own
    // "GGUF's ne[] order is the reverse of PyTorch/Safetensors row-major
    // shape order" reasoning.
    let mut entries: Vec<(String, String, Vec<u64>)> = vec![
        (
            "model.embed_tokens.weight".into(),
            "token_embd.weight".into(),
            vec![HIDDEN_SIZE, VOCAB_SIZE],
        ),
        (
            "model.norm.weight".into(),
            "output_norm.weight".into(),
            vec![HIDDEN_SIZE],
        ),
        (
            "lm_head.weight".into(),
            "output.weight".into(),
            vec![HIDDEN_SIZE, VOCAB_SIZE],
        ),
    ];
    for layer in 0..LAYER_COUNT {
        entries.push((
            format!("model.layers.{layer}.input_layernorm.weight"),
            format!("blk.{layer}.attn_norm.weight"),
            vec![HIDDEN_SIZE],
        ));
        entries.push((
            format!("model.layers.{layer}.self_attn.q_proj.weight"),
            format!("blk.{layer}.attn_q.weight"),
            vec![HIDDEN_SIZE, q_dim],
        ));
        entries.push((
            format!("model.layers.{layer}.self_attn.k_proj.weight"),
            format!("blk.{layer}.attn_k.weight"),
            vec![HIDDEN_SIZE, kv_dim],
        ));
        entries.push((
            format!("model.layers.{layer}.self_attn.v_proj.weight"),
            format!("blk.{layer}.attn_v.weight"),
            vec![HIDDEN_SIZE, kv_dim],
        ));
        entries.push((
            format!("model.layers.{layer}.self_attn.o_proj.weight"),
            format!("blk.{layer}.attn_output.weight"),
            vec![q_dim, HIDDEN_SIZE],
        ));
        entries.push((
            format!("model.layers.{layer}.post_attention_layernorm.weight"),
            format!("blk.{layer}.ffn_norm.weight"),
            vec![HIDDEN_SIZE],
        ));
        entries.push((
            format!("model.layers.{layer}.mlp.gate_proj.weight"),
            format!("blk.{layer}.ffn_gate.weight"),
            vec![HIDDEN_SIZE, INTERMEDIATE_SIZE],
        ));
        entries.push((
            format!("model.layers.{layer}.mlp.up_proj.weight"),
            format!("blk.{layer}.ffn_up.weight"),
            vec![HIDDEN_SIZE, INTERMEDIATE_SIZE],
        ));
        entries.push((
            format!("model.layers.{layer}.mlp.down_proj.weight"),
            format!("blk.{layer}.ffn_down.weight"),
            vec![INTERMEDIATE_SIZE, HIDDEN_SIZE],
        ));
    }

    let tensors: Vec<GgufTensor> = entries
        .into_iter()
        .map(|(hf_name_seed, gguf_name, ne)| {
            let element_count: u64 = ne.iter().product();
            let values = tensor_values(&hf_name_seed, element_count);
            GgufTensor {
                name: gguf_name,
                ne,
                data: f32_bytes(&values),
            }
        })
        .collect();

    fs::write(dir.join("model.gguf"), build_gguf(&kvs, &tensors, 32)).unwrap();
}

#[test]
fn ingests_and_generates_from_a_real_gguf_file() {
    register_real_qwen_component();
    let dir = tempfile::tempdir().unwrap();
    write_tiny_gguf_bundle(dir.path());

    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
        dir.path().to_path_buf(),
    );
    let ingested = GgufIngestor::new()
        .ingest(&source)
        .expect("real GGUF ingestion succeeds");
    assert_eq!(
        ingested.manifest.tensors.len(),
        3 + LAYER_COUNT as usize * 9
    );
    assert!(ingested.manifest.architecture_config.is_some());

    let file_bytes = fs::read(dir.path().join("model.gguf")).unwrap();
    let artifact = magnetar_format_gguf::parse(&file_bytes).unwrap();
    let gguf_tokenizer = magnetar_loader_gguf::GgufTokenizer::from_gguf_metadata(
        &artifact.metadata,
        "gguf-test",
        Some(VOCAB_SIZE),
    )
    .expect("real tokenizer builds from embedded vocabulary");
    let tokenizer_metadata = gguf_tokenizer.metadata().clone();
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(gguf_tokenizer);

    let trust_store =
        ModelTrustStore::default().trust_digest(ingested.manifest.id.digest.value.clone());
    let fixture =
        production_model_fixture(ingested.manifest.clone(), tokenizer_metadata, tokenizer)
            .expect("production fixture builds from real ingested GGUF data");

    let outcome = magnetar_runtime::run_production_qwen_generation(
        fixture,
        ingested.payload_source.as_ref(),
        trust_store,
        "hi",
    )
    .expect(
        "generation runs end to end: real GGUF file -> real embedded tokenizer -> real \
         compiled Qwen Component -> real Reference CPU dispatch",
    );
    assert!(
        !outcome.result.output.generated_token_ids.is_empty(),
        "real GGUF ingestion + loading + generation produced at least one token"
    );
}

/// The strongest correctness proof available without a real downloaded
/// checkpoint: the exact same real (non-trivial) weight values, loaded
/// through the two independent ingestors (`HuggingFaceIngestor` via
/// Safetensors, `GgufIngestor` via GGUF), must produce byte-for-byte
/// identical generation output for the same input token ids. Any bug in
/// `loaders/gguf`'s shape-reversal, projection transpose, or tied-lm_head
/// derivation would almost certainly change the result -- real values are
/// used throughout, not all-zeros or a trivially symmetric case. Drives
/// both paths with `PromptInput::TokenIds` (not text) since the two
/// bundles' tokenizers are deliberately different implementations
/// (`WordLevel` vs. embedded byte-level BPE); this test's subject is
/// weight/architecture correctness, not tokenizer equivalence.
#[test]
fn gguf_and_huggingface_ingestion_of_identical_weights_produce_identical_generation() {
    register_real_qwen_component();

    let hf_dir = tempfile::tempdir().unwrap();
    write_tiny_production_bundle(hf_dir.path());
    let hf_source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::LocalPath(hf_dir.path().to_path_buf()),
        hf_dir.path().to_path_buf(),
    );
    let hf_ingested = magnetar_loader_huggingface::HuggingFaceIngestor::new()
        .ingest(&hf_source)
        .expect("real Hugging Face bundle ingestion succeeds");
    let hf_tokenizer_bytes = fs::read(hf_dir.path().join("tokenizer.json")).unwrap();
    let hf_tokenizer = magnetar_loader_huggingface::HuggingFaceTokenizer::from_bytes(
        &hf_tokenizer_bytes,
        None,
        "hf-cross-check",
        Some(VOCAB_SIZE),
    )
    .expect("real tokenizer.json loads");
    let hf_tokenizer_metadata = hf_tokenizer.metadata().clone();
    let hf_tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(hf_tokenizer);
    let hf_trust_store =
        ModelTrustStore::default().trust_digest(hf_ingested.manifest.id.digest.value.clone());
    let hf_fixture = production_model_fixture(
        hf_ingested.manifest.clone(),
        hf_tokenizer_metadata,
        hf_tokenizer,
    )
    .expect("production fixture builds from real ingested Hugging Face data");
    let hf_outcome = magnetar_runtime::run_production_qwen_generation_for_provider_with_prompt(
        hf_fixture,
        hf_ingested.payload_source.as_ref(),
        hf_trust_store,
        PromptInput::TokenIds(vec![2, 3]),
        None,
        Arc::new(magnetar_runtime::ReferenceCpuProvider::new()),
    )
    .expect("Hugging Face-path generation succeeds");

    let gguf_dir = tempfile::tempdir().unwrap();
    write_tiny_gguf_bundle(gguf_dir.path());
    let gguf_source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::LocalPath(gguf_dir.path().to_path_buf()),
        gguf_dir.path().to_path_buf(),
    );
    let gguf_ingested = GgufIngestor::new()
        .ingest(&gguf_source)
        .expect("real GGUF ingestion succeeds");
    let gguf_file_bytes = fs::read(gguf_dir.path().join("model.gguf")).unwrap();
    let gguf_artifact = magnetar_format_gguf::parse(&gguf_file_bytes).unwrap();
    let gguf_tokenizer = magnetar_loader_gguf::GgufTokenizer::from_gguf_metadata(
        &gguf_artifact.metadata,
        "gguf-cross-check",
        Some(VOCAB_SIZE),
    )
    .expect("real tokenizer builds from embedded vocabulary");
    let gguf_tokenizer_metadata = gguf_tokenizer.metadata().clone();
    let gguf_tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(gguf_tokenizer);
    let gguf_trust_store =
        ModelTrustStore::default().trust_digest(gguf_ingested.manifest.id.digest.value.clone());
    let gguf_fixture = production_model_fixture(
        gguf_ingested.manifest.clone(),
        gguf_tokenizer_metadata,
        gguf_tokenizer,
    )
    .expect("production fixture builds from real ingested GGUF data");
    let gguf_outcome = magnetar_runtime::run_production_qwen_generation_for_provider_with_prompt(
        gguf_fixture,
        gguf_ingested.payload_source.as_ref(),
        gguf_trust_store,
        PromptInput::TokenIds(vec![2, 3]),
        None,
        Arc::new(magnetar_runtime::ReferenceCpuProvider::new()),
    )
    .expect("GGUF-path generation succeeds");

    assert_eq!(
        gguf_outcome.result.output.generated_token_ids,
        hf_outcome.result.output.generated_token_ids,
        "identical real weight values, loaded through the Safetensors and GGUF ingestors \
         respectively, must produce identical generation output -- a mismatch here means a \
         real shape/transpose/naming bug in loaders/gguf"
    );
}

#[test]
fn rejects_a_non_qwen2_gguf_architecture() {
    let dir = tempfile::tempdir().unwrap();
    let kvs = vec![kv_string("general.architecture", "llama")];
    let tensors = [GgufTensor {
        name: "token_embd.weight".into(),
        ne: vec![HIDDEN_SIZE, VOCAB_SIZE],
        data: f32_bytes(&vec![0.0; (HIDDEN_SIZE * VOCAB_SIZE) as usize]),
    }];
    fs::write(
        dir.path().join("model.gguf"),
        build_gguf(&kvs, &tensors, 32),
    )
    .unwrap();
    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
        dir.path().to_path_buf(),
    );
    let error = match GgufIngestor::new().ingest(&source) {
        Err(error) => error,
        Ok(_) => panic!("expected an unsupported-format error"),
    };
    assert!(matches!(
        error,
        magnetar_runtime::production_model_ingestion::ProductionIngestionError::UnsupportedFormat { .. }
    ));
}
