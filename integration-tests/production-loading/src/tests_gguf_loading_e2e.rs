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
    write_tiny_gguf_bundle_for(dir, "qwen2");
}

/// llama.cpp's `LlamaModel.permute` of a `[rows, cols]` row-major
/// projection's rows (`w.reshape(n_head, 2, rows / n_head / 2, cols)
/// .swapaxes(1, 2)`): within each head, upstream row `half * head_dim / 2
/// + i` is stored at row `2 * i + half`.
/// Routed experts of the tiny `qwen2moe` file.
const TINY_MOE_EXPERTS: u64 = 3;

/// The intermediate-neuron order of tiny expert `expert`: identity,
/// rotated by one, reversed.
fn tiny_moe_permutation(expert: usize, neurons: usize) -> Vec<usize> {
    (0..neurons)
        .map(|i| match expert {
            0 => i,
            1 => (i + 1) % neurons,
            _ => neurons - 1 - i,
        })
        .collect()
}

fn llama_cpp_permute_rows(values: &[f32], rows: u64, cols: u64) -> Vec<f32> {
    let (head_dim, cols) = (HEAD_DIMENSION, cols as usize);
    let mut stored = vec![0.0; values.len()];
    for upstream in 0..rows {
        let (head, within) = (upstream / head_dim, upstream % head_dim);
        let (half, i) = (within / (head_dim / 2), within % (head_dim / 2));
        let row = (head * head_dim + 2 * i + half) as usize;
        let upstream = upstream as usize;
        stored[row * cols..(row + 1) * cols]
            .copy_from_slice(&values[upstream * cols..(upstream + 1) * cols]);
    }
    stored
}

/// [`write_tiny_gguf_bundle`] under `architecture`'s key schema. For
/// `llama`, the query and key projections are stored in llama.cpp's
/// permuted row order, as its converter writes them.
fn write_tiny_gguf_bundle_for(dir: &std::path::Path, architecture: &str) {
    let q_dim = ATTENTION_HEAD_COUNT * HEAD_DIMENSION;
    let kv_dim = KV_HEAD_COUNT * HEAD_DIMENSION;
    let key = |name: &str| format!("{architecture}.{name}");

    let mut kvs = vec![
        kv_string("general.architecture", architecture),
        kv_uint32(&key("embedding_length"), HIDDEN_SIZE as u32),
        kv_uint32(&key("feed_forward_length"), INTERMEDIATE_SIZE as u32),
        kv_uint32(&key("block_count"), LAYER_COUNT as u32),
        kv_uint32(&key("attention.head_count"), ATTENTION_HEAD_COUNT as u32),
        kv_uint32(&key("attention.head_count_kv"), KV_HEAD_COUNT as u32),
        kv_uint32("tokenizer.ggml.bos_token_id", 0),
        kv_uint32("tokenizer.ggml.eos_token_id", 1),
        kv_string("tokenizer.ggml.model", "gpt2"),
    ];
    let moe = architecture == "qwen2moe";
    if moe {
        kvs.push(kv_uint32(&key("expert_count"), TINY_MOE_EXPERTS as u32));
        kvs.push(kv_uint32(
            &key("expert_used_count"),
            TINY_MOE_EXPERTS as u32,
        ));
    }
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

    if moe {
        for layer in 0..LAYER_COUNT {
            entries.push((
                format!("model.layers.{layer}.mlp.gate.weight"),
                format!("blk.{layer}.ffn_gate_inp.weight"),
                vec![HIDDEN_SIZE, TINY_MOE_EXPERTS],
            ));
        }
    }
    let tensors: Vec<GgufTensor> = entries
        .into_iter()
        .map(|(hf_name_seed, gguf_name, ne)| {
            let element_count: u64 = ne.iter().product();
            let mut values = tensor_values(&hf_name_seed, element_count);
            // `qwen2moe`: every routed expert is a copy of the dense MLP,
            // stacked as llama.cpp's `[experts, out, in]`.
            let dense_mlp = ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]
                .iter()
                .any(|suffix| gguf_name.ends_with(suffix));
            if moe && dense_mlp {
                // Expert `e` is the dense MLP with its intermediate neurons
                // permuted by `tiny_moe_permutation(e)`: the same function,
                // different bytes per expert, so a slice or offset mix-up
                // between experts changes the result.
                let down = gguf_name.ends_with("ffn_down.weight");
                let (cols, rows) = (ne[0] as usize, ne[1] as usize);
                let mut stacked = Vec::with_capacity(values.len() * TINY_MOE_EXPERTS as usize);
                for expert in 0..TINY_MOE_EXPERTS as usize {
                    let neuron = tiny_moe_permutation(expert, INTERMEDIATE_SIZE as usize);
                    for row in 0..rows {
                        for col in 0..cols {
                            // gate/up store a neuron per row; down stores
                            // one per column.
                            let (r, c) = if down {
                                (row, neuron[col])
                            } else {
                                (neuron[row], col)
                            };
                            stacked.push(values[r * cols + c]);
                        }
                    }
                }
                let mut ne = ne;
                ne.push(TINY_MOE_EXPERTS);
                return GgufTensor {
                    name: gguf_name.replace(".weight", "_exps.weight"),
                    ne,
                    data: f32_bytes(&stacked),
                };
            }
            if architecture == "llama"
                && (gguf_name.ends_with("attn_q.weight") || gguf_name.ends_with("attn_k.weight"))
            {
                values = llama_cpp_permute_rows(&values, ne[1], ne[0]);
            }
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
fn rejects_an_unsupported_gguf_architecture() {
    let dir = tempfile::tempdir().unwrap();
    let kvs = vec![kv_string("general.architecture", "gemma3")];
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

/// The greedy tokens `ingested` generates for token ids `[2, 3]` on the
/// Reference CPU, through the real Model Component declaring its family.
fn reference_cpu_tokens(
    ingested: &magnetar_runtime::production_model_ingestion::ProductionIngestionResult,
    tokenizer: Arc<dyn Tokenizer + Send + Sync>,
) -> Vec<u32> {
    let component = crate::test_support::real_component_for_architecture_family(
        &ingested.manifest.architecture.family,
    );
    let fixture = production_model_fixture(
        ingested.manifest.clone(),
        tokenizer.metadata().clone(),
        tokenizer,
    )
    .expect("production fixture builds from real ingested data");
    let mut model = magnetar_runtime::ProductionLoadedModel::load_with_component(
        fixture,
        ingested.payload_source.as_ref(),
        ModelTrustStore::default().trust_digest(ingested.manifest.id.digest.value.clone()),
        Arc::new(magnetar_runtime::ReferenceCpuProvider::new()),
        Some(component),
    )
    .expect("the model loads with the Component declaring its family");
    let outcome = model
        .generate(
            magnetar_runtime::ProductionGenerationRequest {
                prompt: PromptInput::TokenIds(vec![2, 3]),
                parameters: magnetar_runtime::GenerationParameters::greedy(),
                stop_conditions: magnetar_runtime::StopConditions::default(),
                max_new_tokens: Some(6),
                max_generation_millis: None,
            },
            None,
        )
        .expect("generation succeeds");
    outcome.result.output.generated_token_ids
}

/// A `llama`-architecture GGUF file stores its query and key projections
/// in llama.cpp's permuted row order; the GGUF loader restores the
/// upstream order, so the same weights loaded from a Hugging Face Llama
/// bundle and from the GGUF file generate identical tokens through the
/// Llama Model Component. With heads of 4 rows the permutation moves rows,
/// so a missing or wrong restoration changes the rotary pairs.
#[test]
fn llama_gguf_and_huggingface_ingestion_of_identical_weights_produce_identical_generation() {
    let hf_dir = tempfile::tempdir().unwrap();
    write_tiny_production_bundle(hf_dir.path());
    let config = fs::read_to_string(hf_dir.path().join("config.json"))
        .unwrap()
        .replace("Qwen2ForCausalLM", "LlamaForCausalLM")
        .replace(r#""model_type": "qwen2""#, r#""model_type": "llama""#);
    fs::write(hf_dir.path().join("config.json"), config).unwrap();
    let hf_source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::LocalPath(hf_dir.path().to_path_buf()),
        hf_dir.path().to_path_buf(),
    );
    let hf_ingested = magnetar_loader_huggingface::HuggingFaceIngestor::new()
        .ingest(&hf_source)
        .expect("real Hugging Face Llama bundle ingestion succeeds");
    assert_eq!(hf_ingested.manifest.architecture.family, "llama");
    let hf_tokenizer = magnetar_loader_huggingface::HuggingFaceTokenizer::from_bytes(
        &fs::read(hf_dir.path().join("tokenizer.json")).unwrap(),
        None,
        "hf-llama-cross-check",
        Some(VOCAB_SIZE),
    )
    .expect("real tokenizer.json loads");
    let hf_tokens = reference_cpu_tokens(&hf_ingested, Arc::new(hf_tokenizer));

    let gguf_dir = tempfile::tempdir().unwrap();
    write_tiny_gguf_bundle_for(gguf_dir.path(), "llama");
    let gguf_source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::LocalPath(gguf_dir.path().to_path_buf()),
        gguf_dir.path().to_path_buf(),
    );
    let gguf_ingested = GgufIngestor::new()
        .ingest(&gguf_source)
        .expect("real llama GGUF ingestion succeeds");
    assert_eq!(gguf_ingested.manifest.architecture.family, "llama");
    let gguf_tokenizer =
        magnetar_loader_gguf::load_gguf_tokenizer(&gguf_source, "gguf-llama-cross-check", None)
            .expect("real tokenizer builds from embedded vocabulary");
    let gguf_tokens = reference_cpu_tokens(&gguf_ingested, Arc::new(gguf_tokenizer));

    // Every tensor reaches Model Loading with the same bytes from both
    // ingestors: the restored query/key rows above all.
    let read =
        |ingested: &magnetar_runtime::production_model_ingestion::ProductionIngestionResult,
         name: &str| {
            let tensor = ingested
                .manifest
                .tensors
                .iter()
                .find(|tensor| tensor.name == name)
                .unwrap_or_else(|| panic!("tensor '{name}' is declared"));
            ingested
                .payload_source
                .read_payload(
                    &magnetar_runtime::production_model_ingestion::ProductionPayloadRange {
                        identity: tensor.name.clone(),
                        offset: tensor.offset_bytes.unwrap(),
                        length: tensor.size_bytes.unwrap(),
                        digest: tensor.digest.clone(),
                    },
                )
                .unwrap()
        };
    for tensor in &hf_ingested.manifest.tensors {
        assert_eq!(
            read(&gguf_ingested, &tensor.name),
            read(&hf_ingested, &tensor.name),
            "tensor '{}' differs between the two ingestors",
            tensor.name
        );
    }
    assert_eq!(
        gguf_tokens, hf_tokens,
        "identical weights through the Safetensors and the llama GGUF ingestors must generate \
         identical tokens"
    );
}

/// `execute-and-cache-moe-experts` F02 end to end: a tiny `qwen2moe` GGUF
/// file whose routed experts are all the dense file's MLP, each with its
/// intermediate neurons in a different order (the same function), every
/// expert chosen (`top_k = experts`, so the routing weights sum to one),
/// generates the dense file's tokens through the Qwen Component's
/// `moe-experts` graph on the Reference CPU -- the loader's expert naming
/// and stacking, the Component's MoE graph and the Operator's routing and
/// combination all take part, and any column offset or routing weight
/// error changes the result.
#[test]
fn a_qwen2moe_gguf_of_copied_experts_generates_the_dense_tokens() {
    let tokens = |architecture: &str| {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_gguf_bundle_for(dir.path(), architecture);
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let ingested = GgufIngestor::new()
            .ingest(&source)
            .expect("the tiny GGUF file ingests");
        assert_eq!(ingested.manifest.architecture.family, architecture);
        let tokenizer = magnetar_loader_gguf::load_gguf_tokenizer(&source, "tiny", None)
            .expect("the tokenizer builds");
        (
            reference_cpu_tokens(&ingested, Arc::new(tokenizer)),
            ingested,
            // The directory outlives the ingestion that reads from it.
            dir,
        )
    };
    let (dense, dense_ingested, _dense_dir) = tokens("qwen2");
    let (routed, ingested, _dir) = tokens("qwen2moe");
    // The ingested stacked experts are the `moe-experts` Operator's layout:
    // column `e * I + j` of `experts_{gate,up}_proj [H, E*I]` is the dense
    // projection's column `perm_e(j)`, and row `j` of `experts_down_proj
    // [I, E*H]` columns `[e*H, (e+1)*H)` is the dense row `perm_e(j)`.
    let f32s =
        |ingested: &magnetar_runtime::production_model_ingestion::ProductionIngestionResult,
         name: &str| {
            let tensor = ingested
                .manifest
                .tensors
                .iter()
                .find(|tensor| tensor.name == name)
                .unwrap_or_else(|| panic!("tensor '{name}' is declared"));
            let bytes = ingested
                .payload_source
                .read_payload(
                    &magnetar_runtime::production_model_ingestion::ProductionPayloadRange {
                        identity: tensor.name.clone(),
                        offset: tensor.offset_bytes.unwrap(),
                        length: tensor.size_bytes.unwrap(),
                        digest: tensor.digest.clone(),
                    },
                )
                .unwrap();
            let values: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect();
            (tensor.shape.clone(), values)
        };
    let (h, i, e) = (
        HIDDEN_SIZE as usize,
        INTERMEDIATE_SIZE as usize,
        TINY_MOE_EXPERTS as usize,
    );
    for projection in ["gate", "up"] {
        let (dense_shape, dense_values) =
            f32s(&dense_ingested, &format!("layers.0.mlp.{projection}_proj"));
        let (shape, values) = f32s(
            &ingested,
            &format!("layers.0.moe.experts_{projection}_proj"),
        );
        assert_eq!(dense_shape, vec![h as u64, i as u64]);
        assert_eq!(shape, vec![h as u64, (e * i) as u64]);
        for expert in 0..e {
            let perm = tiny_moe_permutation(expert, i);
            for k in 0..h {
                for j in 0..i {
                    assert_eq!(
                        values[k * e * i + expert * i + j],
                        dense_values[k * i + perm[j]],
                        "{projection} expert {expert} ({k}, {j})"
                    );
                }
            }
        }
    }
    let (_, dense_down) = f32s(&dense_ingested, "layers.0.mlp.down_proj");
    let (down_shape, down) = f32s(&ingested, "layers.0.moe.experts_down_proj");
    assert_eq!(down_shape, vec![i as u64, (e * h) as u64]);
    for expert in 0..e {
        let perm = tiny_moe_permutation(expert, i);
        for j in 0..i {
            for c in 0..h {
                assert_eq!(
                    down[j * e * h + expert * h + c],
                    dense_down[perm[j] * h + c],
                    "down expert {expert} ({j}, {c})"
                );
            }
        }
    }
    let moe = ingested
        .manifest
        .architecture_config
        .as_ref()
        .and_then(|config| config.mixture_of_experts)
        .expect("the qwen2moe file declares a mixture of experts");
    assert_eq!(moe.experts, TINY_MOE_EXPERTS as u32);
    assert_eq!(moe.experts_per_token, TINY_MOE_EXPERTS as u32);
    assert_eq!(moe.expert_intermediate_size, INTERMEDIATE_SIZE);
    assert_eq!(moe.shared_expert_intermediate_size, None);
    assert_eq!(routed, dense);
}
