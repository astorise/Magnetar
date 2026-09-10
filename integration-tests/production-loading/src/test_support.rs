//! Shared test fixtures for this crate's Reference CPU and CUDA
//! production-loading integration tests: a real, production-shaped
//! Hugging Face Qwen bundle (real `config.json`, real `tokenizer.json`/
//! `tokenizer_config.json`, real Safetensors bytes) small enough for a
//! per-PR test, plus the real checked-in Qwen Component artifact
//! registration both test files need.

use magnetar_runtime::register_qwen_component_artifact;
use std::{fs, io::Write};

/// Reads the checked-in, real Qwen Component artifact this repository
/// ships and registers it for production first-native generation to use
/// -- mirrors `integration-tests/cuda-first-native`'s own
/// `register_real_qwen_component`, the same real embedder-facing call
/// path `magnetar-cli` uses, not a test-only shortcut. `register_qwen_
/// component_artifact` is idempotent (a `OnceLock`), so calling it from
/// more than one test file in this crate is safe.
pub(crate) fn register_real_qwen_component() {
    let component_bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../magnetar-runtime/fixtures/components/qwen-real.component.wasm"
    ))
    .expect("checked-in real Qwen Component .wasm is readable");
    let manifest_bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../magnetar-runtime/fixtures/components/qwen-real.component.wasm.magnetar-component.yaml"
    ))
    .expect("checked-in real Qwen Component manifest is readable");
    register_qwen_component_artifact(component_bytes, manifest_bytes);
}

pub(crate) const HIDDEN_SIZE: u64 = 4;
pub(crate) const LAYER_COUNT: u64 = 1;
pub(crate) const ATTENTION_HEAD_COUNT: u64 = 2;
pub(crate) const KV_HEAD_COUNT: u64 = 2;
pub(crate) const HEAD_DIMENSION: u64 = 2;
pub(crate) const INTERMEDIATE_SIZE: u64 = 8;
pub(crate) const VOCAB_SIZE: u64 = 16;

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
pub(crate) fn write_tiny_production_bundle(dir: &std::path::Path) {
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
