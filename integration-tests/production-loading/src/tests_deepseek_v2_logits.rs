//! The DeepSeek Model Component (`components/deepseek`, checked in as
//! `magnetar-runtime/fixtures/components/deepseek-real.component.wasm`) run
//! by the Reference CPU Provider, against `transformers` 5.23's
//! `DeepseekV2ForCausalLM` in float32, on tiny random checkpoints
//! (`fixtures/deepseek_v2/`, see its README for how they were made). Each
//! has a dense first layer, then routed experts beside two ungated shared
//! experts, and multi-head latent attention in its expanded form.
//!
//! The `gguf-*` checkpoints are quantized `deepseek2` GGUF files, compared
//! with `transformers` run on their dequantized weights: they test the GGUF
//! loader's reading of the metadata and of both forms of the latent
//! up-projection, not the quantization error.

use crate::logit_comparison::{Format, compare_on, compare_with_reference};

fn compare(variant: &str) -> f32 {
    let difference =
        compare_with_reference("deepseek_v2", variant, Format::HuggingFace, "deepseek_v2");
    eprintln!("{variant}: max |logit difference| {difference:.3e}");
    difference
}

fn compare_gguf(variant: &str) -> f32 {
    let difference = compare_with_reference("deepseek_v2", variant, Format::Gguf, "deepseek2");
    eprintln!("{variant}: max |logit difference| {difference:.3e}");
    difference
}

/// No low-rank query; YaRN with `mscale == mscale_all_dim` (the published
/// DeepSeek-V2 form: cosines and sines unscaled, the softmax rescaled).
#[test]
fn a_deepseek_v2_lite_shaped_bundle_matches_transformers() {
    compare("lite");
}

/// A low-rank query (`q_a_proj`, `q_a_norm`); YaRN with `mscale !=
/// mscale_all_dim`, so the cosines and sines are scaled too; a routed
/// scaling factor of 2.5.
#[test]
fn a_low_rank_query_bundle_with_scaled_routing_matches_transformers() {
    compare("qlora");
}

/// Latent attention with unscaled RoPE.
#[test]
fn a_deepseek_v2_bundle_without_rope_scaling_matches_transformers() {
    compare("plain");
}

/// A `deepseek2` GGUF file with the latent up-projection in one
/// `attn_kv_b` matrix, as llama.cpp's converter wrote it until 2025 (and as
/// the published DeepSeek-V2-Lite files hold it): `Q8_0` projections,
/// `Q4_0` experts, no `expert_gating_func` and no YaRN betas.
#[test]
fn a_deepseek2_gguf_file_with_one_key_value_projection_matches_transformers() {
    compare_gguf("gguf-kv-b");
}

/// The same decoder with the up-projection split into `attn_k_b` (each
/// head's key rows transposed) and `attn_v_b`, `head_count_kv` 1 and the
/// head widths in `key_length_mla`/`value_length_mla`, as the converter
/// writes it since: `Q4_0` projections (the quantized `attn_k_b` transposed
/// back by the loader), `Q8_0` experts.
#[test]
fn a_deepseek2_gguf_file_with_split_key_value_projections_matches_transformers() {
    compare_gguf("gguf-split");
}

/// A low-rank query (`attn_q_a`, `attn_q_a_norm`, `attn_q_b`) and a routed
/// scaling factor of 2.5 (`expert_weights_scale`), in the split form.
#[test]
fn a_deepseek2_gguf_file_with_a_low_rank_query_matches_transformers() {
    compare_gguf("gguf-qlora");
}

/// Every checkpoint above on CUDA: the CUDA Kernels for the latent
/// attention's forms, YaRN and the routed experts, reading the GGUF files'
/// packed `Q8_0` and `Q4_0` weights, against the same references. Run with
/// f32 activations (`MAGNETAR_CUDA_INT8_ACTIVATIONS=0`): the int8 ones
/// round the activations by more than the comparison allows.
#[test]
#[ignore = "needs a CUDA Device and MAGNETAR_CUDA_INT8_ACTIVATIONS=0"]
fn the_tiny_deepseek_v2_checkpoints_match_transformers_on_cuda() {
    if !magnetar_provider_cuda::CudaProvider::new().is_available() {
        assert!(
            std::env::var("MAGNETAR_REQUIRE_CUDA").as_deref() != Ok("1"),
            "MAGNETAR_REQUIRE_CUDA=1 but no CUDA-capable Device is available"
        );
        eprintln!("skipped: no CUDA-capable Device");
        return;
    }
    assert_eq!(
        std::env::var("MAGNETAR_CUDA_INT8_ACTIVATIONS").as_deref(),
        Ok("0"),
        "run with MAGNETAR_CUDA_INT8_ACTIVATIONS=0: int8 activations round beyond 1e-3"
    );
    let variants = [
        ("lite", Format::HuggingFace, "deepseek_v2"),
        ("qlora", Format::HuggingFace, "deepseek_v2"),
        ("plain", Format::HuggingFace, "deepseek_v2"),
        ("gguf-kv-b", Format::Gguf, "deepseek2"),
        ("gguf-split", Format::Gguf, "deepseek2"),
        ("gguf-qlora", Format::Gguf, "deepseek2"),
    ];
    for (variant, format, family) in variants {
        let difference = compare_on(
            std::sync::Arc::new(magnetar_provider_cuda::CudaProvider::new()),
            "magnetar-cuda",
            "deepseek_v2",
            variant,
            format,
            family,
        );
        eprintln!("cuda {variant}: max |logit difference| {difference:.3e}");
    }
}
