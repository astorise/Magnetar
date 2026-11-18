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

use crate::logit_comparison::{Format, compare_on, compare_with_reference, cpu_providers};
use magnetar_integration_resident_provider::ResidentReferenceCpuProvider;

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

/// Every checkpoint above with every value held "on the Device"
/// ([`ResidentReferenceCpuProvider`]: the Reference CPU, each value it holds
/// reported resident, as a device Provider's). It catches what a device
/// Provider declines, a request for an intermediate value's host bytes, on
/// any machine. A head-wise `concat` that made one failed DeepSeek on CUDA
/// at graph planning.
#[test]
fn the_tiny_deepseek_v2_checkpoints_match_transformers_with_resident_values() {
    for (variant, format, family) in VARIANTS {
        let difference = compare_on(
            std::sync::Arc::new(ResidentReferenceCpuProvider::new()),
            "magnetar-reference-cpu-resident",
            "deepseek_v2",
            variant,
            format,
            family,
        );
        eprintln!("resident {variant}: max |logit difference| {difference:.3e}");
    }
}

/// A latent-attention prefill attends over the expanded keys and values of
/// its own rows while its history holds the latent rows, so a later pass
/// could not read the earlier ones: the model runs every prompt in one pass
/// (`chunk-long-prefills`), and the logits still match the references.
#[test]
fn a_latent_attention_prefill_runs_in_one_pass() {
    for (variant, format, family) in VARIANTS {
        let (difference, chunk) = crate::logit_comparison::compare_chunked_on(
            std::sync::Arc::new(magnetar_runtime::ReferenceCpuProvider::new()),
            "magnetar-reference-cpu",
            "deepseek_v2",
            variant,
            format,
            family,
            4,
        );
        assert_eq!(chunk, 0, "{variant}: the prefill was chunked");
        eprintln!("{variant}, chunking declined: max |logit difference| {difference:.3e}");
    }
}

/// Every checkpoint of this file, with its format and declared family.
const VARIANTS: [(&str, Format, &str); 6] = [
    ("lite", Format::HuggingFace, "deepseek_v2"),
    ("qlora", Format::HuggingFace, "deepseek_v2"),
    ("plain", Format::HuggingFace, "deepseek_v2"),
    ("gguf-kv-b", Format::Gguf, "deepseek2"),
    ("gguf-split", Format::Gguf, "deepseek2"),
    ("gguf-qlora", Format::Gguf, "deepseek2"),
];

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
    for (variant, format, family) in VARIANTS {
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

/// Every checkpoint above on the CPU Providers of `providers/cpu` (see
/// [`cpu_providers`]): their Kernels for the latent attention's forms,
/// YaRN and the routed experts, the optimized one reading the GGUF files'
/// packed weights directly.
#[test]
fn the_tiny_deepseek_v2_checkpoints_match_transformers_on_the_cpu_providers() {
    for (provider, label) in cpu_providers() {
        for (variant, format, family) in VARIANTS {
            let difference = compare_on(
                provider.clone(),
                label,
                "deepseek_v2",
                variant,
                format,
                family,
            );
            eprintln!("{label} {variant}: max |logit difference| {difference:.3e}");
        }
    }
}

/// TurboQuant on the latent history (`turboquant-kv-history`,
/// `cache-latent-attention-history`) on the Reference CPU: the GGUF
/// checkpoints' latent rows (from the `wide` bundle) take `tq4` and `tq3`
/// in groups of gcd(kv_lora_rank, qk_rope_head_dim), and the logits move
/// from the f32 history's by the stores' rounding; the Hugging Face ones'
/// groups of 4 values are too narrow, and they keep f32 with the reason
/// recorded.
#[test]
fn the_tiny_deepseek_v2_checkpoints_run_with_a_turboquant_latent_history() {
    use crate::logit_comparison::kv_precision_drift;
    use magnetar_runtime::{KvPrecisions, KvStorePrecision, ReferenceCpuProvider};
    for precision in [KvStorePrecision::Tq4, KvStorePrecision::Tq3] {
        let requested = KvPrecisions::uniform(precision);
        for (variant, format, family) in VARIANTS {
            let drift = kv_precision_drift(
                &|| std::sync::Arc::new(ReferenceCpuProvider::new()),
                "deepseek_v2",
                variant,
                format,
                family,
                requested,
            );
            if matches!(format, Format::Gguf) {
                assert_eq!(
                    drift.resolution.resolved, requested,
                    "{variant}: {:?}",
                    drift.resolution.fallbacks
                );
                assert!(
                    drift.max_abs_difference > 0.0 && drift.max_abs_difference.is_finite(),
                    "{variant} {precision:?}: {}",
                    drift.max_abs_difference
                );
            } else {
                assert_eq!(
                    drift.resolution.resolved,
                    KvPrecisions::default(),
                    "{variant}"
                );
                assert!(
                    drift
                        .resolution
                        .fallbacks
                        .iter()
                        .all(|reason| reason.contains("4 wide")),
                    "{variant}: {:?}",
                    drift.resolution.fallbacks
                );
                assert_eq!(drift.max_abs_difference, 0.0, "{variant}");
            }
            eprintln!(
                "{} {variant}: max |logit difference| {:.3e} against f32 stores, \
                 {}/{} leading tokens agree",
                precision.as_str(),
                drift.max_abs_difference,
                drift.agreeing_tokens,
                drift.steps
            );
        }
    }
}

/// The CPU Providers of `providers/cpu` hold a TurboQuant latent history
/// as the Runtime's Reference CPU does: with `tq4`, the GGUF checkpoints
/// generate the same tokens with logits within 1e-3 of the Reference
/// CPU's.
#[test]
fn the_cpu_providers_hold_a_turboquant_latent_history_like_the_reference_cpu() {
    use crate::logit_comparison::generate_logits;
    use magnetar_runtime::{KvPrecisions, KvStorePrecision, ReferenceCpuProvider};
    let tq4 = KvPrecisions::uniform(KvStorePrecision::Tq4);
    for (variant, format, family) in VARIANTS {
        if !matches!(format, Format::Gguf) {
            continue;
        }
        let (reference_steps, reference_tokens, _) = generate_logits(
            std::sync::Arc::new(ReferenceCpuProvider::new()),
            "deepseek_v2",
            variant,
            format,
            family,
            tq4,
        );
        for (provider, label) in cpu_providers() {
            let (steps, tokens, resolution) =
                generate_logits(provider, "deepseek_v2", variant, format, family, tq4);
            assert_eq!(resolution.resolved, tq4, "{label} {variant}");
            assert_eq!(tokens, reference_tokens, "{label} {variant}");
            let difference = reference_steps
                .iter()
                .zip(&steps)
                .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
                .fold(0.0_f32, f32::max);
            assert!(difference < 1e-3, "{label} {variant}: {difference}");
            eprintln!(
                "{label} {variant} tq4: max |logit difference| {difference:.3e} against the Reference CPU"
            );
        }
    }
}
