//! The Gemma 4 Model Component (`components/gemma`, checked in as
//! `magnetar-runtime/fixtures/components/gemma-real.component.wasm`) run by
//! the Reference CPU Provider, against `transformers` 5.23's
//! `Gemma4ForCausalLM` in float32, on tiny random checkpoints
//! (`fixtures/gemma4/`, see its README for how they were made).
//!
//! Each case ingests a checkpoint through a real loader, loads it with the
//! Component its architecture family selects, generates 12 greedy tokens
//! (one prefill and eleven decode steps with the KV cache) and compares
//! every step's top-8 logits with the reference: identical tokens, within
//! 1e-3. The quantized GGUF references are computed on the dequantized
//! weights, so they test the handling of the format, not the quantization
//! error.

use crate::logit_comparison::{
    Format, compare_chunked_on, compare_on, compare_with_reference, cpu_providers, generate_logits,
    kv_precision_drift,
};
use magnetar_integration_resident_provider::ResidentReferenceCpuProvider;

fn compare(variant: &str, format: Format, family: &str) -> f32 {
    compare_with_reference("gemma4", variant, format, family)
}

/// Sliding and full layers with their own head widths, V from K, partial
/// RoPE, q/k/v norms, sandwich norms, `layer_scalar`, GeGLU and the
/// logit soft-cap, from a Hugging Face bundle with a nested text config.
#[test]
fn a_dense_gemma4_bundle_matches_transformers() {
    let difference = compare("dense", Format::HuggingFace, "gemma4");
    eprintln!("dense: max |logit difference| {difference:.3e}");
}

/// The same decoder as a `gemma4_unified` bundle (the 12B's model type),
/// whose vision tensor the loader leaves out.
#[test]
fn a_unified_gemma4_bundle_matches_transformers() {
    let difference = compare("unified", Format::HuggingFace, "gemma4_unified");
    eprintln!("unified: max |logit difference| {difference:.3e}");
}

/// Quantized `gemma4` GGUF files (`Q4_0` projections and experts, `Q8_0`
/// embedding and per-layer tables): the layout from the metadata, shared
/// KV with a wider MLP, per-layer input tables cut per layer, and fused
/// experts restacked with the router fold and per-expert scale.
#[test]
fn quantized_gemma4_gguf_files_match_transformers() {
    for variant in ["gguf-dense", "gguf-shared", "gguf-ple", "gguf-moe"] {
        let difference = compare(variant, Format::Gguf, "gemma4");
        eprintln!("{variant}: max |logit difference| {difference:.3e}");
    }
}

/// Every checkpoint of this file, with its format and declared family.
const VARIANTS: [(&str, Format, &str); 6] = [
    ("dense", Format::HuggingFace, "gemma4"),
    ("unified", Format::HuggingFace, "gemma4_unified"),
    ("gguf-dense", Format::Gguf, "gemma4"),
    ("gguf-shared", Format::Gguf, "gemma4"),
    ("gguf-ple", Format::Gguf, "gemma4"),
    ("gguf-moe", Format::Gguf, "gemma4"),
];

/// Every checkpoint above with every value held "on the Device"
/// ([`ResidentReferenceCpuProvider`]), so that no step of the Gemma 4
/// graph asks a device Provider for an intermediate value's host bytes.
#[test]
fn the_tiny_gemma4_checkpoints_match_transformers_with_resident_values() {
    for (variant, format, family) in VARIANTS {
        let difference = compare_on(
            std::sync::Arc::new(ResidentReferenceCpuProvider::new()),
            "magnetar-reference-cpu-resident",
            "gemma4",
            variant,
            format,
            family,
        );
        eprintln!("resident {variant}: max |logit difference| {difference:.3e}");
    }
}

/// Every checkpoint above on CUDA: the CUDA Kernels for the layered forms
/// (per-layer head widths, shared KV, per-layer inputs, the experts and the
/// logit soft-cap), reading the GGUF files' packed `Q4_0` and `Q8_0`
/// weights, against the same references. Run with f32 activations
/// (`MAGNETAR_CUDA_INT8_ACTIVATIONS=0`): the int8 ones round the
/// activations by more than the comparison allows.
#[test]
#[ignore = "needs a CUDA Device and MAGNETAR_CUDA_INT8_ACTIVATIONS=0"]
fn the_tiny_gemma4_checkpoints_match_transformers_on_cuda() {
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
            "gemma4",
            variant,
            format,
            family,
        );
        eprintln!("cuda {variant}: max |logit difference| {difference:.3e}");
        // Prefilled in passes of 3 rows (`chunk-long-prefills`): the CUDA
        // attention Kernels over a history longer than their query rows.
        let (difference, chunk) = compare_chunked_on(
            std::sync::Arc::new(magnetar_provider_cuda::CudaProvider::new()),
            "magnetar-cuda",
            "gemma4",
            variant,
            format,
            family,
            3,
        );
        assert_eq!(chunk, 3, "cuda {variant}: the prefill ran in one pass");
        eprintln!("cuda {variant} in chunks of 3: max |logit difference| {difference:.3e}");
    }
}

/// Every checkpoint above with its 8-token prompt prefilled in passes of 3
/// rows (`chunk-long-prefills`): each pass continues the KV history the
/// passes before it wrote (sliding and full layers, shared KV) from its own
/// position, and the last pass is padded. The logits still match the
/// references, on the Reference CPU and both CPU Providers.
#[test]
fn the_tiny_gemma4_checkpoints_match_transformers_when_prefilled_in_chunks() {
    let reference: (std::sync::Arc<dyn magnetar_runtime::Provider>, &str) = (
        std::sync::Arc::new(magnetar_runtime::ReferenceCpuProvider::new()),
        "magnetar-reference-cpu",
    );
    for (provider, label) in std::iter::once(reference).chain(cpu_providers()) {
        for (variant, format, family) in VARIANTS {
            let (difference, chunk) = compare_chunked_on(
                provider.clone(),
                label,
                "gemma4",
                variant,
                format,
                family,
                3,
            );
            assert_eq!(chunk, 3, "{label} {variant}: the prefill ran in one pass");
            eprintln!("{label} {variant} in chunks of 3: max |logit difference| {difference:.3e}");
        }
    }
}

/// Every checkpoint above on the CPU Providers of `providers/cpu` (see
/// [`cpu_providers`]): their Kernels for the layered forms, the optimized
/// one reading the GGUF files' packed `Q4_0` and `Q8_0` weights directly.
#[test]
fn the_tiny_gemma4_checkpoints_match_transformers_on_the_cpu_providers() {
    for (provider, label) in cpu_providers() {
        for (variant, format, family) in VARIANTS {
            let difference = compare_on(provider.clone(), label, "gemma4", variant, format, family);
            eprintln!("{label} {variant}: max |logit difference| {difference:.3e}");
        }
    }
}

/// Every checkpoint above with TurboQuant KV stores on the Reference CPU
/// (`turboquant-kv-history`): its 16- and 32-wide heads take `tq4` and
/// `tq3`, and the logits move from the f32 history's by the stores'
/// rounding, never by nothing.
#[test]
fn the_tiny_gemma4_checkpoints_run_with_turboquant_kv_stores() {
    use magnetar_runtime::{KvPrecisions, KvStorePrecision, ReferenceCpuProvider};
    for precision in [KvStorePrecision::Tq4, KvStorePrecision::Tq3] {
        for (variant, format, family) in VARIANTS {
            let drift = kv_precision_drift(
                &|| std::sync::Arc::new(ReferenceCpuProvider::new()),
                "gemma4",
                variant,
                format,
                family,
                KvPrecisions::uniform(precision),
            );
            assert_eq!(
                drift.resolution.resolved,
                KvPrecisions::uniform(precision),
                "{variant}: {:?}",
                drift.resolution.fallbacks
            );
            assert!(
                drift.max_abs_difference > 0.0 && drift.max_abs_difference.is_finite(),
                "{variant} {precision:?}: {}",
                drift.max_abs_difference
            );
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

/// The CPU Providers of `providers/cpu` hold TurboQuant stores as the
/// Runtime's Reference CPU does: with `tq4` K and V, every checkpoint
/// above generates the same tokens with logits within 1e-3 of the
/// Reference CPU's (`turboquant-kv-history`).
#[test]
fn the_cpu_providers_hold_turboquant_kv_stores_like_the_reference_cpu() {
    use magnetar_runtime::{KvPrecisions, KvStorePrecision, ReferenceCpuProvider};
    let tq4 = KvPrecisions::uniform(KvStorePrecision::Tq4);
    for (variant, format, family) in VARIANTS {
        let (reference_steps, reference_tokens, _) = generate_logits(
            std::sync::Arc::new(ReferenceCpuProvider::new()),
            "gemma4",
            variant,
            format,
            family,
            tq4,
        );
        for (provider, label) in cpu_providers() {
            let (steps, tokens, resolution) =
                generate_logits(provider, "gemma4", variant, format, family, tq4);
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
