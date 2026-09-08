//! Real first-native CUDA end-to-end proof
//! (`audit-complet-cuda-hot-path-2026-09-08` P1-1).
//!
//! Neither `magnetar-runtime` nor `magnetar-provider-cuda` can prove this
//! on its own: `magnetar-runtime` never depends on
//! `magnetar-provider-cuda` (and must not -- Providers are external,
//! optional extensions, `externalize-runtime-extension-modules`), and
//! `magnetar-provider-cuda`'s own tests (`tests_hardware_hot_path.rs`)
//! deliberately reimplement the generic Kernel dispatch contract rather
//! than reach `magnetar-runtime`'s private, Qwen-specific dispatch
//! functions (`execute_qwen_graph`, `resolve_qwen_weight_edge`,
//! `resident_resource_affinity`, ...) at all -- those are exactly the
//! functions this crate's own previous audit findings lived in.
//!
//! This crate is the third party that can hold both: it depends on
//! `magnetar-runtime`'s public, Provider-generic first-native entrypoint
//! (`run_first_native_graph_with_provider`/
//! `run_first_native_graph_with_provider_and_weights`, added for exactly
//! this purpose) and a real `magnetar_provider_cuda::CudaProvider`,
//! registering the latter with the former -- with zero CUDA-specific code
//! inside `magnetar-runtime` itself. No library code of its own; see
//! `tests/e2e.rs`.

#[cfg(test)]
mod tests_cuda_first_native_e2e;
