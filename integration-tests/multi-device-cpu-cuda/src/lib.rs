//! Real, first native multi-Device execution proof: a single `Runtime`
//! registering two real, heterogeneous Providers (Reference CPU and CUDA)
//! simultaneously, dispatching one logical computation across both real
//! Devices with explicit cross-Device movement between them.
//!
//! Neither `magnetar-runtime` nor either Provider crate can prove this on
//! its own: `magnetar-runtime` never depends on `magnetar-provider-cpu` or
//! `magnetar-provider-cuda` (and must not --
//! `externalize-runtime-extension-modules`), and each Provider crate's own
//! tests exercise only itself. This crate is the third party that can hold
//! all three, mirroring `integration-tests/cuda-first-native`'s own reason
//! for existing as a separate crate.
//!
//! See `tests_multi_device_cpu_cuda.rs` for what this does and does not
//! prove.

#[cfg(test)]
mod tests_multi_device_cpu_cuda;
