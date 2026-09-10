//! Real production Qwen loading end-to-end proof
//! (`implement-production-qwen-model-loading` task group 12).
//!
//! Neither `magnetar-runtime` nor `loaders/huggingface` can prove this on
//! its own: `magnetar-runtime` never depends on `loaders/huggingface`
//! (and must not -- `production-model-ingestion`'s externalization
//! boundary), and `loaders/huggingface`'s own tests deliberately stop at
//! its ingestion contract's output (a normalized `ModelManifest` plus a
//! payload source), never reaching Model Loading, the real Qwen
//! Component, or generation. This crate is the third party that can hold
//! both: a tiny, deterministic, but genuinely production-shaped Hugging
//! Face bundle (real `config.json`, real `tokenizer.json`, real
//! Safetensors bytes) written to a temp directory, ingested through the
//! real external `HuggingFaceIngestor`, loaded through
//! `magnetar_runtime::load_production_qwen_instance`, and executed
//! through the real compiled Qwen Component and Reference CPU Provider --
//! no fixture manifest, no `qwen-test` identity, anywhere in this path.
//! `tests_production_loading_cuda_e2e.rs` proves the same real ingested
//! artifact and Component on real CUDA hardware instead (task 12.3), with
//! zero CUDA-specific code in `magnetar-runtime` itself. No library code
//! of its own.

#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests_production_loading_cuda_e2e;
#[cfg(test)]
mod tests_production_loading_e2e;
#[cfg(test)]
mod tests_real_checkpoint_smoke;
