//! Hardware-agnostic runtime contracts and provider support for Magnetar.
//!
//! The crate root is intentionally a facade. Runtime responsibilities live in
//! architectural modules so the future Component engine and AI domains can be
//! added through dedicated contracts instead of expanding this file again.

pub mod adapter;
pub mod affinity;
pub mod batching;
pub mod capability;
pub mod cli_boundary;
pub mod component;
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub mod component_wasmtime;
#[cfg(all(target_arch = "wasm32", feature = "web-component-engine"))]
pub mod component_web;
pub mod compute;
pub mod conformance;
pub mod device;
pub mod e2e_conformance;
pub mod execution_graph;
pub mod generation;
pub mod inference_api;
pub mod kernel;
pub mod kernel_artifact;
pub mod kernel_artifact_ingestion;
pub mod kernel_artifact_manifest;
pub mod kernel_autotuning;
pub mod kernel_benchmark;
pub mod kernel_cache;
pub mod kernel_compilation;
pub mod kernel_dispatch;
pub mod kernel_optimization_orchestration;
pub mod kernel_performance_model;
pub mod kernel_qualification;
pub mod kernel_registry;
pub mod kernel_selection_policy;
pub mod kv_cache;
pub mod memory;
pub mod model;
pub mod model_component;
pub mod model_format_roadmap;
pub mod model_instance;
pub mod model_loading;
pub mod model_source_cache_roadmap;
pub mod observability;
pub mod operator;
pub mod operator_scope;
pub mod planning;
pub mod prefix_cache;
pub mod provider;
pub mod provider_roadmap;
pub mod qwen_model_component;
pub mod reference_cpu;
pub mod release_cutover;
pub mod release_packaging;
pub mod release_security;
pub mod resolution;
pub mod runtime;
pub mod sampling;
pub mod scheduler;
pub mod server_api_roadmap;
pub mod session;
pub mod tensor;
pub mod tokenizer;

pub use adapter::*;
pub use affinity::*;
pub use batching::*;
pub use capability::*;
pub use cli_boundary::*;
pub use component::*;
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub use component_wasmtime::*;
#[cfg(all(target_arch = "wasm32", feature = "web-component-engine"))]
pub use component_web::*;
pub use compute::*;
pub use conformance::*;
pub use device::*;
pub use e2e_conformance::*;
pub use execution_graph::*;
pub use generation::*;
pub use inference_api::*;
pub use kernel::*;
pub use kernel_artifact::*;
pub use kernel_artifact_ingestion::*;
pub use kernel_artifact_manifest::*;
pub use kernel_autotuning::*;
pub use kernel_benchmark::*;
pub use kernel_cache::*;
pub use kernel_compilation::*;
pub use kernel_dispatch::*;
pub use kernel_performance_model::*;
pub use kernel_qualification::*;
pub use kernel_registry::*;
pub use kernel_selection_policy::*;
pub use kv_cache::*;
pub use memory::*;
pub use model::*;
pub use model_component::*;
pub use model_format_roadmap::*;
pub use model_instance::*;
pub use model_loading::*;
pub use model_source_cache_roadmap::*;
pub use observability::*;
pub use operator::*;
pub use operator_scope::*;
pub use planning::*;
pub use prefix_cache::*;
pub use provider::*;
pub use provider_roadmap::*;
pub use qwen_model_component::*;
pub use reference_cpu::*;
pub use release_cutover::*;
pub use release_packaging::*;
pub use release_security::*;
pub use resolution::*;
pub use runtime::*;
pub use sampling::*;
pub use scheduler::*;
pub use server_api_roadmap::*;
pub use session::*;
pub use tensor::*;
pub use tokenizer::*;
#[cfg(test)]
mod tests;
