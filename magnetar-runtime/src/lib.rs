//! Hardware-agnostic runtime contracts and provider support for Magnetar.
//!
//! The crate root is intentionally a facade. Runtime responsibilities live in
//! architectural modules so the future Component engine and AI domains can be
//! added through dedicated contracts instead of expanding this file again.

pub mod adapter;
pub mod affinity;
pub use magnetar_core::artifact_signature;
pub mod batching;
pub use magnetar_core::binding;
pub use magnetar_core::capability;
pub mod cli_boundary;
pub mod component;
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub mod component_wasmtime;
#[cfg(all(target_arch = "wasm32", feature = "web-component-engine"))]
pub mod component_web;
pub mod compute;
pub use magnetar_core::compute_types;
pub mod conformance;
pub use magnetar_core::device;
#[allow(dead_code)]
mod e2e_conformance;
pub use magnetar_graph::execution_graph;
pub mod execution_stream;
pub mod first_native_implementation_cut;
pub mod first_native_runtime;
pub mod generation;
pub mod graph_builder_capability;
pub use magnetar_graph::graph_fusion;
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
pub mod kernel_execution_plan;
pub mod kernel_optimization_orchestration;
pub mod kernel_performance_model;
pub mod kernel_qualification;
pub mod kernel_registry;
pub mod kernel_selection_policy;
pub mod kv_cache;
pub mod layer_placement;
pub use magnetar_graph::memory;
pub use magnetar_model::model;
pub mod model_component;
pub mod model_config_capability;
pub mod model_format_roadmap;
pub use magnetar_model::model_identity;
pub mod model_instance;
pub mod model_loading;
pub mod multi_device_placement;
pub mod observability;
pub use magnetar_core::operator;
pub mod operator_scope;
pub mod packed_weight;
pub mod phase_attribution;
pub mod planning;
pub mod prefix_cache;
pub mod production_model_ingestion;
pub mod provider;
pub use magnetar_core::provider_abi;
pub mod provider_roadmap;
pub use magnetar_core::provider_status;
pub mod qwen_model_component;
pub mod reference_cpu;
pub use magnetar_core::resolution;
pub mod runtime;
pub mod sampling;
pub mod scheduler;
pub mod session;
pub mod tensor;
pub mod tensor_parallel;
pub mod tokenizer;
pub use magnetar_core::trace_id;

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
pub use e2e_conformance::{
    E2E_EXERCISED_OPERATORS, E2E_FIXTURE_VERSION, E2E_FIXTURE_WEIGHT_DIGEST,
    E2E_FIXTURE_WEIGHT_DIGEST_VERSION, E2E_SUITE_VERSION,
};
pub use execution_graph::*;
pub use execution_stream::*;
pub use first_native_implementation_cut::*;
pub use first_native_runtime::*;
pub use generation::*;
pub use graph_builder_capability::*;
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
pub use kernel_execution_plan::*;
pub use kernel_performance_model::*;
pub use kernel_qualification::*;
pub use kernel_registry::*;
pub use kernel_selection_policy::*;
pub use kv_cache::*;
pub use memory::*;
pub use model::*;
pub use model_component::*;
pub use model_config_capability::*;
pub use model_identity::*;
pub use model_instance::*;
pub use model_loading::*;
pub use multi_device_placement::*;
pub use observability::*;
pub use operator::*;
pub use operator_scope::*;
pub use planning::*;
pub use prefix_cache::*;
pub use production_model_ingestion::*;
pub use provider::*;
pub use qwen_model_component::*;
pub use reference_cpu::*;
pub use resolution::*;
pub use runtime::*;
pub use sampling::*;
pub use scheduler::*;
pub use session::*;
pub use tensor::*;
pub use tokenizer::*;
#[cfg(test)]
mod graph_fusion_runtime_tests;
#[cfg(test)]
mod tests;
