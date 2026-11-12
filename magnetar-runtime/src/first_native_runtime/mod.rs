//! First-native runtime execution and conformance support.
//!
//! Assembles a minimal, deterministic Qwen-like fixture model and drives it
//! through the real Runtime Inference API surface -- model resolution,
//! Model Loading, Model Instance creation, session creation, tokenization,
//! generation, streaming, and cleanup -- using genuine Reference CPU numeric
//! kernels for the forward pass (not canned output), so the success path is
//! a real, if tiny, end-to-end inference run. The `e2e_conformance` module is
//! now a compatibility wrapper around this runtime-owned implementation.

use crate::ProductionArtifactPayloadSource;
use crate::ProviderExecutionResult;
use crate::affinity::*;
use crate::batching::*;
use crate::cli_boundary::*;
use crate::component::*;
use crate::compute;
use crate::compute::*;
use crate::conformance::first_native_model_execution_profile;
use crate::device::DeviceId;
use crate::execution_graph::*;
use crate::generation::*;
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
use crate::graph_builder_capability::*;
use crate::inference_api::*;
use crate::kernel::*;
use crate::kernel_artifact::{
    CompiledKernelArtifactId, PreparedKernel, PreparedKernelGeneration, PreparedKernelIdAllocator,
};
use crate::kernel_dispatch::*;
use crate::kernel_execution_plan::{
    ExecutionGraphSemanticFingerprint, FingerprintedGraph, PlanGuard, PlanGuardContext,
    PlanMemoryRequirements, PlanNodeBinding, PreparedExecutionPhase, PreparedExecutionPlan,
    PreparedExecutionPlanError, PreparedExecutionPlanExecutor, PreparedExecutionPlanGeneration,
    PreparedExecutionPlanId, PreparedExecutionPlanScope, PreparedPlanNodeExecution,
    ResourceBindingPlan, semantic_graph_fingerprint,
};
use crate::kernel_optimization_orchestration::run_kernel_optimization_orchestration_conformance;
use crate::kernel_registry::*;
use crate::kv_cache::*;
use crate::memory::*;
use crate::model::*;
use crate::model_component::*;
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
use crate::model_config_capability::*;
use crate::model_instance::*;
use crate::model_loading::*;
use crate::operator::*;
use crate::operator_scope::*;
use crate::phase_attribution::StepPhase;
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
use crate::phase_attribution::{PhaseAttributionReport, StepKind};
use crate::prefix_cache::*;
use crate::provider::*;
use crate::qwen_model_component;
use crate::qwen_model_component::*;
use crate::reference_cpu::*;
use crate::runtime::*;
use crate::sampling::*;
use crate::session::*;
use crate::tensor::*;
use crate::tokenizer::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest as ShaDigest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};
#[cfg(test)]
mod tests;

mod chat_session;
mod component_graphs;
mod conformance_report;
mod dispatch;
mod engine;
mod executor;
mod fixture;
mod graph_execution;
mod kv_state;
mod prepared_plans;
mod production;
mod segments;
mod weights;

pub use chat_session::*;
pub use component_graphs::*;
pub use conformance_report::*;
pub(crate) use dispatch::*;
pub use engine::*;
pub use executor::*;
pub use fixture::*;
pub use graph_execution::*;
use kv_state::*;
pub use prepared_plans::*;
pub use production::*;
pub use segments::*;
pub use weights::*;

// `e2e_conformance_report_json` (an always-on, production `pub fn`) splices
// in two extra checks only under `#[cfg(test)]` -- both check_* functions
// now live in `tests` (#54/#63's test-code-out-of-production-files move),
// so this re-export is what lets that still-production call site reach
// them.
#[cfg(test)]
pub(crate) use tests::{check_eos_token_stops_generation, check_graph_production_and_execution};
