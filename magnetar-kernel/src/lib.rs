//! Kernel pipeline of the Magnetar Runtime: Kernel Artifacts, ingestion, compilation, qualification, cache, registry, selection, autotuning and prepared execution plans.
//!
//! One layer of the Magnetar Runtime (`quality/layers.toml`). `magnetar-runtime`
//! re-exports every module of this crate at its former path; depend on
//! `magnetar-runtime` unless you need this layer alone.

// Lower layers, visible at this crate's root so that `crate::Name` paths
// keep resolving. Not re-exported: consumers reach them through their own
// crate or through `magnetar-runtime`.
#[allow(unused_imports)]
pub(crate) use magnetar_core::*;

// Lower layers, visible at this crate's root so that `crate::Name` paths
// keep resolving. Not re-exported: consumers reach them through their own
// crate or through `magnetar-runtime`.
#[allow(unused_imports)]
pub(crate) use magnetar_graph::*;

// Lower layers, visible at this crate's root so that `crate::Name` paths
// keep resolving. Not re-exported: consumers reach them through their own
// crate or through `magnetar-runtime`.
#[allow(unused_imports)]
pub(crate) use magnetar_model::*;

pub mod execution_stream;
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
pub mod kernel_performance_model;
pub mod kernel_qualification;
pub mod kernel_registry;
pub mod kernel_selection_policy;

pub use execution_stream::*;
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
