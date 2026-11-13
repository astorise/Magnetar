//! Execution graph, graph fusion and the Memory Manager of the Magnetar Runtime.
//!
//! One layer of the Magnetar Runtime (`quality/layers.toml`). `magnetar-runtime`
//! re-exports every module of this crate at its former path; depend on
//! `magnetar-runtime` unless you need this layer alone.

// Lower layers, visible at this crate's root so that `crate::Name` paths
// keep resolving. Not re-exported: consumers reach them through their own
// crate or through `magnetar-runtime`.
#[allow(unused_imports)]
pub(crate) use magnetar_core::*;

pub mod execution_graph;
pub mod graph_fusion;
pub mod memory;

pub use execution_graph::*;
pub use memory::*;
