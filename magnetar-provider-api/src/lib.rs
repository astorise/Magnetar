//! Provider API of the Magnetar Runtime: the contract Providers implement and call (Provider and ProviderExecutionApi, execution plans and requests, compute advertisements, tensors, packed weights, quantization tables and shared operator semantics).
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
pub(crate) use magnetar_kernel::*;

pub mod compute_support;
pub mod execution;
pub mod layer_placement;
pub mod operator_semantics;
pub mod packed_weight;
pub mod provider_api;
pub mod quantization;
pub mod runtime_event;
pub mod tensor;
pub mod tensor_parallel;

pub use compute_support::*;
pub use execution::*;
pub use layer_placement::*;
pub use operator_semantics::*;
pub use packed_weight::*;
pub use provider_api::*;
pub use quantization::*;
pub use runtime_event::*;
pub use tensor::*;
pub use tensor_parallel::*;
