//! Component engine of the Magnetar Runtime: Component artifacts and trust, the Wasmtime and web engines, and the Model Component graph contract.
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

pub mod component;
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub mod component_wasmtime;
#[cfg(all(target_arch = "wasm32", feature = "web-component-engine"))]
pub mod component_web;
pub mod graph_builder_capability;
pub mod model_component;
pub mod model_config_capability;

pub use component::*;
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub use component_wasmtime::*;
#[cfg(all(target_arch = "wasm32", feature = "web-component-engine"))]
pub use component_web::*;
pub use graph_builder_capability::*;
pub use model_component::*;
pub use model_config_capability::*;
