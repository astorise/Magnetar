//! Model data of the Magnetar Runtime: manifests, architecture and trust data, and the identities of Model Instances, sessions, adapters and tokenizers.
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

pub mod model;
pub mod model_identity;

pub use model::*;
pub use model_identity::*;
