//! Core types of the Magnetar Runtime: capabilities, devices, bindings, compute data types, operators and Provider status.
//!
//! One layer of the Magnetar Runtime (`quality/layers.toml`). `magnetar-runtime`
//! re-exports every module of this crate at its former path; depend on
//! `magnetar-runtime` unless you need this layer alone.

pub mod artifact_signature;
pub mod binding;
pub mod capability;
pub mod compute_types;
pub mod device;
pub mod operator;
pub mod provider_abi;
pub mod provider_status;
pub mod resolution;
pub mod trace_id;

pub use binding::*;
pub use capability::*;
pub use compute_types::*;
pub use device::*;
pub use operator::*;
pub use provider_abi::*;
pub use provider_status::*;
pub use resolution::*;
pub use trace_id::*;
