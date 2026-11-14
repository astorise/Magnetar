//! The Provider contract and Provider loading, under their historical paths.
//!
//! The contract (traits, metadata, registry, tensor value, step capture)
//! lives in `magnetar-provider-api`, below Runtime orchestration; the loader lives in
//! `provider_loading`. This module keeps every item at `provider::…`.

pub use crate::provider_loading::*;
pub use magnetar_core::provider_abi::*;
pub use magnetar_provider_api::provider_api::*;

#[cfg(test)]
mod tests;
