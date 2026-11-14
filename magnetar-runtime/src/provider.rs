//! The Provider contract and Provider loading, under their historical paths.
//!
//! The contract (traits, metadata, registry, tensor value, step capture)
//! lives in `provider_api`, below Runtime orchestration; the loader lives in
//! `provider_loading`. This module keeps every item at `provider::…`.

pub use crate::provider_api::*;
pub use crate::provider_loading::*;
