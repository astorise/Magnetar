//! Executable roadmap and release-process contracts for the Magnetar
//! project.
//!
//! These modules used to live inside `magnetar-runtime` itself (`pub mod` +
//! `pub use ...::*` at its crate root), which made hundreds of items of
//! undone-work description and this repository's own release-process
//! metadata part of every embedder's compiled dependency and public API
//! surface (#62). They are executable Rust, not prose -- each module's own
//! doc comments are explicit that it defines a contract, not an
//! implementation -- so moving them to their own `publish = false` crate
//! preserves that property (their conformance tests still run in CI
//! exactly as before) without shipping any of it to a `magnetar-runtime`
//! consumer.
//!
//! Two of the seven roadmap/release-process modules this issue originally
//! listed, `model_format_roadmap` and `provider_roadmap`, stayed inside
//! `magnetar-runtime` instead: the real, external `loaders/huggingface`
//! submodule already depends on part of `model_format_roadmap` as
//! production API, not an illustrative contract, and `model_format_roadmap`
//! itself depends on `provider_roadmap`. See `model_format_roadmap`'s own
//! doc comment in `magnetar-runtime` for the detail.
//!
//! This crate is a facade in the same sense `magnetar-runtime`'s own root
//! is: flattening every module's items with `pub use` here is harmless,
//! since nothing outside this repository's own CI depends on this crate.

pub mod model_source_cache_roadmap;
pub mod release_cutover;
pub mod release_packaging;
pub mod release_security;
pub mod server_api_roadmap;

pub use model_source_cache_roadmap::*;
pub use release_cutover::*;
pub use release_packaging::*;
pub use release_security::*;
pub use server_api_roadmap::*;

#[cfg(test)]
mod tests;
