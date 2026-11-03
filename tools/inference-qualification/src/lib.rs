//! Qualification tooling for
//! `openspec/changes/qualify-real-checkpoint-performance/`: the run
//! manifests, raw samples, campaign statistics and hardware capability
//! evidence a real-checkpoint campaign records (A01, A03, A04), and the
//! checkpoint-bound logit references Providers are compared against (A02).
//!
//! This crate measures and reports; it never loads models, selects
//! Providers or grants any authority. The harness that drives real
//! generation lives in `integration-tests/production-loading`
//! (`tests_real_checkpoint_qualification.rs`), which feeds this crate the
//! timings it observes at the public streaming boundary.

pub mod campaign;
pub mod hardware;
pub mod numerics;
pub mod resources;
pub mod statistics;
pub mod trace;
