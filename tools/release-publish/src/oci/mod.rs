//! OCI manifest and local-layout construction for Component Artifacts and
//! Kernel Exchange Bundles (tasks.md section 3). Nothing in this module
//! performs a network call: `layout` writes and re-verifies a real local
//! OCI Image Layout directory so a digest mismatch is caught before any
//! registry push, per
//! `openspec/changes/implement-release-publication-automation/design.md`.

pub mod layout;
pub mod manifest;
pub mod media_types;
