//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::tensor::TensorAliasingKind;

#[test]
fn kernel_result_tracks_aliasing_updates_alongside_readiness_and_residency() {
    let result = KernelResult::success(KernelInvocationId::new("aliasing-update"))
        .with_aliasing_update("output", TensorAliasingKind::InputOutputAlias);
    assert_eq!(
        result.updated_aliasing.get("output"),
        Some(&TensorAliasingKind::InputOutputAlias)
    );
}
