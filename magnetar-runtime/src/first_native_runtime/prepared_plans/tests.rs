//! Unit tests for the parent module.
//!
//! Shared mock executors and fixtures live in `first_native_runtime/tests.rs`.

use super::super::tests::*;
use super::*;

#[test]
fn e2e_first_native_generation_requires_ready_model_instance() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_first_native_generation_requires_ready_model_instance(&fixture)
        .expect("non-ready model instance is rejected");
}

#[test]
fn e2e_missing_prepared_plan_fails_closed() {
    check_missing_prepared_plan_fails_closed().expect("missing prepared plan is rejected");
}

#[test]
fn e2e_invalidated_prepared_plan_rejects_new_work() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_invalidated_prepared_plan_rejects_new_work(&fixture)
        .expect("invalidated prepared plan is rejected");
}

#[test]
fn e2e_stale_plan_outside_policy_fails_closed() {
    let fixture = e2e_fixture().expect("fixture builds");
    check_stale_plan_outside_policy_fails_closed(&fixture)
        .expect("plan stale outside its rebuild policy is rejected");
}
