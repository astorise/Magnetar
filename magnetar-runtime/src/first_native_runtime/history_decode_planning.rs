//! Safe plan replacement for one-token, copy-based history graphs.

use super::*;
use crate::kernel_execution_plan::{
    PreparedExecutionPlanObservation, PreparedExecutionPlanObservationKind,
    PreparedExecutionPlanState,
};

type HistorySchema = BTreeMap<String, (String, Vec<u64>)>;

fn history_schema(
    graph: &ExecutionGraph,
    position: u64,
) -> Result<HistorySchema, InferenceApiError> {
    let failure = |reason: &str| InferenceApiError::GraphPlanningFailed {
        reason: format!("invalid decode history graph: {reason}"),
    };
    if graph.phase != ExecutionGraphPhase::Decode {
        return Err(failure("expected decode phase"));
    }
    graph
        .validate(&initial_operator_catalog())
        .map_err(|_| failure("graph validation failed"))?;
    let bindings = graph
        .kv_history_bindings()
        .map_err(|_| failure("history bindings are invalid"))?;
    if bindings.is_empty() {
        return Err(failure("no copy-based history inputs"));
    }
    let output_rows = position
        .checked_add(1)
        .ok_or_else(|| failure("position overflow"))?;
    let mut schema = BTreeMap::new();
    for binding in bindings {
        let input = &graph.edges[&binding.input_edge].descriptor.shape.dimensions;
        let output = &graph.edges[&binding.output_edge]
            .descriptor
            .shape
            .dimensions;
        if input.first() != Some(&position.max(1))
            || output.first() != Some(&output_rows)
            || input.get(1..) != output.get(1..)
            || schema
                .insert(
                    binding.cache_id,
                    (binding.compatibility_key, input[1..].to_vec()),
                )
                .is_some()
        {
            return Err(failure("history length or identity does not match"));
        }
    }
    Ok(schema)
}

/// Replaces a ready decode plan with one prepared for the next history length.
///
/// Both graphs must be validated one-token decode graphs with identical
/// copy-based history and recurrent-state schemas. `next_position` is the
/// number of committed tokens for the new graph, exactly one more than for
/// the old graph. Existing plan guards, instance revision, Kernel bindings,
/// Provider and active leases are checked before a replacement is published.
/// Only canonical first-native plans are accepted; custom plan policies are
/// rejected rather than discarded. A failure leaves the caller's plan
/// untouched. The caller must separately validate request ownership and the
/// actual committed session history.
pub fn advance_first_native_history_decode_plan(
    runtime: &Runtime,
    instance: &ModelInstanceId,
    previous_graph: &ExecutionGraph,
    next_graph: &ExecutionGraph,
    next_position: u64,
    plan: &mut PreparedExecutionPlan,
) -> Result<(), InferenceApiError> {
    let fail = |reason: String| InferenceApiError::GraphPlanningFailed { reason };
    let previous_position = next_position
        .checked_sub(1)
        .ok_or_else(|| fail("the next history position must be positive".into()))?;
    if history_schema(previous_graph, previous_position)?
        != history_schema(next_graph, next_position)?
    {
        return Err(fail("decode history schema changed between steps".into()));
    }
    let previous_states = previous_graph
        .recurrent_state_bindings()
        .map_err(|error| fail(error.to_string()))?;
    let next_states = next_graph
        .recurrent_state_bindings()
        .map_err(|error| fail(error.to_string()))?;
    if !previous_states
        .iter()
        .map(|binding| &binding.layout)
        .eq(next_states.iter().map(|binding| &binding.layout))
    {
        return Err(fail(
            "decode recurrent state schema changed between steps".into(),
        ));
    }
    if plan.state != PreparedExecutionPlanState::Ready || plan.active_leases() != 0 {
        return Err(fail(
            "the previous decode plan is not ready or still leased".into(),
        ));
    }
    let status = require_ready_first_native_instance(runtime, instance)?;
    let revision = status.status().mutation_version;
    let provider = plan
        .scope
        .provider
        .as_ref()
        .ok_or_else(|| fail("the previous plan has no Provider".into()))?;
    let canonical = prepare_first_native_plan_for_graph(
        runtime,
        previous_graph,
        instance,
        revision,
        1,
        plan.generation,
        provider,
    )
    .map_err(|error| fail(error.to_string()))?;
    if plan.id != canonical.id
        || plan.graph_fingerprint != canonical.graph_fingerprint
        || plan.scope != canonical.scope
        || plan.fingerprint != canonical.fingerprint
        || plan.node_bindings != canonical.node_bindings
        || plan.segments != canonical.segments
        || plan.resource_plan != canonical.resource_plan
        || plan.memory_requirements != canonical.memory_requirements
        || plan.kv_requirements != canonical.kv_requirements
        || plan.guards != canonical.guards
    {
        return Err(fail(
            "the previous decode plan is incompatible with its graph or current bindings".into(),
        ));
    }
    let context = first_native_plan_context(PreparedExecutionPhase::Decode, 1);
    let mut verified = plan.clone();
    verified
        .execute_ready_path(&context)
        .map_err(|error| fail(error.to_string()))?;
    let executor = PreparedExecutionPlanExecutor::new();
    for node in previous_graph.nodes.keys() {
        executor
            .prepare_node_execution(
                previous_graph,
                &mut verified,
                runtime.kernel_registry(),
                &context,
                node,
            )
            .map_err(|error| fail(error.to_string()))?;
    }
    let generation = plan
        .generation
        .value()
        .checked_add(1)
        .ok_or_else(|| fail("decode plan generation overflow".into()))?;
    let mut replacement = prepare_first_native_plan_for_graph(
        runtime,
        next_graph,
        instance,
        revision,
        1,
        PreparedExecutionPlanGeneration::new(generation),
        provider,
    )
    .map_err(|error| fail(error.to_string()))?;
    replacement
        .execute_ready_path(&context)
        .map_err(|error| fail(error.to_string()))?;
    verified
        .transition_to(PreparedExecutionPlanState::Retiring)
        .and_then(|_| verified.transition_to(PreparedExecutionPlanState::Retired))
        .map_err(|error| fail(error.to_string()))?;
    for kind in [
        PreparedExecutionPlanObservationKind::PlanReplacementReady,
        PreparedExecutionPlanObservationKind::PlanGenerationPromoted,
    ] {
        replacement
            .observations
            .push(PreparedExecutionPlanObservation::new(
                kind,
                replacement.id.clone(),
                replacement.generation,
            ));
    }
    *plan = replacement;
    Ok(())
}

#[cfg(test)]
mod tests;
