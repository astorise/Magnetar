//! Execution of first-native prefill and decode graphs, node by node.

use super::*;

// ---------------------------------------------------------------------
// Generic graph executor
//
// Walks an `ExecutionGraph`'s dependency order and dispatches each node's
// Operator through the published `PreparedExecutionPlan`, binding graph
// inputs, intermediates, outputs, and (for KV-cache-tagged edges) historical
// Runtime KV state -- so the graph, not a hand-written Rust sequence, is the
// authoritative recipe for which Operators run, in what order, over which
// tensors. This replaces the previous Qwen-specific hard-coded prefill/
// decode call sequences (still available under `#[cfg(test)]` as an
// independent oracle for the tests that compare against them).
// ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum KvRole {
    K,
    V,
}

/// Parses a KV cache identity (`{namespace}.layer{N}.{k|v}`, produced by
/// `GraphBuilderCapability::kv_resource` -- see that method's documentation)
/// back into a layer index and K/V role. Generic over `namespace` rather
/// than hardcoding one: the Runtime-side `graph-builder` Capability accepts
/// whatever `kv_namespace` the caller supplies in `SessionContext` (today
/// always `"qwen"`, since that is the only Model Component that exists),
/// and this parser only needs to know that a namespace precedes `.layer`,
/// not which model family chose it.
pub(super) fn parse_kv_cache_id(cache_id: &str) -> Result<(usize, KvRole), InferenceApiError> {
    let malformed = || InferenceApiError::GraphPlanningFailed {
        reason: format!("malformed graph KV cache id '{cache_id}'"),
    };
    let (_namespace, rest) = cache_id.split_once(".layer").ok_or_else(malformed)?;
    let (layer, role) = rest.split_once('.').ok_or_else(malformed)?;
    let layer = layer.parse::<usize>().map_err(|_| malformed())?;
    let role = match role {
        "k" => KvRole::K,
        "v" => KvRole::V,
        _ => return Err(malformed()),
    };
    Ok((layer, role))
}

/// Maps an execution-graph weight edge id to the canonical Model Loading
/// tensor name `fixture.weights` is keyed by (see `first_native_expected_tensor_names`).
/// A weight edge id is always `weight.{logical_name}` (see
/// `GraphBuilderCapability::weight_edge`), and the Model Component -- not
/// the Runtime -- is the one that chooses `logical_name`; the Qwen
/// Component supplies it already equal to the canonical Model Artifact
/// tensor name (e.g. `layers.0.self_attn.q_proj`, `token_embedding`), so
/// this is a plain prefix strip, not a per-model suffix-mapping table.
/// Returns `None` for a non-weight edge id.
pub(super) fn weight_tensor_name_from_edge(edge_id: &str) -> Option<String> {
    edge_id.strip_prefix("weight.").map(str::to_string)
}

/// Resolves a graph weight edge to its tensor value by looking up its
/// canonical name in `weight_bindings` (the active Model Instance's
/// Runtime-owned weight resource bindings -- see `bind_qwen_fixture_weights`)
/// and reading the bound resource from the registered Provider's storage,
/// rather than a private Rust-side copy of the fixture's tensor bytes (task
/// 6.4). Applies the tied-embeddings `weight.lm_head` -> transposed
/// `token_embedding` substitution `qwen_lm_head_weight_edge` declares via
/// `TensorAliasing::MayAlias` (see that function's doc comment).
///
/// Returns a `NodeValue`, `Opaque` mapping to `Resident` instead of forcing
/// `.into_host()` unconditionally (`make-first-native-cuda-hot-path-device-
/// resident`'s Decision 3/audit P0-1) -- a Device-resident weight now flows
/// to its consuming Kernel without a forced Host download. `descriptor` is
/// the graph's own canonical shape source for this edge (`TensorEdge.
/// descriptor`, never downloaded from the Provider and never recomputed
/// from Qwen config) -- the source `NodeValue::Resident`'s shape needs,
/// since `TensorValue::Opaque` itself carries none.
pub(super) fn resolve_first_native_weight_edge(
    provider: &Arc<dyn ProviderExecutionApi>,
    weight_bindings: &BTreeMap<String, TensorResourceId>,
    edge_id: &str,
    descriptor: &TensorDescriptor,
) -> Result<NodeValue, InferenceApiError> {
    let name = weight_tensor_name_from_edge(edge_id).ok_or_else(|| {
        InferenceApiError::GraphPlanningFailed {
            reason: format!("graph edge '{edge_id}' is neither a bound input nor a known weight"),
        }
    })?;
    // Tied embeddings no longer need special-casing here (design.md's
    // Decision 5): `bind_qwen_fixture_weights` stages the transposed
    // `token_embedding` once, at Model Load, under the name `lm_head`
    // itself -- this function resolves it exactly like any other weight,
    // with no per-dispatch Rust-side transform and no redirect.
    let resource_id =
        weight_bindings
            .get(&name)
            .ok_or_else(|| InferenceApiError::ModelLoadingFailed {
                reason: format!("active Model Instance has no weight resource bound for '{name}'"),
            })?;
    let value = provider.read_tensor_value(resource_id).ok_or_else(|| {
        InferenceApiError::ModelLoadingFailed {
            reason: format!(
                "weight resource '{resource_id}' bound for '{name}' has no materialized data"
            ),
        }
    })?;
    match value {
        TensorValue::Host(tensor) => Ok(NodeValue::Host(tensor)),
        TensorValue::Opaque => Ok(NodeValue::Resident {
            id: resource_id.clone(),
            shape: descriptor.shape.dimensions.clone(),
        }),
    }
}

/// Derives a Kahn's-algorithm topological order for `graph` from
/// `node.inputs`/`node.outputs` directly, rather than relying on
/// `TensorEdge::producer`/`::consumers` (which Qwen's own graph builder does
/// not populate).
/// [`first_native_graph_execution_order`] for sibling modules' tests.
#[cfg(test)]
pub(crate) fn first_native_graph_execution_order_for_tests(
    graph: &ExecutionGraph,
) -> Result<Vec<ExecutionNodeId>, InferenceApiError> {
    first_native_graph_execution_order(graph)
}

pub(super) fn first_native_graph_execution_order(
    graph: &ExecutionGraph,
) -> Result<Vec<ExecutionNodeId>, InferenceApiError> {
    let mut edge_producer: BTreeMap<&TensorEdgeId, &ExecutionNodeId> = BTreeMap::new();
    for (node_id, node) in &graph.nodes {
        for output in &node.outputs {
            edge_producer.insert(output, node_id);
        }
    }
    // Kahn's algorithm, always taking the smallest ready node id: the same
    // order as repeatedly picking the first remaining node whose producers
    // have all run, in O(edges log nodes) instead of O(nodes^2).
    let mut pending_inputs: BTreeMap<&ExecutionNodeId, usize> = BTreeMap::new();
    let mut consumers: BTreeMap<&ExecutionNodeId, Vec<&ExecutionNodeId>> = BTreeMap::new();
    for (node_id, node) in &graph.nodes {
        let mut pending = 0;
        for edge_id in &node.inputs {
            if let Some(producer) = edge_producer.get(edge_id) {
                pending += 1;
                consumers.entry(*producer).or_default().push(node_id);
            }
        }
        pending_inputs.insert(node_id, pending);
    }
    let mut ready: BTreeSet<&ExecutionNodeId> = pending_inputs
        .iter()
        .filter(|(_, pending)| **pending == 0)
        .map(|(node_id, _)| *node_id)
        .collect();
    let mut order = Vec::with_capacity(graph.nodes.len());
    while let Some(node_id) = ready.pop_first() {
        order.push(node_id.clone());
        for consumer in consumers.get(node_id).into_iter().flatten() {
            let pending = pending_inputs
                .get_mut(consumer)
                .expect("every consumer is a graph node");
            *pending -= 1;
            if *pending == 0 {
                ready.insert(consumer);
            }
        }
    }
    if order.len() != graph.nodes.len() {
        return Err(InferenceApiError::GraphPlanningFailed {
            reason: "first-native graph contains a cycle or unresolved producer".into(),
        });
    }
    Ok(order)
}

pub(super) fn node_attribute<'a>(
    node: &'a ExecutionNode,
    name: &str,
) -> Result<&'a OperatorAttributeValue, InferenceApiError> {
    node.attributes
        .get(name)
        .ok_or_else(|| InferenceApiError::GraphPlanningFailed {
            reason: format!("graph node '{}' is missing attribute '{name}'", node.id),
        })
}

pub(super) fn node_attribute_f64(
    node: &ExecutionNode,
    name: &str,
) -> Result<f64, InferenceApiError> {
    match node_attribute(node, name)? {
        OperatorAttributeValue::Float(value) => Ok(*value),
        _ => Err(InferenceApiError::GraphPlanningFailed {
            reason: format!("graph node '{}' attribute '{name}' is not a float", node.id),
        }),
    }
}

pub(super) fn node_attribute_u64(
    node: &ExecutionNode,
    name: &str,
) -> Result<u64, InferenceApiError> {
    match node_attribute(node, name)? {
        OperatorAttributeValue::Integer(value) if *value >= 0 => Ok(*value as u64),
        _ => Err(InferenceApiError::GraphPlanningFailed {
            reason: format!(
                "graph node '{}' attribute '{name}' is not a non-negative integer",
                node.id
            ),
        }),
    }
}

// Task 11.5/12.6: this whole checksum-hash family (through
// `qwen_component_graph_semantics_for_prompt`/`QwenComponentGraphSemantics`/
// `build_first_native_graphs_from_component_output`) is test-oracle only --
// the strict, default production path gets its graph directly from the
// real Component instead of cross-checking a Rust-synthesized one against a
// declared checksum, and no production build has a second, unattested
// graph source to fall back to.

/// Dispatches one graph node's Operator through the published prepared plan,
/// reading structural parameters (epsilon, RoPE base/dimension/mode,
/// attention head geometry) from the node's own graph attributes rather than
/// fixture configuration, so the graph is the authoritative recipe. The one
/// deliberate exception is RoPE's absolute position: the same published
/// decode plan/graph is reused across every decode step (so its baked-in
/// `position_offset` attribute reflects only the first step), so the caller
/// supplies the true per-step position via `absolute_position_override`
/// instead.
pub(super) fn dispatch_first_native_graph_node(
    ctx: &mut FirstNativeDispatchContext<'_>,
    fixture: &E2eFixture,
    node: &ExecutionNode,
    mut inputs: Vec<NodeValue>,
    absolute_position_override: Option<u64>,
    kv_cache_id: &KvCacheId,
) -> Result<(KernelDispatchResult, Vec<NodeValue>), InferenceApiError> {
    let node_id = node.id.as_str();
    let architecture = &fixture.config.architecture;
    // Resolved once, before dispatch: this node's output edge's own final
    // resource identity, and the placement/owner it should be pre-admitted
    // under (`unify-provider-output-admission-and-residency`'s Decision 3).
    // Every operator below except a "rope" whose output carries KV-cache
    // semantics (see that arm) targets this directly instead of
    // letting its own dispatch function synthesize a Kernel-internal id
    // that would later need a separate download-then-reupload to reach the
    // edge.
    let output_target = {
        let output_edge =
            node.outputs
                .first()
                .ok_or_else(|| InferenceApiError::GraphPlanningFailed {
                    reason: format!("graph node '{node_id}' has no output edge"),
                })?;
        let output_resource_id = TensorResourceId::new(format!("edge.{output_edge}"));
        let placement = resolved_output_placement(ctx.prepared_plan.as_deref(), &node.id);
        let owner = MemoryAllocationOwner::Session(kv_cache_id.to_string());
        (output_resource_id, placement, owner)
    };
    match node.operator.name() {
        "embedding" => {
            if inputs.len() != 2 {
                return Err(InferenceApiError::GraphPlanningFailed {
                    reason: format!("graph node '{node_id}' expects 2 embedding inputs"),
                });
            }
            // The graph declares [ids, table]; the embedding kernel's
            // invocation convention (matching every other production
            // dispatch of this operator) is (table, ids).
            let ids = inputs.remove(0);
            let table = inputs.remove(0);
            let sequence_length = ids.shape().first().copied().unwrap_or(0);
            let output_descriptor = TensorDescriptor::new(
                ShapeDescriptor::new([sequence_length, architecture.hidden_size]),
                DTypeDescriptor::portable(ComputeDType::Float32),
                LayoutDescriptor::Contiguous,
            );
            dispatch_reference_cpu_operator_pre_admitted(
                ctx,
                node_id,
                node.operator.clone(),
                vec![
                    node_input_resource(node_id, "table", table),
                    node_input_resource(node_id, "ids", ids),
                ],
                output_descriptor,
                Some(output_target),
                BTreeMap::new(),
            )
            .map(|(dispatch, value)| (dispatch, vec![value]))
        }
        "rmsnorm" => {
            if inputs.len() != 2 {
                return Err(InferenceApiError::GraphPlanningFailed {
                    reason: format!("graph node '{node_id}' expects 2 rmsnorm inputs"),
                });
            }
            // Both inputs are eligible for Resident passthrough now
            // (`make-first-native-cuda-hot-path-device-resident` reverses
            // `enable-device-resident-kernel-chaining` design.md's earlier
            // Non-Goal here, once verified unnecessary: both real Kernels
            // already accept a `[cols]` weight and broadcast internally,
            // so there is no Rust-side transform left that would force
            // materialization). Its output still targets the edge's final
            // identity directly, same as every other operator here.
            let input = inputs.remove(0);
            let weight = inputs.remove(0);
            let epsilon = node_attribute_f64(node, "epsilon")? as f32;
            dispatch_qwen_rmsnorm(ctx, node_id, input, weight, epsilon, Some(output_target))
                .map(|(dispatch, value)| (dispatch, vec![value]))
        }
        "matmul" => {
            if inputs.len() != 2 {
                return Err(InferenceApiError::GraphPlanningFailed {
                    reason: format!("graph node '{node_id}' expects 2 matmul inputs"),
                });
            }
            let a = inputs.remove(0);
            let b = inputs.remove(0);
            dispatch_qwen_matmul(ctx, node_id, a, b, Some(output_target))
                .map(|(dispatch, value)| (dispatch, vec![value]))
        }
        "matmul-bias" | "matmul-residual" | "gated-matmul" => {
            if inputs.len() != 3 {
                return Err(InferenceApiError::GraphPlanningFailed {
                    reason: format!(
                        "graph node '{node_id}' expects 3 {} inputs",
                        node.operator.name()
                    ),
                });
            }
            let a = inputs.remove(0);
            let b = inputs.remove(0);
            let extra = inputs.remove(0);
            dispatch_qwen_matmul_fused(
                ctx,
                node_id,
                node.operator.clone(),
                a,
                b,
                extra,
                Some(output_target),
            )
            .map(|(dispatch, value)| (dispatch, vec![value]))
        }
        "silu-mul" => {
            if inputs.len() != 2 {
                return Err(InferenceApiError::GraphPlanningFailed {
                    reason: format!("graph node '{node_id}' expects 2 silu-mul inputs"),
                });
            }
            let gate = inputs.remove(0);
            let up = inputs.remove(0);
            dispatch_qwen_binary_same_shape(
                ctx,
                node_id,
                "silu-mul",
                OperatorFamily::Activation,
                gate,
                up,
                Some(output_target),
            )
            .map(|(dispatch, value)| (dispatch, vec![value]))
        }
        "rope" => {
            if inputs.len() != 1 {
                return Err(InferenceApiError::GraphPlanningFailed {
                    reason: format!("graph node '{node_id}' expects 1 rope input"),
                });
            }
            // A single Kernel invocation now rotates every head block at
            // once (`make-first-native-cuda-hot-path-device-resident`),
            // so this input is `NodeValue`-eligible for Resident
            // passthrough like every other operator here -- no more
            // forced materialization, no more per-head Rust slicing.
            let input = inputs.remove(0);
            let head_dimension = node_attribute_u64(node, "dimension")?;
            // `head_count` is explicit graph data set by the Qwen Model
            // Component (Q = attention_head_count, K = kv_head_count --
            // correct for GQA/MQA); absent reproduces the portable `rope`
            // Operator's own single-block default (`1`), exactly like
            // every other Provider's "rope" dispatch arm
            // (`reference_cpu.rs`, `providers/cpu`, `providers/cuda`).
            // Runtime never infers it from a node identifier or naming
            // convention (`operator` spec's RoPE Operator requirement).
            //
            // audit-complet-cuda-hot-path-2026-09-08 P1-2 / Correctif D:
            // the id-suffix fallback this used to have
            // (`qwen_rope_head_count`) was removed only after fixing its
            // real prerequisite, not by assuming the audit's premise held
            // -- `components/qwen`'s checked-in, precompiled
            // `qwen-real.component.wasm` test fixture (the real Component
            // `build_first_native_graphs_from_real_qwen_component` invokes
            // for every default `cargo test` run) did not emit
            // `head_count` on its `rope_q`/`rope_k` nodes even though the
            // Rust-side `qwen_model_component.rs` did, so a first attempt
            // at removing this fallback (proven, not assumed) broke
            // `e2e_graph_executor_matches_full_sequence_oracle` with a
            // wrong-but-plausible decode-logits divergence -- exactly the
            // silent-wrong-answer failure mode this design exists to
            // prevent. Fixed at the source instead: `components/qwen`'s
            // `rope_attrs` now sets `head_count` (Q = `ATTENTION_HEAD_COUNT`,
            // K = `KV_HEAD_COUNT`) and the fixture `.wasm` was regenerated
            // from that fix, confirmed green against the full suite before
            // this fallback was removed.
            let head_count = match node.attributes.get("head_count") {
                Some(OperatorAttributeValue::Integer(value)) if *value >= 0 => *value as u64,
                Some(_) => {
                    return Err(InferenceApiError::GraphPlanningFailed {
                        reason: format!(
                            "graph node '{node_id}' attribute 'head_count' is not a non-negative integer"
                        ),
                    });
                }
                None => 1,
            };
            let base = node_attribute_f64(node, "base")?;
            // `fix-rope-rotation-pairing`: the Model Component declares the
            // pairing on the node; absent keeps the interleaved default.
            // Forwarded to the Kernel as-is, never inferred.
            let pairing = match node.attributes.get("rotation_pairing") {
                None => RopePairing::Interleaved,
                Some(OperatorAttributeValue::String(value)) => RopePairing::from_attribute(value)
                    .ok_or_else(|| InferenceApiError::GraphPlanningFailed {
                        reason: format!(
                            "graph node '{node_id}' attribute 'rotation_pairing' value '{value}' is not supported"
                        ),
                    })?,
                Some(_) => {
                    return Err(InferenceApiError::GraphPlanningFailed {
                        reason: format!(
                            "graph node '{node_id}' attribute 'rotation_pairing' is not a string"
                        ),
                    });
                }
            };
            let position_offset = absolute_position_override
                .map(Ok)
                .unwrap_or_else(|| node_attribute_u64(node, "position_offset"))?;
            // Per-frequency scaling is model-family
            // semantics: the Model Component declares it on the node and the
            // Runtime forwards it unchanged, never deriving it itself.
            let frequency_scaling =
                crate::operator::RopeBandedFrequencyScaling::from_attributes(&node.attributes)
                    .map_err(|reason| InferenceApiError::GraphPlanningFailed {
                        reason: format!("graph node '{node_id}': {reason}"),
                    })?;
            let rope_config = FirstNativeRopeConfig {
                base,
                scale: fixture.config.rope.scale,
                dimension: head_dimension,
                position_mode: fixture.config.rope.position_mode,
                dynamic_scaling_supported: fixture.config.rope.dynamic_scaling_supported,
                frequency_scaling: fixture.config.rope.frequency_scaling,
            };
            // `output_target` is not passed when RoPE's output edge carries
            // KV-cache semantics (`rope_k`): `execute_first_native_graph_nodes`
            // appends that output to the history itself, and pre-admitting
            // it here too caused a genuine regression (double-admission
            // racing the KV-cache pending-write path) caught by the full
            // oracle comparison suite. Every other RoPE output (`rope_q`)
            // is written straight into its edge like any other operator's,
            // instead of being downloaded to the host and uploaded back
            // after the dispatch (`attribute-redacted-inference-costs`: 24
            // synchronizing round trips per CUDA decode step).
            let output_has_kv_semantics = node.outputs.first().is_none_or(|edge_id| {
                ctx.graph.as_ref().is_none_or(|graph| {
                    graph
                        .graph()
                        .edges
                        .get(edge_id)
                        .is_none_or(|edge| edge.kv_cache.is_some())
                })
            });
            dispatch_qwen_rope(
                ctx,
                node_id,
                input,
                head_count,
                &rope_config,
                pairing,
                frequency_scaling.as_ref(),
                position_offset,
                (!output_has_kv_semantics).then_some(output_target),
            )
            .map(|(dispatch, value)| (dispatch, vec![value]))
        }
        "attention" => {
            if inputs.len() != 3 {
                return Err(InferenceApiError::GraphPlanningFailed {
                    reason: format!("graph node '{node_id}' expects 3 attention inputs"),
                });
            }
            let q = inputs.remove(0);
            let k = inputs.remove(0);
            let v = inputs.remove(0);
            let key_count = node
                .inputs
                .get(1)
                .and_then(|edge| ctx.bounded_kv_edges.get(edge))
                .copied();
            dispatch_qwen_attention(
                ctx,
                node_id,
                q,
                k,
                v,
                architecture,
                Some(output_target),
                key_count,
            )
            .map(|(dispatch, value)| (dispatch, vec![value]))
        }
        "moe-experts" => {
            if inputs.len() != 5 {
                return Err(InferenceApiError::GraphPlanningFailed {
                    reason: format!("graph node '{node_id}' expects 5 moe-experts inputs"),
                });
            }
            let (rows, cols) = inputs[0].rows_cols()?;
            let output_descriptor = f32_tensor_descriptor_from_shape(&[rows, cols]);
            let resources = ["h", "logits", "gate", "up", "down"]
                .into_iter()
                .zip(inputs)
                .map(|(role, value)| node_input_resource(node_id, role, value))
                .collect();
            dispatch_reference_cpu_operator_pre_admitted(
                ctx,
                node_id,
                dispatch_operator_id("moe-experts", OperatorFamily::LinearAlgebra),
                resources,
                output_descriptor,
                Some(output_target),
                node.attributes.clone(),
            )
            .map(|(dispatch, value)| (dispatch, vec![value]))
        }
        "row-scale" => {
            if inputs.len() != 2 {
                return Err(InferenceApiError::GraphPlanningFailed {
                    reason: format!("graph node '{node_id}' expects 2 row-scale inputs"),
                });
            }
            let x = inputs.remove(0);
            let scale = inputs.remove(0);
            dispatch_qwen_binary_same_shape(
                ctx,
                node_id,
                "row-scale",
                OperatorFamily::Tensor,
                x,
                scale,
                Some(output_target),
            )
            .map(|(dispatch, value)| (dispatch, vec![value]))
        }
        "activation" => {
            if inputs.len() != 1 {
                return Err(InferenceApiError::GraphPlanningFailed {
                    reason: format!("graph node '{node_id}' expects 1 activation input"),
                });
            }
            let input = inputs.remove(0);
            dispatch_qwen_unary(
                ctx,
                node_id,
                "activation",
                OperatorFamily::Activation,
                input,
                node.attributes.clone(),
                Some(output_target),
            )
            .map(|(dispatch, value)| (dispatch, vec![value]))
        }
        "silu" => {
            if inputs.len() != 1 {
                return Err(InferenceApiError::GraphPlanningFailed {
                    reason: format!("graph node '{node_id}' expects 1 silu input"),
                });
            }
            let input = inputs.remove(0);
            dispatch_qwen_unary(
                ctx,
                node_id,
                "silu",
                OperatorFamily::Activation,
                input,
                BTreeMap::new(),
                Some(output_target),
            )
            .map(|(dispatch, value)| (dispatch, vec![value]))
        }
        "add" => {
            if inputs.len() != 2 {
                return Err(InferenceApiError::GraphPlanningFailed {
                    reason: format!("graph node '{node_id}' expects 2 add inputs"),
                });
            }
            let a = inputs.remove(0);
            let b = inputs.remove(0);
            // `dispatch_qwen_binary_same_shape` derives each input's
            // resource descriptor from its own real shape (`node_input_
            // resource`) and the output descriptor from `a`'s shape alone
            // -- exactly `ShapeRule::RowBroadcastAdd`'s contract (task
            // 10.5's sibling QKV bias support: `b` is a real `[dim]`
            // bias, broadcast by the Kernel itself, e.g. `providers/cpu::
            // add`). Its name predates this second use; nothing in it
            // assumes `a` and `b` share a shape.
            dispatch_qwen_binary_same_shape(
                ctx,
                node_id,
                "add",
                OperatorFamily::Tensor,
                a,
                b,
                Some(output_target),
            )
            .map(|(dispatch, value)| (dispatch, vec![value]))
        }
        "mul" => {
            if inputs.len() != 2 {
                return Err(InferenceApiError::GraphPlanningFailed {
                    reason: format!("graph node '{node_id}' expects 2 mul inputs"),
                });
            }
            let a = inputs.remove(0);
            let b = inputs.remove(0);
            dispatch_qwen_binary_same_shape(
                ctx,
                node_id,
                "mul",
                OperatorFamily::Tensor,
                a,
                b,
                Some(output_target),
            )
            .map(|(dispatch, value)| (dispatch, vec![value]))
        }
        "residual-add" => {
            if inputs.len() != 2 {
                return Err(InferenceApiError::GraphPlanningFailed {
                    reason: format!("graph node '{node_id}' expects 2 residual-add inputs"),
                });
            }
            let a = inputs.remove(0);
            let b = inputs.remove(0);
            dispatch_qwen_binary_same_shape(
                ctx,
                node_id,
                "residual-add",
                OperatorFamily::Tensor,
                a,
                b,
                Some(output_target),
            )
            .map(|(dispatch, value)| (dispatch, vec![value]))
        }
        "split" => {
            if inputs.len() != 1 {
                return Err(InferenceApiError::GraphPlanningFailed {
                    reason: format!("graph node '{node_id}' expects 1 split input"),
                });
            }
            let input = inputs.remove(0);
            let second_output_edge =
                node.outputs
                    .get(1)
                    .ok_or_else(|| InferenceApiError::GraphPlanningFailed {
                        reason: format!(
                            "graph node '{node_id}' (operator 'split') expects 2 output edges"
                        ),
                    })?;
            let second_output_target: OutputTarget = (
                TensorResourceId::new(format!("edge.{second_output_edge}")),
                resolved_output_placement(ctx.prepared_plan.as_deref(), &node.id),
                MemoryAllocationOwner::Session(kv_cache_id.to_string()),
            );
            dispatch_qwen_split(
                ctx,
                node_id,
                input,
                [Some(output_target), Some(second_output_target)],
            )
        }
        other => Err(InferenceApiError::OperatorUnsupported {
            reason: format!(
                "first-native graph executor has no dispatch for operator '{other}' at node '{node_id}'"
            ),
        }),
    }
}

/// Generic first-native graph executor: walks `graph`'s dependency order
/// (`first_native_graph_execution_order`), resolving each node's inputs from
/// pre-bound graph inputs (`initial_bindings`), previously computed
/// intermediates, or Model Artifact weights (`resolve_first_native_weight_edge`),
/// concatenating historical Runtime KV state for any edge whose
/// `GraphKvCacheBehavior::Append` metadata says so, dispatching every node
/// through the published `PreparedExecutionPlan`
/// (`dispatch_first_native_graph_node`), and returning every edge's bound value plus
/// the updated per-layer KV state.
/// Resolves the executing Provider's [`ProviderExecutionApi`] from Runtime's
/// own provider registration (see [`ProviderLoader::provider`]) rather than
/// constructing an unregistered throwaway. Unlike the downcast this replaced
/// (Correctif 3), this performs no concrete-type recovery: it only requires
/// that the registered Provider advertises `ProviderExecutionApi`, and
/// dispatch reaches it purely through that trait object. Fails closed with a
/// structured error if the named provider was never registered, or does not
/// implement `ProviderExecutionApi` at all.
pub(super) fn resolve_kernel_execution_provider(
    runtime: &Runtime,
    provider_binding: &ProviderBinding,
) -> Result<Arc<dyn ProviderExecutionApi>, InferenceApiError> {
    let provider = runtime
        .providers()
        .provider(provider_binding.as_str())
        .ok_or_else(|| InferenceApiError::ProviderUnavailable {
            reason: format!("provider '{provider_binding}' is not registered with Runtime"),
        })?;
    provider
        .execution_api()
        .ok_or_else(|| InferenceApiError::ProviderUnavailable {
            reason: format!(
                "registered provider '{provider_binding}' does not implement ProviderExecutionApi"
            ),
        })
}

/// A completed graph execution's dispatch evidence, every edge's bound
/// value (graph inputs, intermediates, and outputs, keyed by
/// [`TensorEdgeId`]), and the updated per-layer KV state.
pub(super) type FirstNativeGraphExecutionOutput = (
    KernelDispatchResult,
    BTreeMap<TensorEdgeId, HostTensor>,
    LayerKvMap,
    ProviderBinding,
);

/// Per-layer KV state, keyed by the *real, global* decoder layer number
/// (parsed from a graph edge's own `kv_cache.cache_id`, never a positional
/// index into whatever subset of layers one particular call happened to
/// touch). A full graph touches every layer `0..layer_count`, so this map
/// is dense for one; a *segment* graph
/// (`add-real-multi-device-model-instance-placement`) touches only its own
/// `[start_layer, end_layer)` range, so this map is correctly sparse (and
/// correctly keyed) for one -- a plain `Vec<FirstNativeLayerKvState>`
/// cannot represent that without silently misaligning "Vec index" against
/// "real layer number" the moment a graph does not touch layer 0..N
/// contiguously from zero.
pub type LayerKvMap = BTreeMap<usize, FirstNativeLayerKvState>;

/// [`execute_first_native_graph_nodes`]'s own return shape -- the same as
/// [`FirstNativeGraphExecutionOutput`] minus the resolved [`ProviderBinding`],
/// which only [`execute_first_native_graph`] (its caller) has resolved.
pub(super) type FirstNativeGraphNodesOutput = (
    KernelDispatchResult,
    BTreeMap<TensorEdgeId, HostTensor>,
    LayerKvMap,
);

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_first_native_graph(
    runtime: &mut Runtime,
    fixture: &E2eFixture,
    model_instance: &ModelInstanceId,
    kv_cache_id: &KvCacheId,
    graph: &ExecutionGraph,
    prepared_plan: &mut PreparedExecutionPlan,
    initial_bindings: BTreeMap<TensorEdgeId, HostTensor>,
    kv_history: Option<&LayerKvMap>,
    absolute_position_override: Option<u64>,
    node_events: &mut Vec<PerNodeCausalEvent>,
) -> Result<FirstNativeGraphExecutionOutput, InferenceApiError> {
    execute_first_native_graph_materializing(
        runtime,
        fixture,
        model_instance,
        kv_cache_id,
        graph,
        prepared_plan,
        initial_bindings,
        kv_history,
        absolute_position_override,
        node_events,
        MaterializedEdges::All,
        None,
        None,
    )
}

/// [`execute_first_native_graph`], downloading only the `materialized`
/// edges once every node has run. `memo`, when given, belongs to `graph`.
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_first_native_graph_materializing(
    runtime: &mut Runtime,
    fixture: &E2eFixture,
    model_instance: &ModelInstanceId,
    kv_cache_id: &KvCacheId,
    graph: &ExecutionGraph,
    prepared_plan: &mut PreparedExecutionPlan,
    initial_bindings: BTreeMap<TensorEdgeId, HostTensor>,
    kv_history: Option<&LayerKvMap>,
    absolute_position_override: Option<u64>,
    node_events: &mut Vec<PerNodeCausalEvent>,
    materialized: MaterializedEdges<'_>,
    memo: Option<&GraphExecutionMemo>,
    kv_store_capacity: Option<u64>,
) -> Result<FirstNativeGraphExecutionOutput, InferenceApiError> {
    execute_first_native_graph_with_resident_input(
        runtime,
        fixture,
        model_instance,
        kv_cache_id,
        graph,
        prepared_plan,
        initial_bindings,
        BTreeMap::new(),
        kv_history,
        absolute_position_override,
        node_events,
        materialized,
        memo,
        kv_store_capacity,
    )
}

/// Which graph edges a graph execution downloads to the host once every
/// node has run. Every download waits for the Device, and a decode step's
/// graph has hundreds of edges, the KV history among them.
#[derive(Clone, Copy, Debug)]
pub(crate) enum MaterializedEdges<'a> {
    /// Every live edge (conformance and test entry points that inspect
    /// intermediate values).
    All,
    /// Only these edges, when the graph produced them.
    Only(&'a [&'a str]),
    /// Only the last row of each of these rank-2 edges (a prefill's
    /// logits: only the last prompt row's distribution is sampled), read
    /// through `read_tensor_rows` so the other rows stay on the Device.
    LastRow(&'a [&'a str]),
}

/// [`execute_first_native_graph`], generalized to additionally accept graph inputs
/// that are ALREADY resident in the executing Provider's own storage
/// (`resident_bindings`) rather than requiring every input to arrive as a
/// fresh `HostTensor` (`add-real-multi-device-model-instance-placement`):
/// the real, zero-Host-round-trip counterpart of a segment's own
/// `input.hidden_states_in` boundary, written directly by a real cross-
/// Device move (`CudaExecutor::copy_tensor_from_peer_admitted`,
/// `add-real-peer-to-peer-gpu-movement`) before this call, instead of a
/// `HostTensor` staged through `initial_bindings`. `execute_first_native_graph`
/// itself is a thin wrapper over this function passing an empty
/// `resident_bindings` map, unchanged in behavior for every one of its
/// existing callers.
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_first_native_graph_with_resident_input(
    runtime: &mut Runtime,
    fixture: &E2eFixture,
    model_instance: &ModelInstanceId,
    kv_cache_id: &KvCacheId,
    graph: &ExecutionGraph,
    prepared_plan: &mut PreparedExecutionPlan,
    initial_bindings: BTreeMap<TensorEdgeId, HostTensor>,
    resident_bindings: BTreeMap<TensorEdgeId, (TensorResourceId, Vec<u64>)>,
    kv_history: Option<&LayerKvMap>,
    absolute_position_override: Option<u64>,
    node_events: &mut Vec<PerNodeCausalEvent>,
    materialized: MaterializedEdges<'_>,
    memo: Option<&GraphExecutionMemo>,
    kv_store_capacity: Option<u64>,
) -> Result<FirstNativeGraphExecutionOutput, InferenceApiError> {
    let prepare_clock = runtime.phase_ledger().clock();
    let computed_order;
    let order = match memo {
        Some(memo) => memo.order(graph)?,
        None => {
            computed_order = first_native_graph_execution_order(graph)?;
            &computed_order[..]
        }
    };
    let fingerprinted = match memo {
        Some(memo) => FingerprintedGraph::with_known_fingerprint(graph, memo.fingerprint(graph)),
        None => FingerprintedGraph::new(graph),
    };
    // Every node in this graph binds to the same Provider; any binding names
    // it (task 5.2: resolve the executing Provider from Runtime provider
    // registration for each prepared binding).
    let provider_binding = prepared_plan
        .node_bindings
        .first()
        .map(|binding| binding.provider.clone())
        .ok_or_else(|| InferenceApiError::KernelUnavailable {
            reason: "prepared execution plan has no node bindings".into(),
        })?;
    let executor = resolve_kernel_execution_provider(runtime, &provider_binding)?;
    // In-place KV history (`append-kv-history-in-place`) only when the
    // Provider supports it and attention is the only reader of KV edges.
    let kv_store_capacity = kv_store_capacity.filter(|_| {
        in_place_kv_supported(runtime, &provider_binding)
            && match memo {
                Some(memo) => memo.kv_edges_feed_only_attention(graph),
                None => kv_edges_feed_only_attention(graph),
            }
    });
    // Cloning this instance's weight bindings (a small `String ->
    // TensorResourceId` map, not tensor bytes) sidesteps holding both this
    // immutable borrow and the mutable `memory_mut()` borrow below at once;
    // the actual weight tensors are read from Provider storage through
    // `resolve_first_native_weight_edge`, by resource id, per node (task 6.3/6.4).
    let weight_bindings = runtime
        .model_instance(model_instance)
        .map_err(InferenceApiError::from)?
        .definition
        .resource_bindings
        .weights
        .clone();
    runtime
        .phase_ledger_mut()
        .record(prepare_clock, StepPhase::GraphPrepare);
    // Runtime's own MemoryManager is used directly (via `FirstNativeDispatchContext`
    // holding `&mut Runtime`, task group 7 / Correctif 13) rather than
    // temporarily taken out with `std::mem::take` and restored afterward --
    // that left any other code path holding `runtime` mid-call seeing a
    // default, empty `MemoryManager`, unsafe under continuous batching /
    // concurrent execution. Walking the graph is delegated to a helper
    // (rather than continuing inline) purely for readability now, not to
    // manage a restore point.
    let (dispatch, bindings, layer_kv) = execute_first_native_graph_nodes(
        order,
        runtime,
        fixture,
        &weight_bindings,
        kv_cache_id,
        fingerprinted,
        prepared_plan,
        &executor,
        initial_bindings,
        resident_bindings,
        kv_history,
        absolute_position_override,
        node_events,
        materialized,
        kv_store_capacity,
    )?;
    Ok((dispatch, bindings, layer_kv, provider_binding))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_first_native_graph_nodes(
    order: &[ExecutionNodeId],
    runtime: &mut Runtime,
    fixture: &E2eFixture,
    weight_bindings: &BTreeMap<String, TensorResourceId>,
    kv_cache_id: &KvCacheId,
    fingerprinted: FingerprintedGraph<'_>,
    prepared_plan: &mut PreparedExecutionPlan,
    executor: &Arc<dyn ProviderExecutionApi>,
    initial_bindings: BTreeMap<TensorEdgeId, HostTensor>,
    resident_bindings: BTreeMap<TensorEdgeId, (TensorResourceId, Vec<u64>)>,
    kv_history: Option<&LayerKvMap>,
    absolute_position_override: Option<u64>,
    node_events: &mut Vec<PerNodeCausalEvent>,
    materialized: MaterializedEdges<'_>,
    kv_store_capacity: Option<u64>,
) -> Result<FirstNativeGraphNodesOutput, InferenceApiError> {
    let graph = fingerprinted.graph();
    // This step's token count, for `PlanGuardContext::sequence_length`
    // (Correctif 4): the same value every dispatched node's Plan guards
    // were built against (`PlanGuard::SequenceRange`), read from the graph
    // input rather than recomputed per node.
    let token_ids = TensorEdgeId::new("input.token_ids");
    let sequence_length = initial_bindings
        .get(&token_ids)
        .map(|tensor| &tensor.shape)
        .or_else(|| resident_bindings.get(&token_ids).map(|(_, shape)| shape))
        .and_then(|shape| shape.first().copied());
    let mut dispatch_ctx = FirstNativeDispatchContext {
        runtime,
        provider: executor.clone(),
        prepared_plan: Some(prepared_plan),
        graph: Some(fingerprinted),
        sequence_length,
        last_provider_execution: None,
        internal_selection: None,
        bounded_kv_edges: BTreeMap::new(),
        node_events,
    };
    let layer_count = fixture.config.architecture.layer_count as usize;
    // Resource-based intermediates (Correctif 5): `bindings` never itself
    // holds tensor bytes. Every graph edge's live value lives exclusively in
    // the registered Provider's storage, keyed by `TensorResourceId`; this
    // map only records which resource each edge currently resolves to.
    // Graph inputs (`initial_bindings`) are the one seam where a caller
    // hands over raw values rather than an existing resource reference, so
    // they are written into Provider storage once, up front, under a
    // resource id derived from the edge id itself.
    //
    // Correctif 1: this write (and every node's `edge.*` output write
    // below) goes through `write_tensor_admitted`, not plain `write_tensor`
    // -- both are under a resource id *stable* across every call this KV
    // cache's session makes (one call per generation step), so a plain
    // `memory.allocate` here without replacement tracking would mint a new,
    // never-released allocation every step even though Provider storage
    // itself does not grow (each write overwrites the same entry) --
    // unbounded ledger growth over a long session. `write_tensor_admitted`
    // tracks and releases the allocation each resource id previously held,
    // Provider-side, so accounting stays bounded to this graph's edge count
    // no matter how many steps a session runs.
    let mut bindings: BTreeMap<TensorEdgeId, (TensorResourceId, Vec<u64>)> = BTreeMap::new();
    let upload_clock = dispatch_ctx.runtime.phase_ledger().clock();
    for (edge_id, tensor) in initial_bindings {
        let resource_id = TensorResourceId::new(edge_id.as_str());
        let shape = tensor.shape.clone();
        dispatch_ctx
            .provider
            .write_tensor_value_admitted(
                dispatch_ctx.runtime.memory_mut(),
                resource_id.clone(),
                TensorValue::Host(tensor),
                MemoryAllocationClass::Tensor,
                MemoryAllocationOwner::Session(kv_cache_id.to_string()),
            )
            .map_err(|error| match error {
                TensorValueAdmissionError::Memory(error) => {
                    InferenceApiError::MemoryAdmissionFailed {
                        reason: format!(
                            "failed to account graph input resource '{resource_id}': {error}"
                        ),
                    }
                }
                TensorValueAdmissionError::Provider(error) => {
                    InferenceApiError::ProviderTensorWriteFailed {
                        reason: format!(
                            "failed to write graph input resource '{resource_id}': {error}"
                        ),
                    }
                }
            })?;
        bindings.insert(edge_id, (resource_id, shape));
    }
    dispatch_ctx
        .runtime
        .phase_ledger_mut()
        .record(upload_clock, StepPhase::InputUpload);
    // Graph inputs ALREADY resident in this executing Provider's own
    // storage (`add-real-multi-device-model-instance-placement`'s real
    // peer-to-peer segment boundary, `CudaExecutor::copy_tensor_from_
    // peer_admitted` having already written the caller-chosen resource id
    // directly -- no Host round trip, no write here at all): just record
    // which resource each such edge resolves to, exactly like a freshly-
    // written `initial_bindings` entry above once it has been written.
    bindings.extend(resident_bindings);
    let mut layer_k: Vec<Option<TensorResourceId>> = vec![None; layer_count];
    let mut layer_v: Vec<Option<TensorResourceId>> = vec![None; layer_count];
    // Valid rows of each layer's in-place KV stores this step, when the
    // history is kept in place (`append-kv-history-in-place`).
    let mut layer_rows: Vec<Option<u64>> = vec![None; layer_count];
    let mut last_dispatch: Option<KernelDispatchResult> = None;
    // A prefill (several rows) keeps no intermediate past its last reader:
    // every edge's own resource stays Provider-resident until the step
    // ends otherwise, which for a long prompt is the whole graph's
    // activations at once (Qwen2.5-1.5B at 4096 tokens: about 9 GB, past a
    // 12 GB Device next to its weights). Decode steps keep theirs: a
    // captured step replays against the same resources.
    let release_intermediates = sequence_length.is_some_and(|rows| rows > 1)
        && matches!(
            materialized,
            MaterializedEdges::Only(_) | MaterializedEdges::LastRow(_)
        );
    let kept_edges: &[&str] = match materialized {
        MaterializedEdges::Only(edges) | MaterializedEdges::LastRow(edges) => edges,
        MaterializedEdges::All => &[],
    };
    let mut last_reader: BTreeMap<&TensorEdgeId, usize> = BTreeMap::new();
    if release_intermediates {
        for (index, node_id) in order.iter().enumerate() {
            if let Some(node) = graph.nodes.get(node_id) {
                for edge_id in &node.inputs {
                    last_reader.insert(edge_id, index);
                }
            }
        }
    }

    for (node_index, node_id) in order.iter().enumerate() {
        let node =
            graph
                .nodes
                .get(node_id)
                .ok_or_else(|| InferenceApiError::GraphPlanningFailed {
                    reason: format!("first-native graph is missing node '{node_id}'"),
                })?;
        // Nodes whose own Rust code never dereferences input bytes directly
        // (they only forward the value to a Kernel dispatch) can accept an
        // already device-resident input by reference. `rmsnorm` and `rope`
        // joined this list in `make-first-native-cuda-hot-path-device-
        // resident` (task groups 2-3): `rmsnorm`'s Rust-side weight
        // broadcast and `rope`'s per-head Rust slicing were both removed --
        // neither operator's own dispatch function dereferences raw floats
        // anymore, both take `NodeValue` straight through to their Kernel
        // dispatch like every operator here.
        let passthrough_eligible = matches!(
            node.operator.name(),
            "matmul"
                | "matmul-bias"
                | "matmul-residual"
                | "gated-matmul"
                | "silu-mul"
                | "moe-experts"
                | "row-scale"
                | "activation"
                | "attention"
                | "silu"
                | "add"
                | "mul"
                | "residual-add"
                | "embedding"
                | "rmsnorm"
                | "rope"
        );
        let inputs_clock = dispatch_ctx.runtime.phase_ledger().clock();
        let mut inputs = Vec::with_capacity(node.inputs.len());
        for edge_id in &node.inputs {
            let value = match bindings.get(edge_id) {
                Some((resource_id, shape)) => {
                    let value = dispatch_ctx
                        .provider
                        .read_tensor_value(resource_id)
                        .ok_or_else(|| InferenceApiError::GraphPlanningFailed {
                            reason: format!(
                                "no materialized data for graph edge '{edge_id}' (resource '{resource_id}')"
                            ),
                        })?;
                    match value {
                        // Same Provider already holds this edge device-
                        // (or otherwise Provider-)resident: reuse it in
                        // place instead of downloading and immediately
                        // re-uploading a copy under a fresh id.
                        TensorValue::Opaque if passthrough_eligible => NodeValue::Resident {
                            id: resource_id.clone(),
                            shape: shape.clone(),
                        },
                        other => {
                            NodeValue::Host(other.into_host(resource_id).map_err(|error| {
                                InferenceApiError::GraphPlanningFailed {
                                    reason: format!(
                                        "graph edge '{edge_id}' (resource '{resource_id}'): {error}"
                                    ),
                                }
                            })?)
                        }
                    }
                }
                None => {
                    let edge = graph.edges.get(edge_id).ok_or_else(|| {
                        InferenceApiError::GraphPlanningFailed {
                            reason: format!("first-native graph is missing edge '{edge_id}'"),
                        }
                    })?;
                    resolve_first_native_weight_edge(
                        executor,
                        weight_bindings,
                        edge_id.as_str(),
                        &edge.descriptor,
                    )?
                }
            };
            inputs.push(value);
        }
        if let MaterializedEdges::LastRow(edges) = materialized
            && last_row_suffices(node, edges)
        {
            last_row_only(&*dispatch_ctx.provider, &mut inputs[0])?;
        }
        dispatch_ctx
            .runtime
            .phase_ledger_mut()
            .record(inputs_clock, StepPhase::NodeInputResolve);
        let dispatch_clock = dispatch_ctx.runtime.phase_ledger().clock();
        let (dispatch_result, output_values) = dispatch_first_native_graph_node(
            &mut dispatch_ctx,
            fixture,
            node,
            inputs,
            absolute_position_override,
            kv_cache_id,
        )?;
        dispatch_ctx.runtime.phase_ledger_mut().record_operator(
            dispatch_clock,
            StepPhase::NodeDispatch,
            node.operator.name(),
        );
        let bind_clock = dispatch_ctx.runtime.phase_ledger().clock();
        if node.outputs.is_empty() {
            return Err(InferenceApiError::GraphPlanningFailed {
                reason: format!("graph node '{node_id}' has no output edge"),
            });
        }
        if output_values.len() != node.outputs.len() {
            return Err(InferenceApiError::GraphPlanningFailed {
                reason: format!(
                    "graph node '{node_id}' dispatch produced {} output value(s) for {} declared output edge(s)",
                    output_values.len(),
                    node.outputs.len()
                ),
            });
        }
        for (output_edge_id, mut output_tensor) in node.outputs.iter().zip(output_values) {
            let output_edge = graph.edges.get(output_edge_id).ok_or_else(|| {
                InferenceApiError::GraphPlanningFailed {
                    reason: format!("first-native graph is missing edge '{output_edge_id}'"),
                }
            })?;
            let output_resource_id = graph_edge_resource_id(output_edge_id.as_str());
            // Whether the Kernel already wrote directly into `output_resource_id`
            // (Decision 3's pre-admission, `dispatch_first_native_graph_node`) or this
            // loop must still explicitly (re-)write it: KV-history
            // concatenation below produces genuinely new, host-computed data,
            // and "rope" nodes have no Kernel-level output identity to begin
            // with (always `NodeValue::Host`, never `Resident` under this id).
            // The resource the edge resolves to for later nodes: its own
            // identity, or the pending KV resource a KV append wrote.
            let mut bound_resource_id = output_resource_id.clone();
            let mut needs_explicit_edge_write = !matches!(
                &output_tensor,
                NodeValue::Resident { id, .. } if *id == output_resource_id
            );
            if let Some(kv_meta) = &output_edge.kv_cache {
                let (layer, role) = parse_kv_cache_id(&kv_meta.cache_id)?;
                if layer >= layer_count {
                    return Err(InferenceApiError::GraphPlanningFailed {
                        reason: format!(
                            "graph KV cache id '{}' has an out-of-range layer",
                            kv_meta.cache_id
                        ),
                    });
                }
                let role_str = match role {
                    KvRole::K => "k",
                    KvRole::V => "v",
                };
                let pending_resource = TensorResourceId::new(format!(
                    "kv.{kv_cache_id}.layer{layer}.{role_str}.pending"
                ));
                // In-place history (`append-kv-history-in-place`): a decode
                // step continues the stores its history lives in; a prefill
                // starts new stores when this execution keeps KV in place.
                let in_place = match kv_history.and_then(|history| history.get(&layer)) {
                    Some(FirstNativeLayerKvState {
                        rows: Some(rows), ..
                    }) => Some((
                        kv_store_capacity.ok_or_else(|| InferenceApiError::KvCacheUnavailable {
                            reason: format!(
                                "layer {layer} keeps its KV history in place, but this \
                                     execution cannot continue it"
                            ),
                        })?,
                        *rows,
                    )),
                    Some(_) => None,
                    None => kv_store_capacity.map(|capacity| (capacity, 0)),
                };
                if let Some((capacity, row_offset)) = in_place {
                    let store = TensorResourceId::new(format!(
                        "kv.{kv_cache_id}.layer{layer}.{role_str}.store"
                    ));
                    let (rows, width) = output_tensor.rows_cols()?;
                    let valid_rows = row_offset
                        .checked_add(rows)
                        .filter(|valid| *valid <= capacity)
                        .ok_or_else(|| InferenceApiError::KvCacheUnavailable {
                            reason: format!(
                                "layer {layer}: {rows} new KV rows at {row_offset} exceed the \
                                 store's capacity of {capacity}"
                            ),
                        })?;
                    let store_descriptor = f32_tensor_descriptor_from_shape(&[capacity, width]);
                    let append_clock = dispatch_ctx.runtime.phase_ledger().clock();
                    if row_offset == 0 {
                        // A new store, admitted once for the whole request.
                        let target: OutputTarget = (
                            store.clone(),
                            resolved_output_placement(
                                dispatch_ctx.prepared_plan.as_deref(),
                                node_id,
                            ),
                            MemoryAllocationOwner::Session(kv_cache_id.to_string()),
                        );
                        let (store_id, _) = resolve_output_target(
                            &mut dispatch_ctx,
                            &format!("{node_id}.kv-store"),
                            store_descriptor.clone(),
                            Some(target),
                        )?;
                        // Zeroed where the store lives when the Provider can,
                        // instead of building and uploading host zeros: at
                        // prefill, one store per layer and role.
                        let zeroed = dispatch_ctx
                            .provider
                            .write_zeroed_tensor(&store_id, &[capacity, width])
                            .map_err(|error| InferenceApiError::ProviderTensorWriteFailed {
                                reason: error.to_string(),
                            })?;
                        if !zeroed {
                            let zeros = HostTensor::new(
                                [capacity, width],
                                vec![0.0; (capacity * width) as usize],
                            )
                            .map_err(runtime_generation_failed)?;
                            dispatch_ctx
                                .provider
                                .write_tensor_value(store_id, TensorValue::Host(zeros))
                                .map_err(|error| InferenceApiError::ProviderTensorWriteFailed {
                                    reason: error.to_string(),
                                })?;
                        }
                    }
                    let mut attributes = BTreeMap::new();
                    attributes.insert(
                        "row_offset".into(),
                        OperatorAttributeValue::Integer(row_offset as i64),
                    );
                    dispatch_runtime_internal_operator(
                        &mut dispatch_ctx,
                        &format!("{node_id}.kv-write"),
                        node_id,
                        dispatch_operator_id("kv-write", OperatorFamily::Tensor),
                        [
                            (
                                "store",
                                NodeValue::Resident {
                                    id: store.clone(),
                                    shape: vec![capacity, width],
                                },
                            ),
                            ("rows", output_tensor),
                        ],
                        (store.clone(), store_descriptor),
                        attributes,
                    )?;
                    dispatch_ctx
                        .runtime
                        .phase_ledger_mut()
                        .record(append_clock, StepPhase::KvAppend);
                    output_tensor = NodeValue::Resident {
                        id: store.clone(),
                        shape: vec![capacity, width],
                    };
                    needs_explicit_edge_write = false;
                    bound_resource_id = store.clone();
                    dispatch_ctx
                        .bounded_kv_edges
                        .insert(output_edge_id.clone(), valid_rows);
                    dispatch_ctx.node_events.push(
                        PerNodeCausalEvent::new(
                            InferenceApiObservationKind::KvUpdatePrepared,
                            node_id.clone(),
                        )
                        .with_resource(store.clone()),
                    );
                    match role {
                        KvRole::K => layer_k[layer] = Some(store),
                        KvRole::V => layer_v[layer] = Some(store),
                    }
                    if layer_rows[layer].is_some_and(|rows| rows != valid_rows) {
                        return Err(InferenceApiError::KvCacheUnavailable {
                            reason: format!(
                                "layer {layer}: K and V in-place stores disagree on their rows"
                            ),
                        });
                    }
                    layer_rows[layer] = Some(valid_rows);
                } else {
                    if kv_meta.behavior == GraphKvCacheBehavior::Append {
                        let history =
                            kv_history.ok_or_else(|| InferenceApiError::KvCacheUnavailable {
                                reason: "decode graph execution requires historical KV state"
                                    .into(),
                            })?;
                        let historical = history.get(&layer).ok_or_else(|| {
                            InferenceApiError::KvCacheUnavailable {
                                reason: format!(
                                    "decode requires historical KV state for layer {layer}"
                                ),
                            }
                        })?;
                        let historical_resource = match role {
                            KvRole::K => &historical.k,
                            KvRole::V => &historical.v,
                        };
                        // Historical KV data stays wherever the registered
                        // Provider already holds it (task 7.2/7.3): a device-
                        // resident Provider (CUDA) is never asked to produce
                        // host-visible bytes for it -- `resident_shape` reads
                        // only structural metadata, and the "concat" Operator
                        // dispatch below runs through the exact same generic
                        // Kernel Registry path every other graph node uses,
                        // targeting this edge's own stable identity directly
                        // (`implement-device-resident-multi-step-cuda-decode`,
                        // replacing this block's former `read_tensor_value(..)
                        // .into_host(..)` + Rust-computed `concat_rows`).
                        let historical_shape = dispatch_ctx
                            .provider
                            .resident_shape(historical_resource)
                            .ok_or_else(|| InferenceApiError::KvCacheUnavailable {
                                reason: format!(
                                    "no materialized historical KV data for layer {layer}"
                                ),
                            })?;
                        let historical_value = NodeValue::Resident {
                            id: historical_resource.clone(),
                            shape: historical_shape,
                        };
                        // The concatenated history is this step's pending KV
                        // write: the concat writes it straight under the
                        // pending identity, and the edge is bound to that
                        // resource, instead of writing the edge and copying it
                        // to the pending identity afterwards.
                        let concat_output_target: OutputTarget = (
                            pending_resource.clone(),
                            resolved_output_placement(
                                dispatch_ctx.prepared_plan.as_deref(),
                                node_id,
                            ),
                            MemoryAllocationOwner::Session(kv_cache_id.to_string()),
                        );
                        let append_clock = dispatch_ctx.runtime.phase_ledger().clock();
                        let concatenated = dispatch_qwen_concat(
                            &mut dispatch_ctx,
                            &format!("{node_id}.kv-append"),
                            node_id,
                            historical_value,
                            output_tensor.clone(),
                            Some(concat_output_target),
                        )?;
                        dispatch_ctx
                            .runtime
                            .phase_ledger_mut()
                            .record(append_clock, StepPhase::KvAppend);
                        // The concat dispatch pre-admitted its output directly
                        // under `output_resource_id` (mirroring Decision 3's
                        // existing optimization for every other node kind), so
                        // the edge is already correctly written -- no explicit
                        // (re-)write, and critically no `into_host` round trip,
                        // needed below.
                        output_tensor = concatenated;
                        needs_explicit_edge_write = false;
                        bound_resource_id = pending_resource.clone();
                    }
                    // Written under a *pending* resource id (task 7.4 prepare):
                    // this generation step's KV update becomes Runtime-owned only
                    // once `commit_generation_step` promotes it after sampling and
                    // token commit succeed; a failure or cancellation before then
                    // simply leaves this pending write unpromoted. When
                    // `output_tensor` is already `Resident` (decode's KV-append
                    // case always is, after the concat dispatch above; prefill's
                    // "Output" case is whenever the producing Kernel itself wrote
                    // directly into `output_resource_id`), this is a real
                    // Provider-side copy (`copy_tensor_admitted`) to the pending
                    // resource's own, separate, stable identity -- never a host
                    // round trip (`implement-device-resident-multi-step-cuda-
                    // decode`). Only a genuinely `Host`-typed value (e.g. "rope",
                    // which has no Kernel-level output identity to begin with)
                    // still needs an explicit host-typed admitted write here.
                    let pending_clock = dispatch_ctx.runtime.phase_ledger().clock();
                    let pending_write_result: Result<(), TensorValueAdmissionError> =
                        match &output_tensor {
                            // Already written under the pending identity (the
                            // KV append above).
                            NodeValue::Resident { id, .. } if *id == pending_resource => Ok(()),
                            NodeValue::Resident { id, .. } => dispatch_ctx
                                .provider
                                .copy_tensor_admitted(
                                    dispatch_ctx.runtime.memory_mut(),
                                    id,
                                    pending_resource.clone(),
                                    MemoryAllocationClass::Tensor,
                                    MemoryAllocationOwner::Session(kv_cache_id.to_string()),
                                )
                                .map(|_allocation_id| ()),
                            NodeValue::Host(_) => {
                                let pending_tensor =
                                    output_tensor.clone().into_host(&dispatch_ctx.provider)?;
                                dispatch_ctx.provider.write_tensor_value_admitted(
                                    dispatch_ctx.runtime.memory_mut(),
                                    pending_resource.clone(),
                                    TensorValue::Host(pending_tensor),
                                    MemoryAllocationClass::Tensor,
                                    MemoryAllocationOwner::Session(kv_cache_id.to_string()),
                                )
                            }
                        };
                    // Correctif 1: admitted (via `write_tensor_admitted`/
                    // `copy_tensor_admitted`), not a bare `write_tensor` -- this
                    // id is stable across every decode step for this layer/role,
                    // so admission replaces (and releases) whatever allocation
                    // the previous step's pending write held for it, the same
                    // reasoning as the graph-edge writes above.
                    // `discard_pending_kv_state` releases this through
                    // `release_admitted_tensor` when a step is cancelled or fails
                    // before commit.
                    pending_write_result.map_err(|error| match error {
                        TensorValueAdmissionError::Memory(error) => {
                            InferenceApiError::MemoryAdmissionFailed {
                                reason: format!(
                                    "failed to account pending KV resource '{pending_resource}': {error}"
                                ),
                            }
                        }
                        TensorValueAdmissionError::Provider(error) => {
                            InferenceApiError::ProviderTensorWriteFailed {
                                reason: format!(
                                    "failed to write pending KV resource '{pending_resource}': {error}"
                                ),
                            }
                        }
                    })?;
                    dispatch_ctx
                        .runtime
                        .phase_ledger_mut()
                        .record(pending_clock, StepPhase::KvPendingWrite);
                    dispatch_ctx.node_events.push(
                        PerNodeCausalEvent::new(
                            InferenceApiObservationKind::KvUpdatePrepared,
                            node_id.clone(),
                        )
                        .with_resource(pending_resource.clone()),
                    );
                    match role {
                        KvRole::K => layer_k[layer] = Some(pending_resource),
                        KvRole::V => layer_v[layer] = Some(pending_resource),
                    }
                }
            }
            // `output_resource_id` (computed above, before the KV block) is
            // this edge's own stable identity. When `needs_explicit_edge_write`
            // is false, the Kernel already wrote directly into it via
            // pre-admission (`dispatch_first_native_graph_node`'s `output_target`) --
            // this loop only needs to record the binding, not re-download and
            // re-upload a copy (`unify-provider-output-admission-and-
            // residency`'s Decision 3: this is what actually eliminates the
            // output-side round-trip `enable-device-resident-kernel-chaining`
            // deferred). When true (KV-history concatenation produced new data,
            // or the node -- "rope" -- has no Kernel-level output identity),
            // the value is explicitly (re-)written exactly as before.
            let output_shape = output_tensor.shape().to_vec();
            if needs_explicit_edge_write {
                let edge_clock = dispatch_ctx.runtime.phase_ledger().clock();
                let output_tensor = output_tensor.into_host(&dispatch_ctx.provider)?;
                dispatch_ctx
                .provider
                .write_tensor_value_admitted(
                    dispatch_ctx.runtime.memory_mut(),
                    output_resource_id.clone(),
                    TensorValue::Host(output_tensor),
                    MemoryAllocationClass::Tensor,
                    MemoryAllocationOwner::Session(kv_cache_id.to_string()),
                )
                .map_err(|error| match error {
                    TensorValueAdmissionError::Memory(error) => {
                        InferenceApiError::MemoryAdmissionFailed {
                            reason: format!(
                                "failed to account graph edge resource '{output_resource_id}': {error}"
                            ),
                        }
                    }
                    TensorValueAdmissionError::Provider(error) => {
                        InferenceApiError::ProviderTensorWriteFailed {
                            reason: format!(
                                "failed to write graph edge resource '{output_resource_id}': {error}"
                            ),
                        }
                    }
                })?;
                dispatch_ctx.runtime.phase_ledger_mut().record_operator(
                    edge_clock,
                    StepPhase::EdgeWrite,
                    node.operator.name(),
                );
            }
            bindings.insert(output_edge_id.clone(), (bound_resource_id, output_shape));
        }
        dispatch_ctx
            .runtime
            .phase_ledger_mut()
            .record(bind_clock, StepPhase::NodeOutputBind);
        if release_intermediates {
            for edge_id in &node.inputs {
                if last_reader.get(edge_id) != Some(&node_index)
                    || kept_edges.contains(&edge_id.as_str())
                    || graph
                        .edges
                        .get(edge_id)
                        .is_none_or(|edge| edge.kv_cache.is_some())
                {
                    continue;
                }
                // Only a node output's own resource: never a graph input,
                // a weight or a KV store an edge resolves to.
                let own = graph_edge_resource_id(edge_id.as_str());
                if bindings.get(edge_id).is_some_and(|(id, _)| *id == own) {
                    bindings.remove(edge_id);
                    let _ = dispatch_ctx
                        .provider
                        .release_admitted_tensor(dispatch_ctx.runtime.memory_mut(), &own);
                }
            }
        }
        last_dispatch = Some(dispatch_result);
    }

    // `0..layer_count` (the *full architecture's* layer count, `fixture`'s
    // only use in this function), not merely "however many layers this
    // particular graph touched" -- true for every graph before segment
    // graphs existed, since a full graph always has nodes for every layer.
    // A prefill-*segment* graph (`add-real-multi-device-model-instance-
    // placement`) only touches its own `[start_layer, end_layer)` range, so
    // layers outside it never populate `layer_k`/`layer_v` at all; skipping
    // those (rather than hard-failing on them) generalizes this loop to
    // segment graphs with zero behavior change for a full graph, which
    // still populates every index here exactly as before. Keyed by real
    // layer number (`LayerKvMap`), not Vec position, so a segment's
    // necessarily sparse result stays correctly addressable by whichever
    // real layer a later decode step actually asks for.
    let mut updated_layer_kv = LayerKvMap::new();
    for layer in 0..layer_count {
        let touched = layer_k[layer].is_some() || layer_v[layer].is_some();
        if !touched {
            continue;
        }
        let k = layer_k[layer]
            .take()
            .ok_or_else(|| InferenceApiError::KvCacheUnavailable {
                reason: format!("first-native graph produced no K state for layer {layer}"),
            })?;
        let v = layer_v[layer]
            .take()
            .ok_or_else(|| InferenceApiError::KvCacheUnavailable {
                reason: format!("first-native graph produced no V state for layer {layer}"),
            })?;
        updated_layer_kv.insert(
            layer,
            FirstNativeLayerKvState {
                k,
                v,
                rows: layer_rows[layer],
            },
        );
    }
    let dispatch_result = last_dispatch.ok_or_else(|| InferenceApiError::GenerationFailed {
        reason: "first-native graph executed no nodes".into(),
    })?;
    let materialize_clock = dispatch_ctx.runtime.phase_ledger().clock();
    if let MaterializedEdges::Only(edges) | MaterializedEdges::LastRow(edges) = materialized {
        bindings.retain(|edge_id, _| edges.contains(&edge_id.as_str()));
    }
    let materialized_bindings = if let MaterializedEdges::LastRow(_) = materialized {
        materialize_last_rows(&*dispatch_ctx.provider, &bindings)?
    } else {
        materialize_first_native_graph_bindings(&*dispatch_ctx.provider, &bindings)?
    };
    dispatch_ctx
        .runtime
        .phase_ledger_mut()
        .record(materialize_clock, StepPhase::OutputMaterialize);
    Ok((dispatch_result, materialized_bindings, updated_layer_kv))
}

/// Replays a request's captured decode step at `absolute_position`
/// (`capture-and-replay-cuda-graphs`). `None` when the capture can no longer
/// be replayed: the step must run per node.
/// A plain greedy step's Device selection: the excluded logits columns and
/// the temperature (`sampling::device_greedy_plan`).
pub(super) type DeviceGreedyPlan = (Vec<std::ops::Range<u64>>, f32);

/// Reports why a decode step was not captured or not replayed, when
/// `MAGNETAR_STEP_CAPTURE_TRACE=1`: either silently falls back to running
/// the step per node, so a Provider's capture support is otherwise only
/// visible in step times.
pub(super) fn trace_step_capture(what: &str, reason: impl std::fmt::Display) {
    if std::env::var("MAGNETAR_STEP_CAPTURE_TRACE").as_deref() == Ok("1") {
        eprintln!("[step capture] {what}: {reason}");
    }
}

/// Replays a recorded decode step. With `greedy`, the Provider selects the
/// step's token on the Device when it can: the output then holds no logits
/// and the selected column comes back instead.
pub(super) fn replay_decode_step(
    runtime: &mut Runtime,
    recorded: &RecordedDecodeStep,
    ids_tensor: HostTensor,
    absolute_position: u64,
    greedy: Option<&DeviceGreedyPlan>,
) -> Result<Option<(FirstNativeGraphExecutionOutput, Option<u64>)>, InferenceApiError> {
    let Some(offset) = absolute_position.checked_sub(recorded.position) else {
        return Ok(None);
    };
    let mut layer_kv = recorded.layer_kv.clone();
    for layer in layer_kv.values_mut() {
        let Some(rows) = layer.rows.and_then(|rows| rows.checked_add(offset)) else {
            return Ok(None);
        };
        layer.rows = Some(rows);
    }
    let executor = recorded.executor.clone();
    let replay_clock = runtime.phase_ledger().clock();
    let replayed = executor.replay_step(
        &recorded.capture,
        offset,
        vec![(recorded.input.clone(), ids_tensor)],
    );
    runtime
        .phase_ledger_mut()
        .record(replay_clock, StepPhase::StepReplay);
    if let Err(error) = replayed {
        trace_step_capture("replay invalidated", format!("{error:?}"));
        return Ok(None);
    }
    let materialize_clock = runtime.phase_ledger().clock();
    let (logits_resource, logits_shape) = &recorded.logits;
    let last_row = logits_shape
        .first()
        .map_or(0, |rows| rows.saturating_sub(1));
    let selected = greedy.and_then(|(excluded, temperature)| {
        executor.select_greedy_column(logits_resource, last_row, excluded, *temperature)
    });
    let bindings = match selected {
        Some(_) => BTreeMap::new(),
        None => materialize_first_native_graph_bindings(
            &*executor,
            &BTreeMap::from([(TensorEdgeId::new("logits"), recorded.logits.clone())]),
        )?,
    };
    runtime
        .phase_ledger_mut()
        .record(materialize_clock, StepPhase::OutputMaterialize);
    Ok(Some((
        (
            recorded.dispatch.clone(),
            bindings,
            layer_kv,
            recorded.provider.clone(),
        ),
        selected,
    )))
}

/// The resource a node output edge is written under, unless it is a KV
/// edge.
pub(super) fn graph_edge_resource_id(edge: &str) -> TensorResourceId {
    TensorResourceId::new(format!("edge.{edge}"))
}

/// The one designated Device-to-Host materialization boundary for a first-
/// native graph dispatch: every remaining live edge (including
/// `"output.logits"`) crosses back into the caller-facing `HostTensor`
/// contract here, once, not per-node -- everywhere inside
/// [`execute_first_native_graph_nodes`]'s own per-node transport loop carries
/// `TensorValue` instead (that function's static guard,
/// `check_execute_qwen_graph_nodes_transport_has_no_host_tensor_typed_calls`,
/// enforces zero raw `HostTensor`-typed `ProviderExecutionApi` calls there
/// specifically -- this function is deliberately a separate one, outside
/// that guard's scan, precisely because this boundary's whole job is the
/// opposite of that loop's: to genuinely materialize, not to defer it).
///
/// Genuinely downloads through `read_tensor` (the `HostTensor`-typed
/// method), not `read_tensor_value(...).into_host(...)` --
/// `audit-complet-cuda-hot-path-2026-09-08` P1-1, found only once a real
/// CUDA Provider actually reached this boundary (`run_first_native_graph_
/// with_provider`'s new generalized entrypoint; every prior call site only
/// ever ran against Reference CPU, which never produces `TensorValue::
/// Opaque` in the first place). `read_tensor_value` is *never* required to
/// answer `Host` for a genuinely Device-resident Provider (`define-
/// provider-prepared-kernel-execution-contract`'s own documented contract:
/// CUDA's `read_tensor_value` always answers `Opaque`, by design), so
/// `.into_host()` on its result is structurally incapable of ever
/// downloading real Device-resident data -- every intermediate edge this
/// dispatch deliberately kept Resident (Decision 3's whole point) would
/// fail exactly this way at this final boundary, Reference CPU's own
/// always-`Host` behavior being the only reason this was never observed
/// before.
pub(super) fn materialize_first_native_graph_bindings(
    provider: &dyn ProviderExecutionApi,
    bindings: &BTreeMap<TensorEdgeId, (TensorResourceId, Vec<u64>)>,
) -> Result<BTreeMap<TensorEdgeId, HostTensor>, InferenceApiError> {
    let mut materialized_bindings = BTreeMap::new();
    for (edge_id, (resource_id, _shape)) in bindings {
        let tensor = provider.read_tensor(resource_id).ok_or_else(|| {
            InferenceApiError::GraphPlanningFailed {
                reason: format!(
                    "no materialized data for graph edge '{edge_id}' (resource '{resource_id}')"
                ),
            }
        })?;
        materialized_bindings.insert(edge_id.clone(), tensor);
    }
    Ok(materialized_bindings)
}

#[cfg(test)]
mod tests;
