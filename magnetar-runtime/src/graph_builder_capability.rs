//! Runtime-side implementation of the `magnetar:model-component-graph`
//! WIT Capability (`model-component-graph-contract`,
//! `magnetar-runtime/wit/model-component-graph.wit`): a
//! [`crate::component::HostCapability`] a Model Component calls into to
//! incrementally build a real [`ExecutionGraph`], which the Runtime owns
//! and validates.
//!
//! Deliberately generic: nothing here names Qwen, Llama, or any other
//! model family. A caller (today, `first_native_runtime.rs`, which *is*
//! still Qwen-aware -- see `reach-architecture-freeze-1` task group 12)
//! supplies a [`SessionContext`] naming the Runtime-recognized weight
//! shapes and the KV cache id namespace this session's graph must produce,
//! so the *capability* stays reusable across model families even though
//! its *caller* is not yet.

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::component::{ComponentError, ComponentValue, HostCapability};
use crate::compute::{
    ComputeDType, DTypeDescriptor, LayoutDescriptor, ShapeDescriptor, TensorDescriptor,
};
use crate::execution_graph::{
    ExecutionGraph, ExecutionGraphId, ExecutionGraphPhase, ExecutionGraphProducer, ExecutionNode,
    ExecutionNodeId, GraphKvCacheBehavior, GraphKvCacheMetadata, GraphModelCompatibility,
    TensorAliasing, TensorEdge, TensorEdgeId,
};
use crate::operator::{
    OperatorAttributeValue, OperatorFamily, OperatorId, initial_operator_catalog,
};

const CAPABILITY_NAME: &str = "magnetar:model-component-graph/graph-builder";

/// Runtime context one graph-building session needs, supplied by the
/// caller before it invokes the Component (`GraphBuilderCapability::
/// prepare_session`), not discovered by the capability itself.
#[derive(Clone, Debug)]
pub struct SessionContext {
    /// Identifies the producing Component in `ExecutionGraphProducer::
    /// ModelComponent` and the graph id.
    pub component_id: String,
    /// Recorded on the produced graph as `ExecutionGraph.fingerprint`.
    pub compatibility_key: String,
    /// KV cache id namespace (e.g. `"qwen"`) this session's `kv-resource`
    /// calls issue identities under, so the existing (not yet
    /// model-family-agnostic -- task group 12) execution path's
    /// `parse_qwen_kv_cache_id`-shaped parsing keeps working against a
    /// Component-produced graph without changes to that parsing itself.
    pub kv_namespace: String,
    /// Real dimensions for every weight this session's Component may
    /// reference via `weight-edge`, keyed by the same logical name the
    /// Component supplies (e.g. `"layer0.q_proj"`, `"token_embedding"`).
    /// The Runtime already knows these from Model Loading; the Component
    /// never supplies or overrides them (Requirement "Graph Builder Tensor
    /// Descriptors and Weight References").
    pub weight_shapes: BTreeMap<String, Vec<u64>>,
    /// The fixed edge id the existing (not yet model-family-agnostic --
    /// task group 12) execution path looks up as the graph's final output
    /// (e.g. `"logits"`). `finish-graph` renames whatever edge id the
    /// Component names as its output to this one, so the produced graph is
    /// consumable by that still-Qwen-specific reader without this
    /// capability itself hardcoding the string `"logits"`.
    pub output_edge_name: String,
}

struct Session {
    context: SessionContext,
    graph: Option<ExecutionGraph>,
    finished: BTreeMap<String, ExecutionGraph>,
    next_handle: u64,
}

/// Runtime-side [`HostCapability`] backing `graph-builder`. One instance is
/// registered once on a `ComponentManager`/engine and shared across every
/// Component instance that imports it; per-instance state (the graph under
/// construction) is keyed by the calling instance's own key -- see
/// [`HostCapability::call`]'s `instance_key` parameter.
#[derive(Default)]
pub struct GraphBuilderCapability {
    sessions: Mutex<BTreeMap<String, Session>>,
}

impl GraphBuilderCapability {
    pub fn new() -> Self {
        Self::default()
    }

    /// Stages `context` for `instance_key` before the caller invokes the
    /// Component's `build-prefill-graph`/`build-decode-graph` export --
    /// the Component's own `begin-graph`/`declare-input`/... calls, which
    /// happen synchronously inside that invocation, look this up by the
    /// same key. Replaces any prior session for this key (a session does
    /// not survive past one export call in practice, since `first_native_
    /// runtime.rs` calls this immediately before each `build-*-graph`
    /// invocation).
    pub fn prepare_session(&self, instance_key: &str, context: SessionContext) {
        let mut sessions = self.sessions.lock().unwrap();
        sessions.insert(
            instance_key.to_string(),
            Session {
                context,
                graph: None,
                finished: BTreeMap::new(),
                next_handle: 0,
            },
        );
    }

    /// Retrieves the real `ExecutionGraph` a prior `finish-graph` call
    /// produced, by the opaque handle string it returned. Removes it from
    /// this session's finished set (a handle is consumed at most once).
    pub fn take_graph(&self, instance_key: &str, handle: &str) -> Option<ExecutionGraph> {
        let mut sessions = self.sessions.lock().unwrap();
        sessions
            .get_mut(instance_key)
            .and_then(|session| session.finished.remove(handle))
    }

    /// Drops all session state for `instance_key` (both any in-progress
    /// graph and any unconsumed finished graph), for the caller to call
    /// once it is done with a Component instance -- a `HostCapability` is
    /// shared for the instance's whole lifetime, so nothing else releases
    /// this state on its own.
    pub fn clear_session(&self, instance_key: &str) {
        self.sessions.lock().unwrap().remove(instance_key);
    }
}

fn rejected(capability: &str, instance_key: &str, message: impl Into<String>) -> ComponentError {
    ComponentError::CapabilityCallRejected {
        capability: capability.to_string(),
        instance_key: instance_key.to_string(),
        message: message.into(),
    }
}

fn expect_arity(
    operation: &str,
    expected: usize,
    arguments: &[ComponentValue],
) -> Result<(), String> {
    if arguments.len() == expected {
        Ok(())
    } else {
        Err(format!(
            "'{operation}' expects {expected} argument(s), got {}",
            arguments.len()
        ))
    }
}

fn expect_string(value: &ComponentValue) -> Result<&str, String> {
    match value {
        ComponentValue::String(value) => Ok(value.as_str()),
        other => Err(format!("expected a string, got {other:?}")),
    }
}

fn expect_u64(value: &ComponentValue) -> Result<u64, String> {
    match value {
        ComponentValue::S64(value) if *value >= 0 => Ok(*value as u64),
        other => Err(format!("expected a non-negative integer, got {other:?}")),
    }
}

fn expect_enum(value: &ComponentValue) -> Result<&str, String> {
    match value {
        ComponentValue::Enum(name) => Ok(name.as_str()),
        other => Err(format!("expected an enum case, got {other:?}")),
    }
}

fn expect_record(value: &ComponentValue) -> Result<&[(String, ComponentValue)], String> {
    match value {
        ComponentValue::Record(fields) => Ok(fields.as_slice()),
        other => Err(format!("expected a record, got {other:?}")),
    }
}

fn expect_list(value: &ComponentValue) -> Result<&[ComponentValue], String> {
    match value {
        ComponentValue::List(items) => Ok(items.as_slice()),
        other => Err(format!("expected a list, got {other:?}")),
    }
}

fn record_field<'a>(
    fields: &'a [(String, ComponentValue)],
    name: &str,
) -> Result<&'a ComponentValue, String> {
    fields
        .iter()
        .find(|(field_name, _)| field_name == name)
        .map(|(_, value)| value)
        .ok_or_else(|| format!("record is missing field '{name}'"))
}

/// Decodes a WIT `tensor-shape` record (`{ dimensions: list<u64> }`) into a
/// `ShapeDescriptor`.
fn decode_tensor_shape(value: &ComponentValue) -> Result<ShapeDescriptor, String> {
    let fields = expect_record(value)?;
    let dimensions = expect_list(record_field(fields, "dimensions")?)?;
    let mut decoded = Vec::with_capacity(dimensions.len());
    for dimension in dimensions {
        decoded.push(expect_u64(dimension)?);
    }
    Ok(ShapeDescriptor::new(decoded))
}

/// Every tensor this Capability's graphs carry is contiguous `float32` --
/// see `tensor-shape`'s own doc comment in the WIT file for why that is a
/// deliberate, scoped simplification for this first Component, not a
/// completed generic descriptor.
fn descriptor_from_shape(shape: ShapeDescriptor) -> TensorDescriptor {
    TensorDescriptor::new(
        shape,
        DTypeDescriptor::portable(ComputeDType::Float32),
        LayoutDescriptor::Contiguous,
    )
}

fn parse_operator_family(name: &str) -> Result<OperatorFamily, String> {
    match name {
        "Tensor" => Ok(OperatorFamily::Tensor),
        "LinearAlgebra" => Ok(OperatorFamily::LinearAlgebra),
        "Normalization" => Ok(OperatorFamily::Normalization),
        "PositionEncoding" => Ok(OperatorFamily::PositionEncoding),
        "Attention" => Ok(OperatorFamily::Attention),
        "Activation" => Ok(OperatorFamily::Activation),
        "Quantization" => Ok(OperatorFamily::Quantization),
        "Layout" => Ok(OperatorFamily::Layout),
        "SamplingSupport" => Ok(OperatorFamily::SamplingSupport),
        "Control" => Ok(OperatorFamily::Control),
        other => Err(format!("unknown Operator family '{other}'")),
    }
}

/// Decodes a WIT `attribute-value` variant into `OperatorAttributeValue`.
/// The WIT variant deliberately omits `DType`/`Layout` (see the WIT file's
/// doc comment on `attribute-value`) -- neither is reachable here.
fn decode_attribute_value(value: &ComponentValue) -> Result<OperatorAttributeValue, String> {
    match value {
        ComponentValue::Variant(case, payload) => {
            let payload = payload
                .as_deref()
                .ok_or_else(|| format!("attribute-value case '{case}' is missing its payload"))?;
            match case.as_str() {
                "boolean" => match payload {
                    ComponentValue::Bool(value) => Ok(OperatorAttributeValue::Boolean(*value)),
                    other => Err(format!(
                        "attribute-value 'boolean' expected a bool, got {other:?}"
                    )),
                },
                "integer" => match payload {
                    ComponentValue::S64(value) => Ok(OperatorAttributeValue::Integer(*value)),
                    other => Err(format!(
                        "attribute-value 'integer' expected an s64, got {other:?}"
                    )),
                },
                "float-value" => match payload {
                    ComponentValue::F64(value) => Ok(OperatorAttributeValue::Float(*value)),
                    other => Err(format!(
                        "attribute-value 'float-value' expected an f64, got {other:?}"
                    )),
                },
                "text" => Ok(OperatorAttributeValue::String(
                    expect_string(payload)?.to_string(),
                )),
                other => Err(format!("unknown attribute-value case '{other}'")),
            }
        }
        other => Err(format!(
            "expected an attribute-value variant, got {other:?}"
        )),
    }
}

/// Decodes a WIT `attribute` record (`{ name: string, value: attribute-value }`).
fn decode_attribute(value: &ComponentValue) -> Result<(String, OperatorAttributeValue), String> {
    let fields = expect_record(value)?;
    let name = expect_string(record_field(fields, "name")?)?.to_string();
    let value = decode_attribute_value(record_field(fields, "value")?)?;
    Ok((name, value))
}

/// Decodes a WIT `kv-involvement` record.
fn decode_kv_involvement(value: &ComponentValue) -> Result<(String, GraphKvCacheBehavior), String> {
    let fields = expect_record(value)?;
    let resource_id = expect_string(record_field(fields, "resource-id")?)?.to_string();
    let append = match record_field(fields, "append")? {
        ComponentValue::Bool(value) => *value,
        other => {
            return Err(format!(
                "kv-involvement 'append' expected a bool, got {other:?}"
            ));
        }
    };
    let behavior = if append {
        GraphKvCacheBehavior::Append
    } else {
        GraphKvCacheBehavior::Output
    };
    Ok((resource_id, behavior))
}

impl GraphBuilderCapability {
    fn begin_graph(
        &self,
        instance_key: &str,
        arguments: &[ComponentValue],
    ) -> Result<Vec<ComponentValue>, ComponentError> {
        expect_arity("begin-graph", 3, arguments)
            .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?;
        let phase = expect_enum(&arguments[0])
            .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?;
        let phase = match phase {
            "prefill" => ExecutionGraphPhase::Prefill,
            "decode" => ExecutionGraphPhase::Decode,
            other => {
                return Err(rejected(
                    CAPABILITY_NAME,
                    instance_key,
                    format!("begin-graph: unknown phase '{other}'"),
                ));
            }
        };
        let sequence_length = expect_u64(&arguments[1])
            .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?;
        let position_offset = expect_u64(&arguments[2])
            .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?;

        let mut sessions = self.sessions.lock().unwrap();
        let session = sessions.get_mut(instance_key).ok_or_else(|| {
            rejected(
                CAPABILITY_NAME,
                instance_key,
                "no prepared session for this instance",
            )
        })?;
        if session.graph.is_some() {
            return Err(rejected(
                CAPABILITY_NAME,
                instance_key,
                "begin-graph called twice without an intervening finish-graph",
            ));
        }
        let graph_id = ExecutionGraphId::new(format!(
            "{}-{phase:?}-{sequence_length}-{position_offset}",
            session.context.component_id
        ));
        let mut graph = ExecutionGraph::new(graph_id, phase).with_producer(
            ExecutionGraphProducer::ModelComponent {
                component_id: session.context.component_id.clone(),
            },
        );
        graph.model = GraphModelCompatibility::default();
        graph.fingerprint = Some(session.context.compatibility_key.clone());
        session.graph = Some(graph);
        Ok(Vec::new())
    }

    fn declare_input(
        &self,
        instance_key: &str,
        arguments: &[ComponentValue],
    ) -> Result<Vec<ComponentValue>, ComponentError> {
        expect_arity("declare-input", 2, arguments)
            .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?;
        let name = expect_string(&arguments[0])
            .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?;
        let shape = decode_tensor_shape(&arguments[1])
            .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?;

        let mut sessions = self.sessions.lock().unwrap();
        let graph = active_graph(&mut sessions, instance_key)?;
        let edge_id = TensorEdgeId::new(format!("input.{name}"));
        graph.edges.insert(
            edge_id.clone(),
            TensorEdge::new(edge_id.clone(), descriptor_from_shape(shape)),
        );
        Ok(vec![ComponentValue::String(edge_id.as_str().to_string())])
    }

    fn weight_edge(
        &self,
        instance_key: &str,
        arguments: &[ComponentValue],
    ) -> Result<Vec<ComponentValue>, ComponentError> {
        expect_arity("weight-edge", 1, arguments)
            .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?;
        let logical_name = expect_string(&arguments[0])
            .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?;

        let mut sessions = self.sessions.lock().unwrap();
        let session = sessions.get_mut(instance_key).ok_or_else(|| {
            rejected(
                CAPABILITY_NAME,
                instance_key,
                "no prepared session for this instance",
            )
        })?;
        let dimensions = session
            .context
            .weight_shapes
            .get(logical_name)
            .cloned()
            .ok_or_else(|| {
                rejected(
                    CAPABILITY_NAME,
                    instance_key,
                    format!("no weight resource bound for logical name '{logical_name}'"),
                )
            })?;
        let edge_id = TensorEdgeId::new(format!("weight.{logical_name}"));
        let graph = session.graph.as_mut().ok_or_else(|| {
            rejected(
                CAPABILITY_NAME,
                instance_key,
                "weight-edge called before begin-graph",
            )
        })?;
        graph.edges.entry(edge_id.clone()).or_insert_with(|| {
            TensorEdge::new(
                edge_id.clone(),
                descriptor_from_shape(ShapeDescriptor::new(dimensions)),
            )
        });
        Ok(vec![ComponentValue::String(edge_id.as_str().to_string())])
    }

    fn alias_weight_edge(
        &self,
        instance_key: &str,
        arguments: &[ComponentValue],
    ) -> Result<Vec<ComponentValue>, ComponentError> {
        expect_arity("alias-weight-edge", 2, arguments)
            .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?;
        let edge_id = expect_string(&arguments[0])
            .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?;
        let target_logical_name = expect_string(&arguments[1])
            .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?;

        let mut sessions = self.sessions.lock().unwrap();
        let graph = active_graph(&mut sessions, instance_key)?;
        let target_edge_id = TensorEdgeId::new(format!("weight.{target_logical_name}"));
        if !graph.edges.contains_key(&target_edge_id) {
            return Err(rejected(
                CAPABILITY_NAME,
                instance_key,
                format!(
                    "alias-weight-edge: target weight edge '{target_edge_id}' does not exist yet"
                ),
            ));
        }
        let edge = graph
            .edges
            .get_mut(&TensorEdgeId::new(edge_id))
            .ok_or_else(|| {
                rejected(
                    CAPABILITY_NAME,
                    instance_key,
                    format!("alias-weight-edge: edge '{edge_id}' does not exist"),
                )
            })?;
        edge.aliasing = TensorAliasing::MayAlias(target_edge_id);
        Ok(Vec::new())
    }

    fn kv_resource(
        &self,
        instance_key: &str,
        arguments: &[ComponentValue],
    ) -> Result<Vec<ComponentValue>, ComponentError> {
        expect_arity("kv-resource", 2, arguments)
            .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?;
        let layer = expect_u64(&arguments[0])
            .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?;
        let role = expect_enum(&arguments[1])
            .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?;
        let role = match role {
            "k" => "k",
            "v" => "v",
            other => {
                return Err(rejected(
                    CAPABILITY_NAME,
                    instance_key,
                    format!("kv-resource: unknown role '{other}'"),
                ));
            }
        };

        let sessions = self.sessions.lock().unwrap();
        let session = sessions.get(instance_key).ok_or_else(|| {
            rejected(
                CAPABILITY_NAME,
                instance_key,
                "no prepared session for this instance",
            )
        })?;
        let resource_id = format!("{}.layer{layer}.{role}", session.context.kv_namespace);
        Ok(vec![ComponentValue::String(resource_id)])
    }

    fn add_node(
        &self,
        instance_key: &str,
        arguments: &[ComponentValue],
    ) -> Result<Vec<ComponentValue>, ComponentError> {
        expect_arity("add-node", 7, arguments)
            .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?;
        let node_id = expect_string(&arguments[0])
            .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?;
        let operator = expect_string(&arguments[1])
            .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?;
        let family = expect_string(&arguments[2])
            .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?;
        let family = parse_operator_family(family)
            .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?;
        let attribute_values = expect_list(&arguments[3])
            .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?;
        let mut attributes = Vec::with_capacity(attribute_values.len());
        for value in attribute_values {
            attributes.push(
                decode_attribute(value)
                    .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?,
            );
        }
        let input_values = expect_list(&arguments[4])
            .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?;
        let mut inputs = Vec::with_capacity(input_values.len());
        for value in input_values {
            inputs.push(
                expect_string(value)
                    .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?
                    .to_string(),
            );
        }
        let output_shape = decode_tensor_shape(&arguments[5])
            .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?;
        let kv = match &arguments[6] {
            ComponentValue::Option(Some(value)) => Some(
                decode_kv_involvement(value)
                    .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?,
            ),
            ComponentValue::Option(None) => None,
            other => {
                return Err(rejected(
                    CAPABILITY_NAME,
                    instance_key,
                    format!("add-node: expected an option for 'kv', got {other:?}"),
                ));
            }
        };

        let mut sessions = self.sessions.lock().unwrap();
        // Read before `active_graph` takes a mutable borrow of `sessions`.
        let compatibility_key = sessions
            .get(instance_key)
            .map(|session| session.context.compatibility_key.clone())
            .unwrap_or_default();
        let graph = active_graph(&mut sessions, instance_key)?;
        for input in &inputs {
            if !graph.edges.contains_key(&TensorEdgeId::new(input.as_str())) {
                return Err(rejected(
                    CAPABILITY_NAME,
                    instance_key,
                    format!("add-node: input edge '{input}' does not exist"),
                ));
            }
        }
        let node_node_id = ExecutionNodeId::new(node_id);
        if graph.nodes.contains_key(&node_node_id) {
            return Err(rejected(
                CAPABILITY_NAME,
                instance_key,
                format!("add-node: node '{node_id}' already exists"),
            ));
        }
        let output_edge_id = TensorEdgeId::new(format!("edge.{node_id}"));
        let mut node = ExecutionNode::new(
            node_node_id.clone(),
            OperatorId::magnetar(operator, 1, family),
        );
        for input in &inputs {
            node = node.with_input(TensorEdgeId::new(input.as_str()));
        }
        node = node.with_output(output_edge_id.clone());
        for (name, value) in attributes {
            node = node.with_attribute(name, value);
        }
        let mut output_edge =
            TensorEdge::new(output_edge_id.clone(), descriptor_from_shape(output_shape))
                .with_producer(node_node_id.clone());
        if let Some((resource_id, behavior)) = kv {
            output_edge.kv_cache = Some(GraphKvCacheMetadata {
                cache_id: resource_id,
                behavior,
                paged: false,
                compatibility_key,
            });
        }
        for input in &inputs {
            if let Some(edge) = graph.edges.get_mut(&TensorEdgeId::new(input.as_str())) {
                edge.consumers.insert(node_node_id.clone());
            }
        }
        graph.nodes.insert(node_node_id, node);
        graph.edges.insert(output_edge_id.clone(), output_edge);
        Ok(vec![ComponentValue::String(
            output_edge_id.as_str().to_string(),
        )])
    }

    fn finish_graph(
        &self,
        instance_key: &str,
        arguments: &[ComponentValue],
    ) -> Result<Vec<ComponentValue>, ComponentError> {
        expect_arity("finish-graph", 1, arguments)
            .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?;
        let output_edge_id = expect_string(&arguments[0])
            .map_err(|message| rejected(CAPABILITY_NAME, instance_key, message))?;

        let mut sessions = self.sessions.lock().unwrap();
        let session = sessions.get_mut(instance_key).ok_or_else(|| {
            rejected(
                CAPABILITY_NAME,
                instance_key,
                "no prepared session for this instance",
            )
        })?;
        let mut graph = session.graph.take().ok_or_else(|| {
            rejected(
                CAPABILITY_NAME,
                instance_key,
                "finish-graph called before begin-graph",
            )
        })?;
        let output_edge_id = TensorEdgeId::new(output_edge_id);
        let output_name = TensorEdgeId::new(session.context.output_edge_name.clone());
        // Rename (not duplicate) the Component-named output edge to the
        // fixed id the existing execution path expects, so the graph has
        // exactly one edge under that id, with its producer node's own
        // `outputs` list updated to match -- a plain second edge sharing
        // the same producer would leave that node claiming two outputs it
        // does not really have.
        let mut edge = graph.edges.remove(&output_edge_id).ok_or_else(|| {
            rejected(
                CAPABILITY_NAME,
                instance_key,
                format!("finish-graph: output edge '{output_edge_id}' does not exist"),
            )
        })?;
        if output_edge_id != output_name {
            if let Some(producer) = &edge.producer
                && let Some(node) = graph.nodes.get_mut(producer)
            {
                for output in &mut node.outputs {
                    if *output == output_edge_id {
                        *output = output_name.clone();
                    }
                }
            }
            edge.id = output_name.clone();
        }
        graph.edges.insert(output_name, edge);
        // Component-produced graphs remain untrusted until validated
        // (Requirement "Component-Produced Graphs Remain Untrusted Until
        // Validated"): this capability's own per-call checks above catch
        // structural problems (a missing edge, a duplicate node id) as
        // they happen, but never checked the *finished* graph against the
        // portable Operator catalog -- topology, Operator identity/arity,
        // and attribute schema compliance. `initial_operator_catalog` is
        // the same generic, Qwen-agnostic catalog `magnetar-runtime`'s own
        // graph execution already validates against elsewhere.
        let catalog = initial_operator_catalog();
        graph.validate(&catalog).map_err(|error| {
            rejected(
                CAPABILITY_NAME,
                instance_key,
                format!("finish-graph: graph failed Runtime validation: {error}"),
            )
        })?;
        let handle = format!("{instance_key}#{}", session.next_handle);
        session.next_handle += 1;
        session.finished.insert(handle.clone(), graph);
        Ok(vec![ComponentValue::String(handle)])
    }
}

fn active_graph<'a>(
    sessions: &'a mut BTreeMap<String, Session>,
    instance_key: &str,
) -> Result<&'a mut ExecutionGraph, ComponentError> {
    sessions
        .get_mut(instance_key)
        .ok_or_else(|| {
            rejected(
                CAPABILITY_NAME,
                instance_key,
                "no prepared session for this instance",
            )
        })?
        .graph
        .as_mut()
        .ok_or_else(|| {
            rejected(
                CAPABILITY_NAME,
                instance_key,
                "no graph under construction (call begin-graph first)",
            )
        })
}

impl HostCapability for GraphBuilderCapability {
    fn call(
        &self,
        instance_key: &str,
        operation: &str,
        arguments: &[ComponentValue],
    ) -> Result<Vec<ComponentValue>, ComponentError> {
        match operation {
            "begin-graph" => self.begin_graph(instance_key, arguments),
            "declare-input" => self.declare_input(instance_key, arguments),
            "weight-edge" => self.weight_edge(instance_key, arguments),
            "alias-weight-edge" => self.alias_weight_edge(instance_key, arguments),
            "kv-resource" => self.kv_resource(instance_key, arguments),
            "add-node" => self.add_node(instance_key, arguments),
            "finish-graph" => self.finish_graph(instance_key, arguments),
            other => Err(rejected(
                CAPABILITY_NAME,
                instance_key,
                format!("unknown graph-builder operation '{other}'"),
            )),
        }
    }
}

#[cfg(test)]
mod tests;
