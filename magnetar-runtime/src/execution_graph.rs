//! Runtime-owned Execution Graph contract.
//!
//! An Execution Graph is a portable inference graph made of Operators and
//! logical tensor/resource edges. Components and architecture implementations
//! may produce graphs, but Runtime validation and planning are mandatory before
//! any Provider dispatch.

use crate::{
    DTypeDescriptor, MemoryAllocationClass, OperatorAttributeValue, OperatorCatalog, OperatorError,
    OperatorId, OperatorObservationKind, ResourceAffinity, TensorDescriptor, TensorLayoutKind,
    initial_operator_catalog, layout_kind, validate_affinity_compatibility,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ExecutionGraphId(String);

impl ExecutionGraphId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ExecutionGraphId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ExecutionGraphVersion(pub u32);

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TensorEdgeId(String);

impl TensorEdgeId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TensorEdgeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ExecutionNodeId(String);

impl ExecutionNodeId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ExecutionNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ExecutionGraphPhase {
    ModelLoad,
    Warmup,
    Prefill,
    Decode,
    AdapterActivation,
    AdapterMerge,
    SamplingHelper,
    Test,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionGraphProducer {
    RuntimeNative,
    ModelComponent { component_id: String },
    ProviderAssistedBuilder { provider_family: String },
    TestFixture { fixture: String },
}

impl ExecutionGraphProducer {
    pub const fn component_has_raw_provider_access(&self) -> bool {
        false
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TensorMutability {
    Immutable,
    Mutable,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TensorLifetimeHint {
    Operator,
    Graph,
    Phase,
    Session,
    Runtime,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TensorAliasing {
    None,
    MayAlias(TensorEdgeId),
    MustAlias(TensorEdgeId),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TensorResidencyConstraint {
    Host,
    Device,
    BrowserLinearMemory,
    ProviderOwnedOpaque,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorEdge {
    pub id: TensorEdgeId,
    pub logical_tensor_id: String,
    pub descriptor: TensorDescriptor,
    pub memory_class: MemoryAllocationClass,
    pub residency: TensorResidencyConstraint,
    pub affinity: Option<ResourceAffinity>,
    pub mutability: TensorMutability,
    pub lifetime: TensorLifetimeHint,
    pub aliasing: TensorAliasing,
    pub producer: Option<ExecutionNodeId>,
    pub consumers: BTreeSet<ExecutionNodeId>,
    pub kv_cache: Option<GraphKvCacheMetadata>,
    pub prefix_cache: Option<GraphPrefixCacheMetadata>,
}

impl TensorEdge {
    pub fn new(id: TensorEdgeId, descriptor: TensorDescriptor) -> Self {
        Self {
            logical_tensor_id: id.as_str().into(),
            id,
            descriptor,
            memory_class: MemoryAllocationClass::Tensor,
            residency: TensorResidencyConstraint::Host,
            affinity: None,
            mutability: TensorMutability::Immutable,
            lifetime: TensorLifetimeHint::Graph,
            aliasing: TensorAliasing::None,
            producer: None,
            consumers: BTreeSet::new(),
            kv_cache: None,
            prefix_cache: None,
        }
    }

    pub fn with_affinity(mut self, affinity: ResourceAffinity) -> Self {
        self.affinity = Some(affinity);
        self
    }

    pub fn with_producer(mut self, producer: ExecutionNodeId) -> Self {
        self.producer = Some(producer);
        self
    }

    pub fn with_consumer(mut self, consumer: ExecutionNodeId) -> Self {
        self.consumers.insert(consumer);
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionNode {
    pub id: ExecutionNodeId,
    pub operator: OperatorId,
    pub attributes: BTreeMap<String, OperatorAttributeValue>,
    pub inputs: Vec<TensorEdgeId>,
    pub outputs: Vec<TensorEdgeId>,
    pub resource_affinity: Option<ResourceAffinity>,
}

impl ExecutionNode {
    pub fn new(id: ExecutionNodeId, operator: OperatorId) -> Self {
        Self {
            id,
            operator,
            attributes: BTreeMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            resource_affinity: None,
        }
    }

    pub fn with_input(mut self, edge: TensorEdgeId) -> Self {
        self.inputs.push(edge);
        self
    }

    pub fn with_output(mut self, edge: TensorEdgeId) -> Self {
        self.outputs.push(edge);
        self
    }

    pub fn with_attribute(
        mut self,
        name: impl Into<String>,
        value: OperatorAttributeValue,
    ) -> Self {
        self.attributes.insert(name.into(), value);
        self
    }

    pub fn with_affinity(mut self, affinity: ResourceAffinity) -> Self {
        self.resource_affinity = Some(affinity);
        self
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GraphModelCompatibility {
    pub model_instance_id: Option<String>,
    pub architecture: Option<String>,
    pub tokenizer_dependency: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GraphAdapterMetadata {
    pub active_adapter_set: Option<String>,
    pub overlay_paths: Vec<String>,
    pub merge_graph: Option<ExecutionGraphId>,
    pub provider_fused_path: Option<String>,
    pub invalidates_dependent_caches: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphKvCacheBehavior {
    Input,
    Output,
    Append,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphKvCacheMetadata {
    pub cache_id: String,
    pub behavior: GraphKvCacheBehavior,
    pub paged: bool,
    pub compatibility_key: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphPrefixCacheMetadata {
    pub reused_prefix_length: u64,
    pub backing_kv_cache: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionGraph {
    pub id: ExecutionGraphId,
    pub version: ExecutionGraphVersion,
    pub phase: ExecutionGraphPhase,
    pub producer: ExecutionGraphProducer,
    pub model: GraphModelCompatibility,
    pub adapter: GraphAdapterMetadata,
    pub fingerprint: Option<String>,
    pub nodes: BTreeMap<ExecutionNodeId, ExecutionNode>,
    pub edges: BTreeMap<TensorEdgeId, TensorEdge>,
}

impl ExecutionGraph {
    pub fn new(id: ExecutionGraphId, phase: ExecutionGraphPhase) -> Self {
        Self {
            id,
            version: ExecutionGraphVersion(1),
            phase,
            producer: ExecutionGraphProducer::RuntimeNative,
            model: GraphModelCompatibility::default(),
            adapter: GraphAdapterMetadata::default(),
            fingerprint: None,
            nodes: BTreeMap::new(),
            edges: BTreeMap::new(),
        }
    }

    pub fn with_producer(mut self, producer: ExecutionGraphProducer) -> Self {
        self.producer = producer;
        self
    }

    pub fn with_node(mut self, node: ExecutionNode) -> Self {
        self.nodes.insert(node.id.clone(), node);
        self
    }

    pub fn with_edge(mut self, edge: TensorEdge) -> Self {
        self.edges.insert(edge.id.clone(), edge);
        self
    }

    pub fn identity_key(&self) -> String {
        format!(
            "{}@{}:{:?}:{:?}:{:?}:{}",
            self.id,
            self.version.0,
            self.phase,
            self.model,
            self.adapter,
            self.fingerprint.as_deref().unwrap_or("no-fingerprint")
        )
    }

    pub fn validate(&self, catalog: &OperatorCatalog) -> Result<GraphValidationReport, GraphError> {
        validate_execution_graph(self, catalog)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphValidationReport {
    pub graph: ExecutionGraphId,
    pub validated_nodes: usize,
    pub validated_edges: usize,
    pub observations: Vec<GraphObservation>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphPlanningPolicy {
    pub allow_data_movement: bool,
    pub allow_dtype_conversion: bool,
    pub allow_layout_conversion: bool,
    pub browser_target: bool,
}

impl Default for GraphPlanningPolicy {
    fn default() -> Self {
        Self {
            allow_data_movement: false,
            allow_dtype_conversion: true,
            allow_layout_conversion: true,
            browser_target: cfg!(target_arch = "wasm32"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphPlanStep {
    ExecuteOperator(ExecutionNodeId),
    InsertDataMovement { edge: TensorEdgeId },
    InsertDTypeConversion { edge: TensorEdgeId },
    InsertLayoutConversion { edge: TensorEdgeId },
    RequestWorkspace { node: ExecutionNodeId },
    PreserveKvCache { edge: TensorEdgeId },
    AdjustPrefillForPrefix { reused_prefix_length: u64 },
    KernelSelectionPlaceholder { node: ExecutionNodeId },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionGraphPlan {
    pub graph: ExecutionGraphId,
    pub phase: ExecutionGraphPhase,
    pub execution_order: Vec<ExecutionNodeId>,
    pub steps: Vec<GraphPlanStep>,
    pub observations: Vec<GraphObservation>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum GraphObservationKind {
    GraphCreated,
    GraphValidationStarted,
    GraphValidationFailed,
    GraphValidated,
    GraphPlanningStarted,
    GraphPlanningCompleted,
    GraphExecutionStarted,
    GraphExecutionCompleted,
    GraphExecutionFailed,
    Operator(OperatorObservationKind),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphObservation {
    pub kind: GraphObservationKind,
    pub graph: ExecutionGraphId,
    pub node: Option<ExecutionNodeId>,
    pub redacted_metadata: BTreeMap<String, String>,
}

impl GraphObservation {
    pub fn new(kind: GraphObservationKind, graph: ExecutionGraphId) -> Self {
        Self {
            kind,
            graph,
            node: None,
            redacted_metadata: BTreeMap::new(),
        }
    }

    pub fn with_node(mut self, node: ExecutionNodeId) -> Self {
        self.node = Some(node);
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphError {
    Validation(OperatorError),
    Planning(OperatorError),
    Execution(OperatorError),
    TensorEdgeMissing(TensorEdgeId),
    NodeMissing(ExecutionNodeId),
    DuplicateProducer(TensorEdgeId),
    AliasingInvalid(TensorEdgeId),
    LifecycleInvalid(String),
    ProviderCapabilityUnavailable(String),
    PolicyRejected(String),
    BrowserFeatureUnsupported(String),
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Validation(error) => write!(f, "graph validation failed: {error}"),
            Self::Planning(error) => write!(f, "graph planning failed: {error}"),
            Self::Execution(error) => write!(f, "graph execution failed: {error}"),
            Self::TensorEdgeMissing(edge) => write!(f, "tensor edge missing: {edge}"),
            Self::NodeMissing(node) => write!(f, "node missing: {node}"),
            Self::DuplicateProducer(edge) => write!(f, "duplicate producer for edge: {edge}"),
            Self::AliasingInvalid(edge) => write!(f, "invalid aliasing for edge: {edge}"),
            Self::LifecycleInvalid(reason) => write!(f, "lifecycle invalid: {reason}"),
            Self::ProviderCapabilityUnavailable(reason) => {
                write!(f, "provider capability unavailable: {reason}")
            }
            Self::PolicyRejected(reason) => write!(f, "policy rejected: {reason}"),
            Self::BrowserFeatureUnsupported(feature) => {
                write!(f, "browser feature unsupported: {feature}")
            }
        }
    }
}

impl Error for GraphError {}

pub fn validate_execution_graph(
    graph: &ExecutionGraph,
    catalog: &OperatorCatalog,
) -> Result<GraphValidationReport, GraphError> {
    if graph.id.as_str().is_empty() {
        return Err(GraphError::LifecycleInvalid(
            "graph identity is empty".into(),
        ));
    }
    if graph.version.0 == 0 {
        return Err(GraphError::LifecycleInvalid("graph version is zero".into()));
    }
    if graph.producer.component_has_raw_provider_access() {
        return Err(GraphError::PolicyRejected(
            "component graph producer has raw Provider access".into(),
        ));
    }

    let mut observations = vec![GraphObservation::new(
        GraphObservationKind::GraphValidationStarted,
        graph.id.clone(),
    )];

    for edge in graph.edges.values() {
        if matches!(edge.aliasing, super::execution_graph::TensorAliasing::MayAlias(ref aliased) | super::execution_graph::TensorAliasing::MustAlias(ref aliased) if !graph.edges.contains_key(aliased))
        {
            return Err(GraphError::AliasingInvalid(edge.id.clone()));
        }
        if let Some(producer) = &edge.producer
            && !graph.nodes.contains_key(producer)
        {
            return Err(GraphError::NodeMissing(producer.clone()));
        }
        for consumer in &edge.consumers {
            if !graph.nodes.contains_key(consumer) {
                return Err(GraphError::NodeMissing(consumer.clone()));
            }
        }
        if let Some(prefix) = &edge.prefix_cache
            && prefix.backing_kv_cache.is_empty()
        {
            return Err(GraphError::LifecycleInvalid(
                "prefix cache metadata requires backing KV cache reference".into(),
            ));
        }
    }

    // `node.inputs`/`node.outputs` are the authoritative topology
    // (Correctif 6): derive the true producer/consumer map from them and
    // reject a graph whose separately-settable `TensorEdge::producer`/
    // `::consumers` metadata was populated but disagrees with it, rather
    // than letting two topology representations silently diverge. A graph
    // producer that leaves `producer`/`consumers` unpopulated (the common
    // case today) is unaffected -- only an explicitly wrong value is
    // rejected. `derive_edge_producers` also rejects an edge that two
    // different nodes both list as an output.
    let derived_producers = derive_edge_producers(graph)?;
    for edge in graph.edges.values() {
        if let Some(declared) = &edge.producer {
            match derived_producers.get(&edge.id) {
                Some(derived) if *derived == declared => {}
                _ => {
                    return Err(GraphError::LifecycleInvalid(format!(
                        "edge '{}' declares producer '{declared}', which does not match any node's outputs",
                        edge.id
                    )));
                }
            }
        }
    }
    let derived_consumers = derive_edge_consumers(graph);
    for edge in graph.edges.values() {
        if edge.consumers.is_empty() {
            continue;
        }
        let derived: BTreeSet<&ExecutionNodeId> =
            derived_consumers.get(&edge.id).cloned().unwrap_or_default();
        let declared: BTreeSet<&ExecutionNodeId> = edge.consumers.iter().collect();
        if declared != derived {
            return Err(GraphError::LifecycleInvalid(format!(
                "edge '{}' declares consumers that do not match any node's inputs",
                edge.id
            )));
        }
    }
    // Reject a cycle (or a node input with no resolvable producer among
    // remaining unscheduled nodes) as part of validation itself, not only
    // when a caller happens to also plan the graph.
    topological_order(graph)?;

    for node in graph.nodes.values() {
        let spec = catalog
            .get(&node.operator)
            .map_err(GraphError::Validation)?;
        let inputs = node
            .inputs
            .iter()
            .map(|id| {
                graph
                    .edges
                    .get(id)
                    .map(|edge| edge.descriptor.clone())
                    .ok_or_else(|| GraphError::TensorEdgeMissing(id.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = node
            .outputs
            .iter()
            .map(|id| {
                graph
                    .edges
                    .get(id)
                    .map(|edge| edge.descriptor.clone())
                    .ok_or_else(|| GraphError::TensorEdgeMissing(id.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        spec.validate_invocation(&inputs, &outputs, &node.attributes)
            .map_err(GraphError::Validation)?;
        validate_node_affinity(node, graph)?;
        observations.push(
            GraphObservation::new(
                GraphObservationKind::Operator(OperatorObservationKind::OperatorPlanned),
                graph.id.clone(),
            )
            .with_node(node.id.clone()),
        );
    }

    observations.push(GraphObservation::new(
        GraphObservationKind::GraphValidated,
        graph.id.clone(),
    ));
    Ok(GraphValidationReport {
        graph: graph.id.clone(),
        validated_nodes: graph.nodes.len(),
        validated_edges: graph.edges.len(),
        observations,
    })
}

pub fn plan_execution_graph(
    graph: &ExecutionGraph,
    catalog: &OperatorCatalog,
    policy: &GraphPlanningPolicy,
    planned_affinity: Option<&ResourceAffinity>,
) -> Result<ExecutionGraphPlan, GraphError> {
    graph.validate(catalog)?;
    let mut steps = Vec::new();
    let mut observations = vec![GraphObservation::new(
        GraphObservationKind::GraphPlanningStarted,
        graph.id.clone(),
    )];
    let order = topological_order(graph)?;

    for edge in graph.edges.values() {
        if let Some(target) = planned_affinity
            && let Some(edge_affinity) = &edge.affinity
            && validate_affinity_compatibility(edge_affinity, target).is_err()
        {
            if policy.allow_data_movement {
                steps.push(GraphPlanStep::InsertDataMovement {
                    edge: edge.id.clone(),
                });
                observations.push(GraphObservation::new(
                    GraphObservationKind::Operator(OperatorObservationKind::DataMovementInserted),
                    graph.id.clone(),
                ));
            } else {
                return Err(GraphError::Planning(
                    OperatorError::ResourceAffinityConflict {
                        reason: "silent data movement is forbidden".into(),
                    },
                ));
            }
        }
        if edge.kv_cache.is_some() {
            steps.push(GraphPlanStep::PreserveKvCache {
                edge: edge.id.clone(),
            });
        }
        if let Some(prefix) = &edge.prefix_cache {
            steps.push(GraphPlanStep::AdjustPrefillForPrefix {
                reused_prefix_length: prefix.reused_prefix_length,
            });
        }
        if policy.browser_target && !browser_layout_supported(layout_kind(&edge.descriptor.layout))
        {
            return Err(GraphError::BrowserFeatureUnsupported(format!(
                "{:?}",
                layout_kind(&edge.descriptor.layout)
            )));
        }
    }

    for node_id in &order {
        let node = graph
            .nodes
            .get(node_id)
            .ok_or_else(|| GraphError::NodeMissing(node_id.clone()))?;
        let spec = catalog.get(&node.operator).map_err(GraphError::Planning)?;
        for edge_id in node.inputs.iter().chain(&node.outputs) {
            let edge = graph
                .edges
                .get(edge_id)
                .ok_or_else(|| GraphError::TensorEdgeMissing(edge_id.clone()))?;
            let layout = layout_kind(&edge.descriptor.layout);
            if !spec.layout_contract.accepts(layout) {
                if policy.allow_layout_conversion {
                    steps.push(GraphPlanStep::InsertLayoutConversion {
                        edge: edge.id.clone(),
                    });
                    observations.push(GraphObservation::new(
                        GraphObservationKind::Operator(
                            OperatorObservationKind::LayoutConversionInserted,
                        ),
                        graph.id.clone(),
                    ));
                } else {
                    return Err(GraphError::Planning(
                        OperatorError::LayoutConversionRequired {
                            from: layout,
                            to: spec
                                .layout_contract
                                .supported
                                .iter()
                                .next()
                                .copied()
                                .unwrap_or(TensorLayoutKind::Contiguous),
                        },
                    ));
                }
            }
            if matches!(
                edge.descriptor.dtype,
                DTypeDescriptor::ProviderSpecific { .. }
            ) {
                if policy.allow_dtype_conversion {
                    steps.push(GraphPlanStep::InsertDTypeConversion {
                        edge: edge.id.clone(),
                    });
                    observations.push(GraphObservation::new(
                        GraphObservationKind::Operator(
                            OperatorObservationKind::DTypeConversionInserted,
                        ),
                        graph.id.clone(),
                    ));
                } else {
                    return Err(GraphError::Planning(
                        OperatorError::DTypeConversionRequired {
                            from: crate::ComputeDType::UInt8,
                            to: crate::ComputeDType::Float32,
                        },
                    ));
                }
            }
        }
        if spec.memory.requires_workspace {
            steps.push(GraphPlanStep::RequestWorkspace {
                node: node.id.clone(),
            });
            observations.push(
                GraphObservation::new(
                    GraphObservationKind::Operator(OperatorObservationKind::WorkspaceRequested),
                    graph.id.clone(),
                )
                .with_node(node.id.clone()),
            );
        }
        steps.push(GraphPlanStep::KernelSelectionPlaceholder {
            node: node.id.clone(),
        });
        steps.push(GraphPlanStep::ExecuteOperator(node.id.clone()));
    }

    observations.push(GraphObservation::new(
        GraphObservationKind::GraphPlanningCompleted,
        graph.id.clone(),
    ));
    Ok(ExecutionGraphPlan {
        graph: graph.id.clone(),
        phase: graph.phase,
        execution_order: order,
        steps,
        observations,
    })
}

pub fn execute_graph_boundary(
    graph: &ExecutionGraph,
    catalog: &OperatorCatalog,
    policy: &GraphPlanningPolicy,
) -> Result<Vec<GraphObservation>, GraphError> {
    let plan = plan_execution_graph(graph, catalog, policy, None)?;
    let mut observations = vec![GraphObservation::new(
        GraphObservationKind::GraphExecutionStarted,
        graph.id.clone(),
    )];
    observations.extend(plan.observations);
    observations.push(GraphObservation::new(
        GraphObservationKind::GraphExecutionCompleted,
        graph.id.clone(),
    ));
    Ok(observations)
}

pub fn default_graph_catalog() -> OperatorCatalog {
    initial_operator_catalog()
}

fn validate_node_affinity(node: &ExecutionNode, graph: &ExecutionGraph) -> Result<(), GraphError> {
    let Some(node_affinity) = &node.resource_affinity else {
        return Ok(());
    };
    for edge_id in node.inputs.iter().chain(&node.outputs) {
        let edge = graph
            .edges
            .get(edge_id)
            .ok_or_else(|| GraphError::TensorEdgeMissing(edge_id.clone()))?;
        if let Some(edge_affinity) = &edge.affinity {
            validate_affinity_compatibility(node_affinity, edge_affinity)
                .map_err(GraphError::Validation)?;
        }
    }
    Ok(())
}

/// Derives each edge's producing node from `node.outputs` -- the
/// authoritative topology (Correctif 6) -- rather than the separately
/// settable `TensorEdge::producer` field, which a graph producer MAY leave
/// unpopulated or let drift out of sync with the node list.
fn derive_edge_producers(
    graph: &ExecutionGraph,
) -> Result<BTreeMap<&TensorEdgeId, &ExecutionNodeId>, GraphError> {
    let mut producers: BTreeMap<&TensorEdgeId, &ExecutionNodeId> = BTreeMap::new();
    for (node_id, node) in &graph.nodes {
        for output in &node.outputs {
            if let Some(existing) = producers.insert(output, node_id)
                && existing != node_id
            {
                return Err(GraphError::DuplicateProducer(output.clone()));
            }
        }
    }
    Ok(producers)
}

/// Derives each edge's consuming nodes from `node.inputs` -- the
/// authoritative topology (Correctif 6) -- rather than the separately
/// settable `TensorEdge::consumers` field.
fn derive_edge_consumers(
    graph: &ExecutionGraph,
) -> BTreeMap<&TensorEdgeId, BTreeSet<&ExecutionNodeId>> {
    let mut consumers: BTreeMap<&TensorEdgeId, BTreeSet<&ExecutionNodeId>> = BTreeMap::new();
    for (node_id, node) in &graph.nodes {
        for input in &node.inputs {
            consumers.entry(input).or_default().insert(node_id);
        }
    }
    consumers
}

fn topological_order(graph: &ExecutionGraph) -> Result<Vec<ExecutionNodeId>, GraphError> {
    let edge_producers = derive_edge_producers(graph)?;
    let mut remaining = graph.nodes.keys().cloned().collect::<BTreeSet<_>>();
    let mut order = Vec::with_capacity(remaining.len());
    while !remaining.is_empty() {
        let ready = remaining
            .iter()
            .find(|node_id| {
                let node = &graph.nodes[*node_id];
                node.inputs.iter().all(|edge_id| {
                    edge_producers
                        .get(edge_id)
                        .is_none_or(|producer| !remaining.contains(*producer))
                })
            })
            .cloned();
        let Some(node_id) = ready else {
            return Err(GraphError::LifecycleInvalid(
                "graph contains a cycle or unresolved producer".into(),
            ));
        };
        remaining.remove(&node_id);
        order.push(node_id);
    }
    Ok(order)
}

fn browser_layout_supported(layout: TensorLayoutKind) -> bool {
    matches!(
        layout,
        TensorLayoutKind::Contiguous
            | TensorLayoutKind::Strided
            | TensorLayoutKind::BrowserCompatible
    )
}

#[cfg(test)]
mod tests;
