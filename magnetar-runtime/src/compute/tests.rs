//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::affinity::{
    CapabilityBinding, CapabilityHealth, FallbackClass, HealthState, ProviderBinding,
    ProviderHealth, ProviderStatusSnapshot, ResourceAffinity,
};
use crate::capability::{Capability, CapabilityId};
use crate::component::WitInterface;
use crate::device::Device;
use crate::kernel::KernelAdvertisement;
use crate::planning::{
    ComputeExecutionClassification, ExecutionStepKind, MemoryPlanningDecision, MemoryRegionKind,
};
use crate::provider::{
    Provider, ProviderError, ProviderExecutionApi, ProviderMetadata, ProviderRegistry,
};
use crate::runtime::Runtime;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

fn provider_with_capabilities(
    name: &str,
    capabilities: impl IntoIterator<Item = Capability>,
) -> TestProvider {
    let mut provider = TestProvider::new(name);
    provider.metadata.capabilities.extend(capabilities);
    provider
}

struct TestProvider {
    metadata: ProviderMetadata,
    initialized: AtomicBool,
    shut_down: AtomicBool,
    fail_initialization: bool,
    health: ProviderHealth,
    status_snapshot: Option<ProviderStatusSnapshot>,
    capability_health: BTreeMap<CapabilityId, HealthState>,
    devices: Vec<Arc<dyn Device>>,
    execution_api: Option<Arc<dyn ProviderExecutionApi>>,
    kernel_advertisements: Vec<KernelAdvertisement>,
}

impl TestProvider {
    fn new(name: &str) -> Self {
        Self {
            metadata: ProviderMetadata::new(name, "1", "test", "test"),
            initialized: AtomicBool::new(false),
            shut_down: AtomicBool::new(false),
            fail_initialization: false,
            health: ProviderHealth::Available,
            status_snapshot: None,
            capability_health: BTreeMap::new(),
            devices: Vec::new(),
            execution_api: None,
            kernel_advertisements: Vec::new(),
        }
    }
}

impl Provider for TestProvider {
    fn metadata(&self) -> ProviderMetadata {
        self.metadata.clone()
    }
    fn kernel_advertisements(&self) -> Vec<KernelAdvertisement> {
        self.kernel_advertisements.clone()
    }
    fn register(&self, _registry: &mut ProviderRegistry) -> Result<(), ProviderError> {
        Ok(())
    }
    fn health(&self) -> ProviderHealth {
        self.health
    }
    fn status_snapshot(&self) -> ProviderStatusSnapshot {
        self.status_snapshot
            .clone()
            .unwrap_or_else(|| ProviderStatusSnapshot::from_health_report(self.health_report()))
    }
    fn capability_health(&self, capability: &CapabilityBinding) -> Option<CapabilityHealth> {
        Some(CapabilityHealth::new(
            ProviderBinding::new(&self.metadata.name),
            capability.clone(),
            self.capability_health
                .get(capability.id())
                .copied()
                .unwrap_or(self.health),
        ))
    }
    fn initialize(&self) -> Result<(), ProviderError> {
        if self.fail_initialization {
            return Err(ProviderError::Lifecycle("unavailable".into()));
        }
        self.initialized.store(true, Ordering::SeqCst);
        Ok(())
    }
    fn devices(&self) -> Vec<Arc<dyn Device>> {
        self.devices.clone()
    }
    fn shutdown(&self) -> Result<(), ProviderError> {
        self.shut_down.store(true, Ordering::SeqCst);
        Ok(())
    }
    fn execution_api(&self) -> Option<Arc<dyn ProviderExecutionApi>> {
        self.execution_api.clone()
    }
}

#[test]
fn compute_capability_has_the_canonical_wit_contract() {
    let compute = compute_capability();
    assert_eq!(compute.id.as_str(), COMPUTE_CAPABILITY_ID);
    assert_eq!(compute.version, COMPUTE_CAPABILITY_VERSION);
    assert_eq!(COMPUTE_WIT_PACKAGE, "magnetar:compute");
    assert_eq!(
        compute.descriptor.contracts,
        BTreeSet::from([WitInterface::new(COMPUTE_WIT_INTERFACE, "2.0.0")])
    );
}

#[test]
fn compute_operation_catalog_defines_stable_family_metadata() {
    let families = ComputeOperationFamily::ALL
        .into_iter()
        .map(|family| family.id())
        .collect::<BTreeSet<_>>();

    assert_eq!(families.len(), 11);
    assert!(families.contains("descriptor-and-view"));
    assert!(families.contains("construction-and-allocation"));
    assert!(families.contains("data-movement-and-conversion"));
    assert!(families.contains("elementwise"));
    assert!(families.contains("comparison-and-selection"));
    assert!(families.contains("reduction"));
    assert!(families.contains("linear-algebra"));
    assert!(families.contains("convolution-and-spatial-transform"));
    assert!(families.contains("indexing-and-update"));
    assert!(families.contains("random-generation"));
    assert!(families.contains("synchronization-and-completion"));

    let metadata = ComputeOperationFamily::LinearAlgebra.metadata();
    assert_eq!(metadata.family, ComputeOperationFamily::LinearAlgebra);
    assert_eq!(metadata.scope, "matrix and batched matrix operations");
    assert!(metadata.examples.contains(&"matmul"));
    assert_eq!(
        ComputeOperationFamily::from_id("elementwise"),
        Some(ComputeOperationFamily::Elementwise)
    );
    assert_eq!(ComputeOperationFamily::from_id("autograd"), None);
}

#[test]
fn compute_operation_validation_uses_provider_advertisements() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32, ComputeDType::Float64])
            .with_layouts([ComputeLayout::Dense])
            .with_precision_modes([ComputePrecision::Default, ComputePrecision::Reduced]),
    );
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::LinearAlgebra,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense])
            .with_precision_modes([ComputePrecision::Default]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();

    runtime
        .validate_compute_operations(
            "portable-compute",
            &[
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                    .with_dtype(ComputeDType::Float32)
                    .with_layout(ComputeLayout::Dense)
                    .with_precision(ComputePrecision::Reduced),
                ComputeOperationDescriptor::new(ComputeOperationFamily::LinearAlgebra)
                    .with_dtype(ComputeDType::Float32)
                    .with_layout(ComputeLayout::Dense),
            ],
        )
        .unwrap();

    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[ComputeOperationDescriptor::new(
                ComputeOperationFamily::Reduction
            )],
        ),
        Err(ComputeValidationError::UnsupportedOperationFamily { .. })
    ));
    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                    .with_dtype(ComputeDType::UInt8)
            ],
        ),
        Err(ComputeValidationError::UnsupportedDType { .. })
    ));
    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                    .with_layout(ComputeLayout::Strided)
            ],
        ),
        Err(ComputeValidationError::UnsupportedLayout { .. })
    ));
}

#[test]
fn provider_compute_advertisement_drives_operation_validation() {
    let schemas = initial_compute_operation_schemas();
    let add = schemas
        .get(&ComputeOperationId::new("elementwise.binary.add"))
        .unwrap();
    let mut provider = provider_with_capabilities("advertised-compute", [compute_capability()]);
    provider.metadata.compute_advertisement = ProviderComputeAdvertisement::new()
        .with_capability(
            ComputeCapabilitySupport::default()
                .with_versions([COMPUTE_CAPABILITY_VERSION])
                .with_operation_catalog_revision("initial")
                .with_operation_schema_revision("initial"),
        )
        .with_operation_schema(OperationSchemaSupport::from_operation_support(
            add.id.clone(),
            add.family,
            ComputeOperationSupport::new()
                .with_dtypes([ComputeDType::Float32])
                .with_layouts([ComputeLayout::Dense])
                .with_precision_modes([ComputePrecision::Default]),
        ));
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let tensor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );

    runtime
        .validate_compute_operations(
            "advertised-compute",
            &[ComputeOperationDescriptor::from_schema(add)
                .with_tensor(tensor.clone())
                .with_tensor(tensor.clone())
                .with_tensor(tensor)],
        )
        .unwrap();
}

#[test]
fn compute_operation_schema_validation_checks_attributes_and_shapes() {
    let schemas = initial_compute_operation_schemas();
    let add = schemas
        .get(&ComputeOperationId::new("elementwise.binary.add"))
        .unwrap();
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_schema_support.insert(
        add.id.clone(),
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let lhs = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 1]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let rhs = TensorDescriptor::materialized(
        ShapeDescriptor::new([1, 3]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let output = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 3]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );

    runtime
        .validate_compute_operations(
            "portable-compute",
            &[ComputeOperationDescriptor::from_schema(add)
                .with_tensor(lhs.clone())
                .with_tensor(rhs.clone())
                .with_tensor(output.clone())],
        )
        .unwrap();

    let bad_output = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[ComputeOperationDescriptor::from_schema(add)
                .with_tensor(lhs.clone())
                .with_tensor(rhs.clone())
                .with_tensor(bad_output)]
        ),
        Err(ComputeValidationError::InvalidOutputDescriptor { .. })
    ));

    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[ComputeOperationDescriptor::from_schema(add)
                .with_attribute("unknown", ComputeOperationAttribute::Boolean(true))
                .with_tensor(lhs)
                .with_tensor(rhs)
                .with_tensor(output)]
        ),
        Err(ComputeValidationError::InvalidOperationAttribute { .. })
    ));
}

#[test]
fn compute_operation_schema_validation_checks_reduction_matmul_and_random_rules() {
    let schemas = initial_compute_operation_schemas();
    let sum = schemas
        .get(&ComputeOperationId::new("reduction.sum"))
        .unwrap();
    let matmul = schemas
        .get(&ComputeOperationId::new("linalg.matmul"))
        .unwrap();
    let random = schemas
        .get(&ComputeOperationId::new("random.uniform"))
        .unwrap();
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    for schema in [sum, matmul, random] {
        provider.metadata.compute_operation_schema_support.insert(
            schema.id.clone(),
            ComputeOperationSupport::new()
                .with_dtypes([ComputeDType::Float32, ComputeDType::SInt64])
                .with_layouts([ComputeLayout::Dense]),
        );
    }
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let f32_2x3 = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 3]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );

    runtime
        .validate_compute_operations(
            "portable-compute",
            &[ComputeOperationDescriptor::from_schema(sum)
                .with_attribute("axes", ComputeOperationAttribute::Axes(vec![1]))
                .with_attribute("keep-dimensions", ComputeOperationAttribute::Boolean(true))
                .with_tensor(f32_2x3.clone())
                .with_tensor(TensorDescriptor::materialized(
                    ShapeDescriptor::new([2, 1]),
                    DTypeDescriptor::portable(ComputeDType::Float32),
                ))],
        )
        .unwrap();
    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[ComputeOperationDescriptor::from_schema(sum)
                .with_attribute("axes", ComputeOperationAttribute::Axes(vec![2]))
                .with_tensor(f32_2x3.clone())
                .with_tensor(f32_2x3.clone())]
        ),
        Err(ComputeValidationError::InvalidShape { .. })
    ));

    runtime
        .validate_compute_operations(
            "portable-compute",
            &[ComputeOperationDescriptor::from_schema(matmul)
                .with_tensor(f32_2x3.clone())
                .with_tensor(TensorDescriptor::materialized(
                    ShapeDescriptor::new([3, 4]),
                    DTypeDescriptor::portable(ComputeDType::Float32),
                ))
                .with_tensor(TensorDescriptor::materialized(
                    ShapeDescriptor::new([2, 4]),
                    DTypeDescriptor::portable(ComputeDType::Float32),
                ))],
        )
        .unwrap();

    runtime
        .validate_compute_operations(
            "portable-compute",
            &[ComputeOperationDescriptor::from_schema(random)
                .with_attribute(
                    "shape",
                    ComputeOperationAttribute::Shape(ShapeDescriptor::new([2, 2])),
                )
                .with_attribute(
                    "dtype",
                    ComputeOperationAttribute::DType(ComputeDType::Float32),
                )
                .with_attribute("seed", ComputeOperationAttribute::Integer(42))
                .with_tensor(TensorDescriptor::materialized(
                    ShapeDescriptor::new([2, 2]),
                    DTypeDescriptor::portable(ComputeDType::Float32),
                ))],
        )
        .unwrap();
}

#[test]
fn tensor_descriptors_validate_shape_dtype_layout_and_provider_support() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::DescriptorAndView,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense, ComputeLayout::Strided])
            .with_descriptor_limits(TensorDescriptorLimits {
                max_rank: 4,
                max_dimension: 1024,
                max_elements: 4096,
                max_bytes: 16_384,
                allow_zero_sized: false,
            }),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let tensor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 3, 4]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );

    runtime
        .validate_compute_operations(
            "portable-compute",
            &[
                ComputeOperationDescriptor::new(ComputeOperationFamily::DescriptorAndView)
                    .with_tensor(tensor.clone()),
            ],
        )
        .unwrap();

    let zero_dim = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 0]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[
                ComputeOperationDescriptor::new(ComputeOperationFamily::DescriptorAndView)
                    .with_tensor(zero_dim)
            ]
        ),
        Err(ComputeValidationError::InvalidShape { .. })
    ));

    let unsupported_dtype = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 3]),
        DTypeDescriptor::portable(ComputeDType::Float64),
    );
    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[
                ComputeOperationDescriptor::new(ComputeOperationFamily::DescriptorAndView)
                    .with_tensor(unsupported_dtype)
            ]
        ),
        Err(ComputeValidationError::UnsupportedDType { .. })
    ));

    let unsupported_layout = TensorDescriptor::new(
        ShapeDescriptor::new([2, 3]),
        DTypeDescriptor::portable(ComputeDType::Float32),
        LayoutDescriptor::ProviderOpaque {
            layout_id: "native-blocked".into(),
        },
    );
    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[
                ComputeOperationDescriptor::new(ComputeOperationFamily::DescriptorAndView)
                    .with_tensor(unsupported_layout)
            ]
        ),
        Err(ComputeValidationError::UnsupportedLayout { .. })
    ));

    let overflowing = TensorDescriptor::materialized(
        ShapeDescriptor::new([64, 65]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    assert!(matches!(
        runtime.validate_compute_operations(
            "portable-compute",
            &[
                ComputeOperationDescriptor::new(ComputeOperationFamily::DescriptorAndView)
                    .with_tensor(overflowing)
            ]
        ),
        Err(ComputeValidationError::SizeOverflow { .. })
    ));
}

#[test]
fn provider_compute_advertisement_drives_data_movement_validation() {
    let mut provider = provider_with_capabilities("advertised-movement", [compute_capability()]);
    provider.metadata.compute_advertisement = ProviderComputeAdvertisement::new()
        .with_capability(
            ComputeCapabilitySupport::default().with_versions([COMPUTE_CAPABILITY_VERSION]),
        )
        .with_data_movement(DataMovementSupport::from_compute_support(
            ComputeDataMovementKind::Upload,
            ComputeDataMovementSupport::new()
                .with_dtypes([ComputeDType::Float32])
                .with_layouts([ComputeLayout::Dense])
                .with_host_encodings([HostBufferEncoding::LittleEndian]),
        ));
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );

    runtime
        .validate_compute_data_movement(
            "advertised-movement",
            &[ComputeDataMovementDescriptor::upload(
                HostBufferDescriptor::new(16, HostBufferEncoding::LittleEndian),
                descriptor,
            )],
        )
        .unwrap();
}

#[test]
fn compute_graph_validation_checks_references_provider_support_and_submission_affinity() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("graph-1"))
        .with_input(ComputeInput::new(
            ComputeInputId::new("x"),
            ComputeInputValue::TensorDescriptor(descriptor.clone()),
        ))
        .with_node(
            ComputeNode::new(
                ComputeNodeId::new("add"),
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                    .with_dtype(ComputeDType::Float32)
                    .with_layout(ComputeLayout::Dense),
            )
            .with_input(ComputeValueRef::Input(ComputeInputId::new("x")))
            .with_output(ComputeNodeOutput::new(
                ComputeOutputId::new("y"),
                descriptor,
            )),
        )
        .with_output(ComputeOutput::new(
            ComputeOutputId::new("result"),
            ComputeValueRef::NodeOutput {
                node: ComputeNodeId::new("add"),
                output: ComputeOutputId::new("y"),
            },
        ));

    let report = runtime
        .validate_compute_graph("portable-compute", &graph)
        .unwrap();
    assert_eq!(report.node_count, 1);
    assert_eq!(report.input_count, 1);
    assert_eq!(report.output_count, 1);

    let submission = runtime
        .submit_validated_compute_graph("portable-compute", &graph)
        .unwrap();
    assert_eq!(submission.graph, ComputeGraphId::new("graph-1"));
    assert_eq!(submission.state(), ComputeSubmissionState::Pending);
    assert_eq!(
        submission.affinity.provider().map(ProviderBinding::as_str),
        Some("portable-compute")
    );
}

#[test]
fn compute_execution_planning_preserves_provider_pinned_resources() {
    let mut provider_a = provider_with_capabilities("provider-a", [compute_capability()]);
    provider_a.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let mut provider_b = provider_with_capabilities("provider-b", [compute_capability()]);
    provider_b.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider_b))
        .register_provider(Arc::new(provider_a))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let resource = TensorResourceDescriptor::new(
        TensorResourceId::new("pinned"),
        descriptor.clone(),
        ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(ProviderBinding::new("provider-b"))
            .with_capability(CapabilityBinding::new(
                CapabilityId::new(COMPUTE_CAPABILITY_ID),
                COMPUTE_CAPABILITY_VERSION,
            )),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("pinned-graph"))
        .with_input(ComputeInput::new(
            ComputeInputId::new("x"),
            ComputeInputValue::TensorResource(resource),
        ))
        .with_node(
            ComputeNode::new(
                ComputeNodeId::new("add"),
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                    .with_dtype(ComputeDType::Float32)
                    .with_layout(ComputeLayout::Dense),
            )
            .with_input(ComputeValueRef::Input(ComputeInputId::new("x")))
            .with_output(ComputeNodeOutput::new(
                ComputeOutputId::new("y"),
                descriptor,
            )),
        );

    let plan = runtime.plan_compute_execution(&graph).unwrap();

    assert_eq!(plan.provider.as_str(), "provider-b");
    assert_eq!(
        plan.classification,
        ComputeExecutionClassification::ProviderPinned
    );
    assert!(
        plan.steps
            .iter()
            .any(|step| step.kind == ExecutionStepKind::PreserveProviderPinnedAffinity)
    );
}

#[test]
fn compute_graph_validation_rejects_missing_and_future_references() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let missing_input = ComputeGraph::new(ComputeGraphId::new("bad-input")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise),
        )
        .with_input(ComputeValueRef::Input(ComputeInputId::new("missing")))
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor.clone(),
        )),
    );
    assert!(matches!(
        runtime.validate_compute_graph("portable-compute", &missing_input),
        Err(ComputeValidationError::MissingInput { .. })
    ));

    let future_reference = ComputeGraph::new(ComputeGraphId::new("cycle"))
        .with_node(
            ComputeNode::new(
                ComputeNodeId::new("first"),
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise),
            )
            .with_input(ComputeValueRef::NodeOutput {
                node: ComputeNodeId::new("second"),
                output: ComputeOutputId::new("out"),
            })
            .with_output(ComputeNodeOutput::new(
                ComputeOutputId::new("out"),
                descriptor.clone(),
            )),
        )
        .with_node(
            ComputeNode::new(
                ComputeNodeId::new("second"),
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise),
            )
            .with_output(ComputeNodeOutput::new(
                ComputeOutputId::new("out"),
                descriptor,
            )),
        );
    assert!(matches!(
        runtime.validate_compute_graph("portable-compute", &future_reference),
        Err(ComputeValidationError::CyclicGraph { .. })
    ));
}

#[test]
fn memory_planning_tracks_lifetimes_reuse_and_output_affinity() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("memory-graph"))
        .with_input(ComputeInput::new(
            ComputeInputId::new("x"),
            ComputeInputValue::TensorDescriptor(descriptor.clone()),
        ))
        .with_node(
            ComputeNode::new(
                ComputeNodeId::new("temp"),
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                    .with_dtype(ComputeDType::Float32)
                    .with_layout(ComputeLayout::Dense),
            )
            .with_input(ComputeValueRef::Input(ComputeInputId::new("x")))
            .with_output(ComputeNodeOutput::new(
                ComputeOutputId::new("tmp"),
                descriptor.clone(),
            )),
        )
        .with_node(
            ComputeNode::new(
                ComputeNodeId::new("result-node"),
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                    .with_dtype(ComputeDType::Float32)
                    .with_layout(ComputeLayout::Dense),
            )
            .with_input(ComputeValueRef::Input(ComputeInputId::new("x")))
            .with_output(ComputeNodeOutput::new(
                ComputeOutputId::new("y"),
                descriptor,
            )),
        )
        .with_output(ComputeOutput::new(
            ComputeOutputId::new("result"),
            ComputeValueRef::NodeOutput {
                node: ComputeNodeId::new("result-node"),
                output: ComputeOutputId::new("y"),
            },
        ));

    let plan = runtime
        .plan_compute_graph_memory("portable-compute", &graph)
        .unwrap();

    assert!(
        plan.decisions
            .iter()
            .any(|decision| matches!(decision, MemoryPlanningDecision::Reuse { .. }))
    );
    assert_eq!(
        plan.output_affinity.provider().map(ProviderBinding::as_str),
        Some("portable-compute")
    );
    assert!(
        plan.requirements
            .iter()
            .any(|requirement| requirement.region == MemoryRegionKind::Intermediate)
    );
}

#[test]
fn tensor_layout_descriptor_kind_covers_every_layout_category() {
    assert_eq!(LayoutDescriptor::Contiguous.kind(), ComputeLayout::Dense);
    assert_eq!(
        LayoutDescriptor::Blocked {
            block_dimensions: vec![2, 2],
        }
        .kind(),
        ComputeLayout::Blocked
    );
    assert_eq!(
        LayoutDescriptor::Paged {
            page_size_elements: 16,
            block_size_elements: 4,
            capacity_pages: None,
            current_length_elements: None,
            logical_to_physical: None,
            append_behavior: None,
        }
        .kind(),
        ComputeLayout::Paged
    );
    assert_eq!(
        LayoutDescriptor::PackedQuantized {
            method: "int4".into(),
            bits_per_value: 4,
            group_size: Some(32),
            scale_dtype: None,
            zero_point_dtype: None,
            packing_order: None,
            dequantization_requirements: None,
        }
        .kind(),
        ComputeLayout::PackedQuantized
    );
    assert_eq!(
        LayoutDescriptor::AttentionSpecific {
            layout_id: "paged-attention".into(),
        }
        .kind(),
        ComputeLayout::AttentionSpecific
    );
    assert_eq!(
        LayoutDescriptor::BrowserCompatible {
            layout_id: "wasm-linear".into(),
        }
        .kind(),
        ComputeLayout::BrowserCompatible
    );
}

#[test]
fn shape_descriptor_row_major_strides_matches_expected_layout() {
    let shape = ShapeDescriptor::new([2, 3, 4]);
    assert_eq!(shape.row_major_strides(), vec![12, 4, 1]);
    assert_eq!(ShapeDescriptor::new([5]).row_major_strides(), vec![1]);
}

#[test]
fn tensor_descriptor_estimated_byte_size_honors_packed_quantized_bits() {
    let packed = TensorDescriptor::new(
        ShapeDescriptor::new([16]),
        DTypeDescriptor::portable(ComputeDType::Float32),
        LayoutDescriptor::PackedQuantized {
            method: "int4".into(),
            bits_per_value: 4,
            group_size: Some(16),
            scale_dtype: None,
            zero_point_dtype: None,
            packing_order: None,
            dequantization_requirements: None,
        },
    );
    assert_eq!(packed.estimated_byte_size().unwrap(), 8);

    let dense = TensorDescriptor::materialized(
        ShapeDescriptor::new([16]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    assert_eq!(
        dense.estimated_byte_size().unwrap(),
        dense.byte_size().unwrap()
    );
}

#[test]
fn tensor_layout_placeholder_variants_validate_without_bounds_checking() {
    let blocked = TensorDescriptor::new(
        ShapeDescriptor::new([4, 4]),
        DTypeDescriptor::portable(ComputeDType::Float32),
        LayoutDescriptor::Blocked {
            block_dimensions: vec![2, 2],
        },
    );
    let paged = TensorDescriptor::new(
        ShapeDescriptor::new([256]),
        DTypeDescriptor::portable(ComputeDType::Float32),
        LayoutDescriptor::Paged {
            page_size_elements: 16,
            block_size_elements: 4,
            capacity_pages: Some(16),
            current_length_elements: Some(200),
            logical_to_physical: None,
            append_behavior: None,
        },
    );
    for descriptor in [blocked, paged] {
        assert!(
            descriptor
                .validate(&TensorDescriptorLimits::default())
                .is_ok()
        );
    }
}

#[test]
fn shape_descriptor_symbolic_dimensions_validate_fixed_consistency() {
    let consistent = ShapeDescriptor::new([2, 8]).with_symbolic_dimensions([
        SymbolicDimension::Symbolic("batch".into()),
        SymbolicDimension::Fixed(8),
    ]);
    assert!(consistent.validate_symbolic().is_ok());

    let inconsistent = ShapeDescriptor::new([2, 8]).with_symbolic_dimensions([
        SymbolicDimension::Symbolic("batch".into()),
        SymbolicDimension::Fixed(16),
    ]);
    assert!(inconsistent.validate_symbolic().is_err());

    let wrong_rank =
        ShapeDescriptor::new([2, 8]).with_symbolic_dimensions([SymbolicDimension::Dynamic]);
    assert!(wrong_rank.validate_symbolic().is_err());
}

#[test]
fn layout_descriptor_paged_tracks_logical_to_physical_and_append_behavior() {
    let mut logical_to_physical = std::collections::BTreeMap::new();
    logical_to_physical.insert(0, 3);
    logical_to_physical.insert(1, 7);
    let paged = LayoutDescriptor::Paged {
        page_size_elements: 16,
        block_size_elements: 4,
        capacity_pages: Some(8),
        current_length_elements: Some(64),
        logical_to_physical: Some(logical_to_physical.clone()),
        append_behavior: Some(PagedAppendBehavior::GrowOnDemand),
    };
    let LayoutDescriptor::Paged {
        logical_to_physical: stored_map,
        append_behavior: stored_behavior,
        ..
    } = &paged
    else {
        unreachable!()
    };
    assert_eq!(stored_map.as_ref(), Some(&logical_to_physical));
    assert_eq!(*stored_behavior, Some(PagedAppendBehavior::GrowOnDemand));
    assert_eq!(paged.kind(), ComputeLayout::Paged);
}
