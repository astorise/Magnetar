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

use crate::affinity::{AffinityError, DeviceBinding};
use crate::capability::CapabilityVersion;
use crate::device::{DeviceDescriptor, DeviceId, DeviceMetadata, DeviceType};
use crate::operator::TensorRole;
use crate::planning::ExecutionConstraint;
use crate::resolution::BuiltInResolutionPolicy;
use crate::tensor::{DimensionRole, TensorAliasingKind, TensorMemoryClass, TensorMutabilityKind};
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

#[test]
fn compute_error_model_maps_validation_resolution_affinity_and_execution_failures() {
    let invalid_shape = ComputeError::from(ComputeValidationError::InvalidShape {
        reason: "rank exceeds limit".into(),
    });
    assert_eq!(invalid_shape.code, ComputeErrorCode::InvalidShape);
    assert_eq!(invalid_shape.phase, ComputeErrorPhase::Validation);
    assert_eq!(invalid_shape.severity, ComputeErrorSeverity::Terminal);
    assert!(
        invalid_shape
            .recovery_hints
            .contains(&RecoveryHint::NotRetryable)
    );

    let materialization = ComputeError::from(ComputeValidationError::MaterializationRequired {
        reason: "view must be materialized".into(),
    });
    assert_eq!(
        materialization.code,
        ComputeErrorCode::MaterializationRequired
    );
    assert!(
        materialization
            .recovery_hints
            .contains(&RecoveryHint::ExplicitMaterializationRequired)
    );

    let affinity = ComputeError::from(AffinityError::BoundProviderUnavailable(
        ProviderBinding::new("provider-a"),
    ));
    assert_eq!(affinity.code, ComputeErrorCode::ProviderUnavailable);
    assert_eq!(affinity.phase, ComputeErrorPhase::Interruption);
    assert!(
        affinity
            .recovery_hints
            .contains(&RecoveryHint::ProviderPinned)
    );

    let policy = ComputeError::from(ProviderError::PolicyRejectedProvider {
        capability: CapabilityBinding::new(
            CapabilityId::new(COMPUTE_CAPABILITY_ID),
            COMPUTE_CAPABILITY_VERSION,
        ),
        policy: BuiltInResolutionPolicy::Availability.id(),
    });
    assert_eq!(policy.code, ComputeErrorCode::PolicyRejectedProvider);
    assert_eq!(policy.phase, ComputeErrorPhase::Resolution);
    assert_eq!(
        policy.diagnostics[0]
            .capability
            .as_ref()
            .unwrap()
            .id()
            .as_str(),
        COMPUTE_CAPABILITY_ID
    );

    let execution = ComputeError::from(ProviderError::Lifecycle("device lost".into()));
    assert_eq!(execution.code, ComputeErrorCode::ExecutionInterrupted);
    assert_eq!(execution.phase, ComputeErrorPhase::Interruption);
    assert!(
        execution
            .recovery_hints
            .contains(&RecoveryHint::RetryBeforeState)
    );
}

#[test]
fn compute_diagnostics_are_optional_redacted_and_non_contractual() {
    let diagnostic = ComputeDiagnostic::new()
        .with_provider(ProviderBinding::new("provider-a"))
        .with_device(DeviceBinding::new(DeviceId::new("gpu:0")))
        .with_operation_family(ComputeOperationFamily::LinearAlgebra)
        .with_backend_message("native handle=0xdeadbeef at C:\\secret\\tensor.bin")
        .with_debug_trace_id("trace-42");

    assert_eq!(
        diagnostic.provider.as_ref().map(ProviderBinding::as_str),
        Some("provider-a")
    );
    assert_eq!(
        diagnostic.backend_message.as_deref(),
        Some("[redacted backend diagnostic]")
    );

    let error = ComputeError::new(
        ComputeErrorCode::ExecutionFailed,
        ComputeErrorPhase::Execution,
        ComputeErrorSeverity::Terminal,
        "provider execution failed",
    )
    .with_diagnostic(diagnostic)
    .with_recovery_hint(RecoveryHint::RestartableWithReplay);

    assert_eq!(error.code, ComputeErrorCode::ExecutionFailed);
    assert_eq!(error.diagnostics.len(), 1);
    assert!(
        error
            .recovery_hints
            .contains(&RecoveryHint::RestartableWithReplay)
    );
}

#[test]
fn compute_operation_schemas_define_initial_portable_operations() {
    let schemas = initial_compute_operation_schemas();

    for id in [
        "tensor.reshape",
        "tensor.transpose",
        "tensor.permute",
        "tensor.slice",
        "tensor.broadcast",
        "tensor.squeeze",
        "tensor.unsqueeze",
        "elementwise.unary.relu",
        "elementwise.binary.add",
        "comparison.eq",
        "selection.where",
        "reduction.sum",
        "linalg.matmul",
        "linalg.batched-matmul",
        "tensor.gather",
        "tensor.index-select",
        "tensor.scatter",
        "tensor.scatter-add",
        "tensor.concat",
        "random.uniform",
        "random.normal",
    ] {
        assert!(schemas.contains_key(&ComputeOperationId::new(id)), "{id}");
    }
    assert!(!schemas.contains_key(&ComputeOperationId::new("convolution.conv2d")));
    assert!(!schemas.contains_key(&ComputeOperationId::new("pooling.max")));
    assert!(!schemas.contains_key(&ComputeOperationId::new("attention.flash")));
    assert!(!schemas.contains_key(&ComputeOperationId::new("quantized.matmul")));
    assert!(!schemas.contains_key(&ComputeOperationId::new("custom.kernel")));
    assert!(!schemas.contains_key(&ComputeOperationId::new("autograd.backward")));

    let scatter = schemas
        .get(&ComputeOperationId::new("tensor.scatter"))
        .unwrap();
    assert!(scatter.provider_specific_semantics);
}

#[test]
fn provider_compute_advertisement_reports_version_and_schema_rejections() {
    let schemas = initial_compute_operation_schemas();
    let add = schemas
        .get(&ComputeOperationId::new("elementwise.binary.add"))
        .unwrap();
    let mut incompatible = TestProvider::new("incompatible-compute");
    incompatible.metadata.compute_advertisement = ProviderComputeAdvertisement::new()
        .with_capability(
            ComputeCapabilitySupport::default().with_versions([CapabilityVersion::new(0, 9, 0)]),
        );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(incompatible))
        .build()
        .unwrap();

    assert!(matches!(
        runtime.validate_compute_operations(
            "incompatible-compute",
            &[ComputeOperationDescriptor::from_schema(add)]
        ),
        Err(ComputeValidationError::UnsupportedAdvertisement { .. })
    ));

    let mut unsupported = provider_with_capabilities("unsupported-schema", [compute_capability()]);
    unsupported.metadata.compute_advertisement = ProviderComputeAdvertisement::new()
        .with_capability(
            ComputeCapabilitySupport::default().with_versions([COMPUTE_CAPABILITY_VERSION]),
        )
        .with_operation_family(OperationFamilySupport::from_operation_support(
            ComputeOperationFamily::Elementwise,
            ComputeOperationSupport::new().with_dtypes([ComputeDType::Float32]),
        ))
        .with_unsupported_operation_schema(add.id.clone());
    let runtime = Runtime::builder()
        .register_provider(Arc::new(unsupported))
        .build()
        .unwrap();

    assert!(matches!(
        runtime.validate_compute_operations(
            "unsupported-schema",
            &[ComputeOperationDescriptor::from_schema(add)]
        ),
        Err(ComputeValidationError::UnsupportedOperationSchema { .. })
    ));
}

#[test]
fn tensor_views_and_resources_preserve_affinity_and_materialization_boundaries() {
    let provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let source = TensorResourceId::new("tensor-1");
    let descriptor = TensorDescriptor::new(
        ShapeDescriptor::new([4, 4]),
        DTypeDescriptor::portable(ComputeDType::Float32),
        LayoutDescriptor::Strided {
            strides_elements: vec![4, 1],
            offset_elements: 0,
        },
    )
    .with_view(ViewDescriptor::from_resource(source.clone(), 4, [4, 1]));
    let resource = TensorResourceDescriptor::new(
        source,
        descriptor,
        ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(ProviderBinding::new("portable-compute"))
            .with_capability(CapabilityBinding::new(
                CapabilityId::new(COMPUTE_CAPABILITY_ID),
                COMPUTE_CAPABILITY_VERSION,
            )),
    );

    runtime
        .validate_compute_tensor_resources("portable-compute", &[resource])
        .unwrap();

    let foreign = TensorResourceDescriptor::new(
        TensorResourceId::new("tensor-2"),
        TensorDescriptor::materialized(
            ShapeDescriptor::new([4, 4]),
            DTypeDescriptor::portable(ComputeDType::Float32),
        ),
        ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(ProviderBinding::new("other-provider")),
    );
    assert!(matches!(
        runtime.validate_compute_tensor_resources("portable-compute", &[foreign]),
        Err(ComputeValidationError::IncompatibleResourceAffinity(_))
    ));
}

#[test]
fn compute_data_movement_validates_host_buffers_affinity_and_provider_support() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_data_movement_support.insert(
        ComputeDataMovementKind::Upload,
        ComputeDataMovementSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense])
            .with_host_encodings([HostBufferEncoding::LittleEndian]),
    );
    provider.metadata.compute_data_movement_support.insert(
        ComputeDataMovementKind::Transfer,
        ComputeDataMovementSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    provider.metadata.compute_data_movement_support.insert(
        ComputeDataMovementKind::Materialize,
        ComputeDataMovementSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense, ComputeLayout::Strided]),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );

    let upload = ComputeDataMovementDescriptor::upload(
        HostBufferDescriptor::new(16, HostBufferEncoding::LittleEndian),
        descriptor.clone(),
    );
    runtime
        .validate_compute_data_movement("portable-compute", std::slice::from_ref(&upload))
        .unwrap();
    let uploaded = runtime
        .wrap_compute_data_movement_output(
            "portable-compute",
            &upload,
            TensorResourceId::new("uploaded"),
        )
        .unwrap();
    assert_eq!(
        uploaded.affinity.provider().map(ProviderBinding::as_str),
        Some("portable-compute")
    );

    let invalid_upload = ComputeDataMovementDescriptor::upload(
        HostBufferDescriptor::new(8, HostBufferEncoding::LittleEndian),
        descriptor.clone(),
    );
    assert!(matches!(
        runtime.validate_compute_data_movement("portable-compute", &[invalid_upload]),
        Err(ComputeValidationError::InvalidHostBuffer { .. })
    ));

    let foreign = TensorResourceDescriptor::new(
        TensorResourceId::new("foreign"),
        descriptor.clone(),
        ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(ProviderBinding::new("other-provider")),
    );
    let transfer = ComputeDataMovementDescriptor::transfer(foreign, descriptor.clone());
    runtime
        .validate_compute_data_movement("portable-compute", &[transfer])
        .unwrap();

    let materialized = TensorResourceDescriptor::new(
        TensorResourceId::new("materialized"),
        descriptor.clone(),
        ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(ProviderBinding::new("portable-compute")),
    );
    let invalid_materialize = ComputeDataMovementDescriptor::materialize(materialized, descriptor);
    assert!(matches!(
        runtime.validate_compute_data_movement("portable-compute", &[invalid_materialize]),
        Err(ComputeValidationError::MaterializationRequired { .. })
    ));
}

#[test]
fn compute_execution_planning_selects_provider_device_and_validates_plan() {
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let mut device = DeviceMetadata::new(
        DeviceId::new("gpu:0"),
        "GPU 0",
        DeviceType::Gpu,
        "portable-compute",
    );
    device.memory_capacity = 1_048_576;
    provider
        .devices
        .push(Arc::new(DeviceDescriptor::new(device)));
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("planned-graph"))
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

    let plan = runtime.plan_compute_execution(&graph).unwrap();

    assert!(plan.is_validated());
    assert_eq!(plan.provider.as_str(), "portable-compute");
    assert_eq!(
        plan.device.as_ref().map(|device| device.id().as_str()),
        Some("gpu:0")
    );
    assert_eq!(plan.policy, BuiltInResolutionPolicy::Deterministic.id());
    assert_eq!(
        plan.classification,
        ComputeExecutionClassification::Transparent
    );
    assert!(
        plan.steps
            .iter()
            .any(|step| step.kind == ExecutionStepKind::SubmitToProvider)
    );
    assert!(
        plan.constraints
            .iter()
            .any(|constraint| matches!(constraint, ExecutionConstraint::NoHiddenCpuStaging))
    );
    assert_eq!(
        plan.memory_plan.graph,
        Some(ComputeGraphId::new("planned-graph"))
    );
}

#[test]
fn tensor_descriptor_builder_sets_intents_and_semantic_role() {
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 3]),
        DTypeDescriptor::portable(ComputeDType::Float16),
    )
    .with_storage_dtype(DTypeDescriptor::portable(ComputeDType::Float16))
    .with_compute_dtype(DTypeDescriptor::portable(ComputeDType::Float32))
    .with_memory_class_intent(TensorMemoryClass::Host)
    .with_mutability_intent(TensorMutabilityKind::Immutable)
    .with_aliasing_intent(TensorAliasingKind::NoAlias)
    .with_affinity_constraints(ResourceAffinity::new(FallbackClass::Transparent))
    .with_semantic_role(TensorRole::Input)
    .with_dimension_roles([DimensionRole::Batch, DimensionRole::Hidden]);

    assert_eq!(
        descriptor.compute_dtype,
        Some(DTypeDescriptor::portable(ComputeDType::Float32))
    );
    assert_eq!(
        descriptor.memory_class_intent,
        Some(TensorMemoryClass::Host)
    );
    assert_eq!(descriptor.semantic_role, Some(TensorRole::Input));
    assert!(
        descriptor
            .validate(&TensorDescriptorLimits::default())
            .is_ok()
    );

    let mismatched = descriptor.with_dimension_roles([DimensionRole::Batch]);
    assert!(
        mismatched
            .validate(&TensorDescriptorLimits::default())
            .is_err()
    );
}

#[test]
fn layout_descriptor_packed_quantized_tracks_dequantization_requirements() {
    let layout = LayoutDescriptor::PackedQuantized {
        method: "int4".into(),
        bits_per_value: 4,
        group_size: Some(32),
        scale_dtype: Some(Box::new(DTypeDescriptor::portable(ComputeDType::Float16))),
        zero_point_dtype: None,
        packing_order: Some("row-major-blocks".into()),
        dequantization_requirements: Some("requires dequantize_placeholder before use".into()),
    };
    let LayoutDescriptor::PackedQuantized {
        dequantization_requirements,
        ..
    } = &layout
    else {
        unreachable!()
    };
    assert_eq!(
        dequantization_requirements.as_deref(),
        Some("requires dequantize_placeholder before use")
    );
}
