use crate::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};
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

fn capability(name: &str, version: CapabilityVersion) -> Capability {
    Capability::new(
        CapabilityId::new(name),
        version,
        CapabilityDescriptor::new("test capability")
            .with_contract(WitInterface::new(name, version.to_string())),
    )
}
fn capability_binding(name: &str, version: CapabilityVersion) -> CapabilityBinding {
    CapabilityBinding::new(CapabilityId::new(name), version)
}
fn provider_with_capabilities(
    name: &str,
    capabilities: impl IntoIterator<Item = Capability>,
) -> TestProvider {
    let mut provider = TestProvider::new(name);
    provider.metadata.capabilities.extend(capabilities);
    provider
}
fn simple_elementwise_compute_graph(name: &str) -> ComputeGraph {
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    ComputeGraph::new(ComputeGraphId::new(name)).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor,
        )),
    )
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
    fn execution_api(&self) -> Option<&dyn ProviderExecutionApi> {
        self.execution_api.as_deref()
    }
}
struct TestProviderExecutionApi {
    submitted: Mutex<Vec<ProviderExecutionRequest>>,
    released: AtomicBool,
    cancel_outcome: Mutex<ProviderCancellationOutcome>,
    outputs: Mutex<Vec<TensorResourceDescriptor>>,
}
impl TestProviderExecutionApi {
    fn new() -> Self {
        Self {
            submitted: Mutex::new(Vec::new()),
            released: AtomicBool::new(false),
            cancel_outcome: Mutex::new(ProviderCancellationOutcome::Unsupported),
            outputs: Mutex::new(Vec::new()),
        }
    }
}
impl ProviderExecutionApi for TestProviderExecutionApi {
    fn submit(
        &self,
        request: ProviderExecutionRequest,
    ) -> Result<ProviderExecutionHandle, ProviderExecutionError> {
        let handle = ProviderExecutionHandle::new(
            request.operation,
            request.plan.id.clone(),
            request.provider.clone(),
            request.device.clone(),
        );
        self.submitted.lock().unwrap().push(request);
        Ok(handle)
    }
    fn status(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionStatus, ProviderExecutionError> {
        let mut status = ProviderExecutionStatus::new(handle.clone(), SchedulingState::Running);
        status.progress = Some(
            ProviderExecutionProgress::new(1, 3).with_message("provider execution is running"),
        );
        Ok(status)
    }
    fn cancel(
        &self,
        _handle: &ProviderExecutionHandle,
    ) -> Result<ProviderCancellationOutcome, ProviderExecutionError> {
        Ok(*self.cancel_outcome.lock().unwrap())
    }
    fn complete(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionResult, ProviderExecutionError> {
        Ok(ProviderExecutionResult::completed(
            handle.clone(),
            self.outputs.lock().unwrap().clone(),
        ))
    }
    fn release(&self, _handle: ProviderExecutionHandle) -> Result<(), ProviderExecutionError> {
        self.released.store(true, Ordering::SeqCst);
        Ok(())
    }
}
#[test]
fn load_valid_provider() {
    let p = Arc::new(TestProvider::new("valid"));
    let mut m = ProviderLoader::new();
    m.register_provider(p.clone()).unwrap();
    assert!(m.provider("valid").is_some());
    assert!(p.initialized.load(Ordering::SeqCst));
}
#[test]
fn runtime_enumerates_provider_devices_with_metadata() {
    let mut provider = TestProvider::new("cuda");
    let mut metadata = DeviceMetadata::new(
        DeviceId::new("cuda:gpu:0"),
        "NVIDIA Test GPU",
        DeviceType::Gpu,
        "cuda",
    );
    metadata.vendor = "NVIDIA".into();
    metadata.architecture = "Ada".into();
    metadata.memory_capacity = 24 * 1024 * 1024 * 1024;
    metadata.compute_units = 128;
    metadata
        .execution_capabilities
        .insert(CapabilityId::new("magnetar:compute/run"));
    provider
        .devices
        .push(Arc::new(DeviceDescriptor::new(metadata)));

    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let devices = runtime.devices().collect::<Vec<_>>();
    assert_eq!(devices.len(), 1);
    assert_eq!(devices[0].id().as_str(), "cuda:gpu:0");
    assert_eq!(devices[0].metadata().vendor, "NVIDIA");
    assert_eq!(
        runtime
            .providers()
            .registry()
            .provider_for_device(&DeviceId::new("cuda:gpu:0")),
        Some("cuda")
    );
}
#[test]
fn device_registration_rejects_duplicate_ids_and_mismatched_owners() {
    let device = |id: &str, provider: &str| {
        Arc::new(DeviceDescriptor::new(DeviceMetadata::new(
            DeviceId::new(id),
            "test",
            DeviceType::Gpu,
            provider,
        ))) as Arc<dyn Device>
    };
    let mut registry = ProviderRegistry::default();
    registry
        .register_devices("cuda", [device("gpu:0", "cuda")])
        .unwrap();
    assert!(matches!(
        registry.register_devices("other", [device("gpu:0", "other")]),
        Err(ProviderError::DeviceAlreadyRegistered(_))
    ));
    assert!(matches!(
        registry.register_devices("cuda", [device("gpu:2", "cuda"), device("gpu:0", "cuda")]),
        Err(ProviderError::DeviceAlreadyRegistered(_))
    ));
    assert!(registry.device(&DeviceId::new("gpu:2")).is_none());
    assert!(matches!(
        registry.register_devices("cuda", [device("gpu:1", "other")]),
        Err(ProviderError::DeviceProviderMismatch { .. })
    ));
}
#[test]
fn registers_capabilities_and_resolves_fallbacks_by_name() {
    let capability = capability("magnetar:runtime/execute", CapabilityVersion::new(1, 0, 0));
    let mut primary = TestProvider::new("a-primary");
    primary.metadata.capabilities.insert(capability.clone());
    let mut fallback = TestProvider::new("z-fallback");
    fallback.metadata.capabilities.insert(capability.clone());
    let mut loader = ProviderLoader::new();
    loader.register_provider(Arc::new(fallback)).unwrap();
    loader.register_provider(Arc::new(primary)).unwrap();

    assert!(loader.registry().has_capability(&capability));
    let names = loader
        .resolve_providers(&capability)
        .into_iter()
        .map(|provider| provider.metadata().name)
        .collect::<Vec<_>>();
    assert_eq!(names, ["a-primary", "z-fallback"]);
}
#[test]
fn semantic_versions_select_the_latest_compatible_capability() {
    let mut registry = ProviderRegistry::default();
    registry
        .register_capability(capability(
            "magnetar:compute/run",
            CapabilityVersion::new(1, 0, 0),
        ))
        .unwrap();
    registry
        .register_capability(capability(
            "magnetar:compute/run",
            CapabilityVersion::new(1, 2, 0),
        ))
        .unwrap();
    registry
        .register_capability(capability(
            "magnetar:compute/run",
            CapabilityVersion::new(2, 0, 0),
        ))
        .unwrap();

    let id = CapabilityId::new("magnetar:compute/run");
    assert_eq!(
        registry
            .resolve_capability(&id, CapabilityVersion::new(1, 1, 0))
            .unwrap()
            .version,
        CapabilityVersion::new(1, 2, 0)
    );
    assert!(
        registry
            .resolve_capability(&id, CapabilityVersion::new(3, 0, 0))
            .is_none()
    );
    assert!(!CapabilityVersion::new(0, 1, 1).is_compatible_with(CapabilityVersion::new(0, 1, 0)));
}
#[test]
fn capability_validation_rejects_invalid_and_conflicting_definitions() {
    let mut registry = ProviderRegistry::default();
    let invalid = Capability::new(
        CapabilityId::new("magnetar:invalid"),
        CapabilityVersion::new(1, 0, 0),
        CapabilityDescriptor::new("missing contract"),
    );
    assert!(matches!(
        registry.register_capability(invalid),
        Err(ProviderError::InvalidCapability(_))
    ));

    let original = capability("magnetar:compute/run", CapabilityVersion::new(1, 0, 0));
    registry.register_capability(original).unwrap();
    let conflicting = Capability::new(
        CapabilityId::new("magnetar:compute/run"),
        CapabilityVersion::new(1, 0, 0),
        CapabilityDescriptor::new("different")
            .with_contract(WitInterface::new("magnetar:compute/other", "1.0.0")),
    );
    assert!(matches!(
        registry.register_capability(conflicting),
        Err(ProviderError::ConflictingCapability { .. })
    ));
}
#[test]
fn capability_dependencies_must_resolve_compatibly() {
    let mut registry = ProviderRegistry::default();
    let dependent = Capability::new(
        CapabilityId::new("magnetar:app/run"),
        CapabilityVersion::new(1, 0, 0),
        CapabilityDescriptor::new("dependent")
            .with_contract(WitInterface::new("magnetar:app/run", "1.0.0"))
            .with_dependency(
                CapabilityId::new("magnetar:compute/run"),
                CapabilityVersion::new(1, 1, 0),
            ),
    );
    registry.register_capability(dependent).unwrap();
    assert!(matches!(
        registry.validate_dependencies(),
        Err(ProviderError::MissingCapabilityDependency { .. })
    ));
    registry
        .register_capability(capability(
            "magnetar:compute/run",
            CapabilityVersion::new(1, 2, 0),
        ))
        .unwrap();
    registry.validate_dependencies().unwrap();
}
#[test]
fn component_import_uses_a_semantic_capability_version() {
    let mut provider = TestProvider::new("compute");
    provider.metadata.capabilities.insert(capability(
        "magnetar:compute/run",
        CapabilityVersion::new(1, 1, 0),
    ));
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    assert_eq!(
        runtime
            .resolve_component_import(&WitInterface::new("magnetar:compute/run", "1.0.0"))
            .unwrap()
            .len(),
        1
    );
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
fn provider_conformance_suite_reports_core_compute_and_data_movement_success() {
    let mut provider = TestProvider::new("magnetar.test.conformant");
    provider.metadata.capabilities.insert(compute_capability());
    let operation_support = ComputeOperationSupport::new()
        .with_dtypes([ComputeDType::Float32])
        .with_layouts([ComputeLayout::Dense])
        .with_precision_modes([ComputePrecision::Default]);
    provider.metadata.compute_advertisement = ProviderComputeAdvertisement::new()
        .with_capability(
            ComputeCapabilitySupport::default().with_versions([COMPUTE_CAPABILITY_VERSION]),
        )
        .with_operation_family(OperationFamilySupport::from_operation_support(
            ComputeOperationFamily::Elementwise,
            operation_support,
        ))
        .with_data_movement(DataMovementSupport::from_compute_support(
            ComputeDataMovementKind::Upload,
            ComputeDataMovementSupport::new()
                .with_dtypes([ComputeDType::Float32])
                .with_layouts([ComputeLayout::Dense])
                .with_host_encodings([HostBufferEncoding::RawBytes]),
        ));

    let suite =
        ProviderConformanceSuite::new(ProviderConformanceConfig::default().with_profiles([
            ProviderConformanceProfile::ProviderCore,
            ProviderConformanceProfile::ProviderCompute,
            ProviderConformanceProfile::ProviderDataMovement,
            ProviderConformanceProfile::ProviderObservability,
        ]));
    let report = suite.run(ProviderConformanceTarget::mock(Arc::new(provider)));

    assert!(report.is_conformant(), "{report:#?}");
    assert_eq!(report.suite_version, PROVIDER_CONFORMANCE_SUITE_VERSION);
    assert_eq!(report.runtime_version, MAGNETAR_RUNTIME_VERSION);
    assert!(report.passed_tests.iter().any(|result| {
        result.profile == ProviderConformanceProfile::ProviderCompute
            && result.requirement.contains("elementwise")
    }));
    assert!(report.passed_tests.iter().any(|result| {
        result.profile == ProviderConformanceProfile::ProviderDataMovement
            && result.requirement.contains("upload")
    }));

    let json = provider_conformance_report_json(&report).unwrap();
    assert!(json.contains("\"provider_identity\": \"magnetar.test.conformant\""));
    assert!(json.contains("\"suite_version\""));
}

#[test]
fn provider_conformance_suite_fails_invalid_public_metadata() {
    let mut provider = TestProvider::new("bad provider");
    provider.metadata.vendor.clear();
    provider.metadata.description = "raw handle 0xdeadbeef".into();

    let report = ProviderConformanceSuite::default()
        .run(ProviderConformanceTarget::mock(Arc::new(provider)));

    assert!(!report.is_conformant());
    assert!(report.failed_tests.iter().any(|result| {
        result.requirement == "ProviderId syntax"
            && result.profile == ProviderConformanceProfile::ProviderCore
    }));
    assert!(report.failed_tests.iter().any(|result| {
        result.requirement == "vendor metadata"
            && result.profile == ProviderConformanceProfile::ProviderCore
    }));
    assert!(report.failed_tests.iter().any(|result| {
        result.requirement == "metadata redaction"
            && result.profile == ProviderConformanceProfile::ProviderCore
    }));
}

#[test]
fn provider_conformance_suite_reports_dynamic_loading_policy() {
    let path = std::env::temp_dir().join("magnetar-provider-fixture.dll");
    let denied = ProviderConformanceSuite::new(
        ProviderConformanceConfig::default()
            .with_profiles([ProviderConformanceProfile::ProviderDynamicAbi]),
    )
    .run(ProviderConformanceTarget::dynamic_library(
        &path,
        ProviderLoadingPolicy::dynamic_library([std::env::temp_dir().join("allowed")]),
    ));
    assert!(!denied.is_conformant());
    assert!(denied.failed_tests.iter().any(|result| {
        result.profile == ProviderConformanceProfile::ProviderDynamicAbi
            && result.requirement == "allowed path loading"
    }));

    let allowed = ProviderConformanceSuite::new(
        ProviderConformanceConfig::default()
            .with_profiles([ProviderConformanceProfile::ProviderDynamicAbi]),
    )
    .run(ProviderConformanceTarget::development(
        &path,
        ProviderLoadingPolicy::development([std::env::temp_dir()]),
    ));
    assert!(allowed.is_conformant(), "{allowed:#?}");
    assert!(allowed.passed_tests.iter().any(|result| {
        result.profile == ProviderConformanceProfile::ProviderDynamicAbi
            && result.requirement == "ABI descriptor structure"
    }));
    assert!(allowed.skipped_tests.iter().any(|result| {
        result.profile == ProviderConformanceProfile::ProviderDynamicAbi
            && result.requirement == "factory symbol exists"
    }));
}

#[test]
fn provider_conformance_profile_ids_mark_hardware_profiles_optional() {
    let profile_ids = provider_conformance_profile_ids([
        ProviderConformanceProfile::ProviderCore,
        ProviderConformanceProfile::Cuda,
        ProviderConformanceProfile::Metal,
        ProviderConformanceProfile::OpenVino,
        ProviderConformanceProfile::Qnn,
    ]);

    assert!(profile_ids["provider-core"]);
    assert!(!profile_ids["provider-hardware-cuda"]);
    assert!(!profile_ids["provider-hardware-metal"]);
    assert!(!profile_ids["provider-hardware-openvino"]);
    assert!(!profile_ids["provider-hardware-qnn"]);
}
#[test]
fn compute_v1_import_is_not_satisfied_by_compute_v2_provider() {
    let provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();

    assert!(matches!(
        runtime.resolve_component_import(&WitInterface::new(COMPUTE_WIT_INTERFACE, "1.1.0")),
        Err(ProviderError::NoCompatibleProvider(capability))
            if capability.id().as_str() == COMPUTE_CAPABILITY_ID
                && capability.version() == CapabilityVersion::new(1, 1, 0)
    ));
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
fn compute_wit_defines_the_stabilized_run_surface() {
    let wit = include_str!("../wit/compute.wit");
    assert!(wit.contains("package magnetar:compute@2.0.0;"));
    assert!(wit.contains("resource tensor"));
    assert!(wit.contains("resource graph"));
    assert!(wit.contains("resource operation"));
    assert!(wit.contains("enum operation-family"));
    assert!(wit.contains("type operation-id = string"));
    assert!(wit.contains("record operation-schema"));
    assert!(wit.contains("record operation-schema-support"));
    assert!(wit.contains("variant operation-attribute"));
    assert!(wit.contains("record operation-input-rule"));
    assert!(wit.contains("record operation-output-rule"));
    assert!(wit.contains("record operation-descriptor"));
    assert!(wit.contains("schema-id: option<operation-id>"));
    assert!(wit.contains("attributes: list<tuple<string, operation-attribute>>"));
    assert!(wit.contains("record graph-descriptor"));
    assert!(wit.contains("record graph-node"));
    assert!(wit.contains("variant graph-value-ref"));
    assert!(wit.contains("record shape-descriptor"));
    assert!(wit.contains("variant dtype-descriptor"));
    assert!(wit.contains("variant layout-descriptor"));
    assert!(wit.contains("record view-descriptor"));
    assert!(wit.contains("record tensor-resource-descriptor"));
    assert!(wit.contains("enum data-movement-kind"));
    assert!(wit.contains("record host-buffer-descriptor"));
    assert!(wit.contains("record data-movement-support"));
    assert!(wit.contains("record data-movement-descriptor"));
    assert!(wit.contains("enum placement-intent"));
    assert!(wit.contains("preserve-source-affinity"));
    assert!(wit.contains("runtime-selected"));
    assert!(wit.contains("host-accessible"));
    assert!(wit.contains("enum host-staging-policy"));
    assert!(wit.contains("forbid"));
    assert!(wit.contains("permit"));
    assert!(wit.contains("placement: placement-intent"));
    assert!(wit.contains("host-staging: host-staging-policy"));
    assert!(wit.contains("world compute-consumer"));
    assert!(wit.contains("import run"));
    assert!(!wit.contains("target-provider"));
    assert!(!wit.contains("target-device"));
    assert!(!wit.contains("target-affinity-group"));
    assert!(wit.contains("invalid-shape"));
    assert!(wit.contains("size-overflow"));
    assert!(wit.contains("unsupported-operation-family"));
    assert!(wit.contains("invalid-tensor-descriptor"));
    assert!(wit.contains("invalid-dtype"));
    assert!(wit.contains("invalid-operation-attribute"));
    assert!(wit.contains("invalid-operation-arity"));
    assert!(wit.contains("invalid-output-descriptor"));
    assert!(wit.contains("unsupported-dtype"));
    assert!(wit.contains("unsupported-layout"));
    assert!(wit.contains("unsupported-data-movement"));
    assert!(wit.contains("no-compatible-provider"));
    assert!(wit.contains("policy-rejected-provider"));
    assert!(wit.contains("provider-unavailable"));
    assert!(wit.contains("device-unavailable"));
    assert!(wit.contains("provider-pinned-resource"));
    assert!(wit.contains("device-bound-resource"));
    assert!(wit.contains("artifact-fingerprint-mismatch"));
    assert!(wit.contains("affinity-group-mismatch"));
    assert!(wit.contains("execution-interrupted"));
    assert!(wit.contains("execution-cancelled"));
    assert!(wit.contains("invalid-host-buffer"));
    assert!(wit.contains("invalid-transfer"));
    assert!(wit.contains("unsupported-conversion"));
    assert!(wit.contains("materialization-required"));
    assert!(wit.contains("enum compute-error-phase"));
    assert!(wit.contains("enum compute-error-severity"));
    assert!(wit.contains("record compute-diagnostic"));
    assert!(wit.contains("enum recovery-hint"));
    assert!(wit.contains("diagnostics: list<compute-diagnostic>"));
    assert!(wit.contains("recovery-hints: list<recovery-hint>"));
    assert!(wit.contains("submit: func("));
    assert!(wit.contains("result<operation, compute-error>"));
    assert!(!wit.contains("BackendStorage"));
    assert!(!wit.contains("Tensor`"));
    assert!(!wit.contains("autograd"));
    assert!(!wit.contains("training"));
    assert!(!wit.contains("kernel-name"));
    assert!(!wit.contains("queue"));
    assert!(!wit.contains("custom-operation"));
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
fn memory_planning_accounts_for_explicit_host_staged_transfers() {
    let mut provider = provider_with_capabilities("movement-compute", [compute_capability()]);
    provider.metadata.compute_data_movement_support.insert(
        ComputeDataMovementKind::Transfer,
        ComputeDataMovementSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense])
            .with_host_staging(),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let source = TensorResourceDescriptor::new(
        TensorResourceId::new("source"),
        descriptor.clone(),
        ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(ProviderBinding::new("other-provider")),
    );
    let movement =
        ComputeDataMovementDescriptor::transfer(source, descriptor).permit_host_staging();

    let plan = runtime
        .plan_compute_data_movement_memory("movement-compute", &[movement])
        .unwrap();

    assert_eq!(plan.pressure.transfer_buffer_cost_bytes, 16);
    assert!(
        plan.decisions.iter().any(|decision| {
            matches!(decision, MemoryPlanningDecision::AccountHostStaging { .. })
        })
    );
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
fn compute_operation_requests_reject_unknown_family_ids() {
    let provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();

    assert!(matches!(
        runtime.validate_compute_operation_requests(
            "portable-compute",
            &[ComputeOperationRequest::new("backend-kernel-name")]
        ),
        Err(ComputeValidationError::UnknownOperationFamily(_))
    ));
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
fn scheduler_accepts_validated_plans_and_runs_fifo() {
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
    let first_graph = ComputeGraph::new(ComputeGraphId::new("first")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor.clone(),
        )),
    );
    let second_graph = ComputeGraph::new(ComputeGraphId::new("second")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor,
        )),
    );
    let first_plan = runtime.plan_compute_execution(&first_graph).unwrap();
    let second_plan = runtime.plan_compute_execution(&second_graph).unwrap();
    let mut scheduler = runtime.scheduler(2);

    let first = runtime
        .schedule_compute_execution(&mut scheduler, first_plan)
        .unwrap();
    let second = runtime
        .schedule_compute_execution(&mut scheduler, second_plan)
        .unwrap();

    assert_eq!(scheduler.policy(), SchedulingPolicy::Fifo);
    assert_eq!(scheduler.submit_next(&runtime).unwrap(), Some(first));
    assert_eq!(
        scheduler.operation(first).unwrap().state(),
        SchedulingState::Running
    );
    assert_eq!(scheduler.submit_next(&runtime).unwrap(), Some(second));
}

#[test]
fn runtime_observability_correlates_plan_scheduler_and_metrics() {
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
    let graph = ComputeGraph::new(ComputeGraphId::new("observable")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor,
        )),
    );
    let plan = runtime.plan_compute_execution(&graph).unwrap();
    let trace = plan.trace_id().clone();

    let plan_events = plan.observations();
    assert!(plan_events.iter().all(|event| event.trace_id == trace));
    assert!(plan_events.iter().any(|event| {
        event.kind == RuntimeEventKind::CapabilityResolution
            && event.provider.as_ref().map(ProviderBinding::as_str) == Some("portable-compute")
    }));
    assert!(plan_events.iter().any(|event| {
        event.kind == RuntimeEventKind::ExecutionPlanning
            && event.execution_plan.as_ref() == Some(&plan.id)
    }));

    let metrics = runtime_metrics_for_execution_plan(&plan);
    assert!(
        metrics
            .iter()
            .any(|metric| metric.kind == RuntimeMetricKind::MemoryUsageEstimate)
    );
    assert!(metrics.iter().all(|metric| {
        metric.trace_id.as_ref() == Some(&trace)
            && metric.provider.as_ref().map(ProviderBinding::as_str) == Some("portable-compute")
    }));

    let mut scheduler = runtime.scheduler(1);
    let operation = runtime
        .schedule_compute_execution(&mut scheduler, plan)
        .unwrap();
    scheduler.submit_next(&runtime).unwrap();
    scheduler.complete(operation).unwrap();

    assert!(
        scheduler
            .observations()
            .iter()
            .all(|event| event.trace_id == trace)
    );
    assert!(
        scheduler
            .observations()
            .iter()
            .any(|event| event.kind == RuntimeEventKind::ExecutionStarted)
    );
    assert!(
        scheduler
            .observations()
            .iter()
            .any(|event| event.kind == RuntimeEventKind::ExecutionCompleted)
    );
}

#[test]
fn runtime_diagnostics_redact_native_details_and_exporters_are_components() {
    let diagnostic = RuntimeDiagnostic::new(
        RuntimeDiagnosticCode::ExecutionFailed,
        "backend handle=0xdeadbeef at C:\\native\\queue",
    )
    .with_trace(TraceId::new("trace:failure"))
    .with_provider(ProviderBinding::new("provider-a"));
    assert_eq!(diagnostic.message, "[redacted backend diagnostic]");

    let component = ComponentMetadata::new("otel-exporter", "1", "exports observations");
    let mut exporter =
        ObservabilityExporterDescriptor::new(component, ObservabilitySink::OpenTelemetry);
    exporter
        .accepted_events
        .insert(RuntimeEventKind::ExecutionCompleted);

    assert_eq!(
        exporter.input_contract,
        WitInterface::new("magnetar:runtime/observability", "1.0.0")
    );
    assert!(
        exporter
            .accepted_events
            .contains(&RuntimeEventKind::ExecutionCompleted)
    );
}

#[test]
fn provider_execution_api_submits_validated_scheduled_work() {
    let api = Arc::new(TestProviderExecutionApi::new());
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.execution_api = Some(api.clone());
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
    let graph = ComputeGraph::new(ComputeGraphId::new("provider-execution")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor.clone(),
        )),
    );
    let plan = runtime.plan_compute_execution(&graph).unwrap();
    let mut scheduler = runtime.scheduler(1);
    let operation_id = runtime
        .schedule_compute_execution(&mut scheduler, plan)
        .unwrap();
    let operation = scheduler.operation(operation_id).unwrap();

    let request = runtime.prepare_provider_execution(operation).unwrap();
    let handle = runtime.submit_provider_execution(request).unwrap();
    let status = runtime.observe_provider_execution(&handle).unwrap();

    assert_eq!(handle.operation, operation_id);
    assert_eq!(handle.provider.as_str(), "portable-compute");
    assert!(handle.id.as_str().contains("provider-execution:"));
    assert_eq!(status.state, SchedulingState::Running);
    assert_eq!(
        status
            .progress
            .as_ref()
            .map(|progress| progress.total_steps),
        Some(3)
    );
    let submitted = api.submitted.lock().unwrap();
    assert_eq!(submitted.len(), 1);
    assert!(submitted[0].plan.is_validated());
    assert!(
        submitted[0].constraints.iter().any(|constraint| matches!(
            constraint,
            ExecutionConstraint::NoImplicitProviderMigration
        ))
    );
}
#[test]
fn provider_execution_completion_preserves_output_affinity_and_releases() {
    let api = Arc::new(TestProviderExecutionApi::new());
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.execution_api = Some(api.clone());
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
    let graph = ComputeGraph::new(ComputeGraphId::new("provider-complete")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor.clone(),
        )),
    );
    let plan = runtime.plan_compute_execution(&graph).unwrap();
    let mut scheduler = runtime.scheduler(1);
    let operation_id = runtime
        .schedule_compute_execution(&mut scheduler, plan)
        .unwrap();
    let request = runtime
        .prepare_provider_execution(scheduler.operation(operation_id).unwrap())
        .unwrap();
    let handle = runtime.submit_provider_execution(request).unwrap();
    api.outputs
        .lock()
        .unwrap()
        .push(TensorResourceDescriptor::new(
            TensorResourceId::new("out-resource"),
            descriptor,
            ResourceAffinity::new(FallbackClass::ProviderPinned)
                .with_provider(ProviderBinding::new("portable-compute"))
                .with_capability(CapabilityBinding::new(
                    CapabilityId::new(COMPUTE_CAPABILITY_ID),
                    COMPUTE_CAPABILITY_VERSION,
                )),
        ));

    let result = runtime.complete_provider_execution(&handle).unwrap();
    runtime.release_provider_execution(handle).unwrap();

    assert_eq!(result.state, SchedulingState::Completed);
    assert_eq!(result.outputs.len(), 1);
    assert_eq!(
        result.outputs[0]
            .affinity
            .provider()
            .map(ProviderBinding::as_str),
        Some("portable-compute")
    );
    assert!(api.released.load(Ordering::SeqCst));
}
#[test]
fn provider_execution_rejects_mismatched_provider_request_and_maps_cancellation() {
    let api = Arc::new(TestProviderExecutionApi::new());
    *api.cancel_outcome.lock().unwrap() = ProviderCancellationOutcome::Accepted;
    let mut provider = provider_with_capabilities("portable-compute", [compute_capability()]);
    provider.execution_api = Some(api);
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
    let graph = ComputeGraph::new(ComputeGraphId::new("provider-cancel")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor,
        )),
    );
    let plan = runtime.plan_compute_execution(&graph).unwrap();
    let mut scheduler = runtime.scheduler(1);
    let operation_id = runtime
        .schedule_compute_execution(&mut scheduler, plan)
        .unwrap();
    let mut request = runtime
        .prepare_provider_execution(scheduler.operation(operation_id).unwrap())
        .unwrap();
    request.provider = ProviderBinding::new("other-provider");

    assert!(matches!(
        runtime.submit_provider_execution(request),
        Err(ProviderExecutionError {
            code: ProviderExecutionErrorCode::InvalidExecutionPlan,
            phase: ProviderExecutionPhase::Submit,
            ..
        })
    ));

    let request = runtime
        .prepare_provider_execution(scheduler.operation(operation_id).unwrap())
        .unwrap();
    let handle = runtime.submit_provider_execution(request).unwrap();
    assert_eq!(
        runtime.cancel_provider_execution(&handle).unwrap(),
        ProviderCancellationOutcome::Accepted
    );
}
#[test]
fn provider_execution_diagnostics_redact_native_details() {
    let diagnostic = ProviderExecutionDiagnostic::new(
        ProviderBinding::new("provider"),
        ProviderExecutionPhase::Submit,
    )
    .with_detail("backend handle=0xdeadbeef");

    assert_eq!(
        diagnostic.detail.as_deref(),
        Some("[redacted backend diagnostic]")
    );
}
#[test]
fn scheduler_rejects_over_capacity_and_cancels_queued_work() {
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
    let graph = ComputeGraph::new(ComputeGraphId::new("queued")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor,
        )),
    );
    let plan = runtime.plan_compute_execution(&graph).unwrap();
    let mut scheduler = runtime.scheduler(1);
    let operation = scheduler.schedule(&runtime, plan.clone()).unwrap();

    assert!(matches!(
        scheduler.schedule(&runtime, plan),
        Err(SchedulerError::QueueCapacityExceeded { capacity: 1 })
    ));

    scheduler.cancel(operation).unwrap();
    assert_eq!(
        scheduler.operation(operation).unwrap().state(),
        SchedulingState::Cancelled
    );
    assert_eq!(
        scheduler.result(operation).unwrap().state,
        SchedulingState::Cancelled
    );
    assert_eq!(scheduler.submit_next(&runtime).unwrap(), None);
}
#[test]
fn scheduler_completion_exposes_terminal_result_without_native_handles() {
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
    let graph = ComputeGraph::new(ComputeGraphId::new("completed"))
        .with_node(
            ComputeNode::new(
                ComputeNodeId::new("node"),
                ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                    .with_dtype(ComputeDType::Float32)
                    .with_layout(ComputeLayout::Dense),
            )
            .with_output(ComputeNodeOutput::new(
                ComputeOutputId::new("out"),
                descriptor,
            )),
        )
        .with_output(ComputeOutput::new(
            ComputeOutputId::new("result"),
            ComputeValueRef::NodeOutput {
                node: ComputeNodeId::new("node"),
                output: ComputeOutputId::new("out"),
            },
        ));
    let plan = runtime.plan_compute_execution(&graph).unwrap();
    let mut scheduler = runtime.scheduler(1);
    let operation = scheduler.schedule(&runtime, plan).unwrap();

    scheduler.submit_next(&runtime).unwrap();
    scheduler.complete(operation).unwrap();

    let result = scheduler.result(operation).unwrap();
    assert_eq!(result.state, SchedulingState::Completed);
    assert_eq!(result.outputs.len(), 1);
    assert!(result.error.is_none());
}
#[test]
fn scheduler_interrupts_when_provider_is_unavailable_before_submission() {
    let mut healthy = provider_with_capabilities("portable-compute", [compute_capability()]);
    healthy.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let planning_runtime = Runtime::builder()
        .register_provider(Arc::new(healthy))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("interrupted")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor,
        )),
    );
    let plan = planning_runtime.plan_compute_execution(&graph).unwrap();
    let mut unavailable = provider_with_capabilities("portable-compute", [compute_capability()]);
    unavailable.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    unavailable.health = ProviderHealth::Unavailable;
    let submission_runtime = Runtime::builder()
        .register_provider(Arc::new(unavailable))
        .build()
        .unwrap();
    let mut scheduler = submission_runtime.scheduler(1);
    let operation = scheduler.schedule(&submission_runtime, plan).unwrap();

    assert!(matches!(
        scheduler.submit_next(&submission_runtime),
        Err(SchedulerError::ProviderUnavailable(provider))
            if provider.as_str() == "portable-compute"
    ));
    assert_eq!(
        scheduler.operation(operation).unwrap().state(),
        SchedulingState::Interrupted
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
fn memory_planning_rejects_provider_and_device_memory_limits() {
    let mut limited_provider =
        provider_with_capabilities("limited-compute", [compute_capability()]);
    limited_provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense])
            .with_descriptor_limits(TensorDescriptorLimits {
                max_bytes: 8,
                ..TensorDescriptorLimits::default()
            }),
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(limited_provider))
        .build()
        .unwrap();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let graph = ComputeGraph::new(ComputeGraphId::new("too-large")).with_node(
        ComputeNode::new(
            ComputeNodeId::new("node"),
            ComputeOperationDescriptor::new(ComputeOperationFamily::Elementwise)
                .with_dtype(ComputeDType::Float32)
                .with_layout(ComputeLayout::Dense),
        )
        .with_output(ComputeNodeOutput::new(
            ComputeOutputId::new("out"),
            descriptor.clone(),
        )),
    );
    assert!(matches!(
        runtime.plan_compute_graph_memory("limited-compute", &graph),
        Err(MemoryPlanningError::ProviderMemoryLimitExceeded { .. })
    ));

    let mut device_provider = provider_with_capabilities("device-limited", [compute_capability()]);
    device_provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    let mut device = DeviceMetadata::new(
        DeviceId::new("gpu:tiny"),
        "Tiny GPU",
        DeviceType::Gpu,
        "device-limited",
    );
    device.memory_capacity = 8;
    device_provider
        .devices
        .push(Arc::new(DeviceDescriptor::new(device)));
    let runtime = Runtime::builder()
        .register_provider(Arc::new(device_provider))
        .build()
        .unwrap();
    assert!(matches!(
        runtime.plan_compute_graph_memory("device-limited", &graph),
        Err(MemoryPlanningError::DeviceMemoryLimitExceeded { .. })
    ));
}
#[test]
fn compute_providers_register_and_resolve_compatibly() {
    let mut provider = TestProvider::new("portable-compute");
    provider.metadata.capabilities.insert(compute_capability());
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();

    assert_eq!(
        runtime
            .resolve_component_import(&WitInterface::new(
                COMPUTE_WIT_INTERFACE,
                COMPUTE_CAPABILITY_VERSION.to_string(),
            ))
            .unwrap()
            .len(),
        1
    );
}
#[test]
fn resolution_policy_records_selected_provider_capability_and_reason() {
    let compute = compute_capability();
    let provider_a = provider_with_capabilities("provider-a", [compute.clone()]);
    let provider_b = provider_with_capabilities("provider-b", [compute.clone()]);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider_b))
        .register_provider(Arc::new(provider_a))
        .build()
        .unwrap();

    let resolution = runtime
        .resolve_with_affinity(&compute, &[], FallbackClass::Transparent)
        .unwrap();
    let decision = resolution.decision();

    assert_eq!(resolution.provider().metadata().name, "provider-a");
    assert_eq!(
        decision
            .selected_provider
            .as_ref()
            .map(ProviderBinding::as_str),
        Some("provider-a")
    );
    assert_eq!(
        decision.selected_capability,
        Some(CapabilityBinding::new(compute.id.clone(), compute.version))
    );
    assert_eq!(
        decision.reason,
        ResolutionDecisionReason::SelectedDeterministically
    );
}
#[test]
fn policy_rejection_is_structured_when_all_candidates_are_unhealthy() {
    let compute = compute_capability();
    let mut provider = provider_with_capabilities("provider-a", [compute.clone()]);
    provider.health = ProviderHealth::Unavailable;
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();

    assert!(matches!(
        runtime.resolve_component_import(&WitInterface::new(
            COMPUTE_WIT_INTERFACE,
            COMPUTE_CAPABILITY_VERSION.to_string(),
        )),
        Err(ProviderError::PolicyRejectedProvider { capability, policy })
            if capability.id() == &compute.id
                && capability.version() == compute.version
                && policy == BuiltInResolutionPolicy::Deterministic.id()
    ));
}
#[test]
fn availability_policy_prefers_healthy_candidates() {
    let compute = compute_capability();
    let mut degraded = provider_with_capabilities("a-degraded", [compute.clone()]);
    degraded.health = ProviderHealth::Degraded;
    let healthy = provider_with_capabilities("z-healthy", [compute.clone()]);
    let runtime = Runtime::builder()
        .config(RuntimeConfig {
            resolution_policy: BuiltInResolutionPolicy::Availability,
            ..RuntimeConfig::default()
        })
        .register_provider(Arc::new(degraded))
        .register_provider(Arc::new(healthy))
        .build()
        .unwrap();

    let providers = runtime.resolve_component_import(&WitInterface::new(
        COMPUTE_WIT_INTERFACE,
        COMPUTE_CAPABILITY_VERSION.to_string(),
    ));

    assert_eq!(providers.unwrap()[0].metadata().name, "z-healthy");
}
#[test]
fn capability_health_rejects_unavailable_implementation() {
    let compute = compute_capability();
    let mut provider = provider_with_capabilities("provider-a", [compute.clone()]);
    provider
        .capability_health
        .insert(compute.id.clone(), HealthState::Unavailable);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();

    assert!(matches!(
        runtime.resolve_component_import(&WitInterface::new(
            COMPUTE_WIT_INTERFACE,
            COMPUTE_CAPABILITY_VERSION.to_string(),
        )),
        Err(ProviderError::PolicyRejectedProvider { .. })
    ));

    let candidates = runtime
        .providers()
        .candidates_for_capability(&compute)
        .unwrap();
    let context = ResolutionContext {
        requested_capability: compute.id.clone(),
        requested_version: compute.version,
        candidates,
        affinity: None,
        fallback: FallbackClass::Transparent,
        execution_phase: ExecutionPhase::BeforeResourceCreation,
        replayable_input: true,
    };
    let decision = BuiltInResolutionPolicy::Deterministic.decide(&context);
    assert_eq!(
        decision.rejected_candidates[0].reason,
        ResolutionRejectionReason::CapabilityUnavailable
    );
}
#[test]
fn health_reports_redact_diagnostics_and_track_freshness() {
    let provider = ProviderBinding::new("provider-a");
    let mut report = ProviderHealthReport::new(provider.clone(), HealthState::Saturated);
    report.timestamp = Some(HealthTimestamp::unix_millis(1_000));
    report.time_to_live = Some(HealthTimeToLive::millis(250));
    report.capacity.queue_depth = Some(8);
    report.diagnostics.push(
        HealthDiagnostic::new(HealthScope::Provider, HealthState::Saturated)
            .with_code("queue-pressure")
            .with_message("cuda stream /tmp/secret token=abc is saturated")
            .with_trace_id("trace-1"),
    );

    assert!(report.is_stale_at(HealthTimestamp::unix_millis(1_251)));
    assert_eq!(report.capacity.queue_depth, Some(8));
    let message = report.diagnostics[0].message.as_deref().unwrap();
    assert!(!message.contains("/tmp/secret"));
    assert!(!message.contains("token=abc"));
    assert_eq!(message, "[redacted backend diagnostic]");
    assert_eq!(report.diagnostics[0].trace_id.as_deref(), Some("trace-1"));

    let device_health = DeviceHealth::new(
        provider.clone(),
        DeviceBinding::new(DeviceId::new("gpu:0")),
        HealthState::Available,
    );
    let capability_health = CapabilityHealth::new(
        provider,
        CapabilityBinding::new(compute_capability().id, COMPUTE_CAPABILITY_VERSION),
        HealthState::Degraded,
    );
    assert!(matches!(
        HealthReport::Device(device_health),
        HealthReport::Device(_)
    ));
    assert!(matches!(
        HealthReport::Capability(capability_health),
        HealthReport::Capability(_)
    ));
}

#[test]
fn provider_status_snapshot_separates_health_readiness_pressure_and_admission() {
    let provider = ProviderBinding::new("provider-a");
    let mut snapshot = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        provider.clone(),
        HealthState::Available,
    ));
    snapshot.health = ProviderHealthState::Healthy;
    snapshot.readiness = ProviderReadinessState::NotReady;
    snapshot.pressure = ProviderPressureLevel::Low;
    snapshot.admission = provider_admission_from_dimensions(
        snapshot.lifecycle,
        snapshot.health,
        snapshot.readiness,
        snapshot.pressure,
    );
    snapshot.timestamp = Some(HealthTimestamp::unix_millis(10));
    snapshot.time_to_live = Some(HealthTimeToLive::millis(5));

    assert_eq!(snapshot.provider, provider);
    assert_eq!(snapshot.health, ProviderHealthState::Healthy);
    assert_eq!(snapshot.readiness, ProviderReadinessState::NotReady);
    assert_eq!(snapshot.pressure, ProviderPressureLevel::Low);
    assert_eq!(snapshot.admission, ProviderAdmissionDecision::Reject);
    assert!(!snapshot.accepts_new_work_by_default());
    assert!(snapshot.is_stale_at(HealthTimestamp::unix_millis(16)));
}

#[test]
fn provider_lifecycle_transitions_and_drain_completion_are_explicit() {
    assert!(ProviderLifecycleState::Registered.can_transition_to(ProviderLifecycleState::Loading));
    assert!(
        ProviderLifecycleState::Loading.can_transition_to(ProviderLifecycleState::Initializing)
    );
    assert!(ProviderLifecycleState::Initializing.can_transition_to(ProviderLifecycleState::Ready));
    assert!(ProviderLifecycleState::Ready.can_transition_to(ProviderLifecycleState::Draining));
    assert!(ProviderLifecycleState::Draining.can_transition_to(ProviderLifecycleState::Stopped));
    assert!(!ProviderLifecycleState::Ready.can_transition_to(ProviderLifecycleState::Removed));

    let mut draining = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        ProviderBinding::new("provider-a"),
        HealthState::Draining,
    ));
    draining.lifecycle = ProviderLifecycleState::Draining;
    draining.readiness = ProviderReadinessState::Draining;
    draining.in_flight_operations = 1;
    assert!(!draining.is_drain_complete());
    assert!(draining.pinned_work_allowed_during_drain());
    draining.in_flight_operations = 0;
    assert!(draining.is_drain_complete());
}

#[test]
fn operation_family_status_falls_back_to_capability_status_when_absent() {
    let provider = ProviderBinding::new("provider-a");
    let mut snapshot = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        provider.clone(),
        HealthState::Available,
    ));
    assert_eq!(
        snapshot.operation_family_or_capability_status(ComputeOperationFamily::LinearAlgebra),
        ProviderReadinessState::Ready
    );

    let unsupported =
        OperationFamilyStatus::unsupported(provider.clone(), ComputeOperationFamily::LinearAlgebra);
    snapshot = snapshot.with_operation_family_status(unsupported);
    assert_eq!(
        snapshot.operation_family_or_capability_status(ComputeOperationFamily::LinearAlgebra),
        ProviderReadinessState::NotReady
    );

    let mut saturated =
        OperationFamilyStatus::available(provider, ComputeOperationFamily::Elementwise);
    saturated.pressure = ProviderPressureLevel::Saturated;
    saturated.readiness = ProviderReadinessState::NotReady;
    snapshot = snapshot.with_operation_family_status(saturated);
    assert_eq!(
        snapshot
            .operation_family_status(ComputeOperationFamily::Elementwise)
            .unwrap()
            .pressure,
        ProviderPressureLevel::Saturated
    );
}

#[test]
fn provider_status_maps_interruption_to_health_readiness_and_admission() {
    let snapshot = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        ProviderBinding::new("provider-a"),
        HealthState::Interrupted,
    ));

    assert_eq!(snapshot.lifecycle, ProviderLifecycleState::Failed);
    assert_eq!(snapshot.health, ProviderHealthState::Failed);
    assert_eq!(snapshot.readiness, ProviderReadinessState::NotReady);
    assert_eq!(snapshot.admission, ProviderAdmissionDecision::Reject);
    assert_eq!(snapshot.health_reason, ProviderStatusReason::Interrupted);
    assert!(matches!(
        snapshot.interruption,
        Some(ProviderInterruptionReason::DriverLoss)
    ));
}

#[test]
fn provider_status_observations_cover_all_status_dimensions() {
    let mut report =
        ProviderHealthReport::new(ProviderBinding::new("provider-a"), HealthState::Draining);
    report.timestamp = Some(HealthTimestamp::unix_millis(10));
    report.time_to_live = Some(HealthTimeToLive::millis(5));
    report.devices.push(DeviceHealth::new(
        ProviderBinding::new("provider-a"),
        DeviceBinding::new(DeviceId::new("gpu:0")),
        HealthState::Available,
    ));
    report.capabilities.push(CapabilityHealth::new(
        ProviderBinding::new("provider-a"),
        CapabilityBinding::new(compute_capability().id, COMPUTE_CAPABILITY_VERSION),
        HealthState::Available,
    ));
    let mut snapshot = ProviderStatusSnapshot::from_health_report(report);
    snapshot.lifecycle = ProviderLifecycleState::Draining;
    snapshot.readiness = ProviderReadinessState::Draining;
    snapshot.in_flight_operations = 0;

    let events =
        runtime_events_for_provider_status(&snapshot, Some(HealthTimestamp::unix_millis(16)));
    let kinds = events
        .iter()
        .map(|event| event.kind.clone())
        .collect::<BTreeSet<_>>();

    assert!(kinds.contains(&RuntimeEventKind::ProviderLifecycleChanged));
    assert!(kinds.contains(&RuntimeEventKind::ProviderHealthChanged));
    assert!(kinds.contains(&RuntimeEventKind::ProviderReadinessChanged));
    assert!(kinds.contains(&RuntimeEventKind::ProviderPressureChanged));
    assert!(kinds.contains(&RuntimeEventKind::ProviderAdmissionChanged));
    assert!(kinds.contains(&RuntimeEventKind::ProviderStatusStale));
    assert!(kinds.contains(&RuntimeEventKind::ProviderDrainStarted));
    assert!(kinds.contains(&RuntimeEventKind::ProviderDrainCompleted));
    assert!(kinds.contains(&RuntimeEventKind::DeviceStatusChanged));
    assert!(kinds.contains(&RuntimeEventKind::CapabilityStatusChanged));
}

#[test]
fn resolution_rejects_healthy_provider_that_is_not_ready() {
    let compute = compute_capability();
    let mut provider = provider_with_capabilities("provider-a", [compute.clone()]);
    let mut snapshot = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        ProviderBinding::new("provider-a"),
        HealthState::Available,
    ));
    snapshot.health = ProviderHealthState::Healthy;
    snapshot.readiness = ProviderReadinessState::NotReady;
    snapshot.admission = provider_admission_from_dimensions(
        snapshot.lifecycle,
        snapshot.health,
        snapshot.readiness,
        snapshot.pressure,
    );
    provider.status_snapshot = Some(snapshot);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();

    let candidates = runtime
        .providers()
        .candidates_for_capability(&compute)
        .unwrap();
    let context = ResolutionContext {
        requested_capability: compute.id.clone(),
        requested_version: compute.version,
        candidates,
        affinity: None,
        fallback: FallbackClass::Transparent,
        execution_phase: ExecutionPhase::BeforeResourceCreation,
        replayable_input: true,
    };
    let decision = BuiltInResolutionPolicy::Deterministic.decide(&context);

    assert_eq!(decision.selected_provider, None);
    assert_eq!(
        decision.rejected_candidates[0].reason,
        ResolutionRejectionReason::ProviderInitializing
    );
}

#[test]
fn resolution_records_selected_provider_status_and_stale_rejection_reason() {
    let compute = compute_capability();
    let ready_provider = provider_with_capabilities("provider-a", [compute.clone()]);
    let mut stale_provider = provider_with_capabilities("provider-b", [compute.clone()]);
    let mut stale = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        ProviderBinding::new("provider-b"),
        HealthState::Available,
    ));
    stale.health_reason = ProviderStatusReason::Stale;
    stale_provider.status_snapshot = Some(stale);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(stale_provider))
        .register_provider(Arc::new(ready_provider))
        .build()
        .unwrap();

    let candidates = runtime
        .providers()
        .candidates_for_capability(&compute)
        .unwrap();
    let context = ResolutionContext {
        requested_capability: compute.id.clone(),
        requested_version: compute.version,
        candidates,
        affinity: None,
        fallback: FallbackClass::Transparent,
        execution_phase: ExecutionPhase::BeforeResourceCreation,
        replayable_input: true,
    };
    let decision = BuiltInResolutionPolicy::Deterministic.decide(&context);

    assert_eq!(
        decision
            .selected_provider_status
            .as_ref()
            .map(|status| status.provider.as_str()),
        Some("provider-a")
    );
    assert!(decision.rejected_candidates.iter().any(|candidate| {
        candidate.provider.as_str() == "provider-b"
            && candidate.reason == ResolutionRejectionReason::ProviderStatusStale
    }));
}

#[test]
fn scheduler_checks_refined_provider_status_before_submission() {
    let compute = compute_capability();
    let mut planning_provider = provider_with_capabilities("provider-a", [compute]);
    planning_provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    planning_provider.execution_api = Some(Arc::new(TestProviderExecutionApi::new()));
    let planning_runtime = Runtime::builder()
        .register_provider(Arc::new(planning_provider))
        .build()
        .unwrap();
    let graph = simple_elementwise_compute_graph("status-change");
    let plan = planning_runtime.plan_compute_execution(&graph).unwrap();

    let mut submission_provider = provider_with_capabilities("provider-a", [compute_capability()]);
    submission_provider
        .metadata
        .compute_operation_support
        .insert(
            ComputeOperationFamily::Elementwise,
            ComputeOperationSupport::new()
                .with_dtypes([ComputeDType::Float32])
                .with_layouts([ComputeLayout::Dense]),
        );
    submission_provider.execution_api = Some(Arc::new(TestProviderExecutionApi::new()));
    let mut snapshot = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        ProviderBinding::new("provider-a"),
        HealthState::Available,
    ));
    snapshot.health = ProviderHealthState::Healthy;
    snapshot.readiness = ProviderReadinessState::Ready;
    snapshot.pressure = ProviderPressureLevel::Saturated;
    snapshot.admission = provider_admission_from_dimensions(
        snapshot.lifecycle,
        snapshot.health,
        snapshot.readiness,
        snapshot.pressure,
    );
    submission_provider.status_snapshot = Some(snapshot);
    let submission_runtime = Runtime::builder()
        .register_provider(Arc::new(submission_provider))
        .build()
        .unwrap();
    let mut scheduler = submission_runtime.scheduler(1);
    let operation = scheduler.schedule(&submission_runtime, plan).unwrap();

    assert!(matches!(
        scheduler.submit_next(&submission_runtime),
        Err(SchedulerError::ProviderSaturated(provider)) if provider.as_str() == "provider-a"
    ));
    assert_eq!(
        scheduler.operation(operation).unwrap().state(),
        SchedulingState::Interrupted
    );
}

#[test]
fn scheduler_and_provider_execution_reject_stale_provider_status() {
    let mut provider = provider_with_capabilities("provider-a", [compute_capability()]);
    provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    provider.execution_api = Some(Arc::new(TestProviderExecutionApi::new()));
    let planning_runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let plan = planning_runtime
        .plan_compute_execution(&simple_elementwise_compute_graph("stale-status"))
        .unwrap();

    let mut stale_provider = provider_with_capabilities("provider-a", [compute_capability()]);
    stale_provider.metadata.compute_operation_support.insert(
        ComputeOperationFamily::Elementwise,
        ComputeOperationSupport::new()
            .with_dtypes([ComputeDType::Float32])
            .with_layouts([ComputeLayout::Dense]),
    );
    stale_provider.execution_api = Some(Arc::new(TestProviderExecutionApi::new()));
    let mut stale = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        ProviderBinding::new("provider-a"),
        HealthState::Available,
    ));
    stale.health_reason = ProviderStatusReason::Stale;
    stale_provider.status_snapshot = Some(stale);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(stale_provider))
        .build()
        .unwrap();
    let mut scheduler = runtime.scheduler(1);
    let operation = scheduler.schedule(&runtime, plan).unwrap();

    assert!(matches!(
        scheduler.submit_next(&runtime),
        Err(SchedulerError::StaleHealthReport(provider)) if provider.as_str() == "provider-a"
    ));
    assert!(matches!(
        runtime.prepare_provider_execution(scheduler.operation(operation).unwrap()),
        Err(error) if error.code == ProviderExecutionErrorCode::StaleHealthReport
    ));
}
#[test]
fn phase_aware_resolution_rejects_restart_after_observable_output() {
    let compute = compute_capability();
    let provider = provider_with_capabilities("provider-a", [compute.clone()]);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();

    assert!(matches!(
        runtime.resolve_with_affinity_at_phase(
            &compute,
            &[],
            FallbackClass::Restartable,
            ExecutionPhase::AfterObservableOutput,
            true,
        ),
        Err(AffinityError::PolicyRejectedProvider { .. })
    ));
}
#[test]
fn builder_isolates_failed_provider_initialization() {
    let mut failed = TestProvider::new("failed");
    failed.fail_initialization = true;
    let runtime = Runtime::builder()
        .register_provider(Arc::new(failed))
        .register_provider(Arc::new(TestProvider::new("available")))
        .build()
        .unwrap();
    assert!(runtime.providers().provider("failed").is_none());
    assert!(runtime.providers().provider("available").is_some());
}
#[test]
fn builder_reports_rejected_provider_instead_of_dropping_it_silently() {
    let mut failed = TestProvider::new("failed");
    failed.fail_initialization = true;
    let runtime = Runtime::builder()
        .register_provider(Arc::new(failed))
        .register_provider(Arc::new(TestProvider::new("available")))
        .build()
        .unwrap();

    let diagnostics = runtime.startup_diagnostics();
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].code, RuntimeDiagnosticCode::ProviderRejected);
    assert_eq!(
        diagnostics[0]
            .provider
            .as_ref()
            .map(ProviderBinding::as_str),
        Some("failed")
    );
    assert!(diagnostics[0].message.contains("failed"));
}
#[test]
fn builder_records_no_diagnostics_when_every_provider_registers() {
    let runtime = Runtime::builder()
        .register_provider(Arc::new(TestProvider::new("available")))
        .build()
        .unwrap();
    assert!(runtime.startup_diagnostics().is_empty());
}
#[test]
fn builder_does_not_register_kernels_for_a_rejected_provider() {
    let mut failed = TestProvider::new("failed");
    failed.fail_initialization = true;
    failed.kernel_advertisements = reference_cpu_kernel_advertisements();
    let runtime = Runtime::builder()
        .register_provider(Arc::new(failed))
        .build()
        .unwrap();

    // The Provider never came up, so its kernels must not be left in the
    // registry as candidates that can never resolve.
    assert!(runtime.providers().provider("failed").is_none());
    assert_eq!(runtime.startup_diagnostics().len(), 1);
}
#[test]
fn reject_incompatible() {
    let mut p = TestProvider::new("old");
    p.metadata.api_version += 1;
    assert!(matches!(
        ProviderLoader::new().register_provider(Arc::new(p)),
        Err(ProviderError::IncompatibleApiVersion { .. })
    ));
}
#[test]
fn reject_duplicate() {
    let mut m = ProviderLoader::new();
    m.register_provider(Arc::new(TestProvider::new("same")))
        .unwrap();
    assert!(matches!(
        m.register_provider(Arc::new(TestProvider::new("same"))),
        Err(ProviderError::ProviderAlreadyRegistered(_))
    ));
}
#[test]
fn provider_shutdown_releases_registered_provider() {
    let p = TestProvider::new("provider");
    let p = Arc::new(p);
    let mut m = ProviderLoader::new();
    m.register_provider(p.clone()).unwrap();
    assert!(m.provider("provider").is_some());
    m.shutdown().unwrap();
    assert!(p.shut_down.load(Ordering::SeqCst));
}

#[test]
fn dynamic_provider_loading_denies_paths_by_default() {
    let mut loader = ProviderLoader::new();
    let path = std::path::PathBuf::from("target/test-provider.dll");

    let result = unsafe { loader.load_dynamic(&path) };

    assert!(matches!(
        result,
        Err(ProviderError::ProviderPathDenied { path: denied }) if denied == path
    ));
}

#[test]
fn dynamic_provider_loading_rejects_legacy_trait_object_factory_contract() {
    let mut loader = ProviderLoader::new();
    let root = std::path::PathBuf::from("target");
    let path = root.join("test-provider.dll");
    let policy = ProviderLoadingPolicy::development([root]);

    let result = unsafe { loader.load_dynamic_with_policy(&path, &policy) };

    assert!(matches!(
        result,
        Err(ProviderError::UnsupportedDynamicAbi {
            expected_symbol: PROVIDER_ABI_FACTORY_SYMBOL_V1,
            expected_descriptor,
            ..
        }) if expected_descriptor.abi_version == ProviderAbiVersion::CURRENT
    ));
    assert!(
        !include_str!("provider.rs").contains("Box<dyn Provider>"),
        "stable dynamic loading must not use Rust trait-object factories"
    );
}

#[test]
fn provider_abi_descriptor_validates_version_layout_functions_and_ownership() {
    let descriptor = ProviderAbiDescriptor::current();
    descriptor.validate().unwrap();
    assert_eq!(descriptor.abi_version, ProviderAbiVersion::CURRENT);
    assert!(descriptor.features.contains(&ProviderAbiFeature::Execution));
    assert_eq!(
        descriptor.threading,
        ProviderAbiThreadingModel::RuntimeSynchronized
    );
    assert_eq!(
        descriptor.execution_behavior,
        ProviderAbiExecutionBehavior::Blocking
    );
    assert_eq!(
        descriptor.unload_policy,
        ProviderAbiUnloadPolicy::NeverUnload
    );
    assert!(descriptor.ownership.strings.release_required);
    assert!(!descriptor.ownership.runtime_buffers.release_required);

    let mut too_small = descriptor.clone();
    too_small.descriptor_size = 1;
    assert!(matches!(
        too_small.validate(),
        Err(ProviderError::InvalidAbiDescriptor(_))
    ));

    let mut unsupported_major = descriptor.clone();
    unsupported_major.abi_version = ProviderAbiVersion::new(PROVIDER_ABI_MAJOR_VERSION + 1, 0);
    assert!(matches!(
        unsupported_major.validate(),
        Err(ProviderError::UnsupportedAbiVersion { .. })
    ));

    let mut unsupported_minor = descriptor.clone();
    unsupported_minor.abi_version =
        ProviderAbiVersion::new(PROVIDER_ABI_MAJOR_VERSION, PROVIDER_ABI_MINOR_VERSION + 1);
    assert!(matches!(
        unsupported_minor.validate(),
        Err(ProviderError::UnsupportedAbiVersion { .. })
    ));

    let mut missing_status = descriptor.clone();
    missing_status.functions.status = false;
    assert!(matches!(
        missing_status.validate(),
        Err(ProviderError::InvalidAbiDescriptor(_))
    ));

    let mut cross_allocator = descriptor;
    cross_allocator.ownership.error_messages = ProviderAbiMemoryRule::runtime_borrowed();
    assert!(matches!(
        cross_allocator.validate(),
        Err(ProviderError::InvalidAbiDescriptor(_))
    ));
}

#[test]
fn provider_abi_handles_lifecycle_and_errors_are_internal_runtime_contracts() {
    let instance = ProviderAbiHandleDescriptor::new(
        ProviderAbiHandleKind::ProviderInstance,
        ProviderAbiHandle::new(7),
    );
    let resource = ProviderAbiHandleDescriptor::new(
        ProviderAbiHandleKind::ProviderResource,
        ProviderAbiHandle::new(8),
    );
    let operation = ProviderAbiHandleDescriptor::new(
        ProviderAbiHandleKind::Operation,
        ProviderAbiHandle::new(9),
    );

    assert!(instance.destroy_required);
    assert_eq!(resource.handle.as_u64(), 8);
    assert_eq!(operation.kind, ProviderAbiHandleKind::Operation);
    assert!(
        ProviderAbiLoadingLifecycle::Discovered
            .can_transition_to(ProviderAbiLoadingLifecycle::LibraryLoaded)
    );
    assert!(
        !ProviderAbiLoadingLifecycle::Discovered
            .can_transition_to(ProviderAbiLoadingLifecycle::Registered)
    );
    assert!(
        ProviderAbiLoadingLifecycle::Failed
            .can_transition_to(ProviderAbiLoadingLifecycle::Destroyed)
    );

    let categories = [
        ProviderAbiErrorCode::InvalidAbiDescriptor,
        ProviderAbiErrorCode::UnsupportedAbiVersion,
        ProviderAbiErrorCode::InvalidMetadata,
        ProviderAbiErrorCode::InvalidAdvertisement,
        ProviderAbiErrorCode::InvalidDeviceMetadata,
        ProviderAbiErrorCode::InitializationFailure,
        ProviderAbiErrorCode::ProviderNotReady,
        ProviderAbiErrorCode::ProviderDraining,
        ProviderAbiErrorCode::ProviderSaturated,
        ProviderAbiErrorCode::ExecutionRejected,
        ProviderAbiErrorCode::ExecutionFailed,
        ProviderAbiErrorCode::CancellationUnsupported,
        ProviderAbiErrorCode::CancellationFailed,
        ProviderAbiErrorCode::ResourceInvalid,
        ProviderAbiErrorCode::InternalProviderError,
        ProviderAbiErrorCode::PanicOrUnwindViolation,
    ];
    assert_eq!(categories.len(), 16);

    let compute_error = ComputeError::from(ProviderError::PanicOrUnwindViolation(
        "panic crossed boundary".into(),
    ));
    assert_eq!(compute_error.code, ComputeErrorCode::ProviderUnavailable);
    assert_eq!(compute_error.phase, ComputeErrorPhase::Resolution);
}

#[test]
fn provider_loading_policy_is_explicit_for_dynamic_and_development_modes() {
    let root = std::path::PathBuf::from("target/providers");
    let provider = root.join("provider.dll");
    let outside = std::path::PathBuf::from("target/other/provider.dll");
    let dynamic = ProviderLoadingPolicy::dynamic_library([root.clone()]);
    let development = ProviderLoadingPolicy::development([root.clone()]);

    assert_eq!(dynamic.mode, ProviderLoadingMode::DynamicLibrary);
    assert!(!dynamic.development_mode);
    assert!(dynamic.allows(&provider));
    assert!(!dynamic.allows(&outside));
    assert_eq!(development.mode, ProviderLoadingMode::DevelopmentProvider);
    assert!(development.development_mode);
    assert!(development.allows(&provider));
    assert!(!ProviderLoadingPolicy::default().allows(&provider));
}

#[test]
fn runtime_source_does_not_restore_legacy_backend_or_plugin_surface() {
    let source = include_str!("lib.rs");
    let forbidden = [
        concat!("trait ", "Backend"),
        concat!("register_", "backend"),
        concat!("preferred_", "backend"),
        concat!("backend_", "names"),
        concat!("trait ", "Plugin"),
        concat!("Plugin", "Registry"),
    ];

    for term in forbidden {
        assert!(
            !source.contains(term),
            "legacy architecture surface remains in runtime source: {term}"
        );
    }
}

#[test]
fn public_component_api_does_not_expose_wasmtime_native_types() {
    let public_sources = [include_str!("lib.rs"), include_str!("component.rs")];
    let forbidden = [
        "wasmtime::Engine",
        "wasmtime::Config",
        "wasmtime::Store",
        "wasmtime::component::Component",
        "wasmtime::component::Linker",
        "wasmtime::component::Instance",
        "wasmtime::Trap",
        "wasmtime::Error",
    ];

    for source in public_sources {
        for term in forbidden {
            assert!(
                !source.contains(term),
                "public Component API exposes concrete engine type: {term}"
            );
        }
    }
}

#[test]
fn component_engine_profiles_declare_platform_capabilities() {
    let native = ComponentEngineCapabilities::native();
    assert_eq!(native.profile, ComponentEngineProfile::Native);
    assert!(native.supports(ComponentEngineFeature::ComponentModel));
    assert!(native.supports(ComponentEngineFeature::NativeProviderEndpoints));
    assert!(!native.supports(ComponentEngineFeature::BrowserCompatible));

    let web = ComponentEngineCapabilities::web();
    assert_eq!(web.profile, ComponentEngineProfile::Web);
    assert!(web.supports(ComponentEngineFeature::BrowserCompatible));
    assert!(web.supports(ComponentEngineFeature::JsMediatedHostCalls));
    assert!(web.supports(ComponentEngineFeature::BrowserMemory));
    assert!(!web.supports(ComponentEngineFeature::NativeProviderEndpoints));

    let test = ComponentEngineCapabilities::test();
    assert_eq!(test.profile, ComponentEngineProfile::Test);
    assert!(test.supports(ComponentEngineFeature::Interruption));
    assert!(!test.supports(ComponentEngineFeature::ControlledWasi));
}

#[test]
fn component_engine_requirements_fail_closed_on_profile_or_feature_mismatch() {
    let requirements = ComponentEngineRequirements::default()
        .require_profile(ComponentEngineProfile::Web)
        .require_feature(ComponentEngineFeature::BrowserCompatible);

    assert!(
        requirements
            .validate("browser-component", &ComponentEngineCapabilities::web())
            .is_ok()
    );

    assert!(matches!(
        requirements.validate("browser-component", &ComponentEngineCapabilities::native()),
        Err(ComponentError::EngineProfileMismatch {
            required: ComponentEngineProfile::Web,
            actual: ComponentEngineProfile::Native,
            ..
        })
    ));

    let requirements = ComponentEngineRequirements::default()
        .require_feature(ComponentEngineFeature::NativeProviderEndpoints);
    assert!(matches!(
        requirements.validate("native-component", &ComponentEngineCapabilities::web()),
        Err(ComponentError::EngineFeatureUnavailable {
            feature: ComponentEngineFeature::NativeProviderEndpoints,
            profile: ComponentEngineProfile::Web,
            ..
        })
    ));
}

#[test]
fn component_manager_observes_engine_selection_and_rejection() {
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component"),
            "component.wasm",
        ))
        .unwrap();

    manager.prepare_component("component").unwrap();
    assert!(
        manager.observations().iter().any(|observation| {
            observation.kind == ComponentObservationKind::EngineSelection
                && observation
                    .message
                    .contains(ComponentEngineProfile::Test.as_str())
        }),
        "selected engine profile should be observable"
    );

    let error = ComponentEngineRequirements::default()
        .require_feature(ComponentEngineFeature::ControlledWasi)
        .validate("component", &manager.engine_capabilities())
        .unwrap_err();
    assert!(matches!(
        error,
        ComponentError::EngineFeatureUnavailable {
            feature: ComponentEngineFeature::ControlledWasi,
            ..
        }
    ));
}

#[test]
fn execution_context_default_allocates_unique_ids() {
    let first = ExecutionContext::default();
    let second = ExecutionContext::default();
    assert_ne!(first.id(), second.id());
    assert_ne!(first.id(), ExecutionContextId::default());
}

#[test]
fn affinity_constraints_preserve_compatible_facts_and_fallback_precedence() {
    let capability_a = capability_binding("magnetar:compute/run", CapabilityVersion::new(1, 1, 0));
    let capability_b = capability_binding("magnetar:tokenize/run", CapabilityVersion::new(1, 0, 0));
    let provider = ProviderBinding::new("provider-a");
    let device = DeviceBinding::new(DeviceId::new("gpu:0"));
    let context = ExecutionContextId::new(42);
    let group = AffinityGroupId::new(7);

    let model = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(provider.clone())
        .with_device(device.clone())
        .with_capability(capability_a.clone())
        .with_artifact(ArtifactBinding::new("model", "sha256:model"))
        .with_artifact(ArtifactBinding::new("bundle", "sha256:bundle"))
        .with_execution_context(context)
        .with_group(group);
    let tokenizer = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(provider)
        .with_device(device)
        .with_capability(capability_b.clone())
        .with_artifact(ArtifactBinding::new("tokenizer", "sha256:tokenizer"))
        .with_artifact(ArtifactBinding::new("bundle", "sha256:bundle"))
        .with_execution_context(context)
        .with_group(group);

    let constraints = AffinityConstraints::try_from_affinities([&model, &tokenizer]).unwrap();
    let aggregate = constraints.affinity();
    assert_eq!(aggregate.capability(capability_a.id()), Some(&capability_a));
    assert_eq!(aggregate.capability(capability_b.id()), Some(&capability_b));
    assert_eq!(
        aggregate.artifact("model").unwrap().fingerprint(),
        "sha256:model"
    );
    assert_eq!(
        aggregate.artifact("tokenizer").unwrap().fingerprint(),
        "sha256:tokenizer"
    );
    assert_eq!(
        aggregate.artifact("bundle").unwrap().fingerprint(),
        "sha256:bundle"
    );
    assert_eq!(aggregate.fallback(), FallbackClass::ProviderPinned);
}

#[test]
fn affinity_constraints_report_each_binding_conflict() {
    let base = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(ProviderBinding::new("provider-a"))
        .with_device(DeviceBinding::new(DeviceId::new("gpu:0")))
        .with_capability(capability_binding(
            "magnetar:compute/run",
            CapabilityVersion::new(1, 1, 0),
        ))
        .with_artifact(ArtifactBinding::new("bundle", "sha256:a"))
        .with_execution_context(ExecutionContextId::new(1))
        .with_group(AffinityGroupId::new(1));

    let provider_conflict = base
        .clone()
        .with_provider(ProviderBinding::new("provider-b"));
    assert!(matches!(
        base.validate_with(&provider_conflict),
        Err(AffinityError::ProviderMismatch { .. })
    ));

    let device_conflict = base
        .clone()
        .with_device(DeviceBinding::new(DeviceId::new("gpu:1")));
    assert!(matches!(
        base.validate_with(&device_conflict),
        Err(AffinityError::DeviceMismatch { .. })
    ));

    let capability_conflict = base.clone().with_capability(capability_binding(
        "magnetar:compute/run",
        CapabilityVersion::new(1, 2, 0),
    ));
    assert!(matches!(
        base.validate_with(&capability_conflict),
        Err(AffinityError::CapabilityMismatch { .. })
    ));

    let artifact_conflict = base
        .clone()
        .with_artifact(ArtifactBinding::new("bundle", "sha256:b"));
    assert!(matches!(
        base.validate_with(&artifact_conflict),
        Err(AffinityError::ArtifactMismatch { .. })
    ));

    let context_conflict = base
        .clone()
        .with_execution_context(ExecutionContextId::new(2));
    assert!(matches!(
        base.validate_with(&context_conflict),
        Err(AffinityError::ExecutionContextMismatch { .. })
    ));

    let group_conflict = base.clone().with_group(AffinityGroupId::new(2));
    assert!(matches!(
        base.validate_with(&group_conflict),
        Err(AffinityError::AffinityGroupMismatch { .. })
    ));
}

#[test]
fn affinity_resource_keeps_value_and_affinity_together() {
    let affinity = ResourceAffinity::new(FallbackClass::Restartable)
        .with_provider(ProviderBinding::new("provider-a"));
    let resource = AffinityResource::new("native-handle", affinity.clone());

    assert_eq!(resource.value(), &"native-handle");
    assert_eq!(resource.affinity(), &affinity);
    assert_eq!(resource.into_parts(), ("native-handle", affinity));
}

#[test]
fn affinity_resolution_uses_provider_local_compatible_version() {
    let requested = capability("magnetar:compute/run", CapabilityVersion::new(1, 1, 0));
    let provider_a = provider_with_capabilities(
        "provider-a",
        [capability(
            "magnetar:compute/run",
            CapabilityVersion::new(1, 1, 0),
        )],
    );
    let provider_b = provider_with_capabilities(
        "provider-b",
        [capability(
            "magnetar:compute/run",
            CapabilityVersion::new(1, 2, 0),
        )],
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider_a))
        .register_provider(Arc::new(provider_b))
        .build()
        .unwrap();
    let dependency = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(ProviderBinding::new("provider-a"));

    let resolution = runtime
        .resolve_with_affinity(&requested, &[&dependency], FallbackClass::ProviderPinned)
        .unwrap();
    assert_eq!(resolution.provider().metadata().name, "provider-a");
    assert_eq!(
        resolution.capability().version,
        CapabilityVersion::new(1, 1, 0)
    );
    assert_eq!(
        resolution
            .affinity()
            .provider()
            .map(ProviderBinding::as_str),
        Some("provider-a")
    );
}

#[test]
fn affinity_resolution_preserves_exact_live_capability_version() {
    let requested = capability("magnetar:compute/run", CapabilityVersion::new(1, 1, 0));
    let provider = provider_with_capabilities(
        "provider-a",
        [
            capability("magnetar:compute/run", CapabilityVersion::new(1, 1, 0)),
            capability("magnetar:compute/run", CapabilityVersion::new(1, 2, 0)),
        ],
    );
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let dependency = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(ProviderBinding::new("provider-a"))
        .with_capability(capability_binding(
            "magnetar:compute/run",
            CapabilityVersion::new(1, 1, 0),
        ));

    let resolution = runtime
        .resolve_with_affinity(&requested, &[&dependency], FallbackClass::ProviderPinned)
        .unwrap();
    assert_eq!(
        resolution.capability().version,
        CapabilityVersion::new(1, 1, 0)
    );
}

#[test]
fn affinity_resolution_requires_selected_provider_to_implement_all_bound_capabilities() {
    let compute = capability("magnetar:compute/run", CapabilityVersion::new(1, 1, 0));
    let provider = provider_with_capabilities("provider-a", [compute.clone()]);
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let dependency = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(ProviderBinding::new("provider-a"))
        .with_capability(capability_binding(
            "magnetar:tokenize/run",
            CapabilityVersion::new(1, 0, 0),
        ));

    assert!(matches!(
        runtime.resolve_with_affinity(&compute, &[&dependency], FallbackClass::ProviderPinned),
        Err(AffinityError::ProviderDoesNotImplementCapability { .. })
    ));
}

#[test]
fn affinity_resolution_reconciles_devices_with_provider_ownership() {
    let compute = compute_capability();
    let device_id = DeviceId::new("gpu:0");
    let mut provider = provider_with_capabilities("provider-a", [compute.clone()]);
    provider
        .devices
        .push(Arc::new(DeviceDescriptor::new(DeviceMetadata::new(
            device_id.clone(),
            "test gpu",
            DeviceType::Gpu,
            "provider-a",
        ))));
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .unwrap();
    let dependency = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_device(DeviceBinding::new(device_id.clone()));

    let resolution = runtime
        .resolve_with_affinity(&compute, &[&dependency], FallbackClass::ProviderPinned)
        .unwrap();
    assert_eq!(
        resolution.affinity().device().map(DeviceBinding::id),
        Some(&device_id)
    );
    assert_eq!(
        resolution
            .affinity()
            .provider()
            .map(ProviderBinding::as_str),
        Some("provider-a")
    );

    let mismatched = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(ProviderBinding::new("other"))
        .with_device(DeviceBinding::new(device_id));
    assert!(matches!(
        runtime.resolve_with_affinity(&compute, &[&mismatched], FallbackClass::ProviderPinned),
        Err(AffinityError::DeviceProviderMismatch { .. })
    ));

    let missing = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_device(DeviceBinding::new(DeviceId::new("missing")));
    assert!(matches!(
        runtime.resolve_with_affinity(&compute, &[&missing], FallbackClass::ProviderPinned),
        Err(AffinityError::BoundDeviceUnavailable(_))
    ));
}

#[test]
fn affinity_resolution_reports_unavailable_bound_provider_without_fallback() {
    let compute = compute_capability();
    let mut fallback = provider_with_capabilities("fallback", [compute.clone()]);
    fallback.metadata.capabilities.insert(compute.clone());
    let runtime = Runtime::builder()
        .register_provider(Arc::new(fallback))
        .build()
        .unwrap();
    let dependency = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(ProviderBinding::new("missing"));

    assert!(matches!(
        runtime.resolve_with_affinity(&compute, &[&dependency], FallbackClass::ProviderPinned),
        Err(AffinityError::BoundProviderUnavailable(provider)) if provider.as_str() == "missing"
    ));
}

#[test]
fn affinity_resolution_rejects_foreign_context_and_preserves_groups() {
    let compute = compute_capability();
    let first = Runtime::builder()
        .register_provider(Arc::new(provider_with_capabilities(
            "provider-a",
            [compute.clone()],
        )))
        .build()
        .unwrap();
    let second = Runtime::builder()
        .register_provider(Arc::new(provider_with_capabilities(
            "provider-a",
            [compute.clone()],
        )))
        .build()
        .unwrap();

    let ungrouped = first
        .resolve_with_affinity(&compute, &[], FallbackClass::ProviderPinned)
        .unwrap()
        .into_affinity();
    assert_eq!(ungrouped.group(), None);

    let grouped = first
        .resolve_with_affinity(&compute, &[&ungrouped], FallbackClass::ProviderPinned)
        .unwrap()
        .into_affinity();
    assert!(grouped.group().is_some());

    let inherited = first
        .resolve_with_affinity(&compute, &[&grouped], FallbackClass::ProviderPinned)
        .unwrap()
        .into_affinity();
    assert_eq!(inherited.group(), grouped.group());

    assert!(matches!(
        second.resolve_with_affinity(&compute, &[&grouped], FallbackClass::ProviderPinned),
        Err(AffinityError::ExecutionContextMismatch { .. })
    ));
}

#[test]
fn affinity_resolution_rejects_shutdown_runtime() {
    let compute = compute_capability();
    let mut runtime = Runtime::builder()
        .register_provider(Arc::new(provider_with_capabilities(
            "provider-a",
            [compute.clone()],
        )))
        .build()
        .unwrap();
    runtime.shutdown();

    assert!(matches!(
        runtime.resolve_with_affinity(&compute, &[], FallbackClass::Transparent),
        Err(AffinityError::RuntimeNotInitialized)
    ));
}

#[test]
fn component_runtime_instantiates_without_generic_start_or_stop() {
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component"),
            "component.wasm",
        ))
        .unwrap();

    let instance = manager.instantiate_component("component").unwrap();
    assert_eq!(
        manager.definition_state("component"),
        Some(ComponentDefinitionState::Prepared)
    );
    assert_eq!(
        manager.instance_state(instance),
        Some(ComponentInstanceState::Ready)
    );

    manager.shutdown();
    assert_eq!(
        manager.instance_state(instance),
        None,
        "shutdown removes Runtime-owned Component instances"
    );
}

#[test]
fn component_artifact_reference_prepares_future_artifact_model_without_trust_policy() {
    let descriptor = ComponentDescriptor::new(
        ComponentMetadata::new("component", "1", "test component"),
        "component.wasm",
    );

    assert!(matches!(
        descriptor.artifact_reference(),
        ComponentArtifactReference::LocalPath(path) if path == std::path::Path::new("component.wasm")
    ));
}

#[test]
fn component_imports_are_authorized_and_linked_explicitly() {
    let interface = WitInterface::new("magnetar:runtime/run", "1.0.0");
    let metadata =
        ComponentMetadata::new("consumer", "1", "test component").with_import(interface.clone());
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(metadata, "consumer.wasm"))
        .unwrap();

    assert!(matches!(
        manager.instantiate_component("consumer"),
        Err(ComponentError::UnauthorizedImport { .. })
    ));

    manager.authorize_interface(interface.clone());
    assert!(matches!(
        manager.instantiate_component("consumer"),
        Err(ComponentError::UnresolvedImport { .. })
    ));

    manager.provide_interface(interface);
    let instance = manager.instantiate_component("consumer").unwrap();
    assert_eq!(
        manager.instance_state(instance),
        Some(ComponentInstanceState::Ready)
    );
}

#[test]
fn component_import_version_must_match_authorized_interface() {
    let authorized = WitInterface::new("magnetar:runtime/run", "1.0.0");
    let requested = WitInterface::new("magnetar:runtime/run", "2.0.0");
    let metadata = ComponentMetadata::new("consumer", "1", "test component").with_import(requested);
    let mut manager = ComponentManager::new();
    manager.provide_interface(authorized);
    manager
        .register_component(ComponentDescriptor::new(metadata, "consumer.wasm"))
        .unwrap();

    assert!(matches!(
        manager.instantiate_component("consumer"),
        Err(ComponentError::UnauthorizedImport { .. })
    ));
}

#[test]
fn component_wasi_imports_fail_closed_without_authorization() {
    let filesystem = WitInterface::new("wasi:filesystem/types", "0.2.0");
    let environment = WitInterface::new("wasi:cli/environment", "0.2.0");
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("wasi-consumer", "1", "test component")
                .with_import(filesystem)
                .with_import(environment),
            "wasi-consumer.wasm",
        ))
        .unwrap();

    assert!(matches!(
        manager.instantiate_component("wasi-consumer"),
        Err(ComponentError::UnauthorizedImport { .. })
    ));
}

#[test]
fn component_ambient_network_process_and_secret_imports_fail_closed() {
    let interfaces = [
        WitInterface::new("wasi:sockets/tcp", "0.2.0"),
        WitInterface::new("wasi:cli/run", "0.2.0"),
        WitInterface::new("magnetar:secrets/read", "1.0.0"),
    ];
    for (index, interface) in interfaces.into_iter().enumerate() {
        let name = format!("authority-{index}");
        let mut manager = ComponentManager::new();
        manager
            .register_component(ComponentDescriptor::new(
                ComponentMetadata::new(&name, "1", "test component").with_import(interface),
                format!("{name}.wasm"),
            ))
            .unwrap();

        assert!(matches!(
            manager.instantiate_component(&name),
            Err(ComponentError::UnauthorizedImport { .. })
        ));
    }
}

#[test]
fn component_link_plan_is_runtime_owned_and_immutable_to_callers() {
    let interface = WitInterface::new("magnetar:runtime/run", "1.0.0");
    let metadata =
        ComponentMetadata::new("consumer", "1", "test component").with_import(interface.clone());
    let mut manager = ComponentManager::new();
    manager.provide_interface(interface.clone());
    manager
        .register_component(ComponentDescriptor::new(metadata, "consumer.wasm"))
        .unwrap();

    let plan = manager.link_plan("consumer").unwrap();
    let links = plan.links().collect::<Vec<_>>();
    assert_eq!(links.len(), 1);
    assert_eq!(links[0].0, &interface);
    assert!(matches!(
        links[0].1,
        ComponentEndpoint::Capability { interface: linked } if linked == &interface
    ));
    assert_eq!(plan.endpoint(&interface), Some(links[0].1));
}

#[test]
fn component_link_plan_rejects_forbidden_external_interfaces_even_if_provided() {
    for interface in [
        WitInterface::new("wasi:filesystem/types", "0.2.0"),
        WitInterface::new("wasi:sockets/tcp", "0.2.0"),
        WitInterface::new("magnetar:workspace/read", "1.0.0"),
        WitInterface::new("magnetar:git/status", "1.0.0"),
        WitInterface::new("magnetar:process/run", "1.0.0"),
        WitInterface::new("magnetar:secrets/read", "1.0.0"),
    ] {
        let mut manager = ComponentManager::new();
        manager.provide_interface(interface.clone());
        manager
            .register_component(ComponentDescriptor::new(
                ComponentMetadata::new("external", "1", "external component")
                    .with_import(interface),
                "external.wasm",
            ))
            .unwrap();

        assert!(matches!(
            manager.link_plan("external"),
            Err(ComponentError::UnauthorizedImport { .. })
        ));
    }
}

#[test]
fn component_authority_requirements_map_to_inference_runtime_endpoints() {
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "compute-capability".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::Capability { interface }
            if interface == WitInterface::new("magnetar:compute/run", "2.0.0")
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "model-artifact-read".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceArtifactRegistry {
            kind: InferenceArtifactKind::Model
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "tokenizer-artifact-read".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceArtifactRegistry {
            kind: InferenceArtifactKind::Tokenizer
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "prompt-template-read".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceArtifactRegistry {
            kind: InferenceArtifactKind::PromptTemplate
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "adapter-artifact-read".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceArtifactRegistry {
            kind: InferenceArtifactKind::Adapter
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "quantization-artifact-read".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceArtifactRegistry {
            kind: InferenceArtifactKind::Quantization
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "kv-cache-access".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceCacheService {
            kind: InferenceCacheKind::Kv
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "prefix-cache-access".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceCacheService {
            kind: InferenceCacheKind::Prefix
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "observability-emit".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::Observability
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "runtime-diagnostics".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::RuntimeDiagnostics
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "generation-capability".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::PendingRuntimeService { .. }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "sampling-capability".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::PendingRuntimeService { .. }
    ));
}

#[test]
fn inference_artifact_registry_uses_identities_not_paths_and_scopes_sessions() {
    let mut manager = ComponentManager::new();
    let digest = ComponentDigest::sha256(b"model");
    let session = InferenceSessionId::new("session-a").unwrap();
    manager
        .register_inference_artifact(
            InferenceArtifactReference::new(InferenceArtifactKind::Model, "qwen-model", digest)
                .unwrap()
                .with_session(session.clone()),
        )
        .unwrap();

    let artifact = manager
        .resolve_inference_artifact(InferenceArtifactKind::Model, "qwen-model", Some(&session))
        .unwrap();
    assert_eq!(artifact.id, "qwen-model");
    assert!(matches!(
        manager.resolve_inference_artifact(InferenceArtifactKind::Model, "../qwen-model", None),
        Err(ComponentError::ArtifactRejected { .. })
    ));
    assert!(matches!(
        manager.resolve_inference_artifact(
            InferenceArtifactKind::Model,
            "qwen-model",
            Some(&InferenceSessionId::new("session-b").unwrap())
        ),
        Err(ComponentError::ArtifactRejected { .. })
    ));
}

#[test]
fn inference_artifact_registry_handles_tokenizer_template_adapter_and_quantization() {
    let mut registry = InferenceArtifactRegistry::default();
    for kind in [
        InferenceArtifactKind::Tokenizer,
        InferenceArtifactKind::PromptTemplate,
        InferenceArtifactKind::Adapter,
        InferenceArtifactKind::Quantization,
    ] {
        let id = format!("{kind:?}").to_ascii_lowercase();
        registry
            .register(
                InferenceArtifactReference::new(kind, &id, ComponentDigest::sha256(id.as_bytes()))
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(registry.resolve(kind, &id, None).unwrap().kind, kind);
    }
    assert!(matches!(
        registry.resolve(InferenceArtifactKind::Tokenizer, "C:\\tokenizer.json", None),
        Err(ComponentError::ArtifactRejected { .. })
    ));
}

#[test]
fn inference_cache_registry_scopes_access_to_session_and_model() {
    let mut registry = InferenceCacheRegistry::default();
    let session = InferenceSessionId::new("session-a").unwrap();
    let authorized =
        InferenceCacheScope::new(InferenceCacheKind::Kv, session.clone(), "qwen-model").unwrap();
    registry.authorize(authorized.clone());

    registry.authorize_access(&authorized).unwrap();
    assert!(matches!(
        registry.authorize_access(
            &InferenceCacheScope::new(InferenceCacheKind::Kv, session, "other-model").unwrap()
        ),
        Err(ComponentError::ArtifactRejected { .. })
    ));
    assert!(matches!(
        registry.authorize_access(
            &InferenceCacheScope::new(
                InferenceCacheKind::Prefix,
                InferenceSessionId::new("session-b").unwrap(),
                "qwen-model"
            )
            .unwrap()
        ),
        Err(ComponentError::ArtifactRejected { .. })
    ));
}

#[test]
fn prepared_component_contract_must_match_declared_imports() {
    let declared = WitInterface::new("example:declared/api", "1.0.0");
    let undeclared = WitInterface::new("example:undeclared/api", "1.0.0");
    let mut contract = ComponentContract::default();
    contract.imports.insert(ComponentImportRequirement::new(
        undeclared,
        ComponentInterfaceShape::Interface,
    ));
    let mut engine = MockComponentEngine::new();
    engine.prepared_contract = Some(contract);
    let mut manager = ComponentManager::with_engine(Box::new(engine));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("consumer", "1", "test component").with_import(declared),
            "consumer.wasm",
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("consumer"),
        Err(ComponentError::ContractValidationFailed { .. })
    ));
}

#[test]
fn prepared_component_contract_must_include_declared_exports() {
    let exported = WitInterface::new("example:export/api", "1.0.0");
    let mut engine = MockComponentEngine::new();
    engine.prepared_contract = Some(ComponentContract::default());
    let mut manager = ComponentManager::with_engine(Box::new(engine));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("producer", "1", "test component").with_export(exported),
            "producer.wasm",
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("producer"),
        Err(ComponentError::ContractValidationFailed { .. })
    ));
}

#[test]
fn component_exports_do_not_automatically_satisfy_imports() {
    let interface = WitInterface::new("example:service/api", "1.0.0");
    let producer =
        ComponentMetadata::new("producer", "1", "producer").with_export(interface.clone());
    let consumer = ComponentMetadata::new("consumer", "1", "consumer").with_import(interface);
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(producer, "producer.wasm"))
        .unwrap();
    manager
        .register_component(ComponentDescriptor::new(consumer, "consumer.wasm"))
        .unwrap();

    assert!(matches!(
        manager.instantiate_component("consumer"),
        Err(ComponentError::UnauthorizedImport { .. })
    ));
}

#[test]
fn component_definition_can_create_multiple_isolated_instances() {
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component"),
            "component.wasm",
        ))
        .unwrap();

    let first = manager.instantiate_component("component").unwrap();
    let second = manager.instantiate_component("component").unwrap();
    assert_ne!(first, second);
    assert_eq!(
        manager.instance_state(first),
        Some(ComponentInstanceState::Ready)
    );
    assert_eq!(
        manager.instance_state(second),
        Some(ComponentInstanceState::Ready)
    );
}

#[test]
fn component_manager_enforces_instance_and_invocation_limits() {
    let mut manager = ComponentManager::new();
    manager.set_resource_limits(ComponentResourceLimits {
        max_instances: Some(1),
        ..ComponentResourceLimits::default()
    });
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component"),
            "component.wasm",
        ))
        .unwrap();

    manager.instantiate_component("component").unwrap();
    assert!(matches!(
        manager.instantiate_component("component"),
        Err(ComponentError::ResourceLimitExceeded {
            limit: "instances",
            ..
        })
    ));

    let interface = WitInterface::new("example:app/run", "1.0.0");
    let mut manager = ComponentManager::new();
    manager.set_resource_limits(ComponentResourceLimits {
        max_concurrent_invocations: Some(0),
        ..ComponentResourceLimits::default()
    });
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("callable", "1", "test component")
                .with_export(interface.clone()),
            "callable.wasm",
        ))
        .unwrap();
    let instance = manager.instantiate_component("callable").unwrap();
    assert!(matches!(
        manager.invoke(ComponentInvocation::new(instance, interface, "run")),
        Err(ComponentError::ResourceLimitExceeded {
            limit: "concurrent invocations",
            ..
        })
    ));
}

#[test]
fn component_invocation_after_destruction_fails() {
    let interface = WitInterface::new("example:app/run", "1.0.0");
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component")
                .with_export(interface.clone()),
            "component.wasm",
        ))
        .unwrap();
    let instance = manager.instantiate_component("component").unwrap();
    manager.destroy_instance(instance).unwrap();

    assert!(matches!(
        manager.invoke(ComponentInvocation::new(instance, interface, "run")),
        Err(ComponentError::InstanceNotFound(_))
    ));
}

#[test]
fn component_shutdown_prevents_new_lifecycle_operations() {
    let interface = WitInterface::new("example:app/run", "1.0.0");
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component")
                .with_export(interface.clone()),
            "component.wasm",
        ))
        .unwrap();
    let instance = manager.instantiate_component("component").unwrap();
    manager.shutdown();

    assert!(matches!(
        manager.invoke(ComponentInvocation::new(instance, interface, "run")),
        Err(ComponentError::RuntimeShutdown)
    ));
    assert!(matches!(
        manager.instantiate_component("component"),
        Err(ComponentError::RuntimeShutdown)
    ));
    assert!(matches!(
        manager.register_component(ComponentDescriptor::new(
            ComponentMetadata::new("other", "1", "test component"),
            "other.wasm",
        )),
        Err(ComponentError::RuntimeShutdown)
    ));
}

#[test]
fn component_observations_are_non_authoritative_and_redacted() {
    let interface = WitInterface::new("example:app/run", "1.0.0");
    let mut engine = MockComponentEngine::new();
    engine.trap_on_invoke = Some(ComponentTrapKind::Trap);
    let mut manager = ComponentManager::with_engine(Box::new(engine));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component")
                .with_export(interface.clone()),
            "component.wasm",
        ))
        .unwrap();
    let instance = manager.instantiate_component("component").unwrap();

    assert!(matches!(
        manager.invoke(ComponentInvocation::new(instance, interface, "run")),
        Err(ComponentError::Trap { .. })
    ));
    assert!(
        manager
            .observations()
            .iter()
            .any(
                |observation| observation.kind == ComponentObservationKind::Trap
                    && observation.instance == Some(instance)
                    && observation.message.contains("[redacted component trap]")
            )
    );
    assert!(
        !manager
            .observations()
            .iter()
            .any(|observation| observation.message.contains("wasmtime::"))
    );
    assert!(!manager.observations().iter().any(|observation| {
        observation.message.contains("Provider")
            || observation.message.contains("Device")
            || observation.message.contains("Store")
    }));
}

#[test]
fn component_engine_normalizes_traps_interruptions_and_limit_failures() {
    let interface = WitInterface::new("example:app/run", "1.0.0");
    let mut trapping_engine = MockComponentEngine::new();
    trapping_engine.trap_on_invoke = Some(ComponentTrapKind::Trap);
    let mut manager = ComponentManager::with_engine(Box::new(trapping_engine));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component")
                .with_export(interface.clone()),
            "component.wasm",
        ))
        .unwrap();
    let instance = manager.instantiate_component("component").unwrap();
    assert!(matches!(
        manager.invoke(ComponentInvocation::new(instance, interface, "run")),
        Err(ComponentError::Trap {
            kind: ComponentTrapKind::Trap,
            ..
        })
    ));

    let mut manager = ComponentManager::with_engine(Box::new(
        MockComponentEngine::new().without_resource_limits(),
    ));
    manager.set_resource_limits(ComponentResourceLimits {
        require_memory_limit: true,
        max_memory_bytes: Some(1024),
        ..ComponentResourceLimits::default()
    });
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("limited", "1", "test component"),
            "limited.wasm",
        ))
        .unwrap();
    assert!(matches!(
        manager.instantiate_component("limited"),
        Err(ComponentError::ResourceLimitUnsupported { .. })
    ));
}

#[test]
fn component_discovery_returns_only_wasm_artifacts() {
    let directory =
        std::env::temp_dir().join(format!("magnetar-components-{}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    fs::write(directory.join("valid.wasm"), []).unwrap();
    fs::write(directory.join("ignored.txt"), []).unwrap();

    let discovered = ComponentManager::discover([&directory]).unwrap();
    fs::remove_dir_all(&directory).unwrap();
    assert_eq!(discovered, vec![directory.join("valid.wasm")]);
}

fn temp_component_artifact_dir(label: &str) -> std::path::PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "magnetar-component-artifact-{label}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir_all(&directory).unwrap();
    directory
}

fn manifest_yaml(digest: &str, runtime_version: &str) -> String {
    format!(
        r#"schema: magnetar-component-artifact
schema_version: 1
artifact:
  kind: component
  digest:
    algorithm: sha256
    value: "{digest}"
component:
  name: "magnetar.examples.hello"
  version: "0.1.0"
  description: "Minimal Magnetar Component fixture"
  role: "test-fixture"
runtime:
  magnetar:
    min_version: "{runtime_version}"
wit:
  imports:
    - package: "magnetar:test"
      interface: "echo"
      version: "1.0.0"
  exports:
    - package: "magnetar:test"
      interface: "run"
      version: "1.0.0"
capabilities:
  requires:
    - id: "magnetar:test/echo"
      version: "1.0.0"
authority:
  requires: []
publisher:
  id: "local-dev"
  name: "Local Development"
source:
  kind: "local"
  uri: "./fixtures/hello.component.wasm"
signatures: []
"#
    )
}

fn tokenizer_manifest_yaml(digest: &str) -> String {
    format!(
        r#"schema: magnetar-component-artifact
schema_version: 1
artifact:
  kind: component
  digest:
    algorithm: sha256
    value: "{digest}"
component:
  name: "magnetar.examples.tokenizer"
  version: "0.1.0"
  description: "Tokenizer Component fixture"
  role: "tokenizer"
runtime:
  magnetar:
    min_version: "0.1.0"
wit:
  imports:
    - package: "magnetar:compute"
      interface: "run"
      version: "2.0.0"
  exports:
    - package: "magnetar:tokenizer"
      interface: "tokenize"
      version: "1.0.0"
capabilities:
  requires:
    - id: "magnetar:compute/run"
      version: "2.0.0"
authority:
  requires:
    - tokenizer-artifact-read
    - compute-capability
    - observability-emit
publisher:
  id: "local-dev"
  name: "Local Development"
source:
  kind: "local"
  uri: "./fixtures/tokenizer.component.wasm"
signatures: []
"#
    )
}

fn manifest_yaml_with_authority(digest: &str, authorities: &[&str]) -> String {
    let requires = authorities
        .iter()
        .map(|authority| format!("    - {authority}"))
        .collect::<Vec<_>>()
        .join("\n");
    manifest_yaml(digest, MAGNETAR_RUNTIME_VERSION).replace(
        "authority:\n  requires: []",
        &format!("authority:\n  requires:\n{requires}"),
    )
}

fn trust_store_yaml(digest: &str) -> String {
    format!(
        r#"schema: magnetar-component-trust
schema_version: 1
trusted_digests:
  - "{digest}"
rejected_digests: []
revoked_digests: []
trusted_publishers: []
trusted_sources: []
development:
  allow_unsigned_local: false
"#
    )
}

fn component_artifact_package(
    bytes: &[u8],
    source_kind: ComponentDistributionSourceKind,
) -> ComponentArtifactPackage {
    let digest = ComponentDigest::sha256(bytes);
    ComponentArtifactPackage::new(
        bytes.to_vec(),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION).into_bytes(),
        digest,
        ComponentDistributionSource::new(source_kind, "test-source"),
    )
}

#[derive(Clone)]
struct TestComponentDistributionSource {
    package: ComponentArtifactPackage,
    candidates: Vec<ComponentDigest>,
}

impl ComponentDistributionSourceProvider for TestComponentDistributionSource {
    fn resolve(
        &self,
        component: &str,
        _version_requirement: Option<&str>,
    ) -> Result<Vec<ComponentDigest>, ComponentError> {
        if component == "magnetar.examples.hello" {
            Ok(self.candidates.clone())
        } else {
            Ok(Vec::new())
        }
    }

    fn fetch(&self, digest: &ComponentDigest) -> Result<ComponentArtifactPackage, ComponentError> {
        if self.package.declared_digest == *digest {
            Ok(self.package.clone())
        } else {
            Err(ComponentError::Distribution {
                category: ComponentDistributionErrorCategory::ArtifactNotFound,
                message: "digest not found".into(),
            })
        }
    }
}

#[test]
fn pushed_component_package_is_validated_before_preparation() {
    let bytes = b"component-bytes";
    let digest = ComponentDigest::sha256(bytes);
    let package =
        component_artifact_package(bytes, ComponentDistributionSourceKind::ClientProvided);
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value.clone()));

    manager.prepare_pushed_package(package).unwrap();

    let definition = manager.definition("magnetar.examples.hello").unwrap();
    assert_eq!(definition.artifact_digest, Some(digest));
    assert!(matches!(
        definition.trust_decision,
        Some(ComponentTrustDecision {
            status: ComponentTrustStatus::Trusted,
            ..
        })
    ));
    assert!(manager.observations().iter().any(|observation| {
        observation.kind == ComponentObservationKind::Distribution
            && observation.message.contains("client-provided")
    }));
}

#[test]
fn pushed_component_package_rejects_source_digest_mismatch() {
    let mut package =
        component_artifact_package(b"component-bytes", ComponentDistributionSourceKind::Tachyon);
    package.declared_digest = ComponentDigest::sha256(b"different");
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_source("tachyon"));

    assert!(matches!(
        manager.prepare_pushed_package(package),
        Err(ComponentError::Distribution {
            category: ComponentDistributionErrorCategory::DigestMismatch,
            ..
        })
    ));
}

#[test]
fn distribution_source_identity_does_not_imply_trust() {
    let package =
        component_artifact_package(b"component-bytes", ComponentDistributionSourceKind::Tachyon);
    let mut manager = ComponentManager::new();

    assert!(matches!(
        manager.prepare_pushed_package(package),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Unknown,
            ..
        })
    ));
}

#[test]
fn trusted_distribution_source_still_rejects_forbidden_authority() {
    let bytes = b"component-bytes";
    let digest = ComponentDigest::sha256(bytes);
    let manifest = manifest_yaml_with_authority(&digest.value, &["filesystem"])
        .replace("  kind: \"local\"", "  kind: \"tachyon\"");
    let package = ComponentArtifactPackage::new(
        bytes.to_vec(),
        manifest.into_bytes(),
        digest,
        ComponentDistributionSource::new(ComponentDistributionSourceKind::Tachyon, "tachyon"),
    );
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_source("tachyon"));

    assert!(matches!(
        manager.prepare_pushed_package(package),
        Err(ComponentError::Manifest { message, .. })
            if message == "authority kind is outside Magnetar inference scope"
    ));
}

#[test]
fn pulled_component_package_resolves_fetches_and_validates_locally() {
    let bytes = b"component-bytes";
    let digest = ComponentDigest::sha256(bytes);
    let package =
        component_artifact_package(bytes, ComponentDistributionSourceKind::LocalDirectory);
    let source = TestComponentDistributionSource {
        package,
        candidates: vec![digest.clone()],
    };
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value.clone()));

    manager
        .prepare_pulled_package(&source, "magnetar.examples.hello", Some(">=0.1.0,<1.0.0"))
        .unwrap();

    assert_eq!(
        manager
            .definition("magnetar.examples.hello")
            .and_then(|definition| definition.artifact_digest.clone()),
        Some(digest)
    );
    assert!(manager.observations().iter().any(|observation| {
        observation.kind == ComponentObservationKind::Distribution
            && observation.message.contains("candidate digest")
    }));
}

#[test]
fn pulled_component_package_rejects_empty_candidate_list() {
    let package = component_artifact_package(
        b"component-bytes",
        ComponentDistributionSourceKind::LocalCache,
    );
    let source = TestComponentDistributionSource {
        package,
        candidates: Vec::new(),
    };
    let mut manager = ComponentManager::new();

    assert!(matches!(
        manager.prepare_pulled_package(&source, "magnetar.examples.hello", None),
        Err(ComponentError::Distribution {
            category: ComponentDistributionErrorCategory::ArtifactNotFound,
            ..
        })
    ));
}

#[test]
fn local_distribution_does_not_require_tachyon_or_network() {
    let bytes = b"component-bytes";
    let digest = ComponentDigest::sha256(bytes);
    let package =
        component_artifact_package(bytes, ComponentDistributionSourceKind::LocalDirectory);
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));

    manager.prepare_pushed_package(package).unwrap();
    assert_eq!(
        manager.definition_state("magnetar.examples.hello"),
        Some(ComponentDefinitionState::Prepared)
    );
}

#[test]
fn component_artifact_accepts_target_tokenizer_manifest_authorities() {
    let directory = temp_component_artifact_dir("tokenizer-authority");
    let artifact = directory.join("tokenizer.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("tokenizer.component.wasm.magnetar-component.yaml"),
        tokenizer_manifest_yaml(&digest.value),
    )
    .unwrap();

    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.tokenizer", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:compute/run", "2.0.0"))
                .with_export(WitInterface::new("magnetar:tokenizer/tokenize", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    manager
        .prepare_component("magnetar.examples.tokenizer")
        .unwrap();
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_rejects_broad_authority_kinds() {
    for authority in [
        "filesystem",
        "network",
        "secrets",
        "git",
        "workspace",
        "process",
    ] {
        let directory = temp_component_artifact_dir(authority);
        let artifact = directory.join("hello.component.wasm");
        let bytes = b"component-bytes";
        fs::write(&artifact, bytes).unwrap();
        let digest = ComponentDigest::sha256(bytes);
        fs::write(
            directory.join("hello.component.wasm.magnetar-component.yaml"),
            manifest_yaml_with_authority(&digest.value, &[authority]),
        )
        .unwrap();
        let mut manager = ComponentManager::new();
        manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
        manager
            .register_component(ComponentDescriptor::new(
                ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                    .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                    .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
                &artifact,
            ))
            .unwrap();

        assert!(matches!(
            manager.prepare_component("magnetar.examples.hello"),
            Err(ComponentError::Manifest { message, .. })
                if message == "authority kind is outside Magnetar inference scope"
        ));
        fs::remove_dir_all(directory).unwrap();
    }
}

#[test]
fn component_artifact_rejects_unknown_authority_kinds() {
    let directory = temp_component_artifact_dir("unknown-authority");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml_with_authority(&digest.value, &["workspace-admin"]),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::Manifest { message, .. }) if message == "unsupported authority kind"
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_accepts_model_artifact_read_authority_when_trusted() {
    let directory = temp_component_artifact_dir("model-artifact-authority");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml_with_authority(&digest.value, &["model-artifact-read"]),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    manager
        .prepare_component("magnetar.examples.hello")
        .unwrap();
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_trust_overrides_do_not_allow_forbidden_authority() {
    for (label, trust_store, source_kind) in [
        ("trusted-digest", ComponentTrustStore::default(), "local"),
        (
            "trusted-publisher",
            ComponentTrustStore::default().trust_publisher("local-dev"),
            "local",
        ),
        (
            "trusted-tachyon-source",
            ComponentTrustStore::default().trust_source("tachyon"),
            "tachyon",
        ),
        (
            "trusted-local-source",
            ComponentTrustStore::default().trust_source("local"),
            "local",
        ),
        (
            "development-mode",
            ComponentTrustStore::default().allow_unsigned_local_development(true),
            "local",
        ),
    ] {
        let directory = temp_component_artifact_dir(label);
        let artifact = directory.join("hello.component.wasm");
        let bytes = b"component-bytes";
        fs::write(&artifact, bytes).unwrap();
        let digest = ComponentDigest::sha256(bytes);
        let mut trust_store = trust_store;
        if label == "trusted-digest" {
            trust_store = trust_store.trust_digest(digest.value.clone());
        }
        let manifest = manifest_yaml_with_authority(&digest.value, &["filesystem"])
            .replace("  kind: \"local\"", &format!("  kind: \"{source_kind}\""));
        fs::write(
            directory.join("hello.component.wasm.magnetar-component.yaml"),
            manifest,
        )
        .unwrap();
        let mut manager = ComponentManager::new();
        manager.set_trust_store(trust_store);
        manager
            .register_component(ComponentDescriptor::new(
                ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                    .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                    .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
                &artifact,
            ))
            .unwrap();

        assert!(matches!(
            manager.prepare_component("magnetar.examples.hello"),
            Err(ComponentError::Manifest { message, .. })
                if message == "authority kind is outside Magnetar inference scope"
        ));
        fs::remove_dir_all(directory).unwrap();
    }
}

#[test]
fn component_authority_rejection_is_observed_with_reason_before_prepare() {
    let directory = temp_component_artifact_dir("authority-diagnostic");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml_with_authority(&digest.value, &["filesystem"]),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::Manifest { .. })
    ));
    assert_eq!(
        manager.definition_state("magnetar.examples.hello"),
        Some(ComponentDefinitionState::Failed)
    );
    assert!(manager.observations().iter().any(|observation| {
        observation.kind == ComponentObservationKind::Validation
            && observation.message.contains("component authority rejected")
            && observation
                .message
                .contains("authority kind is outside Magnetar inference scope")
    }));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_validation_observations_redact_paths_and_secrets() {
    let directory = temp_component_artifact_dir("redacted-diagnostic");
    let artifact = directory.join("hello.component.wasm");
    fs::write(&artifact, b"component-bytes").unwrap();
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture"),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ManifestMissing { .. })
    ));
    let messages = manager
        .observations()
        .iter()
        .map(|observation| observation.message.as_str())
        .collect::<Vec<_>>();
    assert!(
        messages
            .iter()
            .any(|message| message.contains("[redacted]"))
    );
    assert!(!messages.iter().any(|message| {
        message.contains(directory.to_string_lossy().as_ref())
            || message.to_ascii_lowercase().contains("secret")
    }));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_pipeline_requires_external_trust_policy_before_prepare() {
    let directory = temp_component_artifact_dir("trusted");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    let manifest = directory.join("hello.component.wasm.magnetar-component.yaml");
    fs::write(
        &manifest,
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();

    let import = WitInterface::new("magnetar:test/echo", "1.0.0");
    let export = WitInterface::new("magnetar:test/run", "1.0.0");
    let metadata = ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
        .with_import(import)
        .with_export(export);
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value.clone()));
    manager
        .register_component(ComponentDescriptor::new(metadata, &artifact))
        .unwrap();

    manager
        .prepare_component("magnetar.examples.hello")
        .unwrap();
    let definition = manager.definition("magnetar.examples.hello").unwrap();
    assert_eq!(definition.artifact_digest, Some(digest));
    assert!(matches!(
        definition.trust_decision,
        Some(ComponentTrustDecision {
            status: ComponentTrustStatus::Trusted,
            ..
        })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_rejects_local_wasm_without_manifest() {
    let directory = temp_component_artifact_dir("missing-manifest");
    let artifact = directory.join("unknown.component.wasm");
    fs::write(&artifact, b"component-bytes").unwrap();
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("unknown", "0.1.0", "fixture"),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("unknown"),
        Err(ComponentError::ManifestMissing { .. })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_rejects_digest_mismatch_before_prepare() {
    let directory = temp_component_artifact_dir("digest-mismatch");
    let artifact = directory.join("hello.component.wasm");
    fs::write(&artifact, b"component-bytes").unwrap();
    let digest = ComponentDigest::sha256(b"different-bytes");
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Rejected,
            ..
        })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_rejects_manifest_wit_that_differs_from_actual_contract() {
    let directory = temp_component_artifact_dir("wit-mismatch");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();

    let mut engine = MockComponentEngine::new();
    let mut actual = ComponentContract::default();
    actual.imports.insert(ComponentImportRequirement::new(
        WitInterface::new("magnetar:test/other", "1.0.0"),
        ComponentInterfaceShape::Interface,
    ));
    engine.prepared_contract = Some(actual);
    let mut manager = ComponentManager::with_engine(Box::new(engine));
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture"),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ContractValidationFailed { .. })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_trust_store_revocation_overrides_digest_allowlist() {
    let directory = temp_component_artifact_dir("revoked");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(
        ComponentTrustStore::default()
            .trust_digest(digest.value.clone())
            .revoke_digest(digest.value),
    );
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Revoked,
            ..
        })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_trust_store_loads_minimal_yaml_format() {
    let directory = temp_component_artifact_dir("trust-store");
    let digest = ComponentDigest::sha256(b"component-bytes");
    let trust_store = directory.join("trust.yaml");
    fs::write(&trust_store, trust_store_yaml(&digest.value)).unwrap();

    let loaded = ComponentTrustStore::load_yaml(&trust_store).unwrap();
    assert!(loaded.trusted_digests.contains(&digest.value));
    assert!(!loaded.allow_unsigned_local_development);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_manifest_may_declare_optional_wit_import_metadata() {
    let directory = temp_component_artifact_dir("optional-import");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    let manifest = manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION).replace(
        "  exports:",
        "    - package: \"magnetar:optional\"\n      interface: \"telemetry\"\n      version: \"1.0.0\"\n      optional: true\n  exports:",
    );
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest,
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    manager
        .prepare_component("magnetar.examples.hello")
        .unwrap();
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_rejects_runtime_max_version_incompatibility() {
    let directory = temp_component_artifact_dir("runtime-max");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    let manifest = manifest_yaml(&digest.value, "0.0.1").replace(
        "    min_version: \"0.0.1\"",
        "    min_version: \"0.0.1\"\n    max_version: \"0.0.1\"",
    );
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest,
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Rejected,
            ..
        })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_rejects_incompatible_capability_versions() {
    for (label, capability_block) in [
        (
            "cap-major",
            "    - id: \"magnetar:test/echo\"\n      version: \"2.0.0\"",
        ),
        (
            "cap-range",
            "    - id: \"magnetar:test/echo\"\n      version: \"1.0.0\"\n      max_version: \"0.9.0\"",
        ),
    ] {
        let directory = temp_component_artifact_dir(label);
        let artifact = directory.join("hello.component.wasm");
        let bytes = b"component-bytes";
        fs::write(&artifact, bytes).unwrap();
        let digest = ComponentDigest::sha256(bytes);
        let manifest = manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION).replace(
            "    - id: \"magnetar:test/echo\"\n      version: \"1.0.0\"",
            capability_block,
        );
        fs::write(
            directory.join("hello.component.wasm.magnetar-component.yaml"),
            manifest,
        )
        .unwrap();
        let mut manager = ComponentManager::new();
        manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
        manager
            .register_component(ComponentDescriptor::new(
                ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                    .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                    .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
                &artifact,
            ))
            .unwrap();

        assert!(matches!(
            manager.prepare_component("magnetar.examples.hello"),
            Err(ComponentError::ArtifactRejected {
                status: ComponentTrustStatus::Rejected,
                ..
            })
        ));
        fs::remove_dir_all(directory).unwrap();
    }
}

#[test]
fn component_publisher_trust_is_only_policy_driven() {
    let directory = temp_component_artifact_dir("publisher-policy");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();
    let descriptor = || {
        ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        )
    };
    let mut untrusted = ComponentManager::new();
    untrusted.register_component(descriptor()).unwrap();
    assert!(matches!(
        untrusted.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Unknown,
            ..
        })
    ));

    let mut trusted = ComponentManager::new();
    trusted.set_trust_store(ComponentTrustStore::default().trust_publisher("local-dev"));
    trusted.register_component(descriptor()).unwrap();
    trusted
        .prepare_component("magnetar.examples.hello")
        .unwrap();
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_signature_metadata_is_recorded_but_not_trusted_by_itself() {
    let directory = temp_component_artifact_dir("signature");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    let signed_manifest = manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION).replace(
        "signatures: []",
        &format!(
            "signatures:\n  - algorithm: \"test\"\n    digest: \"{}\"",
            digest.value
        ),
    );
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        signed_manifest,
    )
    .unwrap();
    let descriptor = || {
        ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        )
    };
    let mut no_trust = ComponentManager::new();
    no_trust.register_component(descriptor()).unwrap();
    assert!(matches!(
        no_trust.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Unknown,
            ..
        })
    ));

    let mut trusted = ComponentManager::new();
    trusted.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value.clone()));
    trusted.register_component(descriptor()).unwrap();
    trusted
        .prepare_component("magnetar.examples.hello")
        .unwrap();

    let bad_manifest = manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION).replace(
        "signatures: []",
        "signatures:\n  - algorithm: \"test\"\n    digest: \"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"",
    );
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        bad_manifest,
    )
    .unwrap();
    let mut mismatch = ComponentManager::new();
    mismatch.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    mismatch.register_component(descriptor()).unwrap();
    assert!(matches!(
        mismatch.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Rejected,
            ..
        })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_development_mode_is_explicit_and_still_validates_artifact() {
    let directory = temp_component_artifact_dir("development");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().allow_unsigned_local_development(true));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    manager
        .prepare_component("magnetar.examples.hello")
        .unwrap();
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_cache_is_digest_keyed_and_non_authoritative() {
    let bytes = b"component-bytes".to_vec();
    let mut cache = ComponentArtifactCache::default();
    let digest = cache.insert(bytes.clone());
    assert!(cache.contains_untrusted(&digest));
    assert_eq!(cache.get_verified(&digest).unwrap(), Some(bytes.as_slice()));

    let wrong_digest = ComponentDigest::sha256(b"wrong");
    cache.insert_unchecked_for_test(wrong_digest.clone(), bytes);
    assert!(matches!(
        cache.get_verified(&wrong_digest),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Rejected,
            ..
        })
    ));
}

#[test]
fn component_quarantine_prevents_preparation_and_preserves_diagnostic_status() {
    let directory = temp_component_artifact_dir("quarantine");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().quarantine_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Quarantined,
            ..
        })
    ));
    assert!(
        manager
            .observations()
            .iter()
            .any(|observation| observation.message.contains("Quarantined"))
    );
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_validation_emits_structured_observations() {
    let directory = temp_component_artifact_dir("observations");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    manager
        .prepare_component("magnetar.examples.hello")
        .unwrap();
    let messages = manager
        .observations()
        .iter()
        .map(|observation| observation.message.as_str())
        .collect::<Vec<_>>();
    assert!(
        messages
            .iter()
            .any(|message| message.contains("discovered"))
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("digest computed"))
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("manifest loaded"))
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("WIT declarations match"))
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("compatibility validated"))
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("trust decision"))
    );
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn runtime_initializes_memory_manager_as_first_class_service() {
    let runtime = Runtime::initialize(RuntimeConfig::default());

    assert!(runtime.is_initialized());
    assert_eq!(runtime.memory().config(), &MemoryManagerConfig::default());
    assert_eq!(runtime.memory().allocations().count(), 0);
}

#[test]
fn memory_manager_admission_uses_pressure_and_queue_policy() {
    let manager = MemoryManager::default();
    let request = MemoryAllocationRequest::new(
        MemoryAllocationClass::Tensor,
        1024,
        MemoryPlacement::HostOrdinary,
        MemoryAllocationOwner::Runtime,
    );
    let saturated = MemoryPressureSnapshot {
        runtime: MemoryPressureLevel::Saturated,
        ..MemoryPressureSnapshot::default()
    };

    assert!(matches!(
        manager.admit(MemoryAdmissionRequest {
            allocation: request.clone(),
            pressure: saturated.clone(),
            queue_allowed: false,
        }),
        MemoryAdmissionDecision::Reject { .. }
    ));
    assert!(matches!(
        manager.admit(MemoryAdmissionRequest {
            allocation: request,
            pressure: saturated,
            queue_allowed: true,
        }),
        MemoryAdmissionDecision::Queue { .. }
    ));
}

#[test]
fn memory_manager_tracks_allocation_lifetime_and_tensor_residency() {
    let mut manager = MemoryManager::new(MemoryManagerConfig {
        max_runtime_bytes: Some(4096),
        ..MemoryManagerConfig::default()
    });
    let affinity = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(ProviderBinding::new("compute"));
    let allocation = manager
        .allocate(
            MemoryAllocationRequest::new(
                MemoryAllocationClass::Tensor,
                256,
                MemoryPlacement::ProviderOwnedOpaque(ProviderBinding::new("compute")),
                MemoryAllocationOwner::Provider(ProviderBinding::new("compute")),
            )
            .with_affinity(affinity.clone()),
        )
        .unwrap();
    let tensor = TensorResourceId::new("tensor:0");

    manager
        .record_tensor_residency(
            TensorResidency::new(
                tensor.clone(),
                MemoryPlacement::ProviderOwnedOpaque(ProviderBinding::new("compute")),
                affinity,
            )
            .with_allocation(allocation.id),
        )
        .unwrap();

    let residency = manager.tensor_residency(&tensor).unwrap();
    assert_eq!(residency.allocation, Some(allocation.id));
    assert!(residency.provider_owned);
    manager.release(allocation.id).unwrap();
    assert!(manager.observations().iter().any(|observation| {
        observation.kind == MemoryObservationKind::AllocationReleased
            && observation.allocation == Some(allocation.id)
    }));
    assert!(matches!(
        manager.record_tensor_residency(
            TensorResidency::new(
                TensorResourceId::new("tensor:bad"),
                MemoryPlacement::HostOrdinary,
                ResourceAffinity::new(FallbackClass::Transparent),
            )
            .with_allocation(MemoryAllocationId::new(999)),
        ),
        Err(MemoryError::InvalidAllocationHandle(_))
    ));
}

#[test]
fn memory_manager_distinguishes_storage_and_compute_dtype_costs() {
    let dtype = MemoryDTypeRelation::new(
        DTypeDescriptor::portable(ComputeDType::SInt8),
        DTypeDescriptor::portable(ComputeDType::BrainFloat16),
    );

    assert_eq!(dtype.storage_size_bytes(8).unwrap(), 8);
    assert_eq!(dtype.compute_workspace_bytes(8).unwrap(), 16);
}

#[test]
fn memory_manager_rejects_forbidden_staging_and_incompatible_zero_copy() {
    let manager = MemoryManager::default();
    let staging = manager.staging_feasibility(HostStagingPolicy::Forbid, 128);
    assert!(!staging.feasible);
    assert!(staging.reason.contains("forbidden"));

    let affinity = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(ProviderBinding::new("compute"));
    let source = TensorResidency::new(
        TensorResourceId::new("tensor:0"),
        MemoryPlacement::HostOrdinary,
        affinity,
    );
    let zero_copy = manager.zero_copy_feasibility(
        &source,
        &MemoryPlacement::Device(DeviceBinding::new(DeviceId::new("gpu:0"))),
        None,
    );
    assert!(!zero_copy.feasible);
    assert!(zero_copy.reason.contains("incompatible"));
}

#[test]
fn memory_manager_reuses_cached_allocations_and_evicts_over_limit() {
    let mut manager = MemoryManager::new(MemoryManagerConfig {
        max_cached_bytes: 512,
        ..MemoryManagerConfig::default()
    });
    let request = MemoryAllocationRequest::new(
        MemoryAllocationClass::Tensor,
        256,
        MemoryPlacement::HostOrdinary,
        MemoryAllocationOwner::Runtime,
    );

    let first = manager.allocate(request.clone()).unwrap();
    manager.release(first.id).unwrap();
    let second = manager.allocate(request).unwrap();

    assert_eq!(first.id, second.id);
    assert!(manager.observations().iter().any(|observation| {
        observation.kind == MemoryObservationKind::CacheHit
            && observation.allocation == Some(first.id)
    }));

    manager.release(second.id).unwrap();
    let other = manager
        .allocate(MemoryAllocationRequest::new(
            MemoryAllocationClass::TemporaryWorkspace,
            768,
            MemoryPlacement::HostOrdinary,
            MemoryAllocationOwner::Runtime,
        ))
        .unwrap();
    manager.release(other.id).unwrap();

    assert!(
        manager
            .observations()
            .iter()
            .any(|observation| { observation.kind == MemoryObservationKind::CacheEviction })
    );
}

#[test]
fn memory_manager_tracks_arena_growth_shrink_pressure_and_diagnostics() {
    let mut manager = MemoryManager::default();
    let arena = manager
        .create_arena(
            MemoryAllocationClass::TemporaryWorkspace,
            MemoryPlacement::HostOrdinary,
            128,
            MemoryArenaOwner::Runtime,
        )
        .unwrap();
    let id = arena.id;
    *manager.arena_mut(id).unwrap() = MemoryArena::new(
        id,
        MemoryAllocationClass::TemporaryWorkspace,
        MemoryPlacement::HostOrdinary,
        128,
        MemoryArenaOwner::Runtime,
    )
    .with_growth(MemoryArenaGrowthPolicy::GrowOnDemand {
        increment_bytes: 128,
    })
    .with_shrink(MemoryArenaShrinkPolicy::ReleaseReusable);

    manager.reserve_in_arena(id, 192).unwrap();
    manager.shrink_arena(id).unwrap();

    let arena = manager.arenas().find(|arena| arena.id == id).unwrap();
    assert_eq!(arena.used_bytes, 192);
    assert!(arena.capacity_bytes >= 192);
    assert!(matches!(
        arena.pressure,
        MemoryPressureLevel::High | MemoryPressureLevel::Saturated
    ));
    assert!(
        manager
            .observations()
            .iter()
            .any(|observation| { observation.kind == MemoryObservationKind::ArenaPressure })
    );
}

#[test]
fn memory_manager_pending_queue_times_out_cancels_and_retries() {
    let mut timeout_manager = MemoryManager::default();
    timeout_manager
        .submit_pending_allocation(
            MemoryAllocationRequest::new(
                MemoryAllocationClass::Tensor,
                64,
                MemoryPlacement::HostOrdinary,
                MemoryAllocationOwner::Runtime,
            )
            .with_alignment(8)
            .with_deadline_millis(20),
            10,
        )
        .unwrap();
    let errors = timeout_manager.expire_pending_allocations(20);
    assert!(matches!(
        errors.as_slice(),
        [MemoryError::AllocationTimeout { .. }]
    ));

    let mut cancel_manager = MemoryManager::default();
    let pending = cancel_manager
        .submit_pending_allocation(
            MemoryAllocationRequest::new(
                MemoryAllocationClass::Tensor,
                64,
                MemoryPlacement::HostOrdinary,
                MemoryAllocationOwner::Runtime,
            ),
            10,
        )
        .unwrap();
    assert!(matches!(
        cancel_manager.cancel_pending_allocation(pending.allocation.id),
        Err(MemoryError::AllocationCancelled { .. })
    ));

    let mut retry_manager = MemoryManager::default();
    retry_manager
        .submit_pending_allocation(
            MemoryAllocationRequest::new(
                MemoryAllocationClass::Tensor,
                64,
                MemoryPlacement::HostOrdinary,
                MemoryAllocationOwner::Runtime,
            ),
            0,
        )
        .unwrap();
    let results = retry_manager.retry_pending_allocations();
    assert_eq!(results.len(), 1);
    assert!(results[0].is_ok());
    assert!(retry_manager.pending_allocations().next().is_none());
    assert!(
        retry_manager
            .observations()
            .iter()
            .any(|observation| { observation.kind == MemoryObservationKind::PendingQueueDelay })
    );
}

#[test]
fn memory_manager_reports_provider_device_cache_and_class_pressure() {
    let provider = ProviderStatusSnapshot::from_health_report(ProviderHealthReport::new(
        ProviderBinding::new("compute"),
        ProviderHealth::Saturated,
    ));
    let provider_pressure = MemoryManager::pressure_for_provider_status(&provider);
    assert_eq!(provider_pressure.runtime, MemoryPressureLevel::Saturated);
    assert_eq!(
        provider_pressure.provider,
        Some((
            ProviderBinding::new("compute"),
            MemoryPressureLevel::Saturated
        ))
    );

    let mut metadata =
        DeviceMetadata::new(DeviceId::new("gpu:0"), "GPU", DeviceType::Gpu, "compute");
    metadata.memory_capacity = 100;
    let device_pressure =
        MemoryManager::pressure_for_device_metadata(&metadata, 95, DeviceAvailability::Available);
    assert_eq!(device_pressure.runtime, MemoryPressureLevel::Saturated);

    let mut manager = MemoryManager::new(MemoryManagerConfig {
        max_runtime_bytes: Some(100),
        max_cached_bytes: 100,
        ..MemoryManagerConfig::default()
    });
    manager
        .allocate(MemoryAllocationRequest::new(
            MemoryAllocationClass::KvCache,
            80,
            MemoryPlacement::HostOrdinary,
            MemoryAllocationOwner::Session("session:0".into()),
        ))
        .unwrap();
    let pressure = manager.pressure_snapshot();
    assert_eq!(pressure.runtime, MemoryPressureLevel::High);
    assert_eq!(pressure.kv_cache, Some(MemoryPressureLevel::High));
    assert!(pressure.cache.is_some());
}

#[test]
fn memory_manager_observes_zero_copy_staging_pinned_and_browser_policy() {
    let mut manager = MemoryManager::new(MemoryManagerConfig {
        max_pinned_host_bytes: 16,
        allow_browser_linear_memory: false,
        ..MemoryManagerConfig::default()
    });
    let affinity = ResourceAffinity::new(FallbackClass::ProviderPinned)
        .with_provider(ProviderBinding::new("compute"));
    let source = TensorResidency::new(
        TensorResourceId::new("tensor:0"),
        MemoryPlacement::HostOrdinary,
        affinity,
    );

    let accepted =
        manager.observed_zero_copy_feasibility(&source, &MemoryPlacement::HostOrdinary, None);
    assert!(accepted.feasible);
    let rejected = manager.observed_zero_copy_feasibility(
        &source,
        &MemoryPlacement::Device(DeviceBinding::new(DeviceId::new("gpu:0"))),
        None,
    );
    assert!(!rejected.feasible);

    assert!(
        manager
            .observed_staging_feasibility(HostStagingPolicy::Permit, 8)
            .feasible
    );
    assert!(
        !manager
            .observed_staging_feasibility(HostStagingPolicy::Forbid, 8)
            .feasible
    );
    assert!(matches!(
        manager.allocate(MemoryAllocationRequest::new(
            MemoryAllocationClass::BrowserLinearMemory,
            8,
            MemoryPlacement::BrowserLinearMemory,
            MemoryAllocationOwner::Runtime,
        )),
        Err(MemoryError::UnsupportedPlacement(_))
    ));
    assert!(
        manager
            .observations()
            .iter()
            .any(|observation| { observation.kind == MemoryObservationKind::ZeroCopyAccepted })
    );
    assert!(
        manager
            .observations()
            .iter()
            .any(|observation| { observation.kind == MemoryObservationKind::ZeroCopyRejected })
    );
    assert!(
        manager
            .observations()
            .iter()
            .any(|observation| { observation.kind == MemoryObservationKind::StagingInserted })
    );
    assert!(
        manager
            .observations()
            .iter()
            .any(|observation| { observation.kind == MemoryObservationKind::StagingDenied })
    );
}

fn generation_tokenizer_metadata() -> TokenizerMetadata {
    TokenizerMetadata {
        id: TokenizerId::new("fixture").unwrap(),
        artifact: TokenizerArtifactId::new("fixture-tokenizer").unwrap(),
        digest: ModelDigest::sha256(b"tokenizer"),
        family: TokenizerFamily::new("fixture").unwrap(),
        revision: TokenizerRevision::new("1.0.0").unwrap(),
        vocabulary_size: 256,
        added_token_count: 2,
        token_id_range: TokenIdRange::new(1, 300),
        model_max_length: Some(16),
        special_tokens: vec![SpecialToken::new(SpecialTokenKind::Eos, "<eos>", 299)],
        additional_special_tokens: vec![SpecialToken::new(SpecialTokenKind::Stop, "<stop>", 298)],
        byte_fallback: false,
        normalization: None,
        pre_tokenizer: None,
        supports_offsets: true,
        supports_token_type_ids: false,
        supports_browser: true,
    }
}

fn generation_request() -> GenerationRequest {
    let metadata = generation_tokenizer_metadata();
    GenerationRequest {
        request_id: GenerationRequestId::new("gen-1").unwrap(),
        session: None,
        model: GenerationModelReference::LoadedModelContext("model-context".into()),
        tokenizer: GenerationTokenizerReference {
            tokenizer_id: metadata.id.clone(),
            metadata,
        },
        input_token_ids: vec![2, 3, 4],
        prompt_token_count: 3,
        max_new_tokens: 4,
        max_total_tokens: Some(8),
        model_context_length: Some(8),
        parameters: GenerationParameters::default(),
        stop_conditions: StopConditions {
            eos: EosPolicy {
                mode: EosMode::Stop,
                output: EosOutputPolicy::Exclude,
                eos_token_ids: vec![299],
            },
            stop_token_ids: vec![298],
            stop_token_patterns: vec![vec![10, 11]],
            stop_text_sequences: vec!["stop".into()],
            prepared_stop_sequences: vec![TokenStopPattern {
                text: "xy".into(),
                token_ids: vec![121, 122],
                exact: true,
            }],
            ..StopConditions::default()
        },
        streaming: StreamingMode::TokenIds,
        priority: GenerationPriority {
            priority: 3,
            deadline_millis: Some(100),
        },
        cancellation: CancellationMetadata::default(),
        memory: GenerationMemoryEstimate {
            input_token_buffer_bytes: 12,
            output_token_buffer_bytes: 16,
            logits_buffer_bytes: 32,
            sampling_workspace_bytes: 8,
            prefill_workspace_bytes: 8,
            decode_workspace_bytes: 8,
            kv_cache_placeholder_bytes: 8,
            prefix_cache_placeholder_bytes: 0,
            placement: MemoryPlacement::HostOrdinary,
            queue_allowed: false,
        },
        correlation_id: Some(CorrelationId::new("corr-1")),
        trace_id: Some(TraceId::new("trace-1")),
    }
}

fn generation_runtime_tokenizer() -> RuntimeTokenizer<FixtureTokenizer> {
    let metadata = generation_tokenizer_metadata();
    let digest = metadata.digest.clone();
    RuntimeTokenizer::new(
        FixtureTokenizer::new(metadata),
        TokenizerArtifactSet {
            tokenizer: TokenizerArtifactReference::new(
                TokenizerArtifactId::new("fixture-tokenizer").unwrap(),
                ModelArtifactKind::Tokenizer,
                digest,
            )
            .unwrap(),
            tokenizer_config: None,
            vocabulary: None,
            special_tokens: None,
        },
    )
}

#[test]
fn generation_request_validation_is_token_based_and_context_aware() {
    let request = generation_request();
    request.validate().unwrap();
    assert_eq!(request.prompt_token_count, request.input_token_ids.len());
    assert!(matches!(
        request.model,
        GenerationModelReference::LoadedModelContext(_)
    ));
}

#[test]
fn generation_request_rejects_invalid_input_tokens() {
    let mut request = generation_request();
    request.input_token_ids.push(999);
    request.prompt_token_count += 1;

    assert!(matches!(
        request.validate(),
        Err(GenerationError::InputTokensInvalid { .. })
    ));
}

#[test]
fn generation_request_rejects_prompt_that_exceeds_limits_without_truncation() {
    let mut request = generation_request();
    request.max_new_tokens = 20;

    assert!(matches!(
        request.validate(),
        Err(GenerationError::PromptTooLong { .. })
    ));
}

#[test]
fn generation_parameters_validate_temperature_sampling_and_greedy_modes() {
    let mut invalid = generation_request();
    invalid.parameters.temperature = f32::NAN;
    assert!(matches!(
        invalid.validate(),
        Err(GenerationError::ParameterInvalid {
            parameter: "temperature",
            ..
        })
    ));

    let mut greedy = generation_request();
    greedy.parameters = GenerationParameters::greedy();
    greedy.validate().unwrap();
    assert!(!greedy.parameters.sampling_enabled);
}

#[test]
fn generation_stop_conditions_distinguish_length_eos_token_and_sequences() {
    let request = generation_request();

    assert_eq!(
        stop_reason_for(&request, &[1, 2, 3, 4]),
        Some(FinishReason::MaxNewTokens)
    );
    assert_eq!(
        stop_reason_for(&request, &[299]),
        Some(FinishReason::EosToken)
    );
    assert_eq!(
        stop_reason_for(&request, &[298]),
        Some(FinishReason::StopToken)
    );
    assert_eq!(
        stop_reason_for(&request, &[1, 10, 11]),
        Some(FinishReason::StopSequence)
    );
    assert_eq!(
        stop_reason_for(&request, &[1, 121, 122]),
        Some(FinishReason::StopSequence)
    );
}

#[test]
fn generation_can_ignore_eos_by_explicit_policy() {
    let mut request = generation_request();
    request.stop_conditions.eos.mode = EosMode::Ignore;

    assert_eq!(stop_reason_for(&request, &[299]), None);
}

#[test]
fn generation_decode_step_preserves_token_index_and_state_boundary() {
    let request = generation_request();
    let step = decode_step(&request, &[20, 21], 22).unwrap();

    assert_eq!(step.token_id, 22);
    assert_eq!(step.token_index, 2);
    assert!(step.state_update.is_some());
}

#[test]
fn generation_decode_step_delegates_next_token_selection_to_sampling() {
    let mut request = generation_request();
    request.parameters = GenerationParameters::greedy();
    request.stop_conditions = StopConditions::default();
    let mut logits = vec![0.0; request.tokenizer.metadata.vocabulary_size as usize];
    logits[21] = 10.0;

    let (sampling, step) =
        decode_step_from_sampling(&request, &[20, 21], logits, SamplingPolicy::default()).unwrap();

    assert_eq!(sampling.selected_token_id, 22);
    assert_eq!(step.token_id, 22);
    assert_eq!(step.token_index, 2);
    assert!(step.state_update.is_some());
}

#[test]
fn generation_prefill_validates_tokens_and_records_prompt_count() {
    let request = generation_request();
    let state = prefill(&request).unwrap();

    assert_eq!(state.prompt_token_count, 3);
    assert!(state.kv_cache_placeholder.is_some());
    assert!(state.observations.iter().any(|event| {
        event.kind == GenerationEventKind::PrefillStarted && event.request_id == request.request_id
    }));
}

#[test]
fn generation_token_stream_events_preserve_order_and_identity() {
    let request = generation_request();
    let events = token_stream_events(&request, &[10, 11, 12], None).unwrap();

    assert_eq!(events.len(), 3);
    assert_eq!(events[0].token_id, Some(10));
    assert_eq!(events[1].token_index, Some(1));
    assert!(
        events
            .iter()
            .all(|event| event.request_id == request.request_id)
    );
}

#[test]
fn generation_streaming_text_uses_tokenizer_decode() {
    let tokenizer = generation_runtime_tokenizer();
    let output = streaming_text_chunk(
        &tokenizer,
        StreamingDecodeState::default(),
        vec![b'h' as TokenId + 1, b'i' as TokenId + 1],
        true,
    )
    .unwrap();

    assert_eq!(output.text, "hi");
    assert!(output.pending_partial_state.is_none());
}

#[test]
fn generation_prepares_text_stop_sequences_through_tokenizer() {
    let tokenizer = generation_runtime_tokenizer();
    let patterns = prepare_stop_sequences(&tokenizer, &["xy".into()]).unwrap();

    assert_eq!(patterns[0].text, "xy");
    assert_eq!(
        patterns[0].token_ids,
        vec![b'x' as TokenId + 1, b'y' as TokenId + 1]
    );
}

#[test]
fn generation_usage_and_output_account_for_tokens_without_decoded_text() {
    let request = generation_request();
    let output = GenerationOutput::new(&request, vec![10, 11], FinishReason::StopToken);

    output.validate().unwrap();
    assert_eq!(output.generated_token_count, 2);
    assert_eq!(output.usage.prompt_tokens, 3);
    assert_eq!(output.usage.total_tokens, 5);
}

#[test]
fn generation_cancellation_maps_to_stable_finish_reason() {
    let mut request = generation_request();
    request.cancellation.requested = true;

    assert_eq!(
        stop_reason_for(&request, &[]),
        Some(FinishReason::Cancelled)
    );
}

#[test]
fn generation_memory_admission_uses_memory_manager_policy() {
    let mut request = generation_request();
    request.memory.logits_buffer_bytes = 1024;
    let manager = MemoryManager::new(MemoryManagerConfig {
        max_runtime_bytes: Some(64),
        ..MemoryManagerConfig::default()
    });

    assert!(matches!(
        memory_admission(&request, &manager).unwrap(),
        MemoryAdmissionDecision::Reject { .. }
    ));
}

#[test]
fn generation_provider_errors_map_to_finish_reasons() {
    assert_eq!(
        finish_reason_from_provider_error(ProviderExecutionErrorCode::ExecutionInterrupted),
        FinishReason::Interrupted
    );
    assert_eq!(
        finish_reason_from_provider_error(ProviderExecutionErrorCode::OutOfMemory),
        FinishReason::MemoryLimit
    );
    assert_eq!(
        finish_reason_from_provider_error(ProviderExecutionErrorCode::ExecutionFailed),
        FinishReason::ProviderError
    );
}

#[test]
fn generation_contract_has_no_authoritative_provider_or_device_selector() {
    let request = generation_request();

    assert_eq!(request.priority.priority, 3);
    assert!(request.correlation_id.is_some());
    assert!(request.trace_id.is_some());
}

fn reference_cpu_host_tensor(shape: impl Into<Vec<u64>>, data: impl Into<Vec<f32>>) -> HostTensor {
    HostTensor::new(shape, data).unwrap()
}

#[test]
fn reference_cpu_provider_identity_and_device_are_stable() {
    let provider = ReferenceCpuProvider::new();
    let metadata = provider.metadata();
    assert_eq!(metadata.name, REFERENCE_CPU_PROVIDER_NAME);
    assert_eq!(metadata.vendor, REFERENCE_CPU_PROVIDER_VENDOR);

    let devices = provider.devices();
    assert_eq!(devices.len(), 1);
    assert_eq!(devices[0].id().as_str(), REFERENCE_CPU_DEVICE_ID);
    assert_eq!(devices[0].device_type(), DeviceType::Cpu);

    let (min, max) = REFERENCE_CPU_SUPPORTED_RUNTIME_VERSION_RANGE;
    assert!(min <= max);
    assert_eq!(
        REFERENCE_CPU_KERNEL_FAMILY,
        KernelImplementationFamily::CpuScalar
    );
}

#[test]
fn reference_cpu_provider_status_snapshot_reports_health_and_lifecycle() {
    let provider = ReferenceCpuProvider::new();
    let snapshot = provider.status_snapshot();
    assert_eq!(snapshot.provider.as_str(), REFERENCE_CPU_PROVIDER_NAME);
    assert_eq!(snapshot.health, ProviderHealthState::Healthy);
    assert_eq!(snapshot.lifecycle, ProviderLifecycleState::Ready);
    assert_eq!(snapshot.admission, ProviderAdmissionDecision::Admit);
    assert!(snapshot.diagnostics.is_empty());
}

#[test]
fn reference_cpu_provider_pressure_is_explicitly_reportable() {
    let provider = ReferenceCpuProvider::new();
    assert_eq!(
        provider.status_snapshot().pressure,
        ProviderPressureLevel::Low
    );
    provider.report_pressure(ProviderPressureLevel::Saturated);
    assert_eq!(
        provider.status_snapshot().pressure,
        ProviderPressureLevel::Saturated
    );
}

#[test]
fn reference_cpu_device_advertises_dtype_layout_memory_class_and_limits() {
    let device = reference_cpu_device();
    let metadata = device.metadata();
    assert!(metadata.dtype_support.contains(&ComputeDType::Float32));
    assert!(
        metadata
            .layout_support
            .contains(&TensorLayoutKind::Contiguous)
    );
    assert!(
        metadata
            .memory_class_support
            .contains(&KernelMemoryClass::Host)
    );
    assert!(
        metadata
            .execution_limits
            .max_concurrent_operations
            .is_some()
    );
    assert_eq!(metadata.pressure, ProviderPressureLevel::Low);
}

#[test]
fn reference_cpu_initialize_emits_provider_registered_and_device_detected() {
    let provider = ReferenceCpuProvider::new();
    provider.initialize().unwrap();
    let observations = provider.executor().observations();
    assert!(
        observations
            .iter()
            .any(|observation| observation.kind == KernelObservationKind::ProviderRegistered)
    );
    assert!(
        observations
            .iter()
            .any(|observation| observation.kind == KernelObservationKind::DeviceDetected)
    );
}

#[test]
fn reference_cpu_conformance_report_passes_and_is_observed() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let report = executor.run_conformance_checks();
    assert!(
        report.is_conformant(),
        "Reference CPU conformance checks failed: {:?}",
        report.checks
    );
    assert_eq!(report.profile, REFERENCE_CPU_CONFORMANCE_PROFILE);
    assert!(
        executor
            .observations()
            .iter()
            .any(|observation| observation.kind == KernelObservationKind::KernelConformanceResult)
    );
}

#[test]
fn reference_cpu_advertises_only_implemented_kernels() {
    let provider = ReferenceCpuProvider::new();
    let advertisements = provider.kernel_advertisements();
    let names = advertisements
        .iter()
        .map(|advertisement| advertisement.id.name.as_str())
        .collect::<BTreeSet<_>>();
    for expected in [
        "matmul",
        "embedding",
        "rmsnorm",
        "rope",
        "attention",
        "softmax",
        "silu",
        "gelu",
        "activation",
        "add",
        "mul",
        "residual-add",
        "dtype-conversion",
        "layout-conversion",
    ] {
        assert!(names.contains(expected), "missing kernel: {expected}");
    }
    assert!(!names.contains("quantize"));
    assert!(!names.contains("dequantize"));
    for advertisement in &advertisements {
        validate_kernel_advertisement(advertisement).unwrap();
    }
}

#[test]
fn reference_cpu_matmul_known_output() {
    let a = reference_cpu_host_tensor([2, 2], [1.0, 2.0, 3.0, 4.0]);
    let b = reference_cpu_host_tensor([2, 2], [5.0, 6.0, 7.0, 8.0]);
    let result = matmul(&a, &b, false, false).unwrap();
    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(result.data, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn reference_cpu_matmul_rejects_inner_dimension_mismatch() {
    let a = reference_cpu_host_tensor([2, 3], vec![0.0; 6]);
    let b = reference_cpu_host_tensor([2, 2], vec![0.0; 4]);
    assert!(matmul(&a, &b, false, false).is_err());
}

#[test]
fn reference_cpu_embedding_known_output_and_out_of_range() {
    let table = reference_cpu_host_tensor([3, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let ids = reference_cpu_host_tensor([2], [0.0, 2.0]);
    let result = embedding_lookup(&table, &ids).unwrap();
    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(result.data, vec![1.0, 2.0, 5.0, 6.0]);

    let out_of_range = reference_cpu_host_tensor([1], [3.0]);
    assert!(embedding_lookup(&table, &out_of_range).is_err());
}

#[test]
fn reference_cpu_rmsnorm_known_output() {
    let input = reference_cpu_host_tensor([1, 4], [1.0, 2.0, 3.0, 4.0]);
    let weight = reference_cpu_host_tensor([4], [1.0, 1.0, 1.0, 1.0]);
    let result = rmsnorm(&input, &weight, 1e-6).unwrap();
    let mean_square = (1.0_f32 + 4.0 + 9.0 + 16.0) / 4.0;
    let scale = 1.0 / (mean_square + 1e-6).sqrt();
    for (actual, expected) in result.data.iter().zip([1.0, 2.0, 3.0, 4.0]) {
        assert!((actual - expected * scale).abs() < 1e-5);
    }
}

#[test]
fn reference_cpu_rmsnorm_rejects_dtype_shape_mismatch() {
    let input = reference_cpu_host_tensor([1, 4], vec![1.0; 4]);
    let weight = reference_cpu_host_tensor([3], vec![1.0; 3]);
    assert!(rmsnorm(&input, &weight, 1e-6).is_err());
}

#[test]
fn reference_cpu_rope_identity_at_position_zero() {
    let input = reference_cpu_host_tensor([2, 2], [1.0, 2.0, 3.0, 4.0]);
    let result = rope(&input, 10000.0, 1.0, 2, 0).unwrap();
    assert!((result.data[0] - 1.0).abs() < 1e-5);
    assert!((result.data[1] - 2.0).abs() < 1e-5);
}

#[test]
fn reference_cpu_softmax_known_output() {
    let input = reference_cpu_host_tensor([1, 3], [1.0, 1.0, 1.0]);
    let result = softmax_rows(&input).unwrap();
    for value in result.data {
        assert!((value - (1.0 / 3.0)).abs() < 1e-5);
    }
}

#[test]
fn reference_cpu_softmax_rejects_invalid_shape() {
    let input = HostTensor {
        shape: vec![3],
        data: vec![1.0, 2.0, 3.0],
    };
    assert!(softmax_rows(&input).is_err());
}

#[test]
fn reference_cpu_softmax_rejects_fully_masked_row() {
    // Every entry masked out: subtracting the row max would yield NaN for the
    // whole row, so the kernel must reject it rather than return Ok(NaN).
    let input = reference_cpu_host_tensor([1, 3], [f32::NEG_INFINITY; 3]);
    let error = softmax_rows(&input).expect_err("fully masked row must be rejected");
    assert_eq!(error.code, ReferenceCpuErrorCode::ExecutionFailed);
}

#[test]
fn reference_cpu_softmax_allows_partially_masked_row() {
    let input = reference_cpu_host_tensor([1, 3], [f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY]);
    let result = softmax_rows(&input).unwrap();
    assert!(result.data.iter().all(|value| value.is_finite()));
    assert!((result.data[1] - 1.0).abs() < 1e-5);
}

#[test]
fn reference_cpu_silu_known_output() {
    let input = reference_cpu_host_tensor([1], [0.0]);
    let result = silu(&input);
    assert!((result.data[0] - 0.0).abs() < 1e-6);
}

#[test]
fn reference_cpu_elementwise_known_outputs() {
    let a = reference_cpu_host_tensor([2], [1.0, 2.0]);
    let b = reference_cpu_host_tensor([2], [3.0, 4.0]);
    assert_eq!(add(&a, &b).unwrap().data, vec![4.0, 6.0]);
    assert_eq!(mul(&a, &b).unwrap().data, vec![3.0, 8.0]);
    assert_eq!(residual_add(&a, &b).unwrap().data, vec![4.0, 6.0]);

    let mismatched = reference_cpu_host_tensor([3], vec![0.0; 3]);
    assert!(add(&a, &mismatched).is_err());
}

#[test]
fn reference_cpu_attention_causal_masks_future_tokens() {
    let q = reference_cpu_host_tensor([2, 2], [1.0, 0.0, 0.0, 1.0]);
    let k = q.clone();
    let v = reference_cpu_host_tensor([2, 2], [10.0, 10.0, 20.0, 20.0]);
    let result = attention(&q, &k, &v, 1, 2, None, None, true).unwrap();
    // Position 0 can only attend to itself, so its output must equal v[0].
    assert!((result.data[0] - 10.0).abs() < 1e-4);
    assert!((result.data[1] - 10.0).abs() < 1e-4);
}

#[test]
fn reference_cpu_attention_grouped_query_shares_kv_heads() {
    // 2 query heads sharing 1 kv head (head_dimension = 2).
    let q = reference_cpu_host_tensor([1, 4], [1.0, 0.0, 0.0, 1.0]);
    let k = reference_cpu_host_tensor([1, 2], [5.0, 6.0]);
    let v = reference_cpu_host_tensor([1, 2], [7.0, 8.0]);
    let result = attention(&q, &k, &v, 2, 2, Some(1), None, false).unwrap();
    // Single key position: every query head's output must equal v.
    assert_eq!(result.data, vec![7.0, 8.0, 7.0, 8.0]);
}

#[test]
fn reference_cpu_attention_rejects_incompatible_head_grouping() {
    let q = reference_cpu_host_tensor([1, 4], [1.0, 0.0, 0.0, 1.0]);
    let k = reference_cpu_host_tensor([1, 4], [5.0, 6.0, 7.0, 8.0]);
    let v = k.clone();
    // head_count 2 is not a multiple of kv_head_count 3.
    assert!(attention(&q, &k, &v, 2, 2, Some(3), None, false).is_err());
}

#[test]
fn reference_cpu_attention_window_size_restricts_context() {
    let q = reference_cpu_host_tensor([3, 1], [0.0, 0.0, 0.0]);
    let k = q.clone();
    let v = reference_cpu_host_tensor([3, 1], [1.0, 2.0, 3.0]);
    // window_size = 1: each position can only see itself.
    let result = attention(&q, &k, &v, 1, 1, None, Some(1), true).unwrap();
    assert_eq!(result.data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn reference_cpu_attention_rejects_zero_window() {
    let q = reference_cpu_host_tensor([2, 1], [0.0, 0.0]);
    let k = q.clone();
    let v = reference_cpu_host_tensor([2, 1], [1.0, 2.0]);
    // A zero window admits no keys at all; it must not be silently widened to 1.
    let error =
        attention(&q, &k, &v, 1, 1, None, Some(0), true).expect_err("zero window must be rejected");
    assert_eq!(error.code, ReferenceCpuErrorCode::ShapeUnsupported);
}

#[test]
fn reference_cpu_attention_rejects_window_without_causal_mask() {
    let q = reference_cpu_host_tensor([2, 1], [0.0, 0.0]);
    let k = q.clone();
    let v = reference_cpu_host_tensor([2, 1], [1.0, 2.0]);
    // The window is anchored at the query position, which only fully describes
    // the mask under causal attention.
    let error = attention(&q, &k, &v, 1, 1, None, Some(1), false)
        .expect_err("bidirectional sliding window must be rejected");
    assert_eq!(error.code, ReferenceCpuErrorCode::ShapeUnsupported);
}

#[test]
fn reference_cpu_host_tensor_rejects_overflowing_shape() {
    // The product of these dimensions wraps to 0 under unchecked u64
    // multiplication, which would let an empty buffer pass the length check.
    let error = HostTensor::new([1_u64 << 32, 1_u64 << 32], Vec::<f32>::new())
        .expect_err("overflowing shape must be rejected");
    assert_eq!(error.code, ReferenceCpuErrorCode::ShapeUnsupported);
}

#[test]
fn reference_cpu_host_tensor_rejects_shape_beyond_address_space() {
    let error = HostTensor::new([u64::MAX], Vec::<f32>::new())
        .expect_err("shape beyond the address space must be rejected");
    assert_eq!(error.code, ReferenceCpuErrorCode::ShapeUnsupported);
}

fn reference_cpu_kernel_by_name<'a>(
    advertisements: &'a [KernelAdvertisement],
    name: &str,
) -> &'a KernelAdvertisement {
    advertisements
        .iter()
        .find(|advertisement| advertisement.id.name == name)
        .unwrap_or_else(|| panic!("no advertisement named {name}"))
}

#[test]
fn reference_cpu_generic_activation_kernel_dispatches_on_kind() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = reference_cpu_kernel_by_name(&advertisements, "activation");
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();

    let (input_id, input_resource) = reference_cpu_resource("activation-in", [1]);
    let (_out_id, out_resource) = reference_cpu_resource("activation-out", [1]);
    executor.write_tensor(input_id, reference_cpu_host_tensor([1], [0.0]));

    let mut attributes = BTreeMap::new();
    attributes.insert(
        "kind".to_string(),
        OperatorAttributeValue::String("silu".into()),
    );
    let invocation = KernelInvocation::new(
        KernelInvocationId::new("invocation-activation"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(input_resource)
    .with_output(out_resource.clone())
    .with_attributes(attributes);

    let result = executor.execute_invocation(advertisement, operator, &invocation);
    assert_eq!(result.status, KernelResultStatus::Succeeded);
    let output = executor.read_tensor(&out_resource.resource.id).unwrap();
    assert!((output.data[0] - 0.0).abs() < 1e-6);
}

#[test]
fn reference_cpu_generic_activation_kernel_rejects_unknown_kind() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = reference_cpu_kernel_by_name(&advertisements, "activation");
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();

    let (input_id, input_resource) = reference_cpu_resource("activation-bad-in", [1]);
    let (_out_id, out_resource) = reference_cpu_resource("activation-bad-out", [1]);
    executor.write_tensor(input_id, reference_cpu_host_tensor([1], [0.0]));

    let mut attributes = BTreeMap::new();
    attributes.insert(
        "kind".to_string(),
        OperatorAttributeValue::String("relu".into()),
    );
    let invocation = KernelInvocation::new(
        KernelInvocationId::new("invocation-activation-bad"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(input_resource)
    .with_output(out_resource)
    .with_attributes(attributes);

    let result = executor.execute_invocation(advertisement, operator, &invocation);
    assert_eq!(result.status, KernelResultStatus::Failed);
}

#[test]
fn reference_cpu_rope_rejects_unimplemented_position_mode() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = reference_cpu_kernel_by_name(&advertisements, "rope");
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();

    let (input_id, input_resource) = reference_cpu_resource("rope-mode-in", [1, 2]);
    let (_out_id, out_resource) = reference_cpu_resource("rope-mode-out", [1, 2]);
    executor.write_tensor(input_id, reference_cpu_host_tensor([1, 2], [1.0, 2.0]));

    let mut attributes = BTreeMap::new();
    attributes.insert("base".to_string(), OperatorAttributeValue::Float(10000.0));
    attributes.insert("dimension".to_string(), OperatorAttributeValue::Integer(2));
    attributes.insert(
        "position_mode".to_string(),
        OperatorAttributeValue::String("absolute".into()),
    );
    let invocation = KernelInvocation::new(
        KernelInvocationId::new("invocation-rope-mode"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(input_resource)
    .with_output(out_resource)
    .with_attributes(attributes);

    let result = executor.execute_invocation(advertisement, operator, &invocation);
    assert_eq!(result.status, KernelResultStatus::Failed);
}

fn reference_cpu_attention_invocation(
    advertisement: &KernelAdvertisement,
    causal: bool,
    mask_kind: Option<&str>,
    q: KernelResource,
    k: KernelResource,
    v: KernelResource,
    out: KernelResource,
) -> KernelInvocation {
    let mut attributes = BTreeMap::new();
    attributes.insert("head_count".to_string(), OperatorAttributeValue::Integer(1));
    attributes.insert(
        "head_dimension".to_string(),
        OperatorAttributeValue::Integer(2),
    );
    attributes.insert(
        "causal".to_string(),
        OperatorAttributeValue::Boolean(causal),
    );
    if let Some(mask_kind) = mask_kind {
        attributes.insert(
            "attention_mask_kind".to_string(),
            OperatorAttributeValue::String(mask_kind.into()),
        );
    }
    KernelInvocation::new(
        KernelInvocationId::new("invocation-attention"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(q)
    .with_input(k)
    .with_input(v)
    .with_output(out)
    .with_attributes(attributes)
}

#[test]
fn reference_cpu_attention_requires_workspace_from_memory_manager() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = reference_cpu_kernel_by_name(&advertisements, "attention");
    assert!(advertisement.workspace.required);
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();
    let mut memory = MemoryManager::new(MemoryManagerConfig::default());

    let (q_id, q_resource) = reference_cpu_resource("attn-q", [1, 2]);
    let (k_id, k_resource) = reference_cpu_resource("attn-k", [1, 2]);
    let (v_id, v_resource) = reference_cpu_resource("attn-v", [1, 2]);
    let (_out_id, out_resource) = reference_cpu_resource("attn-out", [1, 2]);
    executor.write_tensor(q_id, reference_cpu_host_tensor([1, 2], [1.0, 0.0]));
    executor.write_tensor(k_id, reference_cpu_host_tensor([1, 2], [1.0, 0.0]));
    executor.write_tensor(v_id, reference_cpu_host_tensor([1, 2], [5.0, 6.0]));

    // Without a workspace attached, the shared Kernel Contract validation
    // rejects the invocation before Reference CPU ever runs it.
    let invocation_without_workspace = reference_cpu_attention_invocation(
        advertisement,
        true,
        Some("causal"),
        q_resource.clone(),
        k_resource.clone(),
        v_resource.clone(),
        out_resource.clone(),
    );
    let rejected =
        executor.execute_invocation(advertisement, operator, &invocation_without_workspace);
    assert_eq!(rejected.status, KernelResultStatus::Failed);
    assert_eq!(
        rejected.error,
        Some(KernelError::KernelWorkspaceUnavailable)
    );

    // With a workspace requested through the Memory Manager, execution
    // succeeds.
    let workspace = executor.allocate_workspace(&mut memory, 4096).unwrap();
    let invocation = reference_cpu_attention_invocation(
        advertisement,
        true,
        Some("causal"),
        q_resource,
        k_resource,
        v_resource,
        out_resource.clone(),
    )
    .with_workspace(workspace);
    let result = executor.execute_invocation(advertisement, operator, &invocation);
    assert_eq!(result.status, KernelResultStatus::Succeeded);
    let output = executor.read_tensor(&out_resource.resource.id).unwrap();
    assert_eq!(output.data, vec![5.0, 6.0]);
}

#[test]
fn reference_cpu_attention_mask_kind_must_be_consistent_with_causal_flag() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = reference_cpu_kernel_by_name(&advertisements, "attention");
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();
    let mut memory = MemoryManager::new(MemoryManagerConfig::default());
    let workspace = executor.allocate_workspace(&mut memory, 4096).unwrap();

    let (q_id, q_resource) = reference_cpu_resource("attn-mismatch-q", [1, 2]);
    let (k_id, k_resource) = reference_cpu_resource("attn-mismatch-k", [1, 2]);
    let (v_id, v_resource) = reference_cpu_resource("attn-mismatch-v", [1, 2]);
    let (_out_id, out_resource) = reference_cpu_resource("attn-mismatch-out", [1, 2]);
    executor.write_tensor(q_id, reference_cpu_host_tensor([1, 2], [1.0, 0.0]));
    executor.write_tensor(k_id, reference_cpu_host_tensor([1, 2], [1.0, 0.0]));
    executor.write_tensor(v_id, reference_cpu_host_tensor([1, 2], [5.0, 6.0]));

    // causal=false but attention_mask_kind says "causal": inconsistent.
    let invocation = reference_cpu_attention_invocation(
        advertisement,
        false,
        Some("causal"),
        q_resource,
        k_resource,
        v_resource,
        out_resource,
    )
    .with_workspace(workspace);
    let result = executor.execute_invocation(advertisement, operator, &invocation);
    assert_eq!(result.status, KernelResultStatus::Failed);
}

#[test]
fn reference_cpu_dtype_conversion_rejects_non_f32() {
    let input = reference_cpu_host_tensor([1], [1.0]);
    assert!(dtype_conversion(&input, ComputeDType::Float32, ComputeDType::Float32).is_ok());
    assert!(dtype_conversion(&input, ComputeDType::Float16, ComputeDType::Float32).is_err());
}

#[test]
fn reference_cpu_layout_conversion_rejects_non_contiguous() {
    let input = reference_cpu_host_tensor([1], [1.0]);
    assert!(
        layout_conversion(
            &input,
            TensorLayoutKind::Contiguous,
            TensorLayoutKind::Contiguous
        )
        .is_ok()
    );
    assert!(
        layout_conversion(
            &input,
            TensorLayoutKind::Contiguous,
            TensorLayoutKind::Strided
        )
        .is_err()
    );
}

#[test]
fn reference_cpu_quantization_is_explicitly_unsupported() {
    let error = dequantize_placeholder();
    assert_eq!(error.id(), "reference-cpu-dtype-unsupported");
}

#[test]
fn reference_cpu_fallback_denied_by_default_allowed_by_policy() {
    let pinned = ResourceAffinity::new(FallbackClass::ProviderPinned);
    assert!(evaluate_fallback(&pinned, &FallbackPolicyContext::new(true)).is_err());

    let transparent = ResourceAffinity::new(FallbackClass::Transparent);
    assert!(evaluate_fallback(&transparent, &FallbackPolicyContext::new(false)).is_err());
    assert!(evaluate_fallback(&transparent, &FallbackPolicyContext::new(true)).is_ok());
}

#[test]
fn reference_cpu_fallback_denied_when_dtype_or_layout_conversion_forbidden() {
    let transparent = ResourceAffinity::new(FallbackClass::Transparent);
    let dtype_denied = FallbackPolicyContext::new(true).with_dtype_conversion(true, false);
    assert!(evaluate_fallback(&transparent, &dtype_denied).is_err());

    let layout_denied = FallbackPolicyContext::new(true).with_layout_conversion(true, false);
    assert!(evaluate_fallback(&transparent, &layout_denied).is_err());

    let both_allowed = FallbackPolicyContext::new(true)
        .with_dtype_conversion(true, true)
        .with_layout_conversion(true, true);
    assert!(evaluate_fallback(&transparent, &both_allowed).is_ok());
}

#[test]
fn reference_cpu_fallback_is_observable() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let kernel = provider
        .kernel_advertisements()
        .into_iter()
        .find(|advertisement| advertisement.id.name == "matmul")
        .unwrap()
        .id;
    let transparent = ResourceAffinity::new(FallbackClass::Transparent);

    executor
        .evaluate_fallback_observed(&kernel, &transparent, &FallbackPolicyContext::new(true))
        .unwrap();
    let observations = executor.observations();
    assert!(
        observations
            .iter()
            .any(|observation| observation.kind == KernelObservationKind::KernelFallbackConsidered)
    );
    assert!(
        observations
            .iter()
            .any(|observation| observation.kind == KernelObservationKind::KernelFallbackUsed)
    );

    executor
        .evaluate_fallback_observed(&kernel, &transparent, &FallbackPolicyContext::new(false))
        .unwrap_err();
    assert!(
        executor
            .observations()
            .iter()
            .any(|observation| observation.kind == KernelObservationKind::KernelFallbackFailed)
    );
}

fn reference_cpu_resource(
    id: &str,
    shape: impl Into<Vec<u64>>,
) -> (TensorResourceId, KernelResource) {
    let resource_id = TensorResourceId::new(id);
    let descriptor = TensorResourceDescriptor::new(
        resource_id.clone(),
        TensorDescriptor::materialized(
            ShapeDescriptor::new(shape.into()),
            DTypeDescriptor::portable(ComputeDType::Float32),
        ),
        ResourceAffinity::new(FallbackClass::Transparent),
    );
    (
        resource_id,
        KernelResource::new(descriptor, KernelMemoryClass::Host),
    )
}

#[test]
fn reference_cpu_executes_matmul_invocation_end_to_end() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = advertisements
        .iter()
        .find(|advertisement| advertisement.id.name == "matmul")
        .unwrap();
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();

    let (a_id, a_resource) = reference_cpu_resource("a", [2, 2]);
    let (b_id, b_resource) = reference_cpu_resource("b", [2, 2]);
    let (_out_id, out_resource) = reference_cpu_resource("out", [2, 2]);
    executor.write_tensor(
        a_id,
        reference_cpu_host_tensor([2, 2], [1.0, 2.0, 3.0, 4.0]),
    );
    executor.write_tensor(
        b_id,
        reference_cpu_host_tensor([2, 2], [5.0, 6.0, 7.0, 8.0]),
    );

    let invocation = KernelInvocation::new(
        KernelInvocationId::new("invocation-1"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(a_resource)
    .with_input(b_resource)
    .with_output(out_resource.clone());

    let result = executor.execute_invocation(advertisement, operator, &invocation);
    assert_eq!(result.status, KernelResultStatus::Succeeded);
    let output = executor.read_tensor(&out_resource.resource.id).unwrap();
    assert_eq!(output.data, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn reference_cpu_execution_tracks_output_through_memory_manager() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = advertisements
        .iter()
        .find(|advertisement| advertisement.id.name == "matmul")
        .unwrap();
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();
    let mut memory = MemoryManager::new(MemoryManagerConfig::default());

    let (a_id, a_resource) = reference_cpu_resource("mm-a", [2, 2]);
    let (b_id, b_resource) = reference_cpu_resource("mm-b", [2, 2]);
    let (_out_id, out_resource) = reference_cpu_resource("mm-out", [2, 2]);
    executor.write_tensor(
        a_id,
        reference_cpu_host_tensor([2, 2], [1.0, 0.0, 0.0, 1.0]),
    );
    executor.write_tensor(
        b_id,
        reference_cpu_host_tensor([2, 2], [1.0, 2.0, 3.0, 4.0]),
    );

    let invocation = KernelInvocation::new(
        KernelInvocationId::new("invocation-memory"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(a_resource)
    .with_input(b_resource)
    .with_output(out_resource.clone());

    let result = executor.execute_invocation_with_memory_manager(
        advertisement,
        operator,
        &invocation,
        &mut memory,
    );
    assert_eq!(result.status, KernelResultStatus::Succeeded);

    let residency = memory
        .tensor_residency(&out_resource.resource.id)
        .expect("Memory Manager should record residency for the output tensor");
    assert!(residency.provider_owned);
    assert!(residency.allocation.is_some());
    assert!(memory.allocations().next().is_some());
}

#[test]
fn reference_cpu_emits_memory_feasibility_failed_observation_on_allocation_rejection() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = advertisements
        .iter()
        .find(|advertisement| advertisement.id.name == "matmul")
        .unwrap();
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();
    // A runtime byte budget too small for any tensor allocation to be admitted.
    let mut memory = MemoryManager::new(MemoryManagerConfig {
        max_runtime_bytes: Some(1),
        ..MemoryManagerConfig::default()
    });

    let (a_id, a_resource) = reference_cpu_resource("mem-fail-a", [2, 2]);
    let (b_id, b_resource) = reference_cpu_resource("mem-fail-b", [2, 2]);
    let (_out_id, out_resource) = reference_cpu_resource("mem-fail-out", [2, 2]);
    executor.write_tensor(a_id, reference_cpu_host_tensor([2, 2], vec![0.0; 4]));
    executor.write_tensor(b_id, reference_cpu_host_tensor([2, 2], vec![0.0; 4]));

    let invocation = KernelInvocation::new(
        KernelInvocationId::new("invocation-memory-fail"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(a_resource)
    .with_input(b_resource)
    .with_output(out_resource);

    let result = executor.execute_invocation_with_memory_manager(
        advertisement,
        operator,
        &invocation,
        &mut memory,
    );
    // Kernel execution itself still succeeds (opaque storage is independent);
    // only the Memory Manager accounting request is rejected.
    assert_eq!(result.status, KernelResultStatus::Succeeded);
    assert!(executor.observations().iter().any(
        |observation| observation.kind == KernelObservationKind::KernelMemoryFeasibilityFailed
    ));
}

#[test]
fn reference_cpu_honors_already_elapsed_deadline_as_timeout() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = advertisements
        .iter()
        .find(|advertisement| advertisement.id.name == "matmul")
        .unwrap();
    assert_eq!(
        advertisement.cancellation,
        KernelCancellationSupport::TimeoutOnly
    );
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();

    let (a_id, a_resource) = reference_cpu_resource("deadline-a", [2, 2]);
    let (b_id, b_resource) = reference_cpu_resource("deadline-b", [2, 2]);
    let (_out_id, out_resource) = reference_cpu_resource("deadline-out", [2, 2]);
    executor.write_tensor(a_id, reference_cpu_host_tensor([2, 2], vec![0.0; 4]));
    executor.write_tensor(b_id, reference_cpu_host_tensor([2, 2], vec![0.0; 4]));

    let mut invocation = KernelInvocation::new(
        KernelInvocationId::new("invocation-deadline"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(a_resource)
    .with_input(b_resource)
    .with_output(out_resource);
    invocation.deadline_millis = Some(0);

    let result = executor.execute_invocation(advertisement, operator, &invocation);
    assert_eq!(result.status, KernelResultStatus::Failed);
    assert_eq!(result.error, Some(KernelError::KernelTimeout));
    assert!(
        executor
            .observations()
            .iter()
            .any(|observation| observation.kind == KernelObservationKind::KernelTimeout)
    );
}

#[test]
fn reference_cpu_kernel_registry_selects_registered_candidate() {
    let provider = ReferenceCpuProvider::new();
    let mut registry = KernelRegistry::new();
    for advertisement in provider.kernel_advertisements() {
        registry
            .register_provider_advertisement(advertisement)
            .unwrap();
    }
    registry.set_provider_status(ProviderStatusSnapshot::from_health_report(
        ProviderHealthReport::new(
            ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
            HealthState::Available,
        ),
    ));
    let matmul_operator = OperatorId::magnetar("matmul", 1, OperatorFamily::LinearAlgebra);
    let request = KernelSelectionRequest::new(
        "select-matmul",
        matmul_operator,
        ResourceAffinity::new(FallbackClass::Transparent),
    );
    let selection = registry.select(&request).unwrap();
    let selected = selection
        .selected
        .expect("a compatible Reference CPU candidate should be selected");
    assert_eq!(selected.provider.as_str(), REFERENCE_CPU_PROVIDER_NAME);
    assert!(
        selection
            .observations
            .iter()
            .any(|observation| observation.kind == KernelObservationKind::KernelSelected)
    );
}

#[test]
fn reference_cpu_execution_only_accepts_runtime_created_invocation_shapes() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = advertisements
        .iter()
        .find(|advertisement| advertisement.id.name == "matmul")
        .unwrap();
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();

    // Only one input bound where the Operator requires two: Runtime-level
    // validation must reject it rather than the Provider guessing.
    let (a_id, a_resource) = reference_cpu_resource("a", [2, 2]);
    let (out_id, out_resource) = reference_cpu_resource("out", [2, 2]);
    executor.write_tensor(
        a_id,
        reference_cpu_host_tensor([2, 2], [1.0, 2.0, 3.0, 4.0]),
    );
    let _ = out_id;

    let invocation = KernelInvocation::new(
        KernelInvocationId::new("invocation-2"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(a_resource)
    .with_output(out_resource);

    let result = executor.execute_invocation(advertisement, operator, &invocation);
    assert_eq!(result.status, KernelResultStatus::Failed);
    assert!(result.error.is_some());
}

#[test]
fn reference_cpu_provider_passes_generic_conformance_core_profile() {
    let suite =
        ProviderConformanceSuite::new(ProviderConformanceConfig::default().with_profiles([
            ProviderConformanceProfile::ProviderCore,
            ProviderConformanceProfile::ProviderObservability,
        ]));
    let report = suite.run(ProviderConformanceTarget::BuiltIn {
        provider: Arc::new(ReferenceCpuProvider::new()),
    });
    assert!(
        report.is_conformant(),
        "Reference CPU Provider failed conformance: {:?}",
        report.failed_tests
    );
}

#[test]
fn reference_cpu_kernel_registry_accepts_provider_advertisements() {
    let provider = ReferenceCpuProvider::new();
    let mut registry = KernelRegistry::new();
    for advertisement in provider.kernel_advertisements() {
        registry
            .register_provider_advertisement(advertisement)
            .unwrap();
    }
    let matmul_id = provider
        .kernel_advertisements()
        .into_iter()
        .find(|advertisement| advertisement.id.name == "matmul")
        .unwrap()
        .id;
    assert!(registry.active_advertisement(&matmul_id).is_some());
}

#[test]
fn reference_cpu_no_raw_handles_exposed_in_invocation_or_result() {
    let provider = ReferenceCpuProvider::new();
    let executor = provider.executor();
    let advertisements = provider.kernel_advertisements();
    let advertisement = advertisements
        .iter()
        .find(|advertisement| advertisement.id.name == "matmul")
        .unwrap();
    let catalog = initial_operator_catalog();
    let operator = catalog.get(&advertisement.implemented_operator).unwrap();

    let (a_id, a_resource) = reference_cpu_resource("a", [2, 2]);
    let (b_id, b_resource) = reference_cpu_resource("b", [2, 2]);
    let (_, out_resource) = reference_cpu_resource("out", [2, 2]);
    executor.write_tensor(
        a_id,
        reference_cpu_host_tensor([2, 2], [1.0, 2.0, 3.0, 4.0]),
    );
    executor.write_tensor(
        b_id,
        reference_cpu_host_tensor([2, 2], [5.0, 6.0, 7.0, 8.0]),
    );

    let invocation = KernelInvocation::new(
        KernelInvocationId::new("invocation-3"),
        advertisement.implemented_operator.clone(),
        advertisement.id.clone(),
        ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(a_resource)
    .with_input(b_resource)
    .with_output(out_resource);

    let result = executor.execute_invocation(advertisement, operator, &invocation);
    let text = format!("{invocation:?} {result:?}");
    assert!(!text.contains("0x"));
    assert!(!text.contains("raw handle"));
}

fn tensor_resource_for_test(id: &str) -> TensorResource {
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let residency = TensorResidency::new(
        TensorResourceId::new(id),
        MemoryPlacement::HostOrdinary,
        ResourceAffinity::new(FallbackClass::Transparent),
    );
    TensorResource::new(TensorResourceId::new(id), descriptor, residency)
}

#[test]
fn tensor_lifecycle_allows_declared_to_ready_happy_path() {
    let mut resource = tensor_resource_for_test("tensor-lifecycle-1");
    resource
        .transition_to(TensorLifecycleState::Planned)
        .unwrap();
    resource
        .transition_to(TensorLifecycleState::Allocating)
        .unwrap();
    resource.mark_ready().unwrap();
    assert_eq!(resource.lifecycle, TensorLifecycleState::Ready);
    assert_eq!(resource.readiness, TensorReadiness::Ready);
    assert!(resource.ensure_usable().is_ok());
}

#[test]
fn tensor_lifecycle_rejects_declared_to_ready_skip() {
    let mut resource = tensor_resource_for_test("tensor-lifecycle-2");
    let error = resource
        .transition_to(TensorLifecycleState::Ready)
        .unwrap_err();
    assert!(matches!(error, TensorError::ResourceInvalid { .. }));
}

#[test]
fn tensor_resource_released_is_rejected_for_use() {
    let mut resource = tensor_resource_for_test("tensor-lifecycle-3");
    resource
        .transition_to(TensorLifecycleState::Planned)
        .unwrap();
    resource
        .transition_to(TensorLifecycleState::Allocating)
        .unwrap();
    resource.mark_ready().unwrap();
    resource
        .transition_to(TensorLifecycleState::Released)
        .unwrap();
    assert_eq!(
        resource.ensure_usable().unwrap_err(),
        TensorError::ResourceReleased
    );
}

#[test]
fn tensor_readiness_blocks_dispatch_until_ready() {
    let mut resource = tensor_resource_for_test("tensor-readiness-1");
    resource
        .transition_to(TensorLifecycleState::Planned)
        .unwrap();
    resource
        .transition_to(TensorLifecycleState::Allocating)
        .unwrap();
    resource.transition_to(TensorLifecycleState::Ready).unwrap();
    resource.readiness = TensorReadiness::PendingTransfer;
    assert!(matches!(
        resource.ensure_usable().unwrap_err(),
        TensorError::ResourceNotReady { .. }
    ));
    resource.readiness = TensorReadiness::Ready;
    assert!(resource.ensure_usable().is_ok());
}

#[test]
fn tensor_mutability_denies_mutation_of_immutable_resource() {
    let error =
        validate_mutability_for_dispatch(TensorMutabilityKind::Immutable, true).unwrap_err();
    assert!(matches!(error, TensorError::MutabilityViolation { .. }));
    assert!(validate_mutability_for_dispatch(TensorMutabilityKind::Mutable, true).is_ok());
    assert!(validate_mutability_for_dispatch(TensorMutabilityKind::Immutable, false).is_ok());
}

#[test]
fn tensor_aliasing_requires_in_place_kernel_support() {
    let error =
        validate_aliasing_for_dispatch(TensorAliasingKind::InputOutputAlias, false).unwrap_err();
    assert!(matches!(error, TensorError::AliasingViolation { .. }));
    assert!(validate_aliasing_for_dispatch(TensorAliasingKind::InputOutputAlias, true).is_ok());
    assert!(validate_aliasing_for_dispatch(TensorAliasingKind::NoAlias, false).is_ok());
}

#[test]
fn tensor_memory_class_is_derived_from_memory_placement() {
    assert_eq!(
        TensorMemoryClass::from(&MemoryPlacement::HostOrdinary),
        TensorMemoryClass::Host
    );
    assert_eq!(
        TensorMemoryClass::from(&MemoryPlacement::HostPinned),
        TensorMemoryClass::PinnedHost
    );
    assert_eq!(
        TensorMemoryClass::from(&MemoryPlacement::BrowserLinearMemory),
        TensorMemoryClass::BrowserLinearMemory
    );
    let staged = MemoryPlacement::StagedTemporary(Box::new(MemoryPlacement::HostOrdinary));
    assert_eq!(TensorMemoryClass::from(&staged), TensorMemoryClass::Host);
}

#[test]
fn tensor_memory_class_validation_rejects_unsupported_class() {
    let error = validate_memory_class_for_kernel(
        TensorMemoryClass::Device,
        &[TensorMemoryClass::Host, TensorMemoryClass::PinnedHost],
    )
    .unwrap_err();
    assert!(matches!(error, TensorError::MemoryClassUnsupported { .. }));
    assert!(validate_memory_class_for_kernel(TensorMemoryClass::Device, &[]).is_ok());
    assert!(
        validate_memory_class_for_kernel(TensorMemoryClass::Host, &[TensorMemoryClass::Host])
            .is_ok()
    );
}

#[test]
fn tensor_view_becomes_unavailable_once_base_is_terminal() {
    let view = TensorView::new(
        TensorResourceId::new("base"),
        ViewDescriptor::from_resource(TensorResourceId::new("base"), 0, [1, 1]),
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
        ResourceAffinity::new(FallbackClass::Transparent),
    );
    assert!(
        view.validate_against_base(TensorLifecycleState::Ready)
            .is_ok()
    );
    let error = view
        .validate_against_base(TensorLifecycleState::Released)
        .unwrap_err();
    assert!(matches!(error, TensorError::ViewBaseUnavailable { .. }));
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
fn operator_layout_kind_maps_every_layout_descriptor_variant() {
    assert_eq!(
        layout_kind(&LayoutDescriptor::Blocked {
            block_dimensions: vec![4],
        }),
        TensorLayoutKind::Blocked
    );
    assert_eq!(
        layout_kind(&LayoutDescriptor::Paged {
            page_size_elements: 16,
            block_size_elements: 4,
            capacity_pages: None,
            current_length_elements: None,
            logical_to_physical: None,
            append_behavior: None,
        }),
        TensorLayoutKind::Paged
    );
    assert_eq!(
        layout_kind(&LayoutDescriptor::PackedQuantized {
            method: "int4".into(),
            bits_per_value: 4,
            group_size: None,
            scale_dtype: None,
            zero_point_dtype: None,
            packing_order: None,
            dequantization_requirements: None,
        }),
        TensorLayoutKind::QuantizedPacked
    );
}

#[test]
fn tensor_error_and_observation_redact_backend_diagnostics() {
    let error = TensorError::resource_invalid("native handle=0xdeadbeef");
    assert_eq!(
        error.to_string(),
        "tensor resource invalid: [redacted backend diagnostic]"
    );
    let observation = TensorObservation::new(TensorObservationKind::ResourceReady)
        .with_message("C:\\weights\\model.bin");
    assert_eq!(observation.message, "[redacted backend diagnostic]");
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
fn tensor_residency_tracks_eviction_size_estimate_and_host_visibility() {
    let host = TensorResidency::new(
        TensorResourceId::new("residency-host"),
        MemoryPlacement::HostOrdinary,
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_eviction_eligible(true)
    .with_size_estimate(4096);
    assert!(host.eviction_eligible);
    assert_eq!(host.size_bytes_estimate, Some(4096));
    assert!(host.is_host_visible());
    assert_eq!(host.memory_class(), TensorMemoryClass::Host);

    let device = TensorResidency::new(
        TensorResourceId::new("residency-device"),
        MemoryPlacement::Device(DeviceBinding::new(DeviceId::new("gpu-0"))),
        ResourceAffinity::new(FallbackClass::Transparent),
    );
    assert!(!device.is_host_visible());
    assert_eq!(device.memory_class(), TensorMemoryClass::Device);
}

#[test]
fn tensor_residency_affinity_rejects_forged_device_binding() {
    let claimed = TensorResidency::new(
        TensorResourceId::new("forged"),
        MemoryPlacement::Device(DeviceBinding::new(DeviceId::new("gpu-0"))),
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_device(DeviceBinding::new(DeviceId::new("gpu-0"))),
    );
    let actual = TensorResidency::new(
        TensorResourceId::new("forged"),
        MemoryPlacement::Device(DeviceBinding::new(DeviceId::new("gpu-1"))),
        ResourceAffinity::new(FallbackClass::Transparent)
            .with_device(DeviceBinding::new(DeviceId::new("gpu-1"))),
    );
    let error = AffinityConstraints::try_from_affinities([&claimed.affinity, &actual.affinity])
        .unwrap_err();
    assert!(matches!(error, AffinityError::DeviceMismatch { .. }));
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
fn kernel_result_tracks_aliasing_updates_alongside_readiness_and_residency() {
    let result = KernelResult::success(KernelInvocationId::new("aliasing-update"))
        .with_aliasing_update("output", TensorAliasingKind::InputOutputAlias);
    assert_eq!(
        result.updated_aliasing.get("output"),
        Some(&TensorAliasingKind::InputOutputAlias)
    );
}

#[test]
fn tensor_resource_debug_output_never_exposes_raw_pointers_or_handles() {
    let resource = tensor_resource_for_test("tensor-debug-safety");
    let text = format!("{resource:?}");
    assert!(!text.contains("0x"));
    assert!(!text.contains("handle="));
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

#[test]
fn reference_cpu_quantize_and_dequantize_placeholders_reject_explicitly() {
    for error in [dequantize_placeholder(), quantize_placeholder()] {
        assert_eq!(error.code, ReferenceCpuErrorCode::DTypeUnsupported);
    }
}

#[test]
fn memory_manager_admits_tensor_computed_from_descriptor_size() {
    let manager = MemoryManager::default();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([4, 4]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let decision = manager.admit_tensor(
        &descriptor,
        MemoryPlacement::HostOrdinary,
        MemoryAllocationOwner::Runtime,
        MemoryPressureSnapshot::default(),
    );
    assert!(matches!(decision, MemoryAdmissionDecision::Admit { .. }));
}

#[test]
fn memory_manager_rejects_tensor_admission_when_size_is_unknown() {
    let manager = MemoryManager::default();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new([u64::MAX, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let decision = manager.admit_tensor(
        &descriptor,
        MemoryPlacement::HostOrdinary,
        MemoryAllocationOwner::Runtime,
        MemoryPressureSnapshot::default(),
    );
    assert!(matches!(decision, MemoryAdmissionDecision::Reject { .. }));
}

fn permissive_operator_spec(memory: OperatorMemoryBehavior) -> OperatorSpec {
    OperatorSpec::new(
        OperatorId::new(
            OPERATOR_NAMESPACE,
            "conformance-op",
            1,
            OperatorFamily::Tensor,
        ),
        1,
        1,
    )
    .with_dtype_contract(OperatorDTypeContract::new(TensorRole::Input, []))
    .with_memory(memory)
}

fn contiguous_f32_tensor() -> TensorDescriptor {
    TensorDescriptor::materialized(
        ShapeDescriptor::new([2, 2]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    )
}

#[test]
fn operator_validate_invocation_rejects_aliasing_intent_without_in_place_support() {
    let spec = permissive_operator_spec(OperatorMemoryBehavior::pure());
    let output = contiguous_f32_tensor().with_aliasing_intent(TensorAliasingKind::InputOutputAlias);
    let error = spec
        .validate_invocation(&[contiguous_f32_tensor()], &[output], &BTreeMap::new())
        .unwrap_err();
    assert!(matches!(error, OperatorError::AliasingUnsupported { .. }));
}

#[test]
fn operator_validate_invocation_accepts_aliasing_intent_when_in_place_supported() {
    let memory = OperatorMemoryBehavior {
        supports_in_place: true,
        ..OperatorMemoryBehavior::pure()
    };
    let spec = permissive_operator_spec(memory);
    let output = contiguous_f32_tensor().with_aliasing_intent(TensorAliasingKind::InputOutputAlias);
    assert!(
        spec.validate_invocation(&[contiguous_f32_tensor()], &[output], &BTreeMap::new())
            .is_ok()
    );
}

#[test]
fn operator_validate_invocation_rejects_memory_class_conflict() {
    let memory = OperatorMemoryBehavior {
        requires_host_visible: true,
        ..OperatorMemoryBehavior::pure()
    };
    let spec = permissive_operator_spec(memory);
    let input = contiguous_f32_tensor().with_memory_class_intent(TensorMemoryClass::Device);
    let error = spec
        .validate_invocation(&[input], &[contiguous_f32_tensor()], &BTreeMap::new())
        .unwrap_err();
    assert!(matches!(
        error,
        OperatorError::MemoryBehaviorUnsupported { .. }
    ));
}

#[test]
fn operator_validate_invocation_rejects_mutation_of_immutable_input() {
    let memory = OperatorMemoryBehavior {
        mutates_input: true,
        ..OperatorMemoryBehavior::pure()
    };
    let spec = permissive_operator_spec(memory);
    let input = contiguous_f32_tensor().with_mutability_intent(TensorMutabilityKind::Immutable);
    let error = spec
        .validate_invocation(&[input], &[contiguous_f32_tensor()], &BTreeMap::new())
        .unwrap_err();
    assert!(matches!(
        error,
        OperatorError::MemoryBehaviorUnsupported { .. }
    ));
}

// ---------------------------------------------------------------------
// Runtime Inference API
// ---------------------------------------------------------------------

#[test]
fn inference_api_model_reference_resolves_through_local_registry() {
    let reference = ModelRef::new("qwen-test").unwrap();
    let artifact = ModelArtifactId::new(
        ModelArtifactKind::ModelWeights,
        ModelName::new("qwen").unwrap(),
        ModelRevision::new("1").unwrap(),
        ModelDigest::sha256(b"weights"),
    );
    let mut registry = ModelRegistry::new();
    registry.register(reference.clone(), artifact.clone());

    let result = registry
        .resolve(&ModelResolutionRequest::new(reference))
        .unwrap();
    assert_eq!(result.artifact, artifact);
}

#[test]
fn inference_api_model_reference_resolution_fails_for_unregistered_reference() {
    let registry = ModelRegistry::new();
    let reference = ModelRef::new("unknown-model").unwrap();
    let error = registry
        .resolve(&ModelResolutionRequest::new(reference))
        .unwrap_err();
    assert!(matches!(
        error,
        InferenceApiError::ModelResolutionFailed { .. }
    ));
}

#[test]
fn inference_api_model_reference_rejects_path_like_input() {
    assert!(matches!(
        ModelRef::new("../etc/passwd"),
        Err(InferenceApiError::ModelReferenceInvalid { .. })
    ));
    assert!(matches!(
        ModelRef::new("models/qwen"),
        Err(InferenceApiError::ModelReferenceInvalid { .. })
    ));
}

#[test]
fn inference_api_scope_validation_rejects_non_inference_capabilities() {
    assert!(validate_inference_scope("generation").is_ok());
    for forbidden in [
        "workspace-filesystem",
        "git",
        "shell",
        "secrets",
        "tool-call",
    ] {
        assert!(matches!(
            validate_inference_scope(forbidden),
            Err(InferenceApiError::PolicyDenied { .. })
        ));
    }
}

#[test]
fn inference_api_session_creation_rejects_forbidden_allowed_capabilities() {
    let runtime = &mut Runtime::builder().build().unwrap();
    let mut request = session_creation_request();
    request.allowed_capabilities.insert("shell".into());

    let error = create_inference_session(runtime, request).unwrap_err();
    assert!(matches!(error, InferenceApiError::PolicyDenied { .. }));
}

#[test]
fn inference_api_session_creation_succeeds_with_inference_only_capabilities() {
    let runtime = &mut Runtime::builder().build().unwrap();
    let mut request = session_creation_request();
    request.allowed_capabilities.insert("generation".into());

    let session = create_inference_session(runtime, request).unwrap();
    let status = session_status(
        runtime,
        &session,
        &SessionAccessPolicy::authorize(session.clone()),
    )
    .unwrap();
    assert_eq!(status.id, session);
    assert!(!status.raw_prompt_available);
    assert!(!status.raw_handles_available);
}

#[test]
fn inference_api_tokenize_plain_text_never_stores_raw_prompt_text() {
    let tokenizer = FixtureTokenizer::new(generation_tokenizer_metadata());
    let request = TokenizationRequest::new(PromptInput::PlainText("secret".into()));

    let result = tokenize_prompt_input(&tokenizer, request, None).unwrap();
    assert!(!result.token_ids.is_empty());
    assert!(!format!("{result:?}").contains("secret"));
}

#[test]
fn inference_api_tokenize_chat_messages_requires_authorized_formatter() {
    let tokenizer = FixtureTokenizer::new(generation_tokenizer_metadata());
    let request = TokenizationRequest::new(PromptInput::ChatMessages(vec![ChatMessage::new(
        "user", "hello",
    )]));

    let error = tokenize_prompt_input(&tokenizer, request, None).unwrap_err();
    assert!(matches!(error, InferenceApiError::PolicyDenied { .. }));
}

struct ConcatenationChatTemplate;
impl ChatTemplateFormatter for ConcatenationChatTemplate {
    fn format(&self, messages: &[ChatMessage]) -> Result<String, InferenceApiError> {
        Ok(messages
            .iter()
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>()
            .join(" "))
    }
}

#[test]
fn inference_api_tokenize_chat_messages_formats_through_authorized_contract() {
    let tokenizer = FixtureTokenizer::new(generation_tokenizer_metadata());
    let request = TokenizationRequest::new(PromptInput::ChatMessages(vec![ChatMessage::new(
        "user", "hi",
    )]));

    let result =
        tokenize_prompt_input(&tokenizer, request, Some(&ConcatenationChatTemplate)).unwrap();
    assert!(!result.token_ids.is_empty());
}

#[test]
fn inference_api_tokenize_already_tokenized_input_validates_range() {
    let tokenizer = FixtureTokenizer::new(generation_tokenizer_metadata());
    let in_range = TokenizationRequest::new(PromptInput::TokenIds(vec![2, 3, 4]));
    assert!(tokenize_prompt_input(&tokenizer, in_range, None).is_ok());

    let out_of_range = TokenizationRequest::new(PromptInput::TestTokenSequence(vec![99_999]));
    let error = tokenize_prompt_input(&tokenizer, out_of_range, None).unwrap_err();
    assert!(matches!(
        error,
        InferenceApiError::TokenizationFailed { .. }
    ));
}

#[test]
fn inference_api_build_generation_request_from_tokenized_input() {
    let metadata = generation_tokenizer_metadata();
    let tokenizer = GenerationTokenizerReference {
        tokenizer_id: metadata.id.clone(),
        metadata,
    };
    let tokenized = TokenizationResult {
        token_ids: vec![2, 3, 4],
        token_count: 3,
        offsets: None,
        diagnostics: Vec::new(),
        correlation_id: Some(CorrelationId::new("corr-1")),
    };

    let request = build_generation_request(
        GenerationRequestId::new("gen-api-1").unwrap(),
        None,
        GenerationModelReference::LoadedModelContext("model-context".into()),
        tokenizer,
        tokenized,
        4,
        GenerationParameters::default(),
        StopConditions::default(),
        StreamingMode::TokenIds,
    );

    assert_eq!(request.prompt_token_count, 3);
    request.validate().unwrap();
}

#[test]
fn inference_api_admission_state_reports_structured_backpressure() {
    assert_eq!(
        AdmissionState::from(&MemoryAdmissionDecision::Admit {
            reason: "ok".into()
        }),
        AdmissionState::Accepted
    );
    assert!(matches!(
        AdmissionState::from(&MemoryAdmissionDecision::Queue {
            reason: "busy".into()
        }),
        AdmissionState::Queued { .. }
    ));
    assert!(matches!(
        AdmissionState::from(&MemoryAdmissionDecision::Reject {
            reason: "no memory".into()
        }),
        AdmissionState::Rejected { .. }
    ));
    assert!(matches!(
        AdmissionState::from(&MemoryAdmissionDecision::RetryLater {
            reason: "pressure".into()
        }),
        AdmissionState::Delayed { .. }
    ));
}

#[test]
fn inference_api_submit_generation_admits_compatible_request_into_batch() {
    let mut runtime = Runtime::builder().build().unwrap();
    let policy = BatchingPolicy {
        allow_queueing: false,
        ..BatchingPolicy::default()
    };
    let batch = runtime.create_continuous_batch(policy);
    let request = generation_request();

    let (state, slot) = submit_generation(&mut runtime, &batch, &request).unwrap();
    assert_eq!(state, AdmissionState::Accepted);
    assert!(slot.is_some());
}

#[test]
fn inference_api_submit_generation_reports_queued_when_batch_policy_enqueues() {
    let mut runtime = Runtime::builder().build().unwrap();
    let batch = runtime.create_continuous_batch(BatchingPolicy::default());
    let request = generation_request();

    let (state, slot) = submit_generation(&mut runtime, &batch, &request).unwrap();
    assert!(matches!(state, AdmissionState::Queued { .. }));
    assert!(slot.is_some());
}

#[test]
fn inference_api_submit_generation_reports_rejection_when_batch_policy_denies() {
    let mut runtime = Runtime::builder().build().unwrap();
    let policy = BatchingPolicy {
        max_active_operations: 0,
        ..BatchingPolicy::default()
    };
    let batch = runtime.create_continuous_batch(policy);
    let request = generation_request();

    let error = submit_generation(&mut runtime, &batch, &request).unwrap_err();
    assert!(matches!(
        error,
        InferenceApiError::GenerationRejected { .. }
    ));
}

#[test]
fn inference_api_submit_generation_observed_emits_generation_accepted() {
    let mut runtime = Runtime::builder().build().unwrap();
    let policy = BatchingPolicy {
        allow_queueing: false,
        ..BatchingPolicy::default()
    };
    let batch = runtime.create_continuous_batch(policy);
    let request = generation_request();
    let mut observer = InferenceApiObserver::new();

    let (state, _) =
        submit_generation_observed(&mut runtime, &batch, &request, &mut observer).unwrap();
    assert_eq!(state, AdmissionState::Accepted);
    assert!(
        observer
            .observations()
            .iter()
            .any(|observation| observation.kind == InferenceApiObservationKind::GenerationAccepted)
    );
}

fn model_instance_definition() -> ModelInstanceDefinition {
    ModelInstanceDefinition {
        artifact: ModelArtifactId::new(
            ModelArtifactKind::ModelWeights,
            ModelName::new("qwen").unwrap(),
            ModelRevision::new("1").unwrap(),
            ModelDigest::sha256(b"weights"),
        ),
        architecture: ModelArchitectureImplementation {
            architecture: ModelArchitecture::new("qwen", "qwen2"),
            kind: ModelArchitectureImplementationKind::TestFixture,
            required_capabilities: Vec::new(),
        },
        residencies: BTreeSet::from([ModelResidencyId::new(1)]),
        tokenizer: None,
        placement: ModelInstancePlacement::new(ResourceAffinity::new(FallbackClass::Transparent)),
        policy: ModelInstancePolicy::default(),
        adapter_state: ModelInstanceAdapterState::default(),
        associated_sessions: BTreeSet::new(),
        usage: ModelInstanceUsage::default(),
        compute_dtype: None,
        mutation_version: 0,
        tenant: None,
        owner: None,
        resource_bindings: ModelInstanceResourceBindings::default(),
    }
}

#[test]
fn inference_api_model_instance_suspend_resume_drain_through_api_boundary() {
    let mut runtime = Runtime::builder().build().unwrap();
    let instance = runtime
        .model_instances_mut()
        .create(model_instance_definition())
        .unwrap();

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_eq!(status.lifecycle, ModelInstanceLifecycleState::Ready);
    assert!(!status.raw_provider_handle_available);
    assert!(!status.raw_device_handle_available);
    assert!(!status.raw_weights_available);

    suspend_model_instance(
        &mut runtime,
        &instance,
        ModelInstanceSuspensionReason::AdministrativePolicy,
    )
    .unwrap();
    resume_model_instance(&mut runtime, &instance).unwrap();
    drain_model_instance(&mut runtime, &instance).unwrap();

    let status = model_instance_status(&runtime, &instance).unwrap();
    assert_eq!(status.lifecycle, ModelInstanceLifecycleState::Draining);
}

#[test]
fn inference_api_model_instance_warmup_reports_lifecycle_conflict_when_already_ready() {
    let mut runtime = Runtime::builder().build().unwrap();
    let instance = runtime
        .model_instances_mut()
        .create(model_instance_definition())
        .unwrap();
    let plan = ModelInstanceWarmupPlan {
        policy: ModelInstanceWarmupPolicy::ValidateMetadataOnly,
        steps: Vec::new(),
    };
    let checks = ModelInstanceReadinessChecks {
        residency_available: true,
        provider_ready: true,
        device_ready: true,
        adapter_ready: true,
        memory_pressure: MemoryPressureLevel::Low,
        runtime_policy_allows: true,
        browser_supported: true,
    };

    let error = warm_model_instance(&mut runtime, &instance, &plan, &checks).unwrap_err();
    assert!(matches!(
        error,
        InferenceApiError::ModelInstanceUnavailable { .. }
    ));
}

#[test]
fn inference_api_create_model_instance_observed_emits_model_instance_selected() {
    let mut memory = MemoryManager::new(MemoryManagerConfig::default());
    let manifest = minimal_model_manifest();
    let trust = ModelTrustDecision::new(ModelTrustStatus::Trusted, "test fixture");
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(ModelArchitectureImplementation {
        architecture: manifest.architecture.clone(),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    });
    let core = ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    let loaded = load_model(
        &mut coordinator,
        &mut memory,
        ModelLoadingApiRequest::new(core),
        &manifest,
        &trust,
    )
    .unwrap();

    let mut runtime = Runtime::builder().build().unwrap();
    let architecture = ModelArchitectureImplementation {
        architecture: manifest.architecture.clone(),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    };
    let mut observer = InferenceApiObserver::new();
    let instance = create_model_instance_observed(
        &mut runtime,
        &loaded,
        architecture,
        ResourceAffinity::new(FallbackClass::Transparent),
        &mut observer,
    )
    .unwrap();

    assert_eq!(
        model_instance_status(&runtime, &instance).unwrap().artifact,
        manifest.id
    );
    assert!(
        observer
            .observations()
            .iter()
            .any(|observation| observation.kind
                == InferenceApiObservationKind::ModelInstanceSelected)
    );
}

fn minimal_model_manifest() -> ModelManifest {
    ModelManifest {
        schema_version: MODEL_ARTIFACT_SCHEMA_VERSION,
        id: ModelArtifactId::new(
            ModelArtifactKind::ModelWeights,
            ModelName::new("qwen").unwrap(),
            ModelRevision::new("1").unwrap(),
            ModelDigest::sha256(b"weights"),
        ),
        architecture: ModelArchitecture::new("qwen", "qwen2"),
        parts: BTreeMap::new(),
        storage_dtype: None,
        compute_dtype: None,
        supported_compute_dtypes: BTreeSet::new(),
        tensors: Vec::new(),
        tokenizer: None,
        tokenizer_config: None,
        chat_template: None,
        prompt_template: None,
        generation: None,
        quantization: None,
        shards: Vec::new(),
        runtime_features: BTreeSet::new(),
        memory_features: BTreeSet::new(),
        provider_capabilities: Vec::new(),
        component: None,
        license: None,
        provenance: None,
        signatures: Vec::new(),
        source: None,
    }
}

#[test]
fn inference_api_load_model_wires_coordinator_and_memory_manager() {
    let manifest = minimal_model_manifest();
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(ModelArchitectureImplementation {
        architecture: manifest.architecture.clone(),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    });
    let trust = ModelTrustDecision::new(ModelTrustStatus::Trusted, "test fixture");
    let mut memory = MemoryManager::new(MemoryManagerConfig::default());

    let core = ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    let mut request = ModelLoadingApiRequest::new(core);
    request.tokenizer_reference = Some(TokenizerId::new("fixture").unwrap());
    request.adapter_references = vec![AdapterArtifactId {
        name: AdapterName::new("lora-a").unwrap(),
        revision: AdapterRevision::new("1").unwrap(),
        digest: AdapterDigest {
            algorithm: "sha256".into(),
            value: "deadbeef".into(),
        },
    }];
    request.layout_policy = Some(TensorLayoutKind::Contiguous);
    request.provider_preferences = vec![ProviderBinding::new("reference-cpu")];

    let loaded = load_model(&mut coordinator, &mut memory, request, &manifest, &trust).unwrap();
    assert_eq!(loaded.artifact, manifest.id);
    assert_eq!(loaded.state, ModelLoadingState::Ready);
}

#[test]
fn inference_api_load_model_observed_emits_loading_lifecycle_observations() {
    let manifest = minimal_model_manifest();
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(ModelArchitectureImplementation {
        architecture: manifest.architecture.clone(),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    });
    let trust = ModelTrustDecision::new(ModelTrustStatus::Trusted, "test fixture");
    let mut memory = MemoryManager::new(MemoryManagerConfig::default());
    let core = ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    let mut observer = InferenceApiObserver::new();

    load_model_observed(
        &mut coordinator,
        &mut memory,
        ModelLoadingApiRequest::new(core),
        &manifest,
        &trust,
        &mut observer,
    )
    .unwrap();

    let kinds: Vec<_> = observer
        .observations()
        .iter()
        .map(|observation| observation.kind)
        .collect();
    assert!(kinds.contains(&InferenceApiObservationKind::ModelLoadingRequested));
    assert!(kinds.contains(&InferenceApiObservationKind::ModelLoaded));
}

#[test]
fn inference_api_model_resolution_observed_emits_model_resolved_and_failed() {
    let reference = ModelRef::new("qwen-test").unwrap();
    let artifact = ModelArtifactId::new(
        ModelArtifactKind::ModelWeights,
        ModelName::new("qwen").unwrap(),
        ModelRevision::new("1").unwrap(),
        ModelDigest::sha256(b"weights"),
    );
    let mut registry = ModelRegistry::new();
    registry.register(reference.clone(), artifact);
    let mut observer = InferenceApiObserver::new();

    registry
        .resolve_observed(&ModelResolutionRequest::new(reference), &mut observer)
        .unwrap();
    registry
        .resolve_observed(
            &ModelResolutionRequest::new(ModelRef::new("unknown").unwrap()),
            &mut observer,
        )
        .unwrap_err();

    let kinds: Vec<_> = observer
        .observations()
        .iter()
        .map(|observation| observation.kind)
        .collect();
    assert!(kinds.contains(&InferenceApiObservationKind::ModelResolved));
    assert!(kinds.contains(&InferenceApiObservationKind::ModelResolutionFailed));
}

#[test]
fn inference_api_session_lifecycle_observed_emits_created_and_closed() {
    let mut runtime = Runtime::builder().build().unwrap();
    let mut request = session_creation_request();
    request.allowed_capabilities.insert("generation".into());
    let mut observer = InferenceApiObserver::new();

    let session = create_inference_session_observed(&mut runtime, request, &mut observer).unwrap();
    close_inference_session_observed(&mut runtime, &session, &mut observer).unwrap();

    let kinds: Vec<_> = observer
        .observations()
        .iter()
        .map(|observation| observation.kind)
        .collect();
    assert!(kinds.contains(&InferenceApiObservationKind::SessionCreated));
    assert!(kinds.contains(&InferenceApiObservationKind::SessionClosed));
}

#[test]
fn inference_api_session_close_transitions_lifecycle_to_closed() {
    let mut runtime = Runtime::builder().build().unwrap();
    let mut request = session_creation_request();
    request.allowed_capabilities.insert("generation".into());
    let session = create_inference_session(&mut runtime, request).unwrap();

    close_inference_session(&mut runtime, &session).unwrap();

    let status = session_status(
        &runtime,
        &session,
        &SessionAccessPolicy::authorize(session.clone()),
    )
    .unwrap();
    assert_eq!(status.lifecycle, SessionLifecycleState::Closed);
}

#[test]
fn inference_api_tokenize_prompt_input_observed_emits_tokenized_and_failed() {
    let tokenizer = FixtureTokenizer::new(generation_tokenizer_metadata());
    let mut observer = InferenceApiObserver::new();

    tokenize_prompt_input_observed(
        &tokenizer,
        TokenizationRequest::new(PromptInput::PlainText("secret".into())),
        None,
        &mut observer,
    )
    .unwrap();
    tokenize_prompt_input_observed(
        &tokenizer,
        TokenizationRequest::new(PromptInput::TestTokenSequence(vec![99_999])),
        None,
        &mut observer,
    )
    .unwrap_err();

    let kinds: Vec<_> = observer
        .observations()
        .iter()
        .map(|observation| observation.kind)
        .collect();
    assert!(kinds.contains(&InferenceApiObservationKind::PromptTokenized));
    assert!(kinds.contains(&InferenceApiObservationKind::TokenizationFailed));
}

#[test]
fn inference_api_one_shot_pipeline_uses_session_tokenizer_and_generation_contracts() {
    let mut runtime = Runtime::builder().build().unwrap();
    let mut request = session_creation_request();
    request.allowed_capabilities.insert("generation".into());
    let session = create_one_shot_session(&mut runtime, request).unwrap();

    let tokenizer = FixtureTokenizer::new(generation_tokenizer_metadata());
    let tokenized = tokenize_prompt_input(
        &tokenizer,
        TokenizationRequest::new(PromptInput::PlainText("hi".into())),
        None,
    )
    .unwrap();

    let metadata = generation_tokenizer_metadata();
    let vocabulary_size = metadata.vocabulary_size as usize;
    let generation_request = build_generation_request(
        GenerationRequestId::new("one-shot-1").unwrap(),
        Some(session.clone()),
        GenerationModelReference::LoadedModelContext("model-context".into()),
        GenerationTokenizerReference {
            tokenizer_id: metadata.id.clone(),
            metadata,
        },
        tokenized,
        2,
        GenerationParameters::greedy(),
        StopConditions::default(),
        StreamingMode::TokenIds,
    );
    let prepared = prepare_generation(&runtime, generation_request).unwrap();

    let policy = BatchingPolicy {
        allow_queueing: false,
        ..BatchingPolicy::default()
    };
    let batch = runtime.create_continuous_batch(policy);
    let (state, _) = submit_generation(&mut runtime, &batch, &prepared).unwrap();
    assert_eq!(state, AdmissionState::Accepted);

    // One-shot inference SHALL not bypass Model Instance, Tokenizer,
    // Generation, Sampling, Memory Manager, or Provider/Kernel contracts:
    // this drives the same Generation Contract loop (prefill, per-token
    // Sampling Contract decode, Provider/Kernel readiness gating) that any
    // session-bound generation would use.
    let mut observer = InferenceApiObserver::new();
    let result = run_generation_loop(
        &runtime,
        &prepared,
        true,
        true,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated| vec![0.0f32; vocabulary_size],
        |_generated| false,
        &mut observer,
    )
    .unwrap();
    assert_eq!(result.output.finish_reason, FinishReason::MaxNewTokens);
    assert!(
        observer
            .observations()
            .iter()
            .any(|observation| observation.kind == InferenceApiObservationKind::TokenGenerated)
    );

    close_inference_session(&mut runtime, &session).unwrap();
}

fn adapter_residency_fixture() -> AdapterResidency {
    AdapterResidency {
        id: AdapterResidencyId::new("residency-1").unwrap(),
        artifact: AdapterArtifactId {
            name: AdapterName::new("lora-a").unwrap(),
            revision: AdapterRevision::new("1").unwrap(),
            digest: AdapterDigest {
                algorithm: "sha256".into(),
                value: "deadbeef".into(),
            },
        },
        lifecycle: AdapterLifecycleState::Ready,
        location: AdapterResidencyLocation::Host,
        affinity: None,
        memory_allocation: None,
        provider_resource: None,
    }
}

fn adapter_activation_request_fixture(residency: &AdapterResidency) -> AdapterActivationRequest {
    AdapterActivationRequest {
        residency: residency.id.clone(),
        scope: AdapterActivationScope::Session(InferenceSessionId::new("session-1").unwrap()),
        base_model: GenerationModelReference::LoadedModelContext("model-context".into()),
        adapter_set: AdapterSetId::from_adapters([residency.artifact.clone()]),
        policy: AdapterCompositionPolicy::SingleAdapterOnly,
    }
}

#[test]
fn inference_api_adapter_activation_succeeds_for_ready_residency() {
    let residency = adapter_residency_fixture();
    let request = adapter_activation_request_fixture(&residency);

    activate_adapter(&residency, &request, None, None).unwrap();
}

#[test]
fn inference_api_adapter_activation_rejects_forbidden_operation_scope() {
    let residency = adapter_residency_fixture();
    let mut request = adapter_activation_request_fixture(&residency);
    request.scope = AdapterActivationScope::Operation("shell".into());

    let error = activate_adapter(&residency, &request, None, None).unwrap_err();
    assert!(matches!(error, InferenceApiError::PolicyDenied { .. }));
}

#[test]
fn inference_api_adapter_activation_denied_when_incompatible() {
    let residency = adapter_residency_fixture();
    let mut request = adapter_activation_request_fixture(&residency);
    request.residency = AdapterResidencyId::new("other-residency").unwrap();

    let error = activate_adapter(&residency, &request, None, None).unwrap_err();
    assert!(matches!(
        error,
        InferenceApiError::AdapterActivationFailed { .. }
    ));
}

#[test]
fn inference_api_adapter_activation_observed_emits_adapter_activated() {
    let residency = adapter_residency_fixture();
    let request = adapter_activation_request_fixture(&residency);
    let mut observer = InferenceApiObserver::new();

    activate_adapter_observed(&residency, &request, None, None, &mut observer).unwrap();

    assert!(
        observer
            .observations()
            .iter()
            .any(|observation| observation.kind == InferenceApiObservationKind::AdapterActivated)
    );
}

#[test]
fn inference_api_kv_cache_policy_covers_enabled_scope_budget_reuse_eviction_privacy() {
    let policy = KvCachePolicy {
        enabled: true,
        max_cache_tokens: Some(2048),
        max_cache_memory_bytes: Some(1 << 20),
        sharing: KvCacheSharingPolicy::AllowWithinSession,
        retention: KvCacheRetentionPolicy::RetainForPrefixReuse,
        prefix_reuse_allowed: true,
        privacy_redaction_required: true,
    };
    assert!(policy.enabled);
    assert!(policy.prefix_reuse_allowed);
    assert!(policy.privacy_redaction_required);
    assert_eq!(policy.sharing, KvCacheSharingPolicy::AllowWithinSession);
}

#[test]
fn inference_api_prefix_cache_policy_covers_scope_sharing_ttl_budget_privacy_reuse() {
    let policy = PrefixCachePolicy {
        enabled: true,
        allow_partial_reuse: true,
        require_sealed_kv_cache_for_sharing: true,
        sharing: PrefixCacheSharingPolicy::SessionLocal,
        privacy: PrefixCachePrivacyPolicy::default(),
        max_memory_bytes: Some(1 << 20),
        max_prefix_tokens: Some(1024),
        ttl_millis: Some(60_000),
        idle_ttl_millis: Some(5_000),
        persist_after_session_close: false,
    };
    assert!(policy.enabled);
    assert!(policy.allow_partial_reuse);
    assert_eq!(policy.sharing, PrefixCacheSharingPolicy::SessionLocal);
    assert_eq!(policy.ttl_millis, Some(60_000));
}

#[test]
fn inference_api_runtime_diagnostics_with_inputs_includes_caller_supplied_status() {
    let runtime = Runtime::builder().build().unwrap();
    let inputs = RuntimeDiagnosticsInputs {
        model_resolution_status: Some(ModelResolutionStatus::Resolved),
        model_loading_status: Some(ModelLoadingPhase::PublishModelContext),
        operator_missing_count: 2,
        tokenizer_compatible: Some(true),
        queued_admission_count: 3,
    };

    let diagnostics = runtime_diagnostics_with(&runtime, inputs);
    assert_eq!(
        diagnostics.model_resolution_status,
        Some(ModelResolutionStatus::Resolved)
    );
    assert_eq!(
        diagnostics.model_loading_status,
        Some(ModelLoadingPhase::PublishModelContext)
    );
    assert_eq!(diagnostics.operator_missing_count, 2);
    assert_eq!(diagnostics.tokenizer_compatible, Some(true));
    assert_eq!(diagnostics.queued_admission_count, 3);
    assert!(diagnostics.redacted);
}

#[test]
fn inference_api_generation_result_wraps_output_with_decoded_text_and_cache_usage() {
    let request = generation_request();
    let output = GenerationOutput::new(&request, vec![10, 11, 12], FinishReason::EosToken);

    let result = GenerationResult::new(output)
        .with_decoded_text("hello".into())
        .with_model_instance(ModelInstanceId::new("instance-1").unwrap())
        .with_cache_usage(CacheUsageSummary {
            kv_cache_hit: Some(true),
            prefix_cache_hit: Some(false),
        });

    assert_eq!(result.decoded_text.as_deref(), Some("hello"));
    assert!(result.model_instance.is_some());
    assert_eq!(result.cache_usage.kv_cache_hit, Some(true));
    assert!(result.error.is_none());
    assert!(result.redacted);
}

#[test]
fn inference_api_generation_result_reports_error_for_failed_finish_reason() {
    let request = generation_request();
    let output = GenerationOutput::new(&request, Vec::new(), FinishReason::ProviderError);

    let result = GenerationResult::new(output);
    assert!(result.error.is_some());
}

#[test]
fn inference_api_cancellation_stage_before_dispatch_always_succeeds() {
    let token = CancellationToken::new(GenerationRequestId::new("gen-1").unwrap());
    for stage in [
        CancellationStage::Queued,
        CancellationStage::Tokenization,
        CancellationStage::Prefill,
        CancellationStage::Decode,
        CancellationStage::Sampling,
        CancellationStage::Batching,
        CancellationStage::GraphExecution,
        CancellationStage::KernelDispatch,
    ] {
        assert_eq!(
            request_cancellation_at_stage(&token, stage, false),
            CancellationOutcome::Cancelled
        );
    }
}

#[test]
fn inference_api_cancellation_stage_provider_execution_depends_on_support() {
    let token = CancellationToken::new(GenerationRequestId::new("gen-1").unwrap());
    assert_eq!(
        request_cancellation_at_stage(&token, CancellationStage::ProviderExecution, true),
        CancellationOutcome::Cancelled
    );
    assert!(matches!(
        request_cancellation_at_stage(&token, CancellationStage::ProviderExecution, false),
        CancellationOutcome::LimitationReported { .. }
    ));
}

#[test]
fn inference_api_cancellation_stage_observed_emits_generation_cancelled() {
    let token = CancellationToken::new(GenerationRequestId::new("gen-1").unwrap());
    let mut observer = InferenceApiObserver::new();

    request_cancellation_at_stage_observed(&token, CancellationStage::Decode, false, &mut observer);

    assert!(
        observer
            .observations()
            .iter()
            .any(|observation| observation.kind
                == InferenceApiObservationKind::GenerationCancelled)
    );
}

#[test]
fn inference_api_tachyon_and_cli_boundary_capabilities_are_inference_only() {
    for forbidden in ["git", "shell", "agent-orchestration", "secrets"] {
        assert!(validate_inference_scope(forbidden).is_err());
    }
    assert!(validate_inference_scope("generation").is_ok());
}

#[test]
fn inference_api_streaming_handle_correlates_with_ordered_token_events() {
    let request = generation_request();
    let handle = StreamingHandle::for_request(&request);

    let events = token_stream_events(&request, &[10, 11, 12], None).unwrap();
    assert_eq!(events.len(), 3);
    assert!(
        events
            .iter()
            .all(|event| event.request_id == handle.request)
    );
    let indices: Vec<_> = events
        .iter()
        .filter_map(|event| event.token_index)
        .collect();
    assert_eq!(indices, vec![0, 1, 2]);
}

#[test]
fn inference_api_diagnostics_and_status_debug_output_never_exposes_raw_pointer_markers() {
    let runtime = Runtime::builder().build().unwrap();
    let diagnostics = runtime_diagnostics(&runtime);
    let instance = runtime
        .model_instances()
        .instances()
        .next()
        .map(ModelInstance::status);

    let debug_output = format!("{diagnostics:?} {instance:?}");
    assert!(!debug_output.contains("0x"));
    assert!(!debug_output.to_ascii_lowercase().contains("pointer"));
}

#[test]
fn inference_api_cancellation_reports_limitation_when_unsupported_after_dispatch() {
    let token = CancellationToken::new(GenerationRequestId::new("gen-1").unwrap());
    assert_eq!(
        request_cancellation(&token, true),
        CancellationOutcome::Cancelled
    );
    assert!(matches!(
        request_cancellation(&token, false),
        CancellationOutcome::LimitationReported { .. }
    ));
}

#[test]
fn inference_api_runtime_diagnostics_are_redacted_and_reflect_empty_runtime() {
    let runtime = Runtime::builder().build().unwrap();
    let diagnostics = runtime_diagnostics(&runtime);
    assert!(diagnostics.redacted);
    assert_eq!(diagnostics.model_instance_count, 0);
    assert_eq!(diagnostics.active_session_count, 0);
}

#[test]
fn inference_api_usage_report_never_carries_raw_prompt_text() {
    let usage = GenerationUsage::new(3, 4, FinishReason::EosToken);
    let memory = GenerationMemoryEstimate::default();

    let report = UsageReport::from_generation(&usage, &memory, Some(true), Some(12));
    assert_eq!(report.prompt_token_count, 3);
    assert_eq!(report.generated_token_count, 4);
    assert_eq!(report.cache_hit, Some(true));
    assert!(!report.cancelled);
}

#[test]
fn inference_api_browser_feature_check_only_rejects_on_wasm32() {
    let result = require_browser_supported("wasmtime");
    if cfg!(target_arch = "wasm32") {
        assert!(matches!(
            result,
            Err(InferenceApiError::BrowserFeatureUnsupported { .. })
        ));
    } else {
        assert!(result.is_ok());
    }
}

fn session_creation_request() -> SessionCreationRequest {
    let metadata = generation_tokenizer_metadata();
    SessionCreationRequest {
        model: GenerationModelReference::LoadedModelContext("model-context".into()),
        tokenizer: GenerationTokenizerReference {
            tokenizer_id: metadata.id.clone(),
            metadata,
        },
        generation_defaults: GenerationParameters::default(),
        policy: SessionPolicy::default(),
        memory: SessionMemoryBudget::default(),
        allowed_capabilities: BTreeSet::new(),
        correlation_id: Some(CorrelationId::new("corr-1")),
        created_at_millis: 0,
    }
}

#[test]
fn inference_api_model_resolution_source_placeholders_fail_structured() {
    let mut registry = ModelRegistry::new();
    let reference = ModelRef::new("qwen-test").unwrap();
    registry.register(
        reference.clone(),
        ModelArtifactId::new(
            ModelArtifactKind::ModelWeights,
            ModelName::new("qwen").unwrap(),
            ModelRevision::new("1").unwrap(),
            ModelDigest::sha256(b"weights"),
        ),
    );

    for source in [
        ModelResolutionSource::FutureExternalSource,
        ModelResolutionSource::FutureTachyonSource,
    ] {
        let mut request = ModelResolutionRequest::new(reference.clone());
        request.source = source;
        let error = registry.resolve(&request).unwrap_err();
        assert!(matches!(
            error,
            InferenceApiError::ModelResolutionFailed { .. }
        ));
    }
}

#[test]
fn inference_api_model_resolution_local_registry_source_still_resolves() {
    let reference = ModelRef::new("qwen-test").unwrap();
    let artifact = ModelArtifactId::new(
        ModelArtifactKind::ModelWeights,
        ModelName::new("qwen").unwrap(),
        ModelRevision::new("1").unwrap(),
        ModelDigest::sha256(b"weights"),
    );
    let mut registry = ModelRegistry::new();
    registry.register(reference.clone(), artifact.clone());

    let mut request = ModelResolutionRequest::new(reference);
    request.source = ModelResolutionSource::LocalRegistry;
    assert_eq!(registry.resolve(&request).unwrap().artifact, artifact);
}

#[test]
fn inference_api_streaming_decode_request_carries_state_across_calls() {
    let tokenizer = FixtureTokenizer::new(generation_tokenizer_metadata());
    let mut request = StreamingDecodeRequest::new(vec![2, 3]);
    request.skip_special_tokens = true;

    let output = decode_tokens_streaming(&tokenizer, request).unwrap();
    assert!(output.consumed_token_count > 0 || output.pending_partial_state.is_some());
}

#[test]
fn inference_api_validate_tokenizer_compatibility_accepts_matching_digest() {
    let metadata = generation_tokenizer_metadata();
    let tokenizer = FixtureTokenizer::new(metadata.clone());
    let compatibility = TokenizerCompatibility {
        expected_digest: Some(metadata.digest.clone()),
        expected_vocabulary_size: Some(metadata.vocabulary_size),
        expected_family: Some(metadata.family.clone()),
        expected_model_max_length: None,
        expected_added_tokens: None,
        expected_special_tokens: Vec::new(),
        expected_normalization: None,
    };

    validate_tokenizer_compatibility(&tokenizer, &compatibility).unwrap();
}

#[test]
fn inference_api_validate_tokenizer_compatibility_rejects_digest_mismatch() {
    let metadata = generation_tokenizer_metadata();
    let tokenizer = FixtureTokenizer::new(metadata);
    let compatibility = TokenizerCompatibility {
        expected_digest: Some(ModelDigest::sha256(b"a different tokenizer")),
        expected_vocabulary_size: None,
        expected_family: None,
        expected_model_max_length: None,
        expected_added_tokens: None,
        expected_special_tokens: Vec::new(),
        expected_normalization: None,
    };

    let error = validate_tokenizer_compatibility(&tokenizer, &compatibility).unwrap_err();
    assert!(matches!(
        error,
        InferenceApiError::TokenizerIncompatible { .. }
    ));
}

#[test]
fn inference_api_generation_api_request_carries_privacy_policy() {
    let request = GenerationApiRequest::new(
        generation_request(),
        SessionRedactionPolicy::RedactRawInputs,
    );
    assert_eq!(request.privacy, SessionRedactionPolicy::RedactRawInputs);
}

#[test]
fn inference_api_browser_inference_capabilities_reduced_excludes_kv_cache() {
    let capabilities = BrowserInferenceCapabilities::reduced();
    assert!(capabilities.tokenization);
    assert!(capabilities.generation);
    assert!(capabilities.streaming);
    assert!(!capabilities.kv_cache);
}

#[test]
fn inference_api_run_generation_loop_emits_full_streaming_lifecycle_and_completes() {
    let mut request = generation_request();
    request.parameters = GenerationParameters::greedy();
    request.stop_conditions = StopConditions::default();
    request.max_new_tokens = 2;
    let vocabulary_size = request.tokenizer.metadata.vocabulary_size as usize;

    let runtime = Runtime::builder().build().unwrap();
    let mut observer = InferenceApiObserver::new();
    let mut step = 0usize;

    let result = run_generation_loop(
        &runtime,
        &request,
        true,
        true,
        SamplingPolicy::default(),
        CacheUsageSummary {
            kv_cache_hit: Some(true),
            prefix_cache_hit: Some(false),
        },
        |_generated| {
            step += 1;
            let mut logits = vec![0.0f32; vocabulary_size];
            logits[(10 + step) % vocabulary_size] = 10.0;
            logits
        },
        |_generated| false,
        &mut observer,
    )
    .unwrap();

    assert_eq!(result.output.generated_token_count, 2);
    assert_eq!(result.output.finish_reason, FinishReason::MaxNewTokens);
    assert_eq!(result.cache_usage.kv_cache_hit, Some(true));

    let kinds: Vec<_> = observer
        .observations()
        .iter()
        .map(|observation| observation.kind)
        .collect();
    for expected in [
        InferenceApiObservationKind::GenerationStarted,
        InferenceApiObservationKind::StreamOpened,
        InferenceApiObservationKind::KvCacheUsed,
        InferenceApiObservationKind::PrefixCacheMiss,
        InferenceApiObservationKind::PrefillStarted,
        InferenceApiObservationKind::PrefillCompleted,
        InferenceApiObservationKind::DecodeStarted,
        InferenceApiObservationKind::TokenGenerated,
        InferenceApiObservationKind::GenerationCompleted,
        InferenceApiObservationKind::StreamClosed,
    ] {
        assert!(
            kinds.contains(&expected),
            "missing {expected:?} in {kinds:?}"
        );
    }
}

#[test]
fn inference_api_run_generation_loop_cancels_during_decode() {
    let mut request = generation_request();
    request.parameters = GenerationParameters::greedy();
    request.stop_conditions = StopConditions::default();
    request.max_new_tokens = 5;
    let vocabulary_size = request.tokenizer.metadata.vocabulary_size as usize;

    let runtime = Runtime::builder().build().unwrap();
    let mut observer = InferenceApiObserver::new();

    let result = run_generation_loop(
        &runtime,
        &request,
        true,
        true,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated| vec![0.0f32; vocabulary_size],
        |generated| !generated.is_empty(),
        &mut observer,
    )
    .unwrap();

    assert_eq!(result.output.finish_reason, FinishReason::Cancelled);
    assert_eq!(result.output.generated_token_count, 1);
    let kinds: Vec<_> = observer
        .observations()
        .iter()
        .map(|observation| observation.kind)
        .collect();
    assert!(kinds.contains(&InferenceApiObservationKind::GenerationCancelled));
    assert!(kinds.contains(&InferenceApiObservationKind::StreamInterrupted));
    assert!(!kinds.contains(&InferenceApiObservationKind::GenerationCompleted));
}

#[test]
fn inference_api_run_generation_loop_reports_provider_and_kernel_unavailable() {
    let mut request = generation_request();
    request.parameters = GenerationParameters::greedy();
    request.stop_conditions = StopConditions::default();
    let runtime = Runtime::builder().build().unwrap();

    let mut observer = InferenceApiObserver::new();
    let error = run_generation_loop(
        &runtime,
        &request,
        false,
        true,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated| Vec::new(),
        |_generated| false,
        &mut observer,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        InferenceApiError::ProviderUnavailable { .. }
    ));
    assert!(
        observer
            .observations()
            .iter()
            .any(|observation| observation.kind
                == InferenceApiObservationKind::ProviderUnavailable)
    );

    request.request_id = GenerationRequestId::new("gen-2").unwrap();
    let mut observer = InferenceApiObserver::new();
    let error = run_generation_loop(
        &runtime,
        &request,
        true,
        false,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated| Vec::new(),
        |_generated| false,
        &mut observer,
    )
    .unwrap_err();
    assert!(matches!(error, InferenceApiError::KernelUnavailable { .. }));
    assert!(
        observer
            .observations()
            .iter()
            .any(|observation| observation.kind == InferenceApiObservationKind::KernelUnavailable)
    );
}
