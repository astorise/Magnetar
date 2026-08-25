//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;

const UNIT_EXPORT_COMPONENT: &str =
    include_str!("../../fixtures/components/unit-export.component.wat");
const HOST_ROUNDTRIP_COMPONENT: &str =
    include_str!("../../fixtures/components/host-roundtrip.component.wat");
const HOST_FAILURE_COMPONENT: &str =
    include_str!("../../fixtures/components/host-failure.component.wat");
const COMPUTE_IMPORT_COMPONENT: &str =
    include_str!("../../fixtures/components/compute-import.component.wat");
const LOOP_COMPONENT: &str = include_str!("../../fixtures/components/loop.component.wat");
const TRAPPING_COMPONENT: &str = include_str!("../../fixtures/components/trapping.component.wat");
const U32_EXPORT_COMPONENT: &str =
    include_str!("../../fixtures/components/u32-export.component.wat");
const WASI_FILESYSTEM_COMPONENT: &str =
    include_str!("../../fixtures/components/wasi-filesystem.component.wat");
const WASI_ENVIRONMENT_COMPONENT: &str =
    include_str!("../../fixtures/components/wasi-environment.component.wat");
const RESOURCE_IMPORT_COMPONENT: &str =
    include_str!("../../fixtures/components/resource-import.component.wat");

#[test]
fn wasmtime_engine_reports_component_capabilities() {
    let engine = WasmtimeComponentEngine::new().unwrap();
    let capabilities = engine.capabilities();
    assert!(capabilities.component_model);
    assert!(capabilities.async_host_calls);
    assert!(capabilities.interruption);
    assert!(capabilities.resource_limits);
}

#[test]
fn wasmtime_engine_normalizes_missing_artifact_load_failure() {
    let mut engine = WasmtimeComponentEngine::new().unwrap();
    let definition = ComponentDefinition {
        id: ComponentDefinitionId::new(1),
        metadata: crate::ComponentMetadata::new("missing", "1", "missing component"),
        artifact_path: std::path::PathBuf::from("missing-component.wasm"),
        manifest_path: None,
        artifact_digest: None,
        trust_decision: None,
        state: crate::ComponentDefinitionState::Registered,
    };

    assert!(matches!(
        engine.prepare(&definition, &ComponentResourceLimits::default()),
        Err(ComponentError::ComponentLoadFailed { .. })
    ));
}

#[test]
fn wasmtime_engine_normalizes_invalid_component_bytes() {
    let directory =
        std::env::temp_dir().join(format!("magnetar-wasmtime-invalid-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let artifact = directory.join("invalid.component.wasm");
    std::fs::write(&artifact, b"not a component").unwrap();

    let mut engine = WasmtimeComponentEngine::new().unwrap();
    let definition = ComponentDefinition {
        id: ComponentDefinitionId::new(2),
        metadata: crate::ComponentMetadata::new("invalid", "1", "invalid component"),
        artifact_path: artifact,
        manifest_path: None,
        artifact_digest: None,
        trust_decision: None,
        state: crate::ComponentDefinitionState::Registered,
    };

    assert!(matches!(
        engine.prepare(&definition, &ComponentResourceLimits::default()),
        Err(ComponentError::PreparationFailed { .. })
    ));
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn wasmtime_engine_normalizes_malformed_wat_source() {
    let directory = std::env::temp_dir().join(format!(
        "magnetar-wasmtime-malformed-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let artifact = directory.join("malformed.component.wasm");
    std::fs::write(&artifact, "(component").unwrap();

    let mut engine = WasmtimeComponentEngine::new().unwrap();
    let definition = ComponentDefinition {
        id: ComponentDefinitionId::new(17),
        metadata: crate::ComponentMetadata::new("malformed", "1", "malformed component"),
        artifact_path: artifact,
        manifest_path: None,
        artifact_digest: None,
        trust_decision: None,
        state: crate::ComponentDefinitionState::Registered,
    };

    assert!(matches!(
        engine.prepare(&definition, &ComponentResourceLimits::default()),
        Err(ComponentError::PreparationFailed { .. })
    ));
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn wasmtime_engine_preserves_import_and_export_identity() {
    assert_eq!(
        wit_interface_from_component_name("magnetar:test/api@1.2.3"),
        WitInterface::new("magnetar:test/api", "1.2.3")
    );
    assert_eq!(
        wit_interface_from_component_name("magnetar:test/api"),
        WitInterface::new("magnetar:test/api", "")
    );
}

#[test]
fn wasmtime_engine_instantiates_component_without_imports() {
    let directory = std::env::temp_dir().join(format!("magnetar-wasmtime-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let artifact = directory.join("empty.component.wasm");
    std::fs::write(&artifact, "(component)").unwrap();

    let mut engine = WasmtimeComponentEngine::new().unwrap();
    let definition = ComponentDefinition {
        id: ComponentDefinitionId::new(7),
        metadata: crate::ComponentMetadata::new("empty", "1", "empty component"),
        artifact_path: artifact,
        manifest_path: None,
        artifact_digest: None,
        trust_decision: None,
        state: crate::ComponentDefinitionState::Registered,
    };
    let prepared = engine
        .prepare(&definition, &ComponentResourceLimits::default())
        .unwrap();
    let instance = engine
        .instantiate(&prepared, &ComponentLinkPlan::default())
        .unwrap();

    assert_eq!(instance.definition_id(), definition.id);
    assert!(instance.engine_key().starts_with("wasmtime-instance:7:"));
    engine.destroy(instance).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn wasmtime_engine_links_authorized_unit_host_import() {
    let directory =
        std::env::temp_dir().join(format!("magnetar-wasmtime-link-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let artifact = directory.join("import.component.wasm");
    std::fs::write(
        &artifact,
        r#"(component
            (import "example:test/host@1.0.0" (instance $host
                (export "ping" (func)))) )"#,
    )
    .unwrap();

    let mut engine = WasmtimeComponentEngine::new().unwrap();
    let interface = WitInterface::new("example:test/host", "1.0.0");
    let definition = ComponentDefinition {
        id: ComponentDefinitionId::new(10),
        metadata: crate::ComponentMetadata::new("importer", "1", "importing component")
            .with_import(interface.clone()),
        artifact_path: artifact,
        manifest_path: None,
        artifact_digest: None,
        trust_decision: None,
        state: crate::ComponentDefinitionState::Registered,
    };
    let prepared = engine
        .prepare(&definition, &ComponentResourceLimits::default())
        .unwrap();
    let mut link_plan = ComponentLinkPlan::default();
    link_plan.insert_for_test(crate::ComponentEndpoint::Capability { interface });

    let instance = engine.instantiate(&prepared, &link_plan).unwrap();
    engine.destroy(instance).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn wasmtime_engine_invokes_unit_export() {
    let directory =
        std::env::temp_dir().join(format!("magnetar-wasmtime-ok-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let artifact = directory.join("ok.component.wasm");
    std::fs::write(&artifact, UNIT_EXPORT_COMPONENT).unwrap();

    let mut engine = WasmtimeComponentEngine::new().unwrap();
    let interface = WitInterface::new("example:component/run", "1.0.0");
    let definition = ComponentDefinition {
        id: ComponentDefinitionId::new(11),
        metadata: crate::ComponentMetadata::new("ok", "1", "callable component")
            .with_export(interface.clone()),
        artifact_path: artifact,
        manifest_path: None,
        artifact_digest: None,
        trust_decision: None,
        state: crate::ComponentDefinitionState::Registered,
    };
    let prepared = engine
        .prepare(&definition, &ComponentResourceLimits::default())
        .unwrap();
    let instance = engine
        .instantiate(&prepared, &ComponentLinkPlan::default())
        .unwrap();

    engine
        .invoke(
            &instance,
            &ComponentInvocation::new(crate::ComponentInstanceId::new(100), interface, "run"),
        )
        .unwrap();
    engine.destroy(instance).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn wasmtime_engine_returns_primitive_value() {
    let directory =
        std::env::temp_dir().join(format!("magnetar-wasmtime-u32-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let artifact = directory.join("u32.component.wasm");
    std::fs::write(&artifact, U32_EXPORT_COMPONENT).unwrap();

    let mut engine = WasmtimeComponentEngine::new().unwrap();
    let interface = WitInterface::new("example:component/answer", "1.0.0");
    let definition = ComponentDefinition {
        id: ComponentDefinitionId::new(14),
        metadata: crate::ComponentMetadata::new("u32", "1", "u32 component")
            .with_export(interface.clone()),
        artifact_path: artifact,
        manifest_path: None,
        artifact_digest: None,
        trust_decision: None,
        state: crate::ComponentDefinitionState::Registered,
    };
    let prepared = engine
        .prepare(&definition, &ComponentResourceLimits::default())
        .unwrap();
    let instance = engine
        .instantiate(&prepared, &ComponentLinkPlan::default())
        .unwrap();

    let result = engine
        .invoke(
            &instance,
            &ComponentInvocation::new(crate::ComponentInstanceId::new(103), interface, "answer"),
        )
        .unwrap();
    assert_eq!(
        result,
        ComponentInvocationResult::single(ComponentValue::U32(42))
    );
    engine.destroy(instance).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn wasmtime_engine_normalizes_deadline_interruption() {
    let directory =
        std::env::temp_dir().join(format!("magnetar-wasmtime-deadline-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let artifact = directory.join("deadline.component.wasm");
    std::fs::write(&artifact, LOOP_COMPONENT).unwrap();

    let mut engine = WasmtimeComponentEngine::new().unwrap();
    let interface = WitInterface::new("example:component/run", "1.0.0");
    let definition = ComponentDefinition {
        id: ComponentDefinitionId::new(13),
        metadata: crate::ComponentMetadata::new("deadline", "1", "deadline component")
            .with_export(interface.clone()),
        artifact_path: artifact,
        manifest_path: None,
        artifact_digest: None,
        trust_decision: None,
        state: crate::ComponentDefinitionState::Registered,
    };
    let prepared = engine
        .prepare(&definition, &ComponentResourceLimits::default())
        .unwrap();
    let instance = engine
        .instantiate(&prepared, &ComponentLinkPlan::default())
        .unwrap();
    let mut invocation =
        ComponentInvocation::new(crate::ComponentInstanceId::new(102), interface, "run");
    invocation.deadline_millis = Some(0);

    let result = engine.invoke(&instance, &invocation);
    assert!(matches!(
        result,
        Err(ComponentError::Interrupted {
            reason: ComponentInterruptionReason::Deadline,
            ..
        })
    ));
    engine.destroy(instance).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn wasmtime_engine_interrupts_only_requested_instance() {
    let directory = std::env::temp_dir().join(format!(
        "magnetar-wasmtime-cancel-local-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let artifact = directory.join("cancel.component.wasm");
    std::fs::write(&artifact, UNIT_EXPORT_COMPONENT).unwrap();

    let mut engine = WasmtimeComponentEngine::new().unwrap();
    let interface = WitInterface::new("example:component/run", "1.0.0");
    let definition = ComponentDefinition {
        id: ComponentDefinitionId::new(18),
        metadata: crate::ComponentMetadata::new("cancel-local", "1", "cancel component")
            .with_export(interface.clone()),
        artifact_path: artifact,
        manifest_path: None,
        artifact_digest: None,
        trust_decision: None,
        state: crate::ComponentDefinitionState::Registered,
    };
    let prepared = engine
        .prepare(&definition, &ComponentResourceLimits::default())
        .unwrap();
    let first = engine
        .instantiate(&prepared, &ComponentLinkPlan::default())
        .unwrap();
    let second = engine
        .instantiate(&prepared, &ComponentLinkPlan::default())
        .unwrap();

    engine
        .interrupt(&first, ComponentInterruptionReason::CallerCancelled)
        .unwrap();
    let first_result = engine.invoke(
        &first,
        &ComponentInvocation::new(
            crate::ComponentInstanceId::new(105),
            interface.clone(),
            "run",
        ),
    );
    assert!(matches!(
        first_result,
        Err(ComponentError::Interrupted {
            reason: ComponentInterruptionReason::CallerCancelled,
            ..
        })
    ));
    engine
        .invoke(
            &second,
            &ComponentInvocation::new(crate::ComponentInstanceId::new(106), interface, "run"),
        )
        .unwrap();
    engine.destroy(first).unwrap();
    engine.destroy(second).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn wasmtime_engine_invokes_authorized_host_import_roundtrip() {
    let directory = std::env::temp_dir().join(format!(
        "magnetar-wasmtime-roundtrip-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let artifact = directory.join("roundtrip.component.wasm");
    std::fs::write(&artifact, HOST_ROUNDTRIP_COMPONENT).unwrap();

    let mut engine = WasmtimeComponentEngine::new().unwrap();
    let import = WitInterface::new("example:test/host", "1.0.0");
    let export = WitInterface::new("example:component/run", "1.0.0");
    let definition = ComponentDefinition {
        id: ComponentDefinitionId::new(12),
        metadata: crate::ComponentMetadata::new("roundtrip", "1", "roundtrip component")
            .with_import(import.clone())
            .with_export(export.clone()),
        artifact_path: artifact,
        manifest_path: None,
        artifact_digest: None,
        trust_decision: None,
        state: crate::ComponentDefinitionState::Registered,
    };
    let prepared = engine
        .prepare(&definition, &ComponentResourceLimits::default())
        .unwrap();
    let mut link_plan = ComponentLinkPlan::default();
    link_plan.insert_for_test(crate::ComponentEndpoint::Capability { interface: import });
    let instance = engine.instantiate(&prepared, &link_plan).unwrap();

    engine
        .invoke(
            &instance,
            &ComponentInvocation::new(crate::ComponentInstanceId::new(101), export, "run"),
        )
        .unwrap();
    assert_eq!(
        engine
            .instances
            .get(instance.engine_key())
            .map(|state| state._store.data().host_calls),
        Some(1)
    );
    engine.destroy(instance).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn wasmtime_engine_distinguishes_host_failure_from_component_trap() {
    let directory = std::env::temp_dir().join(format!(
        "magnetar-wasmtime-host-failure-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let artifact = directory.join("host-failure.component.wasm");
    std::fs::write(&artifact, HOST_FAILURE_COMPONENT).unwrap();

    let mut engine = WasmtimeComponentEngine::new().unwrap();
    let import = WitInterface::new("example:test/host", "1.0.0");
    let export = WitInterface::new("example:component/run", "1.0.0");
    let definition = ComponentDefinition {
        id: ComponentDefinitionId::new(19),
        metadata: crate::ComponentMetadata::new("host-failure", "1", "host failure")
            .with_import(import.clone())
            .with_export(export.clone()),
        artifact_path: artifact,
        manifest_path: None,
        artifact_digest: None,
        trust_decision: None,
        state: crate::ComponentDefinitionState::Registered,
    };
    let prepared = engine
        .prepare(&definition, &ComponentResourceLimits::default())
        .unwrap();
    let mut link_plan = ComponentLinkPlan::default();
    link_plan.insert_for_test(crate::ComponentEndpoint::Capability { interface: import });
    let instance = engine.instantiate(&prepared, &link_plan).unwrap();

    assert!(matches!(
        engine.invoke(
            &instance,
            &ComponentInvocation::new(crate::ComponentInstanceId::new(107), export, "run"),
        ),
        Err(ComponentError::InvocationFailed { message, .. })
            if message == "[redacted host adapter error]"
    ));
    engine.destroy(instance).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn wasmtime_engine_keeps_host_state_instance_local() {
    let directory = std::env::temp_dir().join(format!(
        "magnetar-wasmtime-isolation-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let artifact = directory.join("isolation.component.wasm");
    std::fs::write(&artifact, HOST_ROUNDTRIP_COMPONENT).unwrap();

    let mut engine = WasmtimeComponentEngine::new().unwrap();
    let import = WitInterface::new("example:test/host", "1.0.0");
    let export = WitInterface::new("example:component/run", "1.0.0");
    let definition = ComponentDefinition {
        id: ComponentDefinitionId::new(16),
        metadata: crate::ComponentMetadata::new("isolation", "1", "isolation component")
            .with_import(import.clone())
            .with_export(export.clone()),
        artifact_path: artifact,
        manifest_path: None,
        artifact_digest: None,
        trust_decision: None,
        state: crate::ComponentDefinitionState::Registered,
    };
    let prepared = engine
        .prepare(&definition, &ComponentResourceLimits::default())
        .unwrap();
    let mut link_plan = ComponentLinkPlan::default();
    link_plan.insert_for_test(crate::ComponentEndpoint::Capability { interface: import });
    let first = engine.instantiate(&prepared, &link_plan).unwrap();
    let second = engine.instantiate(&prepared, &link_plan).unwrap();

    engine
        .invoke(
            &first,
            &ComponentInvocation::new(crate::ComponentInstanceId::new(104), export, "run"),
        )
        .unwrap();
    assert_eq!(
        engine
            .instances
            .get(first.engine_key())
            .map(|state| state._store.data().host_calls),
        Some(1)
    );
    assert_eq!(
        engine
            .instances
            .get(second.engine_key())
            .map(|state| state._store.data().host_calls),
        Some(0)
    );
    engine.destroy(first).unwrap();
    engine.destroy(second).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn wasmtime_engine_links_compute_import_without_provider_resolution() {
    let directory =
        std::env::temp_dir().join(format!("magnetar-wasmtime-compute-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let artifact = directory.join("compute.component.wasm");
    std::fs::write(&artifact, COMPUTE_IMPORT_COMPONENT).unwrap();

    let mut engine = WasmtimeComponentEngine::new().unwrap();
    let interface = WitInterface::new("magnetar:compute/run", "2.0.0");
    let definition = ComponentDefinition {
        id: ComponentDefinitionId::new(15),
        metadata: crate::ComponentMetadata::new("compute-import", "1", "compute import")
            .with_import(interface.clone()),
        artifact_path: artifact,
        manifest_path: None,
        artifact_digest: None,
        trust_decision: None,
        state: crate::ComponentDefinitionState::Registered,
    };
    let prepared = engine
        .prepare(&definition, &ComponentResourceLimits::default())
        .unwrap();
    let mut link_plan = ComponentLinkPlan::default();
    link_plan.insert_for_test(crate::ComponentEndpoint::Capability { interface });

    let instance = engine.instantiate(&prepared, &link_plan).unwrap();
    assert_eq!(
        engine
            .instances
            .get(instance.engine_key())
            .map(|state| state._store.data().host_calls),
        Some(0)
    );
    engine.destroy(instance).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn wasmtime_engine_rejects_unauthorized_wasi_fixtures_without_link_plan() {
    for (index, (fixture, interface)) in [
        (
            WASI_FILESYSTEM_COMPONENT,
            WitInterface::new("wasi:filesystem/types", "0.2.0"),
        ),
        (
            WASI_ENVIRONMENT_COMPONENT,
            WitInterface::new("wasi:cli/environment", "0.2.0"),
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let directory = std::env::temp_dir().join(format!(
            "magnetar-wasmtime-wasi-{index}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let artifact = directory.join("wasi.component.wasm");
        std::fs::write(&artifact, fixture).unwrap();

        let mut engine = WasmtimeComponentEngine::new().unwrap();
        let definition = ComponentDefinition {
            id: ComponentDefinitionId::new(20 + index as u64),
            metadata: crate::ComponentMetadata::new(format!("wasi-{index}"), "1", "wasi component")
                .with_import(interface),
            artifact_path: artifact,
            manifest_path: None,
            artifact_digest: None,
            trust_decision: None,
            state: crate::ComponentDefinitionState::Registered,
        };
        let prepared = engine
            .prepare(&definition, &ComponentResourceLimits::default())
            .unwrap();

        assert!(matches!(
            engine.instantiate(&prepared, &ComponentLinkPlan::default()),
            Err(ComponentError::InstantiationFailed { .. })
        ));
        std::fs::remove_dir_all(directory).unwrap();
    }
}

#[test]
fn wasmtime_engine_rejects_resource_imports_without_runtime_resource_mapping() {
    let directory =
        std::env::temp_dir().join(format!("magnetar-wasmtime-resource-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let artifact = directory.join("resource.component.wasm");
    std::fs::write(&artifact, RESOURCE_IMPORT_COMPONENT).unwrap();

    let mut engine = WasmtimeComponentEngine::new().unwrap();
    let interface = WitInterface::new("example:test/resources", "1.0.0");
    let definition = ComponentDefinition {
        id: ComponentDefinitionId::new(22),
        metadata: crate::ComponentMetadata::new("resource-import", "1", "resource import")
            .with_import(interface.clone()),
        artifact_path: artifact,
        manifest_path: None,
        artifact_digest: None,
        trust_decision: None,
        state: crate::ComponentDefinitionState::Registered,
    };
    let prepared = engine
        .prepare(&definition, &ComponentResourceLimits::default())
        .unwrap();
    let mut link_plan = ComponentLinkPlan::default();
    link_plan.insert_for_test(crate::ComponentEndpoint::Capability { interface });

    assert!(matches!(
        engine.instantiate(&prepared, &link_plan),
        Err(ComponentError::InstantiationFailed { message, .. })
            if message.contains("unsupported host import item")
    ));
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn wasmtime_engine_applies_memory_limit_to_store() {
    let directory =
        std::env::temp_dir().join(format!("magnetar-wasmtime-limit-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let artifact = directory.join("memory.component.wasm");
    std::fs::write(
        &artifact,
        r#"
        (component
            (core module $m
                (memory 1))
            (core instance $i (instantiate $m))
        )
        "#,
    )
    .unwrap();

    let mut engine = WasmtimeComponentEngine::new().unwrap();
    let definition = ComponentDefinition {
        id: ComponentDefinitionId::new(9),
        metadata: crate::ComponentMetadata::new("memory", "1", "memory component"),
        artifact_path: artifact,
        manifest_path: None,
        artifact_digest: None,
        trust_decision: None,
        state: crate::ComponentDefinitionState::Registered,
    };
    let prepared = engine
        .prepare(
            &definition,
            &ComponentResourceLimits {
                require_memory_limit: true,
                max_memory_bytes: Some(0),
                ..ComponentResourceLimits::default()
            },
        )
        .unwrap();

    assert!(matches!(
        engine.instantiate(&prepared, &ComponentLinkPlan::default()),
        Err(ComponentError::InstantiationFailed { .. })
    ));
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn wasmtime_engine_normalizes_export_trap() {
    let directory =
        std::env::temp_dir().join(format!("magnetar-wasmtime-call-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let artifact = directory.join("call.component.wasm");
    std::fs::write(&artifact, TRAPPING_COMPONENT).unwrap();

    let mut engine = WasmtimeComponentEngine::new().unwrap();
    let interface = WitInterface::new("example:component/run", "1.0.0");
    let definition = ComponentDefinition {
        id: ComponentDefinitionId::new(8),
        metadata: crate::ComponentMetadata::new("callable", "1", "callable component")
            .with_export(interface.clone()),
        artifact_path: artifact,
        manifest_path: None,
        artifact_digest: None,
        trust_decision: None,
        state: crate::ComponentDefinitionState::Registered,
    };
    let prepared = engine
        .prepare(&definition, &ComponentResourceLimits::default())
        .unwrap();
    let instance = engine
        .instantiate(&prepared, &ComponentLinkPlan::default())
        .unwrap();

    assert!(matches!(
        engine.invoke(
            &instance,
            &ComponentInvocation::new(crate::ComponentInstanceId::new(99), interface, "run"),
        ),
        Err(ComponentError::Trap {
            instance,
            kind: ComponentTrapKind::Trap,
            ..
        }) if instance.get() == 99
    ));
    engine.destroy(instance).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}
