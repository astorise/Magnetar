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
const BOUNDED_LOOP_COMPONENT: &str =
    include_str!("../../fixtures/components/bounded-loop.component.wat");
const QWEN_GRAPH_COMPONENT: &str =
    include_str!("../../fixtures/components/qwen-graph.component.wat");
const CAPABILITY_ECHO_COMPONENT: &str =
    include_str!("../../fixtures/components/capability-echo.component.wat");
/// The first real, minimal Qwen Model Component (task 11.4,
/// `model-component-graph-contract`): a compiled `magnetar-component-Qwen`
/// (the `components/qwen` submodule) Component binary, built via
/// `cargo build --target wasm32-unknown-unknown --release` then
/// `wasm-tools component new`. A pre-built binary fixture, not `.wat` text
/// like this file's other fixtures, because its real graph-building logic
/// (calling `graph-builder` through `wit-bindgen`-generated bindings) is
/// not reasonably hand-writable as WAT. Regenerate from
/// `components/qwen`'s source with the same two commands if that source
/// changes.
const QWEN_REAL_COMPONENT: &[u8] =
    include_bytes!("../../fixtures/components/qwen-real.component.wasm");

/// Test-only [`HostCapability`] (`model-component-graph-contract`): answers
/// `echo(x: u32) -> u32` with `x + 1`, real enough to prove the dynamic
/// argument/result marshaling actually round-trips a value through a real
/// Component's own core-wasm call, not just that linking succeeds.
struct AddOneCapability;

impl HostCapability for AddOneCapability {
    fn call(
        &self,
        instance_key: &str,
        operation: &str,
        arguments: &[ComponentValue],
    ) -> Result<Vec<ComponentValue>, ComponentError> {
        assert!(!instance_key.is_empty());
        assert_eq!(operation, "echo");
        let [ComponentValue::U32(x)] = arguments else {
            panic!("expected exactly one u32 argument, got {arguments:?}");
        };
        Ok(vec![ComponentValue::U32(x + 1)])
    }
}

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
fn wasmtime_engine_dispatches_registered_capability_with_real_arguments() {
    let directory = std::env::temp_dir().join(format!(
        "magnetar-wasmtime-capability-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let artifact = directory.join("capability-echo.component.wasm");
    std::fs::write(&artifact, CAPABILITY_ECHO_COMPONENT).unwrap();

    let mut engine = WasmtimeComponentEngine::new().unwrap();
    let interface = WitInterface::new("example:test/capability", "1.0.0");
    engine.register_capability(interface.clone(), Arc::new(AddOneCapability));
    let export_interface = WitInterface::new("example:test/capability-echo", "1.0.0");
    let definition = ComponentDefinition {
        id: ComponentDefinitionId::new(19),
        metadata: crate::ComponentMetadata::new(
            "capability-echo",
            "1",
            "capability echo component",
        )
        .with_import(interface.clone())
        .with_export(export_interface.clone()),
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
    let result = engine
        .invoke(
            &instance,
            &ComponentInvocation::new(
                crate::ComponentInstanceId::new(190),
                export_interface,
                "run",
            ),
        )
        .unwrap();
    // The Component calls the registered capability with a fixed `41` and
    // returns whatever it got back (`AddOneCapability` answers `x + 1`) --
    // `42` here proves the value actually crossed the real host-import
    // boundary and back, not merely that linking succeeded.
    assert_eq!(
        result,
        ComponentInvocationResult::single(ComponentValue::U32(42))
    );
    engine.destroy(instance).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn wasmtime_engine_builds_a_real_qwen_prefill_graph_through_the_graph_builder_capability() {
    let directory = std::env::temp_dir().join(format!(
        "magnetar-wasmtime-qwen-real-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let artifact = directory.join("qwen-real.component.wasm");
    std::fs::write(&artifact, QWEN_REAL_COMPONENT).unwrap();

    let capability = Arc::new(crate::GraphBuilderCapability::new());
    let mut engine = WasmtimeComponentEngine::new().unwrap();
    let graph_builder_interface =
        WitInterface::new("magnetar:model-component-graph/graph-builder", "1.0.0");
    engine.register_capability(
        graph_builder_interface.clone(),
        capability.clone() as Arc<dyn HostCapability>,
    );

    let export_interface = WitInterface::new(
        "magnetar:model-component-graph/model-component-graph-producer",
        "1.0.0",
    );
    let definition = ComponentDefinition {
        id: ComponentDefinitionId::new(200),
        metadata: crate::ComponentMetadata::new(
            "qwen-real",
            "1",
            "the first real, minimal Qwen Model Component",
        )
        .with_import(graph_builder_interface.clone())
        .with_export(export_interface.clone()),
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
    link_plan.insert_for_test(crate::ComponentEndpoint::Capability {
        interface: graph_builder_interface,
    });
    let instance = engine.instantiate(&prepared, &link_plan).unwrap();

    let mut weight_shapes = BTreeMap::new();
    for (name, dims) in [
        ("layers.0.input_norm", vec![4]),
        ("layers.0.self_attn.q_proj", vec![4, 4]),
        ("layers.0.self_attn.k_proj", vec![4, 4]),
        ("layers.0.self_attn.v_proj", vec![4, 4]),
        ("layers.0.self_attn.o_proj", vec![4, 4]),
        ("layers.0.post_attn_norm", vec![4]),
        ("layers.0.mlp.gate_proj", vec![4, 8]),
        ("layers.0.mlp.up_proj", vec![4, 8]),
        ("layers.0.mlp.down_proj", vec![8, 4]),
        ("token_embedding", vec![258, 4]),
        ("final_norm", vec![4]),
        ("lm_head", vec![4, 258]),
    ] {
        weight_shapes.insert(name.to_string(), dims);
    }
    capability.prepare_session(
        instance.engine_key(),
        crate::graph_builder_capability::SessionContext {
            component_id: "qwen-real".to_string(),
            compatibility_key: "qwen-real-fixture".to_string(),
            kv_namespace: "qwen".to_string(),
            weight_shapes,
            output_edge_name: "logits".to_string(),
        },
    );

    let result = engine
        .invoke(
            &instance,
            &ComponentInvocation::new(
                crate::ComponentInstanceId::new(200),
                export_interface,
                "build-prefill-graph",
            )
            .with_arguments(vec![ComponentValue::S64(4)]),
        )
        .unwrap();
    let [ComponentValue::String(handle)] = result.values.as_slice() else {
        panic!("expected a single string handle, got {:?}", result.values);
    };

    let graph = capability
        .take_graph(instance.engine_key(), handle)
        .expect("the finished graph is retrievable by its handle");
    // embedding(1) + 16 nodes per layer (matching qwen_build_graph exactly)
    // for the fixture's one layer + final_norm(1) + lm_head(1) -- the same
    // 19-node count `wasmtime_engine_executes_qwen_graph_component_fixture`
    // already asserts for this architecture via the Rust-synthesized path,
    // now produced through the WIT graph-builder contract instead.
    assert_eq!(graph.nodes.len(), 19);
    assert!(
        graph
            .edges
            .contains_key(&crate::TensorEdgeId::new("logits"))
    );
    assert_eq!(graph.phase, crate::ExecutionGraphPhase::Prefill);
    assert!(matches!(
        &graph.producer,
        crate::ExecutionGraphProducer::ModelComponent { component_id } if component_id == "qwen-real"
    ));
    // Tied embeddings: the lm_head weight edge aliases token_embedding.
    let lm_head_edge = graph
        .edges
        .get(&crate::TensorEdgeId::new("weight.lm_head"))
        .expect("lm_head weight edge exists");
    assert_eq!(
        lm_head_edge.aliasing,
        crate::TensorAliasing::MayAlias(crate::TensorEdgeId::new("weight.token_embedding"))
    );
    // The K/V edges carry KV cache metadata under the qwen.layer0.{k,v}
    // namespace `parse_kv_cache_id` (the existing execution path, generic
    // over the namespace) expects.
    let k_edge = graph
        .edges
        .values()
        .find(|edge| {
            edge.kv_cache
                .as_ref()
                .is_some_and(|kv| kv.cache_id == "qwen.layer0.k")
        })
        .expect("a k edge carries kv metadata under the expected cache id");
    assert_eq!(
        k_edge.kv_cache.as_ref().unwrap().behavior,
        crate::GraphKvCacheBehavior::Output
    );

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
fn wasmtime_engine_executes_qwen_graph_component_fixture() {
    let directory =
        std::env::temp_dir().join(format!("magnetar-wasmtime-qwen-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let artifact = directory.join("qwen-graph.component.wasm");
    std::fs::write(&artifact, QWEN_GRAPH_COMPONENT).unwrap();

    let mut engine = WasmtimeComponentEngine::new().unwrap();
    let interface = WitInterface::new("magnetar:qwen/graph-fixture", "1.0.0");
    let definition = ComponentDefinition {
        id: ComponentDefinitionId::new(18),
        metadata: crate::ComponentMetadata::new(
            "qwen-graph-fixture",
            "1",
            "executable Qwen graph fixture component",
        )
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

    for (operation, expected) in [
        ("prefill-node-count", 19),
        ("decode-node-count", 19),
        ("provider-authority-count", 0),
    ] {
        let result = engine
            .invoke(
                &instance,
                &ComponentInvocation::new(
                    crate::ComponentInstanceId::new(180),
                    interface.clone(),
                    operation,
                ),
            )
            .unwrap();
        assert_eq!(
            result,
            ComponentInvocationResult::single(ComponentValue::U32(expected))
        );
    }
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
fn wasmtime_engine_reports_capability_version_mismatch_distinctly_from_unresolved_import() {
    // `CAPABILITY_ECHO_COMPONENT`'s real compiled import is
    // "example:test/capability@1.0.0" -- the Link Plan below approves that
    // same interface *name* at a different version ("2.0.0"), so the exact
    // `(name, version)` lookup misses while the name itself is still known.
    // `model-component-graph-contract`'s "Component requires unsupported
    // contract version" scenario is exactly this case.
    let directory = std::env::temp_dir().join(format!(
        "magnetar-wasmtime-capability-version-mismatch-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let artifact = directory.join("capability-echo.component.wasm");
    std::fs::write(&artifact, CAPABILITY_ECHO_COMPONENT).unwrap();

    let mut engine = WasmtimeComponentEngine::new().unwrap();
    let requested_interface = WitInterface::new("example:test/capability", "1.0.0");
    let approved_interface = WitInterface::new("example:test/capability", "2.0.0");
    let export_interface = WitInterface::new("example:test/capability-echo", "1.0.0");
    let definition = ComponentDefinition {
        id: ComponentDefinitionId::new(191),
        metadata: crate::ComponentMetadata::new(
            "capability-echo-version-mismatch",
            "1",
            "capability echo component requiring an unapproved version",
        )
        .with_import(requested_interface)
        .with_export(export_interface),
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
    link_plan.insert_for_test(crate::ComponentEndpoint::Capability {
        interface: approved_interface.clone(),
    });

    let error = engine.instantiate(&prepared, &link_plan).unwrap_err();
    match error {
        ComponentError::CapabilityVersionMismatch {
            name,
            requested_version,
            available_version,
            ..
        } => {
            assert_eq!(name, "example:test/capability");
            assert_eq!(requested_version, "1.0.0");
            assert_eq!(available_version, "2.0.0");
        }
        other => panic!("expected CapabilityVersionMismatch, got {other:?}"),
    }

    std::fs::remove_dir_all(directory).unwrap();
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

/// Builds a running instance of the infinite-loop fixture under `limits`.
fn looping_instance(
    engine: &mut WasmtimeComponentEngine,
    directory: &std::path::Path,
    definition_id: u64,
    limits: &ComponentResourceLimits,
) -> (ComponentEngineInstance, WitInterface) {
    std::fs::create_dir_all(directory).unwrap();
    let artifact = directory.join("loop.component.wasm");
    std::fs::write(&artifact, LOOP_COMPONENT).unwrap();

    let interface = WitInterface::new("example:component/run", "1.0.0");
    let definition = ComponentDefinition {
        id: ComponentDefinitionId::new(definition_id),
        metadata: crate::ComponentMetadata::new("loop", "1", "looping component")
            .with_export(interface.clone()),
        artifact_path: artifact,
        manifest_path: None,
        artifact_digest: None,
        trust_decision: None,
        state: crate::ComponentDefinitionState::Registered,
    };
    let prepared = engine.prepare(&definition, limits).unwrap();
    let instance = engine
        .instantiate(&prepared, &ComponentLinkPlan::default())
        .unwrap();
    (instance, interface)
}

/// A declared execution budget must stop a Component that never returns.
///
/// Fuel is deterministic -- it traps after a fixed number of operations, with
/// no dependence on timing -- so this test cannot hang even if the deadline
/// machinery regresses.
#[test]
fn wasmtime_engine_stops_a_runaway_component_at_its_execution_budget() {
    let directory =
        std::env::temp_dir().join(format!("magnetar-wasmtime-fuel-{}", std::process::id()));
    let mut engine = WasmtimeComponentEngine::new().unwrap();
    let limits = ComponentResourceLimits {
        engine_execution_budget: Some(100_000),
        ..ComponentResourceLimits::default()
    };
    let (instance, interface) = looping_instance(&mut engine, &directory, 40, &limits);
    let invocation =
        ComponentInvocation::new(crate::ComponentInstanceId::new(140), interface, "run");

    let result = engine.invoke(&instance, &invocation);

    assert!(
        matches!(
            result,
            Err(ComponentError::Interrupted {
                reason: ComponentInterruptionReason::ResourcePolicy,
                ..
            })
        ),
        "expected the execution budget to stop the loop, got {result:?}"
    );
    engine.destroy(instance).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

/// The budget is per invocation: each call starts from the full declared
/// allowance rather than sharing one tank across the instance's lifetime.
///
/// This has to be shown with a Component that *returns* and that costs
/// measurable fuel. A trapped instance cannot be called again at all --
/// Wasmtime poisons it -- so repeating a call that exhausts its budget would
/// only ever observe that poisoning; and an empty export consumes no fuel at
/// all, so no number of calls to one would ever drain a tank.
#[test]
fn wasmtime_engine_execution_budget_is_replenished_per_invocation() {
    let directory = std::env::temp_dir().join(format!(
        "magnetar-wasmtime-fuel-reset-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let artifact = directory.join("bounded.component.wasm");
    std::fs::write(&artifact, BOUNDED_LOOP_COMPONENT).unwrap();

    let mut engine = WasmtimeComponentEngine::new().unwrap();
    let interface = WitInterface::new("example:component/run", "1.0.0");
    let definition = ComponentDefinition {
        id: ComponentDefinitionId::new(41),
        metadata: crate::ComponentMetadata::new("budget-reset", "1", "returning component")
            .with_export(interface.clone()),
        artifact_path: artifact,
        manifest_path: None,
        artifact_digest: None,
        trust_decision: None,
        state: crate::ComponentDefinitionState::Registered,
    };
    // The fixture runs 10,000 iterations, so one call costs tens of thousands
    // of fuel units. This budget covers a single call with room to spare and
    // would be exhausted within the first few iterations below if the tank
    // were never refilled.
    let limits = ComponentResourceLimits {
        engine_execution_budget: Some(200_000),
        ..ComponentResourceLimits::default()
    };
    let prepared = engine.prepare(&definition, &limits).unwrap();
    let instance = engine
        .instantiate(&prepared, &ComponentLinkPlan::default())
        .unwrap();
    let invocation =
        ComponentInvocation::new(crate::ComponentInstanceId::new(141), interface, "run");

    for attempt in 0..200 {
        let result = engine.invoke(&instance, &invocation);
        assert!(
            result.is_ok(),
            "attempt {attempt}: budget was not replenished, got {result:?}"
        );
    }

    engine.destroy(instance).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

/// A non-zero wall-clock deadline must stop a Component that never returns.
///
/// Before the epoch ticker existed, only the sentinel `Some(0)` interrupted
/// anything: every real deadline took the branch that *disabled* the epoch
/// deadline, and nothing advanced the epoch, so this call ran forever.
///
/// A generous execution budget is declared alongside the deadline purely as a
/// backstop: if the deadline machinery regresses, the call still terminates on
/// fuel and this test fails on the wrong interruption reason instead of
/// hanging CI. The budget is roughly two seconds of execution here, while a
/// 50ms run needs about 50 million units, so it cannot fire first -- and the
/// margin only widens on a slower machine, where the wall-clock deadline still
/// lands at 50ms but fuel is consumed more slowly.
#[test]
fn wasmtime_engine_stops_a_runaway_component_at_its_wall_clock_deadline() {
    let directory = std::env::temp_dir().join(format!(
        "magnetar-wasmtime-real-deadline-{}",
        std::process::id()
    ));
    let mut engine = WasmtimeComponentEngine::new().unwrap();
    let limits = ComponentResourceLimits {
        execution_deadline_millis: Some(50),
        engine_execution_budget: Some(2_000_000_000),
        ..ComponentResourceLimits::default()
    };
    let (instance, interface) = looping_instance(&mut engine, &directory, 42, &limits);
    let invocation =
        ComponentInvocation::new(crate::ComponentInstanceId::new(142), interface, "run");

    let started = std::time::Instant::now();
    let result = engine.invoke(&instance, &invocation);
    let elapsed = started.elapsed();

    assert!(
        matches!(
            result,
            Err(ComponentError::Interrupted {
                reason: ComponentInterruptionReason::Deadline,
                ..
            })
        ),
        "expected the deadline to stop the loop, got {result:?} after {elapsed:?}"
    );
    engine.destroy(instance).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

/// A per-call deadline may tighten the Component's configured limit, never
/// loosen it.
#[test]
fn wasmtime_engine_takes_the_stricter_of_call_and_component_deadlines() {
    let directory = std::env::temp_dir().join(format!(
        "magnetar-wasmtime-deadline-min-{}",
        std::process::id()
    ));
    let mut engine = WasmtimeComponentEngine::new().unwrap();
    let limits = ComponentResourceLimits {
        execution_deadline_millis: Some(50),
        engine_execution_budget: Some(2_000_000_000),
        ..ComponentResourceLimits::default()
    };
    let (instance, interface) = looping_instance(&mut engine, &directory, 43, &limits);
    let mut invocation =
        ComponentInvocation::new(crate::ComponentInstanceId::new(143), interface, "run");
    // A call asking for a far longer deadline must not escape the 50ms limit.
    invocation.deadline_millis = Some(600_000);

    let result = engine.invoke(&instance, &invocation);

    assert!(
        matches!(
            result,
            Err(ComponentError::Interrupted {
                reason: ComponentInterruptionReason::Deadline,
                ..
            })
        ),
        "a longer per-call deadline must not loosen the component limit, got {result:?}"
    );
    engine.destroy(instance).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn epoch_deadline_ticks_rounds_up_and_never_expires_immediately() {
    // 0 ticks would mean "already expired"; a sub-tick deadline must still get
    // one tick of runway.
    assert_eq!(epoch_deadline_ticks(Some(1)), 1);
    assert_eq!(epoch_deadline_ticks(Some(50)), 50);
    assert_eq!(epoch_deadline_ticks(None), DISABLED_EPOCH_DEADLINE);
}
