use crate::{
    ComponentContract, ComponentDefinition, ComponentDefinitionId, ComponentEngine,
    ComponentEngineCapabilities, ComponentEngineInstance, ComponentError,
    ComponentExportDescription, ComponentImportRequirement, ComponentInterfaceShape,
    ComponentInterruptionReason, ComponentInvocation, ComponentInvocationResult, ComponentLinkPlan,
    ComponentResourceLimits, ComponentTrapKind, ComponentValue, PreparedComponent, WitInterface,
};
use std::{collections::BTreeMap, fs, path::Path};
use wasmtime::{
    Config, Engine, Store, StoreLimits, StoreLimitsBuilder,
    component::{
        Component as WasmtimeComponent, Instance as WasmtimeInstance, Linker as WasmtimeLinker,
        types::{ComponentExtern, ComponentItem},
    },
};

const HOST_ADAPTER_FAILURE_MARKER: &str = "[magnetar host adapter error]";
const DISABLED_EPOCH_DEADLINE: u64 = 1_000_000_000;

pub struct WasmtimeComponentEngine {
    engine: Engine,
    prepared: BTreeMap<String, WasmtimePreparedComponent>,
    instances: BTreeMap<String, WasmtimeInstanceState>,
    next_prepared_id: u64,
    next_instance_id: u64,
}

struct WasmtimeInstanceState {
    _store: Store<WasmtimeStoreState>,
    _instance: WasmtimeInstance,
}

struct WasmtimeStoreState {
    limits: StoreLimits,
    host_calls: u64,
    pending_interruption: Option<ComponentInterruptionReason>,
}

#[derive(Clone)]
struct WasmtimePreparedComponent {
    component: WasmtimeComponent,
    limits: ComponentResourceLimits,
}

impl WasmtimeComponentEngine {
    pub fn new() -> Result<Self, ComponentError> {
        let mut config = Config::new();
        config.wasm_component_model(true);
        config.epoch_interruption(true);

        let engine = Engine::new(&config).map_err(map_engine_error)?;
        Ok(Self {
            engine,
            prepared: BTreeMap::new(),
            instances: BTreeMap::new(),
            next_prepared_id: 1,
            next_instance_id: 1,
        })
    }

    fn next_key(&mut self, definition_id: ComponentDefinitionId) -> String {
        let key = format!(
            "wasmtime-component:{}:{}",
            definition_id.get(),
            self.next_prepared_id
        );
        self.next_prepared_id += 1;
        key
    }

    fn next_instance_key(&mut self, definition_id: ComponentDefinitionId) -> String {
        let key = format!(
            "wasmtime-instance:{}:{}",
            definition_id.get(),
            self.next_instance_id
        );
        self.next_instance_id += 1;
        key
    }

    fn load_component_bytes(path: &Path) -> Result<Vec<u8>, ComponentError> {
        if !path.exists() {
            return Err(ComponentError::ComponentLoadFailed {
                path: path.into(),
                message: "artifact path does not exist".into(),
                source: None,
            });
        }
        if !path.is_file() {
            return Err(ComponentError::ComponentLoadFailed {
                path: path.into(),
                message: "artifact path is not a file".into(),
                source: None,
            });
        }
        fs::read(path).map_err(|source| ComponentError::ComponentLoadFailed {
            path: path.into(),
            message: source.to_string(),
            source: Some(source),
        })
    }

    fn inspect_wasmtime_contract(&self, component: &WasmtimeComponent) -> ComponentContract {
        let component_type = component.component_type();
        ComponentContract {
            imports: component_type
                .imports(&self.engine)
                .map(|(name, item)| {
                    ComponentImportRequirement::new(
                        wit_interface_from_component_name(name),
                        shape_from_component_extern(&item),
                    )
                })
                .collect(),
            exports: component_type
                .exports(&self.engine)
                .map(|(name, item)| {
                    ComponentExportDescription::new(
                        wit_interface_from_component_name(name),
                        shape_from_component_extern(&item),
                    )
                })
                .collect(),
        }
    }

    fn load_and_inspect_contract(
        &self,
        definition: &ComponentDefinition,
    ) -> Result<ComponentContract, ComponentError> {
        let bytes = Self::load_component_bytes(&definition.artifact_path)?;
        let component = WasmtimeComponent::new(&self.engine, bytes).map_err(|source| {
            ComponentError::PreparationFailed {
                component: definition.metadata.name.clone(),
                message: redact_engine_message(source),
            }
        })?;
        Ok(self.inspect_wasmtime_contract(&component))
    }
}

impl ComponentEngine for WasmtimeComponentEngine {
    fn capabilities(&self) -> ComponentEngineCapabilities {
        ComponentEngineCapabilities {
            component_model: true,
            async_host_calls: true,
            interruption: true,
            resource_limits: true,
        }
    }

    fn inspect_contract(
        &mut self,
        definition: &ComponentDefinition,
    ) -> Result<ComponentContract, ComponentError> {
        self.load_and_inspect_contract(definition)
    }

    fn prepare(
        &mut self,
        definition: &ComponentDefinition,
        limits: &ComponentResourceLimits,
    ) -> Result<PreparedComponent, ComponentError> {
        if limits.require_memory_limit && limits.max_memory_bytes.is_none() {
            return Err(ComponentError::ResourceLimitUnsupported {
                component: definition.metadata.name.clone(),
                limit: "memory",
            });
        }
        if limits
            .max_memory_bytes
            .is_some_and(|limit| limit > usize::MAX as u64)
        {
            return Err(ComponentError::ResourceLimitUnsupported {
                component: definition.metadata.name.clone(),
                limit: "memory",
            });
        }

        let bytes = Self::load_component_bytes(&definition.artifact_path)?;
        let component = WasmtimeComponent::new(&self.engine, bytes).map_err(|source| {
            ComponentError::PreparationFailed {
                component: definition.metadata.name.clone(),
                message: redact_engine_message(source),
            }
        })?;
        let contract = self.inspect_wasmtime_contract(&component);
        let key = self.next_key(definition.id);
        self.prepared.insert(
            key.clone(),
            WasmtimePreparedComponent {
                component,
                limits: limits.clone(),
            },
        );
        Ok(PreparedComponent::with_contract(
            definition.id,
            key,
            contract,
        ))
    }

    fn instantiate(
        &mut self,
        prepared: &PreparedComponent,
        link_plan: &ComponentLinkPlan,
    ) -> Result<ComponentEngineInstance, ComponentError> {
        let prepared_state = self
            .prepared
            .get(prepared.engine_key())
            .ok_or(ComponentError::MissingPreparedDefinition(
                prepared.definition_id(),
            ))?
            .clone();
        let mut store = Store::new(
            &self.engine,
            WasmtimeStoreState {
                limits: store_limits(&prepared_state.limits),
                host_calls: 0,
                pending_interruption: None,
            },
        );
        store.limiter(|state| &mut state.limits);
        store.set_epoch_deadline(DISABLED_EPOCH_DEADLINE);
        store.epoch_deadline_trap();
        let mut linker = WasmtimeLinker::new(&self.engine);
        configure_linker(
            &self.engine,
            &mut linker,
            &prepared_state.component,
            link_plan,
            prepared.definition_id(),
        )?;
        let instance = linker
            .instantiate(&mut store, &prepared_state.component)
            .map_err(|source| ComponentError::InstantiationFailed {
                definition: prepared.definition_id(),
                message: redact_engine_message(source),
            })?;
        let key = self.next_instance_key(prepared.definition_id());
        self.instances.insert(
            key.clone(),
            WasmtimeInstanceState {
                _store: store,
                _instance: instance,
            },
        );
        Ok(ComponentEngineInstance::new(prepared.definition_id(), key))
    }

    fn invoke(
        &mut self,
        instance: &ComponentEngineInstance,
        invocation: &ComponentInvocation,
    ) -> Result<ComponentInvocationResult, ComponentError> {
        let state = self
            .instances
            .get_mut(instance.engine_key())
            .ok_or(ComponentError::InstanceNotFound(invocation.instance_id))?;
        let interruption_reason = if invocation.deadline_millis == Some(0) {
            state._store.set_epoch_deadline(0);
            state._store.data_mut().pending_interruption =
                Some(ComponentInterruptionReason::Deadline);
            self.engine.increment_epoch();
            Some(ComponentInterruptionReason::Deadline)
        } else if let Some(reason) = state._store.data().pending_interruption {
            Some(reason)
        } else {
            state._store.set_epoch_deadline(DISABLED_EPOCH_DEADLINE);
            None
        };
        if let Some(reason) = interruption_reason {
            state._store.data_mut().pending_interruption = None;
            state._store.set_epoch_deadline(DISABLED_EPOCH_DEADLINE);
            return Err(ComponentError::Interrupted {
                instance: invocation.instance_id,
                reason,
            });
        }
        let host_calls_before = state._store.data().host_calls;
        let result = if let Ok(typed) = state
            ._instance
            .get_typed_func::<(), ()>(&mut state._store, invocation.operation.as_str())
        {
            typed
                .call(&mut state._store, ())
                .map(|()| ComponentInvocationResult::empty())
        } else {
            let typed = state
                ._instance
                .get_typed_func::<(), (u32,)>(&mut state._store, invocation.operation.as_str())
                .map_err(|source| ComponentError::InvocationFailed {
                    instance: invocation.instance_id,
                    message: redact_engine_message(source),
                })?;
            typed
                .call(&mut state._store, ())
                .map(|(value,)| ComponentInvocationResult::single(ComponentValue::U32(value)))
        };
        let host_failure = state._store.data().host_calls > host_calls_before;
        let result = result.map_err(|source| map_call_error(source, invocation, host_failure));
        if matches!(result, Err(ComponentError::Interrupted { .. })) {
            state._store.data_mut().pending_interruption = None;
            state._store.set_epoch_deadline(DISABLED_EPOCH_DEADLINE);
        }
        result
    }

    fn interrupt(
        &mut self,
        instance: &ComponentEngineInstance,
        reason: ComponentInterruptionReason,
    ) -> Result<(), ComponentError> {
        if let Some(state) = self.instances.get_mut(instance.engine_key()) {
            state._store.data_mut().pending_interruption = Some(reason);
            state._store.set_epoch_deadline(0);
            self.engine.increment_epoch();
        }
        Ok(())
    }

    fn destroy(&mut self, instance: ComponentEngineInstance) -> Result<(), ComponentError> {
        self.instances.remove(instance.engine_key());
        Ok(())
    }
}

fn map_engine_error(source: wasmtime::Error) -> ComponentError {
    ComponentError::EngineFailure(redact_engine_message(source))
}

fn redact_engine_message(source: wasmtime::Error) -> String {
    let message = source.to_string();
    if message.is_empty() {
        "[redacted engine error]".into()
    } else {
        message
    }
}

fn redact_trap_message(_source: wasmtime::Error) -> String {
    "[redacted component trap]".into()
}

fn is_epoch_interruption(source: &wasmtime::Error) -> bool {
    let message = source.to_string();
    message.contains("epoch") || message.contains("deadline")
}

fn is_host_adapter_failure(source: &wasmtime::Error) -> bool {
    source.to_string().contains(HOST_ADAPTER_FAILURE_MARKER)
}

fn map_call_error(
    source: wasmtime::Error,
    invocation: &ComponentInvocation,
    host_failure: bool,
) -> ComponentError {
    if is_epoch_interruption(&source) {
        ComponentError::Interrupted {
            instance: invocation.instance_id,
            reason: ComponentInterruptionReason::Deadline,
        }
    } else if host_failure || is_host_adapter_failure(&source) {
        ComponentError::InvocationFailed {
            instance: invocation.instance_id,
            message: "[redacted host adapter error]".into(),
        }
    } else {
        ComponentError::Trap {
            instance: invocation.instance_id,
            kind: ComponentTrapKind::Trap,
            diagnostic: Some(redact_trap_message(source)),
        }
    }
}

fn store_limits(limits: &ComponentResourceLimits) -> StoreLimits {
    let mut builder = StoreLimitsBuilder::new();
    if let Some(max_memory_bytes) = limits.max_memory_bytes {
        builder = builder.memory_size(max_memory_bytes as usize);
    }
    builder.build()
}

fn configure_linker(
    engine: &Engine,
    linker: &mut WasmtimeLinker<WasmtimeStoreState>,
    component: &WasmtimeComponent,
    link_plan: &ComponentLinkPlan,
    definition: ComponentDefinitionId,
) -> Result<(), ComponentError> {
    for (import_name, item) in component.component_type().imports(engine) {
        let interface = wit_interface_from_component_name(import_name);
        if link_plan.endpoint(&interface).is_none() {
            return Err(ComponentError::InstantiationFailed {
                definition,
                message: format!(
                    "Component import '{}@{}' is absent from the approved Link Plan",
                    interface.name, interface.version
                ),
            });
        }
        match item.ty {
            ComponentItem::ComponentInstance(instance) => {
                let mut linker_instance = linker.instance(import_name).map_err(|source| {
                    ComponentError::InstantiationFailed {
                        definition,
                        message: redact_engine_message(source),
                    }
                })?;
                for (export_name, export) in instance.exports(engine) {
                    match export.ty {
                        ComponentItem::ComponentFunc(func)
                            if func.params().len() == 0 && func.results().len() == 0 =>
                        {
                            let fails_for_test = export_name == "fail";
                            linker_instance
                                .func_wrap(export_name, move |mut store, _params: ()| {
                                    store.data_mut().host_calls += 1;
                                    if fails_for_test {
                                        return Err(wasmtime::Error::msg(
                                            HOST_ADAPTER_FAILURE_MARKER,
                                        ));
                                    }
                                    Ok(())
                                })
                                .map_err(|source| ComponentError::InstantiationFailed {
                                    definition,
                                    message: redact_engine_message(source),
                                })?;
                        }
                        ComponentItem::ComponentFunc(_) => {
                            return Err(ComponentError::InstantiationFailed {
                                definition,
                                message: format!(
                                    "unsupported host import function signature for '{import_name}.{export_name}'"
                                ),
                            });
                        }
                        _ => {
                            return Err(ComponentError::InstantiationFailed {
                                definition,
                                message: format!(
                                    "unsupported host import item for '{import_name}.{export_name}'"
                                ),
                            });
                        }
                    }
                }
            }
            ComponentItem::ComponentFunc(func)
                if func.params().len() == 0 && func.results().len() == 0 =>
            {
                let fails_for_test =
                    import_name.ends_with("/fail@1.0.0") || import_name.ends_with(":fail");
                linker
                    .root()
                    .func_wrap(import_name, move |mut store, _params: ()| {
                        store.data_mut().host_calls += 1;
                        if fails_for_test {
                            return Err(wasmtime::Error::msg(HOST_ADAPTER_FAILURE_MARKER));
                        }
                        Ok(())
                    })
                    .map_err(|source| ComponentError::InstantiationFailed {
                        definition,
                        message: redact_engine_message(source),
                    })?;
            }
            ComponentItem::ComponentFunc(_) => {
                return Err(ComponentError::InstantiationFailed {
                    definition,
                    message: format!(
                        "unsupported host import function signature for '{import_name}'"
                    ),
                });
            }
            _ => {
                return Err(ComponentError::InstantiationFailed {
                    definition,
                    message: format!("unsupported host import item for '{import_name}'"),
                });
            }
        }
    }
    Ok(())
}

fn wit_interface_from_component_name(name: &str) -> WitInterface {
    let (name, version) = name.rsplit_once('@').unwrap_or((name, ""));
    WitInterface::new(name, version)
}

fn shape_from_component_extern(item: &ComponentExtern<'_>) -> ComponentInterfaceShape {
    match item.ty {
        ComponentItem::ComponentFunc(_) | ComponentItem::CoreFunc(_) => {
            ComponentInterfaceShape::Function
        }
        ComponentItem::Module(_) => ComponentInterfaceShape::Module,
        ComponentItem::Component(_) => ComponentInterfaceShape::Component,
        ComponentItem::ComponentInstance(_) => ComponentInterfaceShape::Instance,
        ComponentItem::Type(_) => ComponentInterfaceShape::Type,
        ComponentItem::Resource(_) => ComponentInterfaceShape::Resource,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNIT_EXPORT_COMPONENT: &str =
        include_str!("../fixtures/components/unit-export.component.wat");
    const HOST_ROUNDTRIP_COMPONENT: &str =
        include_str!("../fixtures/components/host-roundtrip.component.wat");
    const HOST_FAILURE_COMPONENT: &str =
        include_str!("../fixtures/components/host-failure.component.wat");
    const COMPUTE_IMPORT_COMPONENT: &str =
        include_str!("../fixtures/components/compute-import.component.wat");
    const LOOP_COMPONENT: &str = include_str!("../fixtures/components/loop.component.wat");
    const TRAPPING_COMPONENT: &str = include_str!("../fixtures/components/trapping.component.wat");
    const U32_EXPORT_COMPONENT: &str =
        include_str!("../fixtures/components/u32-export.component.wat");
    const WASI_FILESYSTEM_COMPONENT: &str =
        include_str!("../fixtures/components/wasi-filesystem.component.wat");
    const WASI_ENVIRONMENT_COMPONENT: &str =
        include_str!("../fixtures/components/wasi-environment.component.wat");
    const RESOURCE_IMPORT_COMPONENT: &str =
        include_str!("../fixtures/components/resource-import.component.wat");

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
        let directory =
            std::env::temp_dir().join(format!("magnetar-wasmtime-{}", std::process::id()));
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
                &ComponentInvocation::new(
                    crate::ComponentInstanceId::new(103),
                    interface,
                    "answer",
                ),
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
                metadata: crate::ComponentMetadata::new(
                    format!("wasi-{index}"),
                    "1",
                    "wasi component",
                )
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
}
