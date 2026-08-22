use crate::{
    ComponentDefinition, ComponentDefinitionId, ComponentEngine, ComponentEngineCapabilities,
    ComponentEngineInstance, ComponentError, ComponentInterruptionReason, ComponentInvocation,
    ComponentLinkPlan, ComponentResourceLimits, PreparedComponent,
};
use std::{collections::BTreeMap, fs, path::Path};
use wasmtime::{Config, Engine, component::Component as WasmtimeComponent};

pub struct WasmtimeComponentEngine {
    engine: Engine,
    prepared: BTreeMap<String, WasmtimeComponent>,
    next_prepared_id: u64,
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
            next_prepared_id: 1,
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

    fn load_component_bytes(path: &Path) -> Result<Vec<u8>, ComponentError> {
        if !path.exists() {
            return Err(ComponentError::ComponentLoadFailed {
                path: path.into(),
                message: "artifact path does not exist".into(),
            });
        }
        if !path.is_file() {
            return Err(ComponentError::ComponentLoadFailed {
                path: path.into(),
                message: "artifact path is not a file".into(),
            });
        }
        fs::read(path).map_err(|source| ComponentError::ComponentLoadFailed {
            path: path.into(),
            message: source.to_string(),
        })
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

        let bytes = Self::load_component_bytes(&definition.artifact_path)?;
        let component = WasmtimeComponent::new(&self.engine, bytes).map_err(|source| {
            ComponentError::PreparationFailed {
                component: definition.metadata.name.clone(),
                message: redact_engine_message(source),
            }
        })?;
        let key = self.next_key(definition.id);
        self.prepared.insert(key.clone(), component);
        Ok(PreparedComponent::new(definition.id, key))
    }

    fn instantiate(
        &mut self,
        prepared: &PreparedComponent,
        _link_plan: &ComponentLinkPlan,
    ) -> Result<ComponentEngineInstance, ComponentError> {
        if !self.prepared.contains_key(prepared.engine_key()) {
            return Err(ComponentError::MissingPreparedDefinition(
                prepared.definition_id(),
            ));
        }
        Err(ComponentError::InstantiationFailed {
            definition: prepared.definition_id(),
            message: "Wasmtime host adapter linking is not implemented yet".into(),
        })
    }

    fn invoke(&mut self, invocation: &ComponentInvocation) -> Result<(), ComponentError> {
        Err(ComponentError::EngineFailure(format!(
            "Wasmtime invocation is not implemented for instance '{}'",
            invocation.instance_id.get()
        )))
    }

    fn interrupt(
        &mut self,
        _instance: &ComponentEngineInstance,
        _reason: ComponentInterruptionReason,
    ) -> Result<(), ComponentError> {
        Ok(())
    }

    fn destroy(&mut self, instance: ComponentEngineInstance) -> Result<(), ComponentError> {
        self.prepared.remove(instance.engine_key());
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

#[cfg(test)]
mod tests {
    use super::*;

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
            state: crate::ComponentDefinitionState::Registered,
        };

        assert!(matches!(
            engine.prepare(&definition, &ComponentResourceLimits::default()),
            Err(ComponentError::ComponentLoadFailed { .. })
        ));
    }
}
