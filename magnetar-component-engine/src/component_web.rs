use crate::{
    ComponentContract, ComponentDefinition, ComponentEngine, ComponentEngineCapabilities,
    ComponentEngineInstance, ComponentError, ComponentInterruptionReason, ComponentInvocation,
    ComponentInvocationResult, ComponentLinkPlan, ComponentResourceLimits, PreparedComponent,
};

/// Browser-target Component Engine adapter.
///
/// This placeholder defines the platform boundary for `wasm32` builds without
/// pulling in Wasmtime or native Provider loading. Browser host binding support
/// is intentionally fail-closed until the JavaScript adapter is implemented.
#[derive(Default)]
pub struct WebComponentEngine {
    next_prepared_id: u64,
}

impl WebComponentEngine {
    pub const fn new() -> Self {
        Self {
            next_prepared_id: 1,
        }
    }
}

impl ComponentEngine for WebComponentEngine {
    fn capabilities(&self) -> ComponentEngineCapabilities {
        ComponentEngineCapabilities::web()
    }

    fn inspect_contract(
        &mut self,
        definition: &ComponentDefinition,
    ) -> Result<ComponentContract, ComponentError> {
        Ok(ComponentContract::from_metadata(&definition.metadata))
    }

    fn prepare(
        &mut self,
        definition: &ComponentDefinition,
        limits: &ComponentResourceLimits,
    ) -> Result<PreparedComponent, ComponentError> {
        if limits.require_memory_limit {
            return Err(ComponentError::EngineFeatureUnavailable {
                component: definition.metadata.name.clone(),
                feature: crate::ComponentEngineFeature::ResourceLimits,
                profile: crate::ComponentEngineProfile::Web,
            });
        }
        let key = format!(
            "web-component:{}:{}",
            definition.id.get(),
            self.next_prepared_id
        );
        self.next_prepared_id += 1;
        Ok(PreparedComponent::with_contract(
            definition.id,
            key,
            ComponentContract::from_metadata(&definition.metadata),
        ))
    }

    fn instantiate(
        &mut self,
        prepared: &PreparedComponent,
        link_plan: &ComponentLinkPlan,
    ) -> Result<ComponentEngineInstance, ComponentError> {
        if link_plan.links().next().is_some() {
            return Err(ComponentError::HostBindingFailed {
                component: format!("definition:{}", prepared.definition_id().get()),
                message: "browser host binding adapter is not implemented".into(),
            });
        }
        Ok(ComponentEngineInstance::new(
            prepared.definition_id(),
            format!("web-instance:{}", prepared.definition_id().get()),
        ))
    }

    fn invoke(
        &mut self,
        instance: &ComponentEngineInstance,
        _invocation: &ComponentInvocation,
    ) -> Result<ComponentInvocationResult, ComponentError> {
        Err(ComponentError::HostBindingFailed {
            component: format!("definition:{}", instance.definition_id().get()),
            message: "browser invocation adapter is not implemented".into(),
        })
    }

    fn interrupt(
        &mut self,
        _instance: &ComponentEngineInstance,
        _reason: ComponentInterruptionReason,
    ) -> Result<(), ComponentError> {
        Err(ComponentError::EngineFeatureUnavailable {
            component: "web-component-engine".into(),
            feature: crate::ComponentEngineFeature::Interruption,
            profile: crate::ComponentEngineProfile::Web,
        })
    }

    fn destroy(&mut self, _instance: ComponentEngineInstance) -> Result<(), ComponentError> {
        Ok(())
    }
}
