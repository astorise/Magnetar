use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
    path::{Path, PathBuf},
};

static NEXT_COMPONENT_DEFINITION_ID: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);
static NEXT_COMPONENT_INSTANCE_ID: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);

fn next_component_definition_id() -> ComponentDefinitionId {
    ComponentDefinitionId(
        NEXT_COMPONENT_DEFINITION_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    )
}

fn next_component_instance_id() -> ComponentInstanceId {
    ComponentInstanceId(
        NEXT_COMPONENT_INSTANCE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    )
}

/// A WIT interface identified by its package-qualified name and version.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct WitInterface {
    pub name: String,
    pub version: String,
}
impl WitInterface {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
        }
    }
}

/// Declarative metadata for portable WebAssembly Component contracts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub imports: BTreeSet<WitInterface>,
    pub exports: BTreeSet<WitInterface>,
}
impl ComponentMetadata {
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            description: description.into(),
            imports: BTreeSet::new(),
            exports: BTreeSet::new(),
        }
    }

    pub fn with_import(mut self, interface: WitInterface) -> Self {
        self.imports.insert(interface);
        self
    }

    pub fn with_export(mut self, interface: WitInterface) -> Self {
        self.exports.insert(interface);
        self
    }
}

/// A discovered Component artifact and its declared metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentDescriptor {
    pub metadata: ComponentMetadata,
    pub artifact_path: PathBuf,
}

impl ComponentDescriptor {
    pub fn new(metadata: ComponentMetadata, artifact_path: impl Into<PathBuf>) -> Self {
        Self {
            metadata,
            artifact_path: artifact_path.into(),
        }
    }

    pub fn artifact_reference(&self) -> ComponentArtifactReference<'_> {
        ComponentArtifactReference::LocalPath(&self.artifact_path)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComponentArtifactReference<'a> {
    LocalPath(&'a Path),
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ComponentInterfaceShape {
    Function,
    Interface,
    Resource,
    Instance,
    Component,
    Module,
    Type,
    Unknown,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ComponentImportRequirement {
    pub interface: WitInterface,
    pub shape: ComponentInterfaceShape,
}

impl ComponentImportRequirement {
    pub fn new(interface: WitInterface, shape: ComponentInterfaceShape) -> Self {
        Self { interface, shape }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ComponentExportDescription {
    pub interface: WitInterface,
    pub shape: ComponentInterfaceShape,
}

impl ComponentExportDescription {
    pub fn new(interface: WitInterface, shape: ComponentInterfaceShape) -> Self {
        Self { interface, shape }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ComponentContract {
    pub imports: BTreeSet<ComponentImportRequirement>,
    pub exports: BTreeSet<ComponentExportDescription>,
}

impl ComponentContract {
    pub fn from_metadata(metadata: &ComponentMetadata) -> Self {
        Self {
            imports: metadata
                .imports
                .iter()
                .cloned()
                .map(|interface| {
                    ComponentImportRequirement::new(interface, ComponentInterfaceShape::Interface)
                })
                .collect(),
            exports: metadata
                .exports
                .iter()
                .cloned()
                .map(|interface| {
                    ComponentExportDescription::new(interface, ComponentInterfaceShape::Interface)
                })
                .collect(),
        }
    }

    pub fn import_interfaces(&self) -> BTreeSet<WitInterface> {
        self.imports
            .iter()
            .map(|requirement| requirement.interface.clone())
            .collect()
    }

    pub fn export_interfaces(&self) -> BTreeSet<WitInterface> {
        self.exports
            .iter()
            .map(|description| description.interface.clone())
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ComponentDefinitionId(u64);
impl ComponentDefinitionId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ComponentInstanceId(u64);
impl ComponentInstanceId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComponentDefinitionState {
    Registered,
    Validated,
    Prepared,
    Failed,
    Removed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComponentInstanceState {
    Instantiating,
    Ready,
    Failed,
    Destroyed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentDefinition {
    pub id: ComponentDefinitionId,
    pub metadata: ComponentMetadata,
    pub artifact_path: PathBuf,
    pub state: ComponentDefinitionState,
}

impl ComponentDefinition {
    fn registered(descriptor: ComponentDescriptor) -> Self {
        Self {
            id: next_component_definition_id(),
            metadata: descriptor.metadata,
            artifact_path: descriptor.artifact_path,
            state: ComponentDefinitionState::Registered,
        }
    }

    pub fn artifact_reference(&self) -> ComponentArtifactReference<'_> {
        ComponentArtifactReference::LocalPath(&self.artifact_path)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedComponent {
    definition_id: ComponentDefinitionId,
    engine_key: String,
    contract: ComponentContract,
}

impl PreparedComponent {
    pub fn new(definition_id: ComponentDefinitionId, engine_key: impl Into<String>) -> Self {
        Self::with_contract(definition_id, engine_key, ComponentContract::default())
    }

    pub fn with_contract(
        definition_id: ComponentDefinitionId,
        engine_key: impl Into<String>,
        contract: ComponentContract,
    ) -> Self {
        Self {
            definition_id,
            engine_key: engine_key.into(),
            contract,
        }
    }

    pub const fn definition_id(&self) -> ComponentDefinitionId {
        self.definition_id
    }

    pub fn engine_key(&self) -> &str {
        &self.engine_key
    }

    pub const fn contract(&self) -> &ComponentContract {
        &self.contract
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentEngineInstance {
    definition_id: ComponentDefinitionId,
    engine_key: String,
}

impl ComponentEngineInstance {
    pub fn new(definition_id: ComponentDefinitionId, engine_key: impl Into<String>) -> Self {
        Self {
            definition_id,
            engine_key: engine_key.into(),
        }
    }

    pub const fn definition_id(&self) -> ComponentDefinitionId {
        self.definition_id
    }

    pub fn engine_key(&self) -> &str {
        &self.engine_key
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentInstance {
    pub id: ComponentInstanceId,
    pub definition_id: ComponentDefinitionId,
    pub state: ComponentInstanceState,
    engine_instance: ComponentEngineInstance,
}

impl ComponentInstance {
    pub const fn engine_instance(&self) -> &ComponentEngineInstance {
        &self.engine_instance
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ComponentEndpoint {
    RuntimeService { interface: WitInterface },
    Capability { interface: WitInterface },
}

impl ComponentEndpoint {
    pub const fn interface(&self) -> &WitInterface {
        match self {
            Self::RuntimeService { interface } | Self::Capability { interface } => interface,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ComponentLinkPlan {
    links: BTreeMap<WitInterface, ComponentEndpoint>,
}

impl ComponentLinkPlan {
    pub fn links(&self) -> impl Iterator<Item = (&WitInterface, &ComponentEndpoint)> {
        self.links.iter()
    }

    pub fn endpoint(&self, interface: &WitInterface) -> Option<&ComponentEndpoint> {
        self.links.get(interface)
    }

    #[cfg(all(test, feature = "wasmtime-component-engine"))]
    pub(crate) fn insert_for_test(&mut self, endpoint: ComponentEndpoint) {
        self.links.insert(endpoint.interface().clone(), endpoint);
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentResourceLimits {
    pub max_memory_bytes: Option<u64>,
    pub execution_deadline_millis: Option<u64>,
    pub max_concurrent_invocations: Option<u32>,
    pub max_instances: Option<u32>,
    pub engine_execution_budget: Option<u64>,
    pub require_memory_limit: bool,
}

impl Default for ComponentResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_bytes: None,
            execution_deadline_millis: None,
            max_concurrent_invocations: Some(1),
            max_instances: None,
            engine_execution_budget: None,
            require_memory_limit: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComponentObservationKind {
    Validation,
    Preparation,
    LinkPlan,
    Instantiation,
    Invocation,
    Trap,
    Interruption,
    ResourceLimit,
    Destruction,
    Shutdown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentObservation {
    pub kind: ComponentObservationKind,
    pub component: Option<String>,
    pub instance: Option<ComponentInstanceId>,
    pub message: String,
}

impl ComponentObservation {
    pub fn new(
        kind: ComponentObservationKind,
        component: Option<String>,
        instance: Option<ComponentInstanceId>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            component,
            instance,
            message: message.into(),
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ComponentEngineCapabilities {
    pub component_model: bool,
    pub async_host_calls: bool,
    pub interruption: bool,
    pub resource_limits: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComponentTrapKind {
    Trap,
    Unreachable,
    MemoryFault,
    ResourceLimit,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComponentInterruptionReason {
    CallerCancelled,
    RuntimeShutdown,
    Deadline,
    ResourcePolicy,
    Administrative,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentInvocation {
    pub instance_id: ComponentInstanceId,
    pub interface: WitInterface,
    pub operation: String,
    pub deadline_millis: Option<u64>,
}

impl ComponentInvocation {
    pub fn new(
        instance_id: ComponentInstanceId,
        interface: WitInterface,
        operation: impl Into<String>,
    ) -> Self {
        Self {
            instance_id,
            interface,
            operation: operation.into(),
            deadline_millis: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ComponentValue {
    U32(u32),
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ComponentInvocationResult {
    pub values: Vec<ComponentValue>,
}

impl ComponentInvocationResult {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn single(value: ComponentValue) -> Self {
        Self {
            values: vec![value],
        }
    }
}

pub trait ComponentEngine: Send {
    fn capabilities(&self) -> ComponentEngineCapabilities;
    fn prepare(
        &mut self,
        definition: &ComponentDefinition,
        limits: &ComponentResourceLimits,
    ) -> Result<PreparedComponent, ComponentError>;
    fn instantiate(
        &mut self,
        prepared: &PreparedComponent,
        link_plan: &ComponentLinkPlan,
    ) -> Result<ComponentEngineInstance, ComponentError>;
    fn invoke(
        &mut self,
        instance: &ComponentEngineInstance,
        invocation: &ComponentInvocation,
    ) -> Result<ComponentInvocationResult, ComponentError>;
    fn interrupt(
        &mut self,
        instance: &ComponentEngineInstance,
        reason: ComponentInterruptionReason,
    ) -> Result<(), ComponentError>;
    fn destroy(&mut self, instance: ComponentEngineInstance) -> Result<(), ComponentError>;
}

#[derive(Default)]
pub struct MockComponentEngine {
    capabilities: ComponentEngineCapabilities,
    pub prepared: Vec<ComponentDefinitionId>,
    pub instantiated: Vec<ComponentDefinitionId>,
    pub destroyed: Vec<String>,
    pub invoked: Vec<ComponentInvocation>,
    pub interrupted: Vec<ComponentInterruptionReason>,
    pub fail_prepare: Option<String>,
    pub fail_instantiate: Option<String>,
    pub trap_on_invoke: Option<ComponentTrapKind>,
    pub interrupt_on_invoke: Option<ComponentInterruptionReason>,
    pub prepared_contract: Option<ComponentContract>,
}

impl MockComponentEngine {
    pub fn new() -> Self {
        Self {
            capabilities: ComponentEngineCapabilities {
                component_model: true,
                async_host_calls: true,
                interruption: true,
                resource_limits: true,
            },
            ..Self::default()
        }
    }

    pub fn without_resource_limits(mut self) -> Self {
        self.capabilities.resource_limits = false;
        self
    }
}

impl ComponentEngine for MockComponentEngine {
    fn capabilities(&self) -> ComponentEngineCapabilities {
        self.capabilities.clone()
    }

    fn prepare(
        &mut self,
        definition: &ComponentDefinition,
        limits: &ComponentResourceLimits,
    ) -> Result<PreparedComponent, ComponentError> {
        if limits.require_memory_limit && !self.capabilities.resource_limits {
            return Err(ComponentError::ResourceLimitUnsupported {
                component: definition.metadata.name.clone(),
                limit: "memory",
            });
        }
        if let Some(message) = &self.fail_prepare {
            return Err(ComponentError::PreparationFailed {
                component: definition.metadata.name.clone(),
                message: message.clone(),
            });
        }
        self.prepared.push(definition.id);
        Ok(PreparedComponent::with_contract(
            definition.id,
            format!("prepared:{}", definition.metadata.name),
            self.prepared_contract
                .clone()
                .unwrap_or_else(|| ComponentContract::from_metadata(&definition.metadata)),
        ))
    }

    fn instantiate(
        &mut self,
        prepared: &PreparedComponent,
        _link_plan: &ComponentLinkPlan,
    ) -> Result<ComponentEngineInstance, ComponentError> {
        if let Some(message) = &self.fail_instantiate {
            return Err(ComponentError::InstantiationFailed {
                definition: prepared.definition_id(),
                message: message.clone(),
            });
        }
        self.instantiated.push(prepared.definition_id());
        Ok(ComponentEngineInstance::new(
            prepared.definition_id(),
            format!("instance:{}", prepared.engine_key()),
        ))
    }

    fn invoke(
        &mut self,
        _instance: &ComponentEngineInstance,
        invocation: &ComponentInvocation,
    ) -> Result<ComponentInvocationResult, ComponentError> {
        self.invoked.push(invocation.clone());
        if let Some(reason) = self.interrupt_on_invoke {
            return Err(ComponentError::Interrupted {
                instance: invocation.instance_id,
                reason,
            });
        }
        if let Some(kind) = self.trap_on_invoke {
            return Err(ComponentError::Trap {
                instance: invocation.instance_id,
                kind,
                diagnostic: Some("[redacted component trap]".into()),
            });
        }
        Ok(ComponentInvocationResult::empty())
    }

    fn interrupt(
        &mut self,
        _instance: &ComponentEngineInstance,
        reason: ComponentInterruptionReason,
    ) -> Result<(), ComponentError> {
        self.interrupted.push(reason);
        Ok(())
    }

    fn destroy(&mut self, instance: ComponentEngineInstance) -> Result<(), ComponentError> {
        self.destroyed.push(instance.engine_key);
        Ok(())
    }
}

pub struct ComponentManager {
    engine: Box<dyn ComponentEngine>,
    host_interfaces: BTreeSet<WitInterface>,
    authorized_interfaces: BTreeSet<WitInterface>,
    definitions: BTreeMap<String, ComponentDefinition>,
    prepared: BTreeMap<ComponentDefinitionId, PreparedComponent>,
    instances: BTreeMap<ComponentInstanceId, ComponentInstance>,
    active_invocations: BTreeMap<ComponentInstanceId, u32>,
    observations: Vec<ComponentObservation>,
    limits: ComponentResourceLimits,
    shutdown: bool,
}

impl Default for ComponentManager {
    fn default() -> Self {
        Self::with_engine(Box::new(MockComponentEngine::new()))
    }
}

impl ComponentManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_engine(engine: Box<dyn ComponentEngine>) -> Self {
        Self {
            engine,
            host_interfaces: BTreeSet::new(),
            authorized_interfaces: BTreeSet::new(),
            definitions: BTreeMap::new(),
            prepared: BTreeMap::new(),
            instances: BTreeMap::new(),
            active_invocations: BTreeMap::new(),
            observations: Vec::new(),
            limits: ComponentResourceLimits::default(),
            shutdown: false,
        }
    }

    pub fn provide_interface(&mut self, interface: WitInterface) {
        self.host_interfaces.insert(interface.clone());
        self.authorized_interfaces.insert(interface);
    }

    pub fn authorize_interface(&mut self, interface: WitInterface) {
        self.authorized_interfaces.insert(interface);
    }

    pub fn set_resource_limits(&mut self, limits: ComponentResourceLimits) {
        self.limits = limits;
    }

    pub fn observations(&self) -> &[ComponentObservation] {
        &self.observations
    }

    pub fn register_component(
        &mut self,
        descriptor: ComponentDescriptor,
    ) -> Result<ComponentDefinitionId, ComponentError> {
        if self.shutdown {
            return Err(ComponentError::RuntimeShutdown);
        }
        if self.definitions.contains_key(&descriptor.metadata.name) {
            return Err(ComponentError::AlreadyRegistered(descriptor.metadata.name));
        }
        let definition = ComponentDefinition::registered(descriptor);
        let id = definition.id;
        self.definitions
            .insert(definition.metadata.name.clone(), definition);
        Ok(id)
    }

    pub fn definition(&self, name: &str) -> Option<&ComponentDefinition> {
        self.definitions.get(name)
    }

    pub fn definition_state(&self, name: &str) -> Option<ComponentDefinitionState> {
        self.definitions
            .get(name)
            .map(|definition| definition.state)
    }

    pub fn instance_state(&self, id: ComponentInstanceId) -> Option<ComponentInstanceState> {
        self.instances.get(&id).map(|instance| instance.state)
    }

    pub fn link_plan(&self, name: &str) -> Result<ComponentLinkPlan, ComponentError> {
        let definition = self
            .definitions
            .get(name)
            .ok_or_else(|| ComponentError::NotFound(name.into()))?;
        self.build_link_plan(definition)
    }

    pub fn prepare_component(
        &mut self,
        name: &str,
    ) -> Result<ComponentDefinitionId, ComponentError> {
        if self.shutdown {
            return Err(ComponentError::RuntimeShutdown);
        }
        let definition = self
            .definitions
            .get_mut(name)
            .ok_or_else(|| ComponentError::NotFound(name.into()))?;
        definition.state = ComponentDefinitionState::Validated;
        self.observations.push(ComponentObservation::new(
            ComponentObservationKind::Validation,
            Some(definition.metadata.name.clone()),
            None,
            "component definition validated",
        ));
        let prepared = match self.engine.prepare(definition, &self.limits) {
            Ok(prepared) => prepared,
            Err(error) => {
                definition.state = ComponentDefinitionState::Failed;
                self.observe_error(None, &error);
                return Err(error);
            }
        };
        if let Err(error) = validate_prepared_contract(definition, prepared.contract()) {
            definition.state = ComponentDefinitionState::Failed;
            self.observe_error(None, &error);
            return Err(error);
        }
        definition.state = ComponentDefinitionState::Prepared;
        let id = definition.id;
        self.prepared.insert(id, prepared);
        self.observations.push(ComponentObservation::new(
            ComponentObservationKind::Preparation,
            Some(definition.metadata.name.clone()),
            None,
            "component prepared by engine",
        ));
        Ok(id)
    }

    pub fn instantiate_component(
        &mut self,
        name: &str,
    ) -> Result<ComponentInstanceId, ComponentError> {
        if self.shutdown {
            return Err(ComponentError::RuntimeShutdown);
        }
        if let Some(max_instances) = self.limits.max_instances
            && self.instances.len() >= max_instances as usize
        {
            let error = ComponentError::ResourceLimitExceeded {
                component: name.into(),
                limit: "instances",
            };
            self.observe_error(None, &error);
            return Err(error);
        }
        let link_plan = self.link_plan(name)?;
        self.observations.push(ComponentObservation::new(
            ComponentObservationKind::LinkPlan,
            Some(name.into()),
            None,
            "runtime-owned link plan built",
        ));
        let definition_id = self.prepare_component(name)?;
        let prepared = self
            .prepared
            .get(&definition_id)
            .ok_or(ComponentError::MissingPreparedDefinition(definition_id))?;
        let engine_instance = self.engine.instantiate(prepared, &link_plan)?;
        let id = next_component_instance_id();
        self.instances.insert(
            id,
            ComponentInstance {
                id,
                definition_id,
                state: ComponentInstanceState::Ready,
                engine_instance,
            },
        );
        self.observations.push(ComponentObservation::new(
            ComponentObservationKind::Instantiation,
            Some(name.into()),
            Some(id),
            "component instance ready",
        ));
        Ok(id)
    }

    pub fn invoke(
        &mut self,
        invocation: ComponentInvocation,
    ) -> Result<ComponentInvocationResult, ComponentError> {
        if self.shutdown {
            return Err(ComponentError::RuntimeShutdown);
        }
        if invocation.deadline_millis.is_some() && !self.engine.capabilities().interruption {
            let error = ComponentError::ResourceLimitUnsupported {
                component: format!("instance:{}", invocation.instance_id.get()),
                limit: "deadline",
            };
            self.observe_error(Some(invocation.instance_id), &error);
            return Err(error);
        }
        let instance = self
            .instances
            .get(&invocation.instance_id)
            .ok_or(ComponentError::InstanceNotFound(invocation.instance_id))?;
        if instance.state != ComponentInstanceState::Ready {
            return Err(ComponentError::InvalidInstanceTransition {
                instance: invocation.instance_id,
                state: instance.state,
                operation: "invoke",
            });
        }
        let active = self
            .active_invocations
            .entry(invocation.instance_id)
            .or_default();
        if let Some(max) = self.limits.max_concurrent_invocations
            && *active >= max
        {
            let error = ComponentError::ResourceLimitExceeded {
                component: format!("instance:{}", invocation.instance_id.get()),
                limit: "concurrent invocations",
            };
            self.observe_error(Some(invocation.instance_id), &error);
            return Err(error);
        }
        *active += 1;
        let engine_instance = instance.engine_instance.clone();
        let result = self.engine.invoke(&engine_instance, &invocation);
        if let Some(active) = self.active_invocations.get_mut(&invocation.instance_id) {
            *active = active.saturating_sub(1);
            if *active == 0 {
                self.active_invocations.remove(&invocation.instance_id);
            }
        }
        match &result {
            Ok(_) => self.observations.push(ComponentObservation::new(
                ComponentObservationKind::Invocation,
                None,
                Some(invocation.instance_id),
                "component invocation completed",
            )),
            Err(error) => self.observe_error(Some(invocation.instance_id), error),
        }
        result
    }

    pub fn destroy_instance(&mut self, id: ComponentInstanceId) -> Result<(), ComponentError> {
        let mut instance = self
            .instances
            .remove(&id)
            .ok_or(ComponentError::InstanceNotFound(id))?;
        if instance.state == ComponentInstanceState::Destroyed {
            return Err(ComponentError::InvalidInstanceTransition {
                instance: id,
                state: instance.state,
                operation: "destroy",
            });
        }
        instance.state = ComponentInstanceState::Destroyed;
        self.active_invocations.remove(&id);
        let result = self.engine.destroy(instance.engine_instance);
        self.observations.push(ComponentObservation::new(
            ComponentObservationKind::Destruction,
            None,
            Some(id),
            "component instance destroyed",
        ));
        result
    }

    pub fn shutdown(&mut self) {
        self.shutdown = true;
        let ids = self.instances.keys().copied().collect::<Vec<_>>();
        for id in ids {
            if let Some(instance) = self.instances.get(&id) {
                let _ = self.engine.interrupt(
                    &instance.engine_instance,
                    ComponentInterruptionReason::RuntimeShutdown,
                );
            }
            let _ = self.destroy_instance(id);
        }
        self.prepared.clear();
        self.observations.push(ComponentObservation::new(
            ComponentObservationKind::Shutdown,
            None,
            None,
            "component manager shutdown",
        ));
    }

    pub fn discover(
        paths: impl IntoIterator<Item = impl AsRef<Path>>,
    ) -> Result<Vec<PathBuf>, ComponentError> {
        let mut found = BTreeSet::new();
        for directory in paths {
            let directory = directory.as_ref();
            for entry in
                std::fs::read_dir(directory).map_err(|source| ComponentError::Discovery {
                    path: directory.into(),
                    source,
                })?
            {
                let path = entry
                    .map_err(|source| ComponentError::Discovery {
                        path: directory.into(),
                        source,
                    })?
                    .path();
                if path.is_file()
                    && path.extension().and_then(|extension| extension.to_str()) == Some("wasm")
                {
                    found.insert(path);
                }
            }
        }
        Ok(found.into_iter().collect())
    }

    fn build_link_plan(
        &self,
        definition: &ComponentDefinition,
    ) -> Result<ComponentLinkPlan, ComponentError> {
        let mut links = BTreeMap::new();
        for interface in &definition.metadata.imports {
            if !self.authorized_interfaces.contains(interface) {
                return Err(ComponentError::UnauthorizedImport {
                    component: definition.metadata.name.clone(),
                    interface: interface.clone(),
                });
            }
            if !self.host_interfaces.contains(interface) {
                return Err(ComponentError::UnresolvedImport {
                    component: definition.metadata.name.clone(),
                    interface: interface.clone(),
                });
            }
            links.insert(
                interface.clone(),
                ComponentEndpoint::Capability {
                    interface: interface.clone(),
                },
            );
        }
        Ok(ComponentLinkPlan { links })
    }

    fn observe_error(&mut self, instance: Option<ComponentInstanceId>, error: &ComponentError) {
        let kind = match error {
            ComponentError::Trap { .. } => ComponentObservationKind::Trap,
            ComponentError::Interrupted { .. } => ComponentObservationKind::Interruption,
            ComponentError::ResourceLimitUnsupported { .. }
            | ComponentError::ResourceLimitExceeded { .. } => {
                ComponentObservationKind::ResourceLimit
            }
            ComponentError::InstantiationFailed { .. } => ComponentObservationKind::Instantiation,
            ComponentError::InvocationFailed { .. } => ComponentObservationKind::Invocation,
            ComponentError::PreparationFailed { .. }
            | ComponentError::ComponentLoadFailed { .. } => ComponentObservationKind::Preparation,
            ComponentError::UnauthorizedImport { .. }
            | ComponentError::UnresolvedImport { .. }
            | ComponentError::ContractValidationFailed { .. } => {
                ComponentObservationKind::Validation
            }
            ComponentError::RuntimeShutdown => ComponentObservationKind::Shutdown,
            _ => ComponentObservationKind::Validation,
        };
        self.observations.push(ComponentObservation::new(
            kind,
            None,
            instance,
            error.to_string(),
        ));
    }
}

fn validate_prepared_contract(
    definition: &ComponentDefinition,
    contract: &ComponentContract,
) -> Result<(), ComponentError> {
    let prepared_imports = contract.import_interfaces();
    for interface in &prepared_imports {
        if !definition.metadata.imports.contains(interface) {
            return Err(ComponentError::ContractValidationFailed {
                component: definition.metadata.name.clone(),
                message: format!(
                    "prepared Component imports undeclared interface '{}@{}'",
                    interface.name, interface.version
                ),
            });
        }
    }
    for interface in &definition.metadata.imports {
        if !prepared_imports.contains(interface) {
            return Err(ComponentError::ContractValidationFailed {
                component: definition.metadata.name.clone(),
                message: format!(
                    "declared import '{}@{}' was not found in prepared Component",
                    interface.name, interface.version
                ),
            });
        }
    }

    let prepared_exports = contract.export_interfaces();
    for interface in &definition.metadata.exports {
        if !prepared_exports.contains(interface) {
            return Err(ComponentError::ContractValidationFailed {
                component: definition.metadata.name.clone(),
                message: format!(
                    "declared export '{}@{}' was not found in prepared Component",
                    interface.name, interface.version
                ),
            });
        }
    }

    Ok(())
}

#[derive(Debug)]
pub enum ComponentError {
    AlreadyRegistered(String),
    NotFound(String),
    InstanceNotFound(ComponentInstanceId),
    MissingPreparedDefinition(ComponentDefinitionId),
    UnresolvedImport {
        component: String,
        interface: WitInterface,
    },
    UnauthorizedImport {
        component: String,
        interface: WitInterface,
    },
    PreparationFailed {
        component: String,
        message: String,
    },
    ContractValidationFailed {
        component: String,
        message: String,
    },
    InstantiationFailed {
        definition: ComponentDefinitionId,
        message: String,
    },
    InvocationFailed {
        instance: ComponentInstanceId,
        message: String,
    },
    Trap {
        instance: ComponentInstanceId,
        kind: ComponentTrapKind,
        diagnostic: Option<String>,
    },
    Interrupted {
        instance: ComponentInstanceId,
        reason: ComponentInterruptionReason,
    },
    ResourceLimitUnsupported {
        component: String,
        limit: &'static str,
    },
    ResourceLimitExceeded {
        component: String,
        limit: &'static str,
    },
    ComponentLoadFailed {
        path: PathBuf,
        message: String,
        source: Option<std::io::Error>,
    },
    InvalidInstanceTransition {
        instance: ComponentInstanceId,
        state: ComponentInstanceState,
        operation: &'static str,
    },
    EngineFailure(String),
    RuntimeShutdown,
    Discovery {
        path: PathBuf,
        source: std::io::Error,
    },
}
impl fmt::Display for ComponentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyRegistered(name) => write!(f, "component '{name}' is already registered"),
            Self::NotFound(name) => write!(f, "component '{name}' is not registered"),
            Self::InstanceNotFound(id) => {
                write!(f, "component instance '{}' is not registered", id.get())
            }
            Self::MissingPreparedDefinition(id) => {
                write!(
                    f,
                    "component definition '{}' has not been prepared",
                    id.get()
                )
            }
            Self::UnresolvedImport {
                component,
                interface,
            } => write!(
                f,
                "component '{component}' requires unresolved WIT interface '{}@{}'",
                interface.name, interface.version
            ),
            Self::UnauthorizedImport {
                component,
                interface,
            } => write!(
                f,
                "component '{component}' is not authorized to import WIT interface '{}@{}'",
                interface.name, interface.version
            ),
            Self::PreparationFailed { component, message } => {
                write!(f, "component '{component}' preparation failed: {message}")
            }
            Self::ContractValidationFailed { component, message } => {
                write!(
                    f,
                    "component '{component}' contract validation failed: {message}"
                )
            }
            Self::InstantiationFailed {
                definition,
                message,
            } => write!(
                f,
                "component definition '{}' instantiation failed: {message}",
                definition.get()
            ),
            Self::InvocationFailed { instance, message } => write!(
                f,
                "component instance '{}' invocation failed: {message}",
                instance.get()
            ),
            Self::Trap {
                instance,
                kind,
                diagnostic,
            } => {
                write!(
                    f,
                    "component instance '{}' trapped as {kind:?}",
                    instance.get()
                )?;
                if let Some(diagnostic) = diagnostic {
                    write!(f, ": {diagnostic}")?;
                }
                Ok(())
            }
            Self::Interrupted { instance, reason } => write!(
                f,
                "component instance '{}' was interrupted: {reason:?}",
                instance.get()
            ),
            Self::ResourceLimitUnsupported { component, limit } => write!(
                f,
                "component '{component}' requires unsupported resource limit '{limit}'"
            ),
            Self::ResourceLimitExceeded { component, limit } => write!(
                f,
                "component '{component}' exceeded resource limit '{limit}'"
            ),
            Self::ComponentLoadFailed { path, message, .. } => write!(
                f,
                "component artifact '{}' could not be loaded: {message}",
                path.display()
            ),
            Self::InvalidInstanceTransition {
                instance,
                state,
                operation,
            } => write!(
                f,
                "cannot {operation} component instance '{}' from state {state:?}",
                instance.get()
            ),
            Self::EngineFailure(message) => write!(f, "component engine failed: {message}"),
            Self::RuntimeShutdown => write!(f, "component runtime is shut down"),
            Self::Discovery { path, source } => write!(
                f,
                "could not discover components in '{}': {source}",
                path.display()
            ),
        }
    }
}
impl Error for ComponentError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ComponentLoadFailed {
                source: Some(source),
                ..
            } => Some(source),
            Self::Discovery { source, .. } => Some(source),
            _ => None,
        }
    }
}
