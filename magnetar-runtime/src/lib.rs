//! Hardware-agnostic runtime contracts and provider support for Magnetar.

use libloading::Library;
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    error::Error,
    fmt,
    path::{Path, PathBuf},
    sync::Arc,
};

pub const PROVIDER_API_VERSION: u32 = 1;

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

/// Declarative metadata for a portable WebAssembly component.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub imports: BTreeSet<WitInterface>,
    pub exports: BTreeSet<WitInterface>,
    pub dependencies: BTreeSet<String>,
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
            dependencies: BTreeSet::new(),
        }
    }
}

/// A discovered component artifact and its declared metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentDescriptor {
    pub metadata: ComponentMetadata,
    pub artifact_path: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComponentState {
    Registered,
    Instantiated,
    Started,
    Stopped,
}

/// A portable component lifecycle, implemented by a future WASM-engine adapter.
pub trait Component: Send {
    fn metadata(&self) -> ComponentMetadata;
    fn instantiate(&mut self) -> Result<(), ComponentError>;
    fn start(&mut self) -> Result<(), ComponentError>;
    fn stop(&mut self) -> Result<(), ComponentError>;
    fn destroy(&mut self) -> Result<(), ComponentError>;
}

struct ManagedComponent {
    component: Box<dyn Component>,
    metadata: ComponentMetadata,
    state: ComponentState,
}

/// Owns components, validates their contracts, and drives their lifecycle.
#[derive(Default)]
pub struct ComponentManager {
    host_interfaces: BTreeSet<WitInterface>,
    components: BTreeMap<String, ManagedComponent>,
    start_order: Vec<String>,
}
impl ComponentManager {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn provide_interface(&mut self, interface: WitInterface) {
        self.host_interfaces.insert(interface);
    }
    pub fn register_component(
        &mut self,
        component: Box<dyn Component>,
    ) -> Result<(), ComponentError> {
        let metadata = component.metadata();
        if self.components.contains_key(&metadata.name) {
            return Err(ComponentError::AlreadyRegistered(metadata.name));
        }
        self.components.insert(
            metadata.name.clone(),
            ManagedComponent {
                component,
                metadata,
                state: ComponentState::Registered,
            },
        );
        Ok(())
    }
    pub fn component_state(&self, name: &str) -> Option<ComponentState> {
        self.components.get(name).map(|component| component.state)
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
    pub fn validate_interfaces(&self) -> Result<(), ComponentError> {
        let available = self
            .host_interfaces
            .iter()
            .chain(
                self.components
                    .values()
                    .flat_map(|component| component.metadata.exports.iter()),
            )
            .collect::<BTreeSet<_>>();
        for component in self.components.values() {
            for interface in &component.metadata.imports {
                if !available.contains(interface) {
                    return Err(ComponentError::MissingInterface {
                        component: component.metadata.name.clone(),
                        interface: interface.clone(),
                    });
                }
            }
        }
        Ok(())
    }
    pub fn resolve_dependencies(&self) -> Result<Vec<String>, ComponentError> {
        fn visit(
            name: &str,
            components: &BTreeMap<String, ManagedComponent>,
            visiting: &mut BTreeSet<String>,
            visited: &mut BTreeSet<String>,
            order: &mut Vec<String>,
        ) -> Result<(), ComponentError> {
            if visited.contains(name) {
                return Ok(());
            }
            if !visiting.insert(name.into()) {
                return Err(ComponentError::DependencyCycle(name.into()));
            }
            let component =
                components
                    .get(name)
                    .ok_or_else(|| ComponentError::MissingDependency {
                        component: name.into(),
                        dependency: name.into(),
                    })?;
            for dependency in &component.metadata.dependencies {
                if !components.contains_key(dependency) {
                    return Err(ComponentError::MissingDependency {
                        component: name.into(),
                        dependency: dependency.clone(),
                    });
                }
                visit(dependency, components, visiting, visited, order)?;
            }
            visiting.remove(name);
            visited.insert(name.into());
            order.push(name.into());
            Ok(())
        }

        let mut order = Vec::new();
        let mut visiting = BTreeSet::new();
        let mut visited = BTreeSet::new();
        for name in self.components.keys() {
            visit(
                name,
                &self.components,
                &mut visiting,
                &mut visited,
                &mut order,
            )?;
        }
        Ok(order)
    }
    pub fn instantiate_component(&mut self, name: &str) -> Result<(), ComponentError> {
        let component = self.component_mut(name)?;
        match component.state {
            ComponentState::Registered => {
                component.component.instantiate()?;
                component.state = ComponentState::Instantiated;
                Ok(())
            }
            state => Err(ComponentError::InvalidTransition {
                name: name.into(),
                state,
                operation: "instantiate",
            }),
        }
    }
    pub fn start_all(&mut self) -> Result<(), ComponentError> {
        self.validate_interfaces()?;
        let order = self.resolve_dependencies()?;
        for name in order {
            if self.component_state(&name) == Some(ComponentState::Registered) {
                self.instantiate_component(&name)?;
            }
            let component = self.component_mut(&name)?;
            if component.state != ComponentState::Instantiated
                && component.state != ComponentState::Stopped
            {
                return Err(ComponentError::InvalidTransition {
                    name,
                    state: component.state,
                    operation: "start",
                });
            }
            if let Err(error) = component.component.start() {
                self.stop_started();
                return Err(error);
            }
            component.state = ComponentState::Started;
            self.start_order.push(name);
        }
        Ok(())
    }
    pub fn stop_started(&mut self) {
        for name in std::mem::take(&mut self.start_order).into_iter().rev() {
            if let Some(component) = self.components.get_mut(&name) {
                let _ = component.component.stop();
                component.state = ComponentState::Stopped;
            }
        }
    }
    pub fn destroy_component(&mut self, name: &str) -> Result<(), ComponentError> {
        if self.component_state(name) == Some(ComponentState::Started) {
            self.stop_started();
        }
        let mut component = self
            .components
            .remove(name)
            .ok_or_else(|| ComponentError::NotFound(name.into()))?;
        component.component.destroy()
    }
    pub fn shutdown(&mut self) {
        self.stop_started();
        let names = self.components.keys().cloned().collect::<Vec<_>>();
        for name in names {
            let _ = self.destroy_component(&name);
        }
    }
    fn component_mut(&mut self, name: &str) -> Result<&mut ManagedComponent, ComponentError> {
        self.components
            .get_mut(name)
            .ok_or_else(|| ComponentError::NotFound(name.into()))
    }
}

#[derive(Debug)]
pub enum ComponentError {
    AlreadyRegistered(String),
    NotFound(String),
    MissingDependency {
        component: String,
        dependency: String,
    },
    DependencyCycle(String),
    MissingInterface {
        component: String,
        interface: WitInterface,
    },
    InvalidTransition {
        name: String,
        state: ComponentState,
        operation: &'static str,
    },
    Discovery {
        path: PathBuf,
        source: std::io::Error,
    },
    Lifecycle(String),
}
impl fmt::Display for ComponentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyRegistered(name) => write!(f, "component '{name}' is already registered"),
            Self::NotFound(name) => write!(f, "component '{name}' is not registered"),
            Self::MissingDependency {
                component,
                dependency,
            } => write!(
                f,
                "component '{component}' depends on unavailable component '{dependency}'"
            ),
            Self::DependencyCycle(name) => {
                write!(f, "component dependency cycle includes '{name}'")
            }
            Self::MissingInterface {
                component,
                interface,
            } => write!(
                f,
                "component '{component}' requires unavailable WIT interface '{}@{}'",
                interface.name, interface.version
            ),
            Self::InvalidTransition {
                name,
                state,
                operation,
            } => write!(
                f,
                "cannot {operation} component '{name}' from state {state:?}"
            ),
            Self::Discovery { path, source } => write!(
                f,
                "could not discover components in '{}': {source}",
                path.display()
            ),
            Self::Lifecycle(message) => {
                write!(f, "component lifecycle operation failed: {message}")
            }
        }
    }
}
impl Error for ComponentError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        if let Self::Discovery { source, .. } = self {
            Some(source)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DeviceId(String);
impl DeviceId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DeviceType {
    Cpu,
    Gpu,
    Npu,
    Tpu,
    Other,
}

/// Immutable metadata describing a hardware execution target.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceMetadata {
    pub id: DeviceId,
    pub name: String,
    pub device_type: DeviceType,
    pub vendor: String,
    pub architecture: String,
    pub memory_capacity: u64,
    pub compute_units: u32,
    pub execution_capabilities: BTreeSet<CapabilityId>,
    /// Stable name of the Provider that discovered this device.
    pub provider: String,
}
impl DeviceMetadata {
    pub fn new(
        id: DeviceId,
        name: impl Into<String>,
        device_type: DeviceType,
        provider: impl Into<String>,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            device_type,
            vendor: String::new(),
            architecture: String::new(),
            memory_capacity: 0,
            compute_units: 0,
            execution_capabilities: BTreeSet::new(),
            provider: provider.into(),
        }
    }
}

/// A reusable concrete device implementation backed by [`DeviceMetadata`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceDescriptor {
    pub metadata: DeviceMetadata,
}
impl DeviceDescriptor {
    pub fn new(metadata: DeviceMetadata) -> Self {
        Self { metadata }
    }
}
pub trait Device: Send + Sync {
    fn metadata(&self) -> &DeviceMetadata;
    fn id(&self) -> &DeviceId {
        &self.metadata().id
    }
    fn device_type(&self) -> DeviceType {
        self.metadata().device_type
    }
    fn availability(&self) -> DeviceAvailability {
        DeviceAvailability::Available
    }
}
impl Device for DeviceDescriptor {
    fn metadata(&self) -> &DeviceMetadata {
        &self.metadata
    }
}

/// A hardware execution contribution. It is distinct from [`Provider`].
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    fn devices(&self) -> Vec<Arc<dyn Device>>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderMetadata {
    pub name: String,
    pub version: String,
    pub vendor: String,
    pub api_version: u32,
    pub description: String,
    pub capabilities: BTreeSet<Capability>,
    pub compute_advertisement: ProviderComputeAdvertisement,
    pub compute_operation_schema_support: BTreeMap<ComputeOperationId, ComputeOperationSupport>,
    pub compute_operation_support: BTreeMap<ComputeOperationFamily, ComputeOperationSupport>,
    pub compute_data_movement_support:
        BTreeMap<ComputeDataMovementKind, ComputeDataMovementSupport>,
}
impl ProviderMetadata {
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        vendor: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            vendor: vendor.into(),
            api_version: PROVIDER_API_VERSION,
            description: description.into(),
            capabilities: BTreeSet::new(),
            compute_advertisement: ProviderComputeAdvertisement::default(),
            compute_operation_schema_support: BTreeMap::new(),
            compute_operation_support: BTreeMap::new(),
            compute_data_movement_support: BTreeMap::new(),
        }
    }
}

/// A discovered provider library and the metadata it declares.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderDescriptor {
    pub metadata: ProviderMetadata,
    pub artifact_path: PathBuf,
}

/// A globally unique, package-qualified capability identifier.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CapabilityId(String);
impl CapabilityId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl fmt::Display for CapabilityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// A semantic capability version.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CapabilityVersion {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
}
impl CapabilityVersion {
    pub const fn new(major: u64, minor: u64, patch: u64) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }
    /// Whether this available version satisfies `required`.
    pub fn is_compatible_with(&self, required: Self) -> bool {
        if self.major != required.major {
            return false;
        }
        if self.major == 0 {
            return self == &required;
        }
        self >= &required
    }
}
impl fmt::Display for CapabilityVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Declarative contracts and dependencies of a capability.
#[derive(Clone, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct CapabilityDescriptor {
    pub description: String,
    pub contracts: BTreeSet<WitInterface>,
    pub dependencies: BTreeMap<CapabilityId, CapabilityVersion>,
}
impl CapabilityDescriptor {
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            ..Self::default()
        }
    }
    pub fn with_contract(mut self, contract: WitInterface) -> Self {
        self.contracts.insert(contract);
        self
    }
    pub fn with_dependency(mut self, id: CapabilityId, version: CapabilityVersion) -> Self {
        self.dependencies.insert(id, version);
        self
    }
}

/// A versioned, independently registered runtime capability contract.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Capability {
    pub id: CapabilityId,
    pub version: CapabilityVersion,
    pub descriptor: CapabilityDescriptor,
}
impl Capability {
    pub fn new(
        id: CapabilityId,
        version: CapabilityVersion,
        descriptor: CapabilityDescriptor,
    ) -> Self {
        Self {
            id,
            version,
            descriptor,
        }
    }
    pub fn from_wit(interface: WitInterface) -> Result<Self, ProviderError> {
        let version = parse_capability_version(&interface.version)?;
        Ok(Self::new(
            CapabilityId::new(&interface.name),
            version,
            CapabilityDescriptor::default().with_contract(interface),
        ))
    }
}

/// Process-local identity of a Runtime execution context.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ExecutionContextId(u64);
impl ExecutionContextId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}
impl fmt::Display for ExecutionContextId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Process-local identity for resources resolved as one affinity cohort.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AffinityGroupId(u64);
impl AffinityGroupId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}
impl fmt::Display for AffinityGroupId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Stable name of the Provider that owns a live resource in one Runtime.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProviderBinding(String);
impl ProviderBinding {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl fmt::Display for ProviderBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Globally unique Device selected for a device-resident resource.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DeviceBinding(DeviceId);
impl DeviceBinding {
    pub fn new(id: DeviceId) -> Self {
        Self(id)
    }
    pub fn id(&self) -> &DeviceId {
        &self.0
    }
}
impl fmt::Display for DeviceBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Exact Capability implementation that created or constrains a live resource.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CapabilityBinding {
    id: CapabilityId,
    version: CapabilityVersion,
}
impl CapabilityBinding {
    pub fn new(id: CapabilityId, version: CapabilityVersion) -> Self {
        Self { id, version }
    }
    pub fn id(&self) -> &CapabilityId {
        &self.id
    }
    pub const fn version(&self) -> CapabilityVersion {
        self.version
    }
}
impl fmt::Display for CapabilityBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.id, self.version)
    }
}

/// Canonical content fingerprint attached under a semantic artifact role.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ArtifactBinding {
    role: String,
    fingerprint: String,
}
impl ArtifactBinding {
    pub fn new(role: impl Into<String>, fingerprint: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            fingerprint: fingerprint.into(),
        }
    }
    pub fn role(&self) -> &str {
        &self.role
    }
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
}

/// Recovery classification for state associated with an affinity.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum FallbackClass {
    Transparent,
    Restartable,
    ProviderPinned,
}

/// Logical point at which resolution or re-resolution is being considered.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ExecutionPhase {
    #[default]
    BeforeResourceCreation,
    AfterResourceCreation,
    AfterSubmittedWork,
    AfterObservableOutput,
}

/// Stable health category reported by the host-facing Provider wrapper.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProviderHealth {
    #[default]
    Healthy,
    Degraded,
    Unavailable,
}

/// Stable availability category for a candidate Device.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DeviceAvailability {
    #[default]
    Available,
    Busy,
    Unavailable,
}

/// Immutable ownership and compatibility facts carried by one opaque resource.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceAffinity {
    provider: Option<ProviderBinding>,
    device: Option<DeviceBinding>,
    capabilities: BTreeMap<CapabilityId, CapabilityBinding>,
    artifacts: BTreeMap<String, ArtifactBinding>,
    execution_context: Option<ExecutionContextId>,
    group: Option<AffinityGroupId>,
    fallback: FallbackClass,
}
impl ResourceAffinity {
    pub fn new(fallback: FallbackClass) -> Self {
        Self {
            provider: None,
            device: None,
            capabilities: BTreeMap::new(),
            artifacts: BTreeMap::new(),
            execution_context: None,
            group: None,
            fallback,
        }
    }
    pub fn with_provider(mut self, binding: ProviderBinding) -> Self {
        self.provider = Some(binding);
        self
    }
    pub fn with_device(mut self, binding: DeviceBinding) -> Self {
        self.device = Some(binding);
        self
    }
    pub fn with_capability(mut self, binding: CapabilityBinding) -> Self {
        self.capabilities.insert(binding.id.clone(), binding);
        self
    }
    pub fn with_artifact(mut self, binding: ArtifactBinding) -> Self {
        self.artifacts.insert(binding.role.clone(), binding);
        self
    }
    pub fn with_execution_context(mut self, id: ExecutionContextId) -> Self {
        self.execution_context = Some(id);
        self
    }
    pub fn with_group(mut self, id: AffinityGroupId) -> Self {
        self.group = Some(id);
        self
    }
    pub fn with_fallback(mut self, fallback: FallbackClass) -> Self {
        self.fallback = fallback;
        self
    }
    pub fn provider(&self) -> Option<&ProviderBinding> {
        self.provider.as_ref()
    }
    pub fn device(&self) -> Option<&DeviceBinding> {
        self.device.as_ref()
    }
    pub fn capability(&self, id: &CapabilityId) -> Option<&CapabilityBinding> {
        self.capabilities.get(id)
    }
    pub fn capabilities(&self) -> impl Iterator<Item = &CapabilityBinding> {
        self.capabilities.values()
    }
    pub fn artifact(&self, role: &str) -> Option<&ArtifactBinding> {
        self.artifacts.get(role)
    }
    pub fn artifacts(&self) -> impl Iterator<Item = &ArtifactBinding> {
        self.artifacts.values()
    }
    pub const fn execution_context(&self) -> Option<ExecutionContextId> {
        self.execution_context
    }
    pub const fn group(&self) -> Option<AffinityGroupId> {
        self.group
    }
    pub const fn fallback(&self) -> FallbackClass {
        self.fallback
    }
    pub fn validate_with(&self, other: &Self) -> Result<(), AffinityError> {
        AffinityConstraints::try_from_affinities([self, other]).map(|_| ())
    }
}

/// A conflict-checked aggregation of all affinities consumed by one call.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AffinityConstraints {
    affinity: ResourceAffinity,
}
impl AffinityConstraints {
    pub fn new(fallback: FallbackClass) -> Self {
        Self {
            affinity: ResourceAffinity::new(fallback),
        }
    }
    pub fn try_from_affinities<'a>(
        affinities: impl IntoIterator<Item = &'a ResourceAffinity>,
    ) -> Result<Self, AffinityError> {
        let mut constraints = Self::new(FallbackClass::Transparent);
        for affinity in affinities {
            constraints.merge(affinity)?;
        }
        Ok(constraints)
    }
    pub fn affinity(&self) -> &ResourceAffinity {
        &self.affinity
    }
    pub fn into_affinity(self) -> ResourceAffinity {
        self.affinity
    }
    pub fn require_fallback(&mut self, fallback: FallbackClass) {
        self.affinity.fallback = self.affinity.fallback.max(fallback);
    }
    fn merge(&mut self, incoming: &ResourceAffinity) -> Result<(), AffinityError> {
        if let Some(found) = &incoming.provider {
            match &self.affinity.provider {
                Some(expected) if expected != found => {
                    return Err(AffinityError::ProviderMismatch {
                        expected: expected.clone(),
                        found: found.clone(),
                    });
                }
                None => self.affinity.provider = Some(found.clone()),
                _ => {}
            }
        }
        if let Some(found) = &incoming.device {
            match &self.affinity.device {
                Some(expected) if expected != found => {
                    return Err(AffinityError::DeviceMismatch {
                        expected: expected.clone(),
                        found: found.clone(),
                    });
                }
                None => self.affinity.device = Some(found.clone()),
                _ => {}
            }
        }
        if let Some(found) = incoming.execution_context {
            match self.affinity.execution_context {
                Some(expected) if expected != found => {
                    return Err(AffinityError::ExecutionContextMismatch { expected, found });
                }
                None => self.affinity.execution_context = Some(found),
                _ => {}
            }
        }
        if let Some(found) = incoming.group {
            match self.affinity.group {
                Some(expected) if expected != found => {
                    return Err(AffinityError::AffinityGroupMismatch { expected, found });
                }
                None => self.affinity.group = Some(found),
                _ => {}
            }
        }
        for (id, found) in &incoming.capabilities {
            match self.affinity.capabilities.get(id) {
                Some(expected) if expected != found => {
                    return Err(AffinityError::CapabilityMismatch {
                        id: id.clone(),
                        expected: expected.version,
                        found: found.version,
                    });
                }
                None => {
                    self.affinity.capabilities.insert(id.clone(), found.clone());
                }
                _ => {}
            }
        }
        for (role, found) in &incoming.artifacts {
            match self.affinity.artifacts.get(role) {
                Some(expected) if expected != found => {
                    return Err(AffinityError::ArtifactMismatch {
                        role: role.clone(),
                        expected: expected.fingerprint.clone(),
                        found: found.fingerprint.clone(),
                    });
                }
                None => {
                    self.affinity.artifacts.insert(role.clone(), found.clone());
                }
                _ => {}
            }
        }
        self.affinity.fallback = self.affinity.fallback.max(incoming.fallback);
        Ok(())
    }
}

/// Structured validation and constrained-resolution failures for affinities.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AffinityError {
    ProviderMismatch {
        expected: ProviderBinding,
        found: ProviderBinding,
    },
    DeviceMismatch {
        expected: DeviceBinding,
        found: DeviceBinding,
    },
    CapabilityMismatch {
        id: CapabilityId,
        expected: CapabilityVersion,
        found: CapabilityVersion,
    },
    ArtifactMismatch {
        role: String,
        expected: String,
        found: String,
    },
    ExecutionContextMismatch {
        expected: ExecutionContextId,
        found: ExecutionContextId,
    },
    AffinityGroupMismatch {
        expected: AffinityGroupId,
        found: AffinityGroupId,
    },
    BoundProviderUnavailable(ProviderBinding),
    BoundDeviceUnavailable(DeviceBinding),
    DeviceProviderMismatch {
        device: DeviceBinding,
        provider: ProviderBinding,
        owner: ProviderBinding,
    },
    ProviderDoesNotImplementCapability {
        provider: ProviderBinding,
        capability: CapabilityBinding,
    },
    NoCompatibleProvider(CapabilityBinding),
    PolicyRejectedProvider {
        capability: CapabilityBinding,
        policy: ResolutionPolicyId,
    },
    RuntimeNotInitialized,
}
impl fmt::Display for AffinityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProviderMismatch { expected, found } => write!(
                f,
                "resource provider mismatch: expected '{expected}', found '{found}'"
            ),
            Self::DeviceMismatch { expected, found } => write!(
                f,
                "resource device mismatch: expected '{expected}', found '{found}'"
            ),
            Self::CapabilityMismatch {
                id,
                expected,
                found,
            } => write!(
                f,
                "resource capability mismatch for '{id}': expected {expected}, found {found}"
            ),
            Self::ArtifactMismatch {
                role,
                expected,
                found,
            } => write!(
                f,
                "resource artifact mismatch for role '{role}': expected '{expected}', found '{found}'"
            ),
            Self::ExecutionContextMismatch { expected, found } => write!(
                f,
                "resource execution-context mismatch: expected {expected}, found {found}"
            ),
            Self::AffinityGroupMismatch { expected, found } => write!(
                f,
                "resource affinity-group mismatch: expected {expected}, found {found}"
            ),
            Self::BoundProviderUnavailable(provider) => {
                write!(f, "bound provider '{provider}' is unavailable")
            }
            Self::BoundDeviceUnavailable(device) => {
                write!(f, "bound device '{device}' is unavailable")
            }
            Self::DeviceProviderMismatch {
                device,
                provider,
                owner,
            } => write!(
                f,
                "bound device '{device}' belongs to provider '{owner}', not '{provider}'"
            ),
            Self::ProviderDoesNotImplementCapability {
                provider,
                capability,
            } => write!(
                f,
                "bound provider '{provider}' does not implement compatible capability '{capability}'"
            ),
            Self::NoCompatibleProvider(capability) => {
                write!(
                    f,
                    "no Provider implements compatible capability '{capability}'"
                )
            }
            Self::PolicyRejectedProvider { capability, policy } => write!(
                f,
                "resolution policy '{policy}' rejected every Provider for capability '{capability}'"
            ),
            Self::RuntimeNotInitialized => {
                write!(f, "runtime is not initialized for affinity resolution")
            }
        }
    }
}
impl Error for AffinityError {}

impl From<AffinityError> for ComputeError {
    fn from(error: AffinityError) -> Self {
        let message = error.to_string();
        match error {
            AffinityError::ProviderMismatch { expected, found } => ComputeError::new(
                ComputeErrorCode::ProviderPinnedResource,
                ComputeErrorPhase::AffinityValidation,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(
                ComputeDiagnostic::new()
                    .with_provider(found)
                    .with_rejected_candidate(expected),
            )
            .with_recovery_hint(RecoveryHint::ProviderPinned),
            AffinityError::DeviceMismatch { expected: _, found } => ComputeError::new(
                ComputeErrorCode::DeviceBoundResource,
                ComputeErrorPhase::AffinityValidation,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_device(found))
            .with_recovery_hint(RecoveryHint::ExplicitTransferRequired),
            AffinityError::CapabilityMismatch {
                id,
                expected,
                found: _,
            } => ComputeError::new(
                ComputeErrorCode::CapabilityVersionMismatch,
                ComputeErrorPhase::AffinityValidation,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(
                ComputeDiagnostic::new().with_capability(CapabilityBinding::new(id, expected)),
            ),
            AffinityError::ArtifactMismatch { .. } => ComputeError::new(
                ComputeErrorCode::ArtifactFingerprintMismatch,
                ComputeErrorPhase::AffinityValidation,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::ProviderPinned),
            AffinityError::ExecutionContextMismatch { .. } => ComputeError::new(
                ComputeErrorCode::ProviderPinnedResource,
                ComputeErrorPhase::Interruption,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::ProviderPinned),
            AffinityError::AffinityGroupMismatch { .. } => ComputeError::new(
                ComputeErrorCode::AffinityGroupMismatch,
                ComputeErrorPhase::AffinityValidation,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::ExplicitTransferRequired),
            AffinityError::BoundProviderUnavailable(provider) => ComputeError::new(
                ComputeErrorCode::ProviderUnavailable,
                ComputeErrorPhase::Interruption,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_provider(provider))
            .with_recovery_hint(RecoveryHint::ProviderPinned),
            AffinityError::BoundDeviceUnavailable(device) => ComputeError::new(
                ComputeErrorCode::DeviceUnavailable,
                ComputeErrorPhase::Interruption,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_device(device))
            .with_recovery_hint(RecoveryHint::ProviderPinned),
            AffinityError::DeviceProviderMismatch {
                device,
                provider,
                owner,
            } => ComputeError::new(
                ComputeErrorCode::DeviceBoundResource,
                ComputeErrorPhase::AffinityValidation,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(
                ComputeDiagnostic::new()
                    .with_device(device)
                    .with_provider(owner)
                    .with_rejected_candidate(provider),
            ),
            AffinityError::ProviderDoesNotImplementCapability {
                provider,
                capability,
            } => ComputeError::new(
                ComputeErrorCode::UnsupportedOperation,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(
                ComputeDiagnostic::new()
                    .with_provider(provider)
                    .with_capability(capability),
            ),
            AffinityError::NoCompatibleProvider(capability) => ComputeError::new(
                ComputeErrorCode::NoCompatibleProvider,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_capability(capability)),
            AffinityError::PolicyRejectedProvider {
                capability,
                policy: _,
            } => ComputeError::new(
                ComputeErrorCode::PolicyRejectedProvider,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_capability(capability)),
            AffinityError::RuntimeNotInitialized => ComputeError::new(
                ComputeErrorCode::ProviderUnavailable,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_recovery_hint(RecoveryHint::RetryBeforeState),
        }
    }
}

/// Host-side opaque value paired with immutable Resource Affinity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AffinityResource<T> {
    value: T,
    affinity: ResourceAffinity,
}
impl<T> AffinityResource<T> {
    pub fn new(value: T, affinity: ResourceAffinity) -> Self {
        Self { value, affinity }
    }
    pub fn value(&self) -> &T {
        &self.value
    }
    pub fn affinity(&self) -> &ResourceAffinity {
        &self.affinity
    }
    pub fn into_parts(self) -> (T, ResourceAffinity) {
        (self.value, self.affinity)
    }
}

/// One coherent Provider and Capability selection plus affinity for its output.
pub struct AffinityResolution<'a> {
    provider: &'a dyn Provider,
    capability: &'a Capability,
    affinity: ResourceAffinity,
    decision: ResolutionDecision,
}
impl<'a> AffinityResolution<'a> {
    pub fn provider(&self) -> &'a dyn Provider {
        self.provider
    }
    pub fn capability(&self) -> &'a Capability {
        self.capability
    }
    pub fn affinity(&self) -> &ResourceAffinity {
        &self.affinity
    }
    pub fn decision(&self) -> &ResolutionDecision {
        &self.decision
    }
    pub fn into_affinity(self) -> ResourceAffinity {
        self.affinity
    }
}

/// Stable identifier for a resolution policy.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ResolutionPolicyId(String);
impl ResolutionPolicyId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl fmt::Display for ResolutionPolicyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Built-in runtime policy families. Preference placeholders currently use the
/// deterministic ordering after applying their eligibility gates.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum BuiltInResolutionPolicy {
    #[default]
    Deterministic,
    Priority,
    Availability,
    PerformancePreferred,
    EnergyPreferred,
    MemoryConstrained,
}
impl BuiltInResolutionPolicy {
    pub fn id(self) -> ResolutionPolicyId {
        let id = match self {
            Self::Deterministic => "magnetar:policy/deterministic",
            Self::Priority => "magnetar:policy/priority",
            Self::Availability => "magnetar:policy/availability",
            Self::PerformancePreferred => "magnetar:policy/performance-preferred",
            Self::EnergyPreferred => "magnetar:policy/energy-preferred",
            Self::MemoryConstrained => "magnetar:policy/memory-constrained",
        };
        ResolutionPolicyId::new(id)
    }
}

/// A stable, inspectable candidate considered by a [`ResolutionPolicy`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolutionCandidate {
    pub provider: ProviderBinding,
    pub capability: CapabilityBinding,
    pub device: Option<DeviceBinding>,
    pub provider_health: ProviderHealth,
    pub device_availability: DeviceAvailability,
    pub affinity_compatible: bool,
    pub priority: i32,
}
impl ResolutionCandidate {
    fn sort_key(&self) -> (&ProviderBinding, &CapabilityBinding, Option<&DeviceBinding>) {
        (&self.provider, &self.capability, self.device.as_ref())
    }
}

/// Policy input assembled by the Runtime before execution begins.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolutionContext {
    pub requested_capability: CapabilityId,
    pub requested_version: CapabilityVersion,
    pub candidates: Vec<ResolutionCandidate>,
    pub affinity: Option<ResourceAffinity>,
    pub fallback: FallbackClass,
    pub execution_phase: ExecutionPhase,
    pub replayable_input: bool,
}

/// Stable reason a candidate was rejected before or by policy selection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResolutionRejectionReason {
    IncompatibleCapability,
    ProviderUnavailable,
    DeviceUnavailable,
    AffinityIncompatible,
    FallbackNotAllowed,
    PolicyRejected,
}

/// Stable rejection record; backend-specific strings are intentionally absent.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolutionCandidateRejection {
    pub provider: ProviderBinding,
    pub capability: CapabilityBinding,
    pub reason: ResolutionRejectionReason,
}

/// Stable decision reason emitted by policy evaluation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResolutionDecisionReason {
    SelectedDeterministically,
    SelectedByPriority,
    SelectedByAvailability,
    SelectedByPreferencePlaceholder,
    PreservedAffinity,
    NoCompatibleProvider,
    PolicyRejectedAllCandidates,
}

/// Inspectable result of applying a Resolution Policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolutionDecision {
    pub policy_id: ResolutionPolicyId,
    pub selected_provider: Option<ProviderBinding>,
    pub selected_device: Option<DeviceBinding>,
    pub selected_capability: Option<CapabilityBinding>,
    pub reason: ResolutionDecisionReason,
    pub rejected_candidates: Vec<ResolutionCandidateRejection>,
}

/// Selects one candidate from a deterministic context.
pub trait ResolutionPolicy {
    fn id(&self) -> ResolutionPolicyId;
    fn decide(&self, context: &ResolutionContext) -> ResolutionDecision;
}
impl ResolutionPolicy for BuiltInResolutionPolicy {
    fn id(&self) -> ResolutionPolicyId {
        (*self).id()
    }

    fn decide(&self, context: &ResolutionContext) -> ResolutionDecision {
        let mut rejected_candidates = Vec::new();
        let mut eligible = Vec::new();
        for candidate in &context.candidates {
            if !candidate
                .capability
                .version()
                .is_compatible_with(context.requested_version)
            {
                rejected_candidates.push(rejection(
                    candidate,
                    ResolutionRejectionReason::IncompatibleCapability,
                ));
            } else if candidate.provider_health == ProviderHealth::Unavailable {
                rejected_candidates.push(rejection(
                    candidate,
                    ResolutionRejectionReason::ProviderUnavailable,
                ));
            } else if candidate.device_availability == DeviceAvailability::Unavailable {
                rejected_candidates.push(rejection(
                    candidate,
                    ResolutionRejectionReason::DeviceUnavailable,
                ));
            } else if !candidate.affinity_compatible {
                rejected_candidates.push(rejection(
                    candidate,
                    ResolutionRejectionReason::AffinityIncompatible,
                ));
            } else if context.execution_phase >= ExecutionPhase::AfterObservableOutput
                || (context.fallback >= FallbackClass::Restartable && !context.replayable_input)
            {
                rejected_candidates.push(rejection(
                    candidate,
                    ResolutionRejectionReason::FallbackNotAllowed,
                ));
            } else {
                eligible.push(candidate);
            }
        }

        eligible.sort_by(|left, right| match self {
            Self::Priority => right
                .priority
                .cmp(&left.priority)
                .then_with(|| left.sort_key().cmp(&right.sort_key())),
            Self::Availability => left
                .provider_health
                .cmp(&right.provider_health)
                .then_with(|| left.device_availability.cmp(&right.device_availability))
                .then_with(|| left.sort_key().cmp(&right.sort_key())),
            _ => left.sort_key().cmp(&right.sort_key()),
        });

        let reason = match self {
            Self::Deterministic => ResolutionDecisionReason::SelectedDeterministically,
            Self::Priority => ResolutionDecisionReason::SelectedByPriority,
            Self::Availability => ResolutionDecisionReason::SelectedByAvailability,
            Self::PerformancePreferred | Self::EnergyPreferred | Self::MemoryConstrained => {
                ResolutionDecisionReason::SelectedByPreferencePlaceholder
            }
        };
        let selected = eligible.first();
        ResolutionDecision {
            policy_id: self.id(),
            selected_provider: selected.map(|candidate| candidate.provider.clone()),
            selected_device: selected.and_then(|candidate| candidate.device.clone()),
            selected_capability: selected.map(|candidate| candidate.capability.clone()),
            reason: if selected.is_some() {
                reason
            } else if context.candidates.is_empty() {
                ResolutionDecisionReason::NoCompatibleProvider
            } else {
                ResolutionDecisionReason::PolicyRejectedAllCandidates
            },
            rejected_candidates,
        }
    }
}

fn rejection(
    candidate: &ResolutionCandidate,
    reason: ResolutionRejectionReason,
) -> ResolutionCandidateRejection {
    ResolutionCandidateRejection {
        provider: candidate.provider.clone(),
        capability: candidate.capability.clone(),
        reason,
    }
}

/// Package-qualified identifier of the portable Compute capability.
pub const COMPUTE_CAPABILITY_ID: &str = "magnetar:compute/run";
/// WIT package that defines the Compute capability contract.
pub const COMPUTE_WIT_PACKAGE: &str = "magnetar:compute";
/// WIT interface implemented by Compute providers.
pub const COMPUTE_WIT_INTERFACE: &str = COMPUTE_CAPABILITY_ID;
/// Current stable version of the executable Compute capability WIT contract.
pub const COMPUTE_CAPABILITY_VERSION: CapabilityVersion = CapabilityVersion::new(1, 1, 0);

/// Semantic operation families covered by the portable Compute capability.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeOperationFamily {
    DescriptorAndView,
    ConstructionAndAllocation,
    DataMovementAndConversion,
    Elementwise,
    ComparisonAndSelection,
    Reduction,
    LinearAlgebra,
    ConvolutionAndSpatialTransform,
    IndexingAndUpdate,
    RandomGeneration,
    SynchronizationAndCompletion,
}
impl ComputeOperationFamily {
    pub const ALL: [Self; 11] = [
        Self::DescriptorAndView,
        Self::ConstructionAndAllocation,
        Self::DataMovementAndConversion,
        Self::Elementwise,
        Self::ComparisonAndSelection,
        Self::Reduction,
        Self::LinearAlgebra,
        Self::ConvolutionAndSpatialTransform,
        Self::IndexingAndUpdate,
        Self::RandomGeneration,
        Self::SynchronizationAndCompletion,
    ];
    pub const fn id(self) -> &'static str {
        match self {
            Self::DescriptorAndView => "descriptor-and-view",
            Self::ConstructionAndAllocation => "construction-and-allocation",
            Self::DataMovementAndConversion => "data-movement-and-conversion",
            Self::Elementwise => "elementwise",
            Self::ComparisonAndSelection => "comparison-and-selection",
            Self::Reduction => "reduction",
            Self::LinearAlgebra => "linear-algebra",
            Self::ConvolutionAndSpatialTransform => "convolution-and-spatial-transform",
            Self::IndexingAndUpdate => "indexing-and-update",
            Self::RandomGeneration => "random-generation",
            Self::SynchronizationAndCompletion => "synchronization-and-completion",
        }
    }
    pub const fn metadata(self) -> ComputeOperationFamilyMetadata {
        match self {
            Self::DescriptorAndView => ComputeOperationFamilyMetadata {
                family: self,
                name: "Descriptor and view",
                scope: "tensor metadata and view transformations",
                examples: &[
                    "shape",
                    "dtype",
                    "reshape",
                    "flatten",
                    "squeeze",
                    "unsqueeze",
                    "transpose",
                    "permute",
                    "narrow",
                    "slice",
                    "broadcast",
                ],
            },
            Self::ConstructionAndAllocation => ComputeOperationFamilyMetadata {
                family: self,
                name: "Construction and allocation",
                scope: "portable tensor construction and allocation requests",
                examples: &["scalar", "zeros", "ones", "range", "allocate"],
            },
            Self::DataMovementAndConversion => ComputeOperationFamilyMetadata {
                family: self,
                name: "Data movement and conversion",
                scope: "explicit transfer, copy, materialization and dtype conversion",
                examples: &[
                    "upload",
                    "download",
                    "copy",
                    "materialize",
                    "convert",
                    "transfer",
                ],
            },
            Self::Elementwise => ComputeOperationFamilyMetadata {
                family: self,
                name: "Elementwise",
                scope: "portable unary, binary, activation and affine tensor operations",
                examples: &["add", "sub", "mul", "div", "exp", "log", "relu", "pow"],
            },
            Self::ComparisonAndSelection => ComputeOperationFamilyMetadata {
                family: self,
                name: "Comparison and selection",
                scope: "comparisons and conditional selection",
                examples: &["eq", "lt", "gt", "where"],
            },
            Self::Reduction => ComputeOperationFamilyMetadata {
                family: self,
                name: "Reduction",
                scope: "axis-based reductions with future schema-defined edge behavior",
                examples: &["sum", "mean", "min", "max", "argmin", "argmax"],
            },
            Self::LinearAlgebra => ComputeOperationFamilyMetadata {
                family: self,
                name: "Linear algebra",
                scope: "matrix and batched matrix operations",
                examples: &["matmul", "batched-matmul", "broadcast-matmul"],
            },
            Self::ConvolutionAndSpatialTransform => ComputeOperationFamilyMetadata {
                family: self,
                name: "Convolution and spatial transform",
                scope: "convolutions, pooling and spatial resampling",
                examples: &[
                    "conv",
                    "conv-transpose",
                    "pool",
                    "upsample-nearest",
                    "upsample-bilinear",
                ],
            },
            Self::IndexingAndUpdate => ComputeOperationFamilyMetadata {
                family: self,
                name: "Indexing and update",
                scope: "indexing, scatter/gather and explicit update-like result semantics",
                examples: &[
                    "gather",
                    "index-select",
                    "index-add",
                    "scatter",
                    "scatter-add",
                    "concat",
                ],
            },
            Self::RandomGeneration => ComputeOperationFamilyMetadata {
                family: self,
                name: "Random generation",
                scope: "provider-owned random tensor generation with optional seeds",
                examples: &["uniform", "normal", "seeded-generation"],
            },
            Self::SynchronizationAndCompletion => ComputeOperationFamilyMetadata {
                family: self,
                name: "Synchronization and completion",
                scope: "coarse operation status, await, cancellation and output retrieval",
                examples: &["status", "await", "cancel", "take-outputs"],
            },
        }
    }
    pub fn from_id(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|family| family.id() == value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ComputeOperationFamilyMetadata {
    pub family: ComputeOperationFamily,
    pub name: &'static str,
    pub scope: &'static str,
    pub examples: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeDType {
    Boolean,
    UInt8,
    SInt8,
    UInt16,
    SInt16,
    UInt32,
    SInt32,
    UInt64,
    SInt64,
    Float16,
    BrainFloat16,
    Float32,
    Float64,
}
impl ComputeDType {
    pub const fn size_bytes(self) -> u64 {
        match self {
            Self::Boolean | Self::UInt8 | Self::SInt8 => 1,
            Self::UInt16 | Self::SInt16 | Self::Float16 | Self::BrainFloat16 => 2,
            Self::UInt32 | Self::SInt32 | Self::Float32 => 4,
            Self::UInt64 | Self::SInt64 | Self::Float64 => 8,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DTypeDescriptor {
    Portable(ComputeDType),
    ProviderSpecific { id: String, size_bytes: u64 },
}
impl DTypeDescriptor {
    pub const fn portable(dtype: ComputeDType) -> Self {
        Self::Portable(dtype)
    }
    pub fn size_bytes(&self) -> u64 {
        match self {
            Self::Portable(dtype) => dtype.size_bytes(),
            Self::ProviderSpecific { size_bytes, .. } => *size_bytes,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeLayout {
    Dense,
    Strided,
    ProviderOpaque,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LayoutDescriptor {
    Contiguous,
    Strided {
        strides_elements: Vec<i64>,
        offset_elements: u64,
    },
    ProviderOpaque {
        layout_id: String,
    },
}
impl LayoutDescriptor {
    pub const fn kind(&self) -> ComputeLayout {
        match self {
            Self::Contiguous => ComputeLayout::Dense,
            Self::Strided { .. } => ComputeLayout::Strided,
            Self::ProviderOpaque { .. } => ComputeLayout::ProviderOpaque,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TensorResourceId(String);
impl TensorResourceId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl fmt::Display for TensorResourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShapeDescriptor {
    pub dimensions: Vec<u64>,
}
impl ShapeDescriptor {
    pub fn new(dimensions: impl Into<Vec<u64>>) -> Self {
        Self {
            dimensions: dimensions.into(),
        }
    }
    pub fn rank(&self) -> u64 {
        self.dimensions.len() as u64
    }
    pub fn element_count(&self) -> Result<u64, ComputeValidationError> {
        self.dimensions.iter().try_fold(1_u64, |acc, dimension| {
            acc.checked_mul(*dimension)
                .ok_or(ComputeValidationError::SizeOverflow {
                    reason: "tensor element count overflows u64".into(),
                })
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TensorViewSource {
    Descriptor,
    Resource(TensorResourceId),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ViewDescriptor {
    pub source: TensorViewSource,
    pub offset_elements: u64,
    pub strides_elements: Vec<i64>,
}
impl ViewDescriptor {
    pub fn from_resource(
        source: TensorResourceId,
        offset_elements: u64,
        strides_elements: impl Into<Vec<i64>>,
    ) -> Self {
        Self {
            source: TensorViewSource::Resource(source),
            offset_elements,
            strides_elements: strides_elements.into(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorDescriptor {
    pub shape: ShapeDescriptor,
    pub dtype: DTypeDescriptor,
    pub layout: LayoutDescriptor,
    pub view: Option<ViewDescriptor>,
}
impl TensorDescriptor {
    pub fn new(shape: ShapeDescriptor, dtype: DTypeDescriptor, layout: LayoutDescriptor) -> Self {
        Self {
            shape,
            dtype,
            layout,
            view: None,
        }
    }
    pub fn with_view(mut self, view: ViewDescriptor) -> Self {
        self.view = Some(view);
        self
    }
    pub fn materialized(shape: ShapeDescriptor, dtype: DTypeDescriptor) -> Self {
        Self::new(shape, dtype, LayoutDescriptor::Contiguous)
    }
    pub fn byte_size(&self) -> Result<u64, ComputeValidationError> {
        self.shape
            .element_count()?
            .checked_mul(self.dtype.size_bytes())
            .ok_or(ComputeValidationError::SizeOverflow {
                reason: "tensor byte size overflows u64".into(),
            })
    }
    pub fn validate(&self, limits: &TensorDescriptorLimits) -> Result<(), ComputeValidationError> {
        limits.validate_shape(&self.shape)?;
        let byte_size = self.byte_size()?;
        if byte_size > limits.max_bytes {
            return Err(ComputeValidationError::SizeOverflow {
                reason: format!(
                    "tensor byte size {byte_size} exceeds provider limit {}",
                    limits.max_bytes
                ),
            });
        }
        validate_layout_bounds(&self.shape, &self.layout)?;
        if let Some(view) = &self.view {
            validate_strides(&self.shape, &view.strides_elements, view.offset_elements)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorResourceDescriptor {
    pub id: TensorResourceId,
    pub descriptor: TensorDescriptor,
    pub affinity: ResourceAffinity,
}
impl TensorResourceDescriptor {
    pub fn new(
        id: TensorResourceId,
        descriptor: TensorDescriptor,
        affinity: ResourceAffinity,
    ) -> Self {
        Self {
            id,
            descriptor,
            affinity,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorDescriptorLimits {
    pub max_rank: u64,
    pub max_dimension: u64,
    pub max_elements: u64,
    pub max_bytes: u64,
    pub allow_zero_sized: bool,
}
impl Default for TensorDescriptorLimits {
    fn default() -> Self {
        Self {
            max_rank: 64,
            max_dimension: u64::MAX,
            max_elements: u64::MAX,
            max_bytes: u64::MAX,
            allow_zero_sized: false,
        }
    }
}
impl TensorDescriptorLimits {
    pub fn validate_shape(&self, shape: &ShapeDescriptor) -> Result<(), ComputeValidationError> {
        if shape.rank() > self.max_rank {
            return Err(ComputeValidationError::InvalidShape {
                reason: format!(
                    "tensor rank {} exceeds provider limit {}",
                    shape.rank(),
                    self.max_rank
                ),
            });
        }
        for dimension in &shape.dimensions {
            if *dimension > self.max_dimension {
                return Err(ComputeValidationError::InvalidShape {
                    reason: format!(
                        "tensor dimension {dimension} exceeds provider limit {}",
                        self.max_dimension
                    ),
                });
            }
            if *dimension == 0 && !self.allow_zero_sized {
                return Err(ComputeValidationError::InvalidShape {
                    reason: "zero-sized tensor dimensions are not supported".into(),
                });
            }
        }
        let element_count = shape.element_count()?;
        if element_count > self.max_elements {
            return Err(ComputeValidationError::SizeOverflow {
                reason: format!(
                    "tensor element count {element_count} exceeds provider limit {}",
                    self.max_elements
                ),
            });
        }
        Ok(())
    }
}

fn validate_layout_bounds(
    shape: &ShapeDescriptor,
    layout: &LayoutDescriptor,
) -> Result<(), ComputeValidationError> {
    match layout {
        LayoutDescriptor::Contiguous | LayoutDescriptor::ProviderOpaque { .. } => Ok(()),
        LayoutDescriptor::Strided {
            strides_elements,
            offset_elements,
        } => validate_strides(shape, strides_elements, *offset_elements),
    }
}

fn validate_strides(
    shape: &ShapeDescriptor,
    strides_elements: &[i64],
    offset_elements: u64,
) -> Result<(), ComputeValidationError> {
    if strides_elements.len() as u64 != shape.rank() {
        return Err(ComputeValidationError::InvalidLayout {
            reason: format!(
                "stride rank {} does not match tensor rank {}",
                strides_elements.len(),
                shape.rank()
            ),
        });
    }
    let element_count = shape.element_count()?;
    if element_count == 0 {
        return Ok(());
    }
    if offset_elements >= element_count {
        return Err(ComputeValidationError::InvalidLayout {
            reason: "view offset is outside tensor bounds".into(),
        });
    }
    let max_relative_offset = shape.dimensions.iter().zip(strides_elements).try_fold(
        0_u64,
        |acc, (dimension, stride)| {
            let stride = stride.unsigned_abs();
            let extent = dimension.saturating_sub(1);
            let span = extent
                .checked_mul(stride)
                .ok_or(ComputeValidationError::SizeOverflow {
                    reason: "strided layout span overflows u64".into(),
                })?;
            acc.checked_add(span)
                .ok_or(ComputeValidationError::SizeOverflow {
                    reason: "strided layout span overflows u64".into(),
                })
        },
    )?;
    let max_offset = offset_elements.checked_add(max_relative_offset).ok_or(
        ComputeValidationError::SizeOverflow {
            reason: "strided layout offset overflows u64".into(),
        },
    )?;
    if max_offset >= element_count {
        return Err(ComputeValidationError::InvalidLayout {
            reason: "strided layout addresses elements outside tensor bounds".into(),
        });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputePrecision {
    Exact,
    Default,
    Reduced,
    Mixed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DTypeSupport {
    pub portable: BTreeSet<ComputeDType>,
    pub provider_specific: BTreeSet<String>,
}
impl DTypeSupport {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_portable(mut self, dtypes: impl IntoIterator<Item = ComputeDType>) -> Self {
        self.portable.extend(dtypes);
        self
    }
    pub fn with_provider_specific(
        mut self,
        dtypes: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.provider_specific
            .extend(dtypes.into_iter().map(Into::into));
        self
    }
}
impl Default for DTypeSupport {
    fn default() -> Self {
        Self {
            portable: BTreeSet::new(),
            provider_specific: BTreeSet::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayoutSupport {
    pub input: BTreeSet<ComputeLayout>,
    pub output: BTreeSet<ComputeLayout>,
    pub consumes_views: bool,
    pub requires_materialization: bool,
}
impl LayoutSupport {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_input(mut self, layouts: impl IntoIterator<Item = ComputeLayout>) -> Self {
        self.input.extend(layouts);
        self
    }
    pub fn with_output(mut self, layouts: impl IntoIterator<Item = ComputeLayout>) -> Self {
        self.output.extend(layouts);
        self
    }
    pub const fn with_view_consumption(mut self) -> Self {
        self.consumes_views = true;
        self
    }
    pub const fn with_materialization_required(mut self) -> Self {
        self.requires_materialization = true;
        self
    }
}
impl Default for LayoutSupport {
    fn default() -> Self {
        Self {
            input: BTreeSet::new(),
            output: BTreeSet::new(),
            consumes_views: true,
            requires_materialization: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShapeLimitSupport {
    pub descriptor_limits: TensorDescriptorLimits,
    pub max_broadcast_rank: Option<u64>,
    pub max_batch_dimensions: Option<u64>,
}
impl ShapeLimitSupport {
    pub fn new(limits: TensorDescriptorLimits) -> Self {
        Self {
            descriptor_limits: limits,
            max_broadcast_rank: None,
            max_batch_dimensions: None,
        }
    }
    pub const fn with_broadcast_rank(mut self, rank: u64) -> Self {
        self.max_broadcast_rank = Some(rank);
        self
    }
    pub const fn with_batch_dimensions(mut self, dimensions: u64) -> Self {
        self.max_batch_dimensions = Some(dimensions);
        self
    }
}
impl Default for ShapeLimitSupport {
    fn default() -> Self {
        Self::new(TensorDescriptorLimits::default())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrecisionSupport {
    pub modes: BTreeSet<ComputePrecision>,
    pub accumulation_dtypes: BTreeSet<ComputeDType>,
    pub approximate_math: bool,
    pub deterministic_execution: bool,
    pub deterministic_random_generation: bool,
}
impl PrecisionSupport {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_modes(mut self, modes: impl IntoIterator<Item = ComputePrecision>) -> Self {
        self.modes.extend(modes);
        self
    }
    pub fn with_accumulation_dtypes(
        mut self,
        dtypes: impl IntoIterator<Item = ComputeDType>,
    ) -> Self {
        self.accumulation_dtypes.extend(dtypes);
        self
    }
    pub const fn with_approximate_math(mut self) -> Self {
        self.approximate_math = true;
        self
    }
    pub const fn with_deterministic_execution(mut self) -> Self {
        self.deterministic_execution = true;
        self
    }
    pub const fn with_deterministic_random_generation(mut self) -> Self {
        self.deterministic_random_generation = true;
        self
    }
}
impl Default for PrecisionSupport {
    fn default() -> Self {
        Self {
            modes: BTreeSet::new(),
            accumulation_dtypes: BTreeSet::new(),
            approximate_math: false,
            deterministic_execution: false,
            deterministic_random_generation: false,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ComputeOperationSupport {
    pub dtypes: BTreeSet<ComputeDType>,
    pub provider_specific_dtypes: BTreeSet<String>,
    pub layouts: BTreeSet<ComputeLayout>,
    pub precision_modes: BTreeSet<ComputePrecision>,
    pub descriptor_limits: TensorDescriptorLimits,
}
impl ComputeOperationSupport {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_dtypes(mut self, dtypes: impl IntoIterator<Item = ComputeDType>) -> Self {
        self.dtypes.extend(dtypes);
        self
    }
    pub fn with_provider_specific_dtypes(
        mut self,
        dtypes: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.provider_specific_dtypes
            .extend(dtypes.into_iter().map(Into::into));
        self
    }
    pub fn with_layouts(mut self, layouts: impl IntoIterator<Item = ComputeLayout>) -> Self {
        self.layouts.extend(layouts);
        self
    }
    pub fn with_descriptor_limits(mut self, limits: TensorDescriptorLimits) -> Self {
        self.descriptor_limits = limits;
        self
    }
    pub fn with_precision_modes(
        mut self,
        precision_modes: impl IntoIterator<Item = ComputePrecision>,
    ) -> Self {
        self.precision_modes.extend(precision_modes);
        self
    }
    fn supports(
        &self,
        operation: &ComputeOperationDescriptor,
    ) -> Result<(), ComputeValidationError> {
        if let Some(dtype) = operation.dtype
            && !self.dtypes.is_empty()
            && !self.dtypes.contains(&dtype)
        {
            return Err(ComputeValidationError::UnsupportedDType {
                family: operation.family,
                dtype,
            });
        }
        if let Some(layout) = operation.layout
            && !self.layouts.is_empty()
            && !self.layouts.contains(&layout)
        {
            return Err(ComputeValidationError::UnsupportedLayout {
                family: operation.family,
                layout,
            });
        }
        for tensor in &operation.tensors {
            tensor.validate(&self.descriptor_limits)?;
            self.supports_dtype(&tensor.dtype, operation.family)?;
            self.supports_layout(tensor.layout.kind(), operation.family)?;
            if let Some(view) = &tensor.view {
                self.supports_layout(ComputeLayout::Strided, operation.family)?;
                validate_strides(&tensor.shape, &view.strides_elements, view.offset_elements)?;
            }
        }
        if let Some(precision) = operation.precision
            && !self.precision_modes.is_empty()
            && !self.precision_modes.contains(&precision)
        {
            return Err(ComputeValidationError::UnsupportedPrecision {
                family: operation.family,
                precision,
            });
        }
        Ok(())
    }
    fn supports_dtype(
        &self,
        dtype: &DTypeDescriptor,
        family: ComputeOperationFamily,
    ) -> Result<(), ComputeValidationError> {
        match dtype {
            DTypeDescriptor::Portable(dtype) => {
                if !self.dtypes.is_empty() && !self.dtypes.contains(dtype) {
                    return Err(ComputeValidationError::UnsupportedDType {
                        family,
                        dtype: *dtype,
                    });
                }
            }
            DTypeDescriptor::ProviderSpecific { id, .. } => {
                if !self.provider_specific_dtypes.contains(id) {
                    return Err(ComputeValidationError::UnsupportedProviderDType {
                        family,
                        dtype: id.clone(),
                    });
                }
            }
        }
        Ok(())
    }
    fn supports_layout(
        &self,
        layout: ComputeLayout,
        family: ComputeOperationFamily,
    ) -> Result<(), ComputeValidationError> {
        if !self.layouts.is_empty() && !self.layouts.contains(&layout) {
            return Err(ComputeValidationError::UnsupportedLayout { family, layout });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeCapabilitySupport {
    pub capability_id: CapabilityId,
    pub versions: BTreeSet<CapabilityVersion>,
    pub operation_catalog_revision: String,
    pub operation_schema_revision: String,
    pub experimental_extensions: BTreeSet<String>,
}
impl Default for ComputeCapabilitySupport {
    fn default() -> Self {
        Self {
            capability_id: CapabilityId::new(COMPUTE_CAPABILITY_ID),
            versions: BTreeSet::new(),
            operation_catalog_revision: String::new(),
            operation_schema_revision: String::new(),
            experimental_extensions: BTreeSet::new(),
        }
    }
}
impl ComputeCapabilitySupport {
    pub fn with_versions(mut self, versions: impl IntoIterator<Item = CapabilityVersion>) -> Self {
        self.versions.extend(versions);
        self
    }
    pub fn with_operation_catalog_revision(mut self, revision: impl Into<String>) -> Self {
        self.operation_catalog_revision = revision.into();
        self
    }
    pub fn with_operation_schema_revision(mut self, revision: impl Into<String>) -> Self {
        self.operation_schema_revision = revision.into();
        self
    }
    pub fn with_experimental_extensions(
        mut self,
        extensions: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.experimental_extensions
            .extend(extensions.into_iter().map(Into::into));
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperationFamilySupport {
    pub family: ComputeOperationFamily,
    pub dtypes: DTypeSupport,
    pub layouts: LayoutSupport,
    pub shapes: ShapeLimitSupport,
    pub precision: PrecisionSupport,
    pub portable: bool,
}
impl OperationFamilySupport {
    pub fn new(family: ComputeOperationFamily) -> Self {
        Self {
            family,
            dtypes: DTypeSupport::default(),
            layouts: LayoutSupport::default(),
            shapes: ShapeLimitSupport::default(),
            precision: PrecisionSupport::default(),
            portable: true,
        }
    }
    pub fn from_operation_support(
        family: ComputeOperationFamily,
        support: ComputeOperationSupport,
    ) -> Self {
        Self {
            family,
            dtypes: DTypeSupport {
                portable: support.dtypes,
                provider_specific: support.provider_specific_dtypes,
            },
            layouts: LayoutSupport {
                input: support.layouts.clone(),
                output: support.layouts,
                ..LayoutSupport::default()
            },
            shapes: ShapeLimitSupport::new(support.descriptor_limits),
            precision: PrecisionSupport {
                modes: support.precision_modes,
                ..PrecisionSupport::default()
            },
            portable: true,
        }
    }
    fn operation_support(&self) -> ComputeOperationSupport {
        ComputeOperationSupport {
            dtypes: self.dtypes.portable.clone(),
            provider_specific_dtypes: self.dtypes.provider_specific.clone(),
            layouts: self
                .layouts
                .input
                .union(&self.layouts.output)
                .copied()
                .collect(),
            precision_modes: self.precision.modes.clone(),
            descriptor_limits: self.shapes.descriptor_limits.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperationSchemaSupport {
    pub operation: ComputeOperationId,
    pub family: ComputeOperationFamily,
    pub dtypes: DTypeSupport,
    pub layouts: LayoutSupport,
    pub shapes: ShapeLimitSupport,
    pub precision: PrecisionSupport,
    pub portable: bool,
}
impl OperationSchemaSupport {
    pub fn new(operation: ComputeOperationId, family: ComputeOperationFamily) -> Self {
        Self {
            operation,
            family,
            dtypes: DTypeSupport::default(),
            layouts: LayoutSupport::default(),
            shapes: ShapeLimitSupport::default(),
            precision: PrecisionSupport::default(),
            portable: true,
        }
    }
    pub fn from_operation_support(
        operation: ComputeOperationId,
        family: ComputeOperationFamily,
        support: ComputeOperationSupport,
    ) -> Self {
        Self {
            operation,
            family,
            dtypes: DTypeSupport {
                portable: support.dtypes,
                provider_specific: support.provider_specific_dtypes,
            },
            layouts: LayoutSupport {
                input: support.layouts.clone(),
                output: support.layouts,
                ..LayoutSupport::default()
            },
            shapes: ShapeLimitSupport::new(support.descriptor_limits),
            precision: PrecisionSupport {
                modes: support.precision_modes,
                ..PrecisionSupport::default()
            },
            portable: true,
        }
    }
    fn operation_support(&self) -> ComputeOperationSupport {
        OperationFamilySupport {
            family: self.family,
            dtypes: self.dtypes.clone(),
            layouts: self.layouts.clone(),
            shapes: self.shapes.clone(),
            precision: self.precision.clone(),
            portable: self.portable,
        }
        .operation_support()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeDataMovementKind {
    Upload,
    Download,
    Copy,
    Materialize,
    Transfer,
    DTypeConversion,
    PlacementConversion,
}
impl ComputeDataMovementKind {
    pub const ALL: [Self; 7] = [
        Self::Upload,
        Self::Download,
        Self::Copy,
        Self::Materialize,
        Self::Transfer,
        Self::DTypeConversion,
        Self::PlacementConversion,
    ];
    pub const fn id(self) -> &'static str {
        match self {
            Self::Upload => "upload",
            Self::Download => "download",
            Self::Copy => "copy",
            Self::Materialize => "materialize",
            Self::Transfer => "transfer",
            Self::DTypeConversion => "dtype-conversion",
            Self::PlacementConversion => "placement-conversion",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DataMovementSupport {
    pub kind: ComputeDataMovementKind,
    pub dtypes: DTypeSupport,
    pub layouts: LayoutSupport,
    pub host_encodings: BTreeSet<HostBufferEncoding>,
    pub shapes: ShapeLimitSupport,
    pub allow_host_staging: bool,
}
impl DataMovementSupport {
    pub fn new(kind: ComputeDataMovementKind) -> Self {
        Self {
            kind,
            dtypes: DTypeSupport::default(),
            layouts: LayoutSupport::default(),
            host_encodings: BTreeSet::new(),
            shapes: ShapeLimitSupport::default(),
            allow_host_staging: false,
        }
    }
    pub fn from_compute_support(
        kind: ComputeDataMovementKind,
        support: ComputeDataMovementSupport,
    ) -> Self {
        Self {
            kind,
            dtypes: DTypeSupport {
                portable: support.dtypes,
                provider_specific: support.provider_specific_dtypes,
            },
            layouts: LayoutSupport {
                input: support.layouts.clone(),
                output: support.layouts,
                ..LayoutSupport::default()
            },
            host_encodings: support.host_encodings,
            shapes: ShapeLimitSupport::new(support.descriptor_limits),
            allow_host_staging: support.allow_host_staging,
        }
    }
    fn movement_support(&self) -> ComputeDataMovementSupport {
        ComputeDataMovementSupport {
            dtypes: self.dtypes.portable.clone(),
            provider_specific_dtypes: self.dtypes.provider_specific.clone(),
            layouts: self
                .layouts
                .input
                .union(&self.layouts.output)
                .copied()
                .collect(),
            host_encodings: self.host_encodings.clone(),
            descriptor_limits: self.shapes.descriptor_limits.clone(),
            allow_host_staging: self.allow_host_staging,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum HostBufferEncoding {
    RawBytes,
    NativeEndian,
    LittleEndian,
    BigEndian,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostBufferDescriptor {
    pub byte_len: u64,
    pub encoding: HostBufferEncoding,
}
impl HostBufferDescriptor {
    pub const fn new(byte_len: u64, encoding: HostBufferEncoding) -> Self {
        Self { byte_len, encoding }
    }
    pub fn validate_for(&self, tensor: &TensorDescriptor) -> Result<(), ComputeValidationError> {
        let expected = tensor.byte_size()?;
        if self.byte_len != expected {
            return Err(ComputeValidationError::InvalidHostBuffer {
                reason: format!(
                    "host buffer byte length {} does not match tensor byte size {expected}",
                    self.byte_len
                ),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ComputeDataMovementSource {
    Host(HostBufferDescriptor),
    Tensor(TensorResourceDescriptor),
}
impl ComputeDataMovementSource {
    fn tensor(&self) -> Option<&TensorResourceDescriptor> {
        match self {
            Self::Host(_) => None,
            Self::Tensor(tensor) => Some(tensor),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeDataMovementDescriptor {
    pub kind: ComputeDataMovementKind,
    pub source: ComputeDataMovementSource,
    pub output: TensorDescriptor,
    pub target_provider: Option<ProviderBinding>,
    pub target_device: Option<DeviceBinding>,
    pub target_group: Option<AffinityGroupId>,
    pub allow_host_staging: bool,
}
impl ComputeDataMovementDescriptor {
    pub fn upload(host: HostBufferDescriptor, output: TensorDescriptor) -> Self {
        Self::new(
            ComputeDataMovementKind::Upload,
            ComputeDataMovementSource::Host(host),
            output,
        )
    }
    pub fn download(source: TensorResourceDescriptor, output: TensorDescriptor) -> Self {
        Self::new(
            ComputeDataMovementKind::Download,
            ComputeDataMovementSource::Tensor(source),
            output,
        )
    }
    pub fn copy(source: TensorResourceDescriptor, output: TensorDescriptor) -> Self {
        Self::new(
            ComputeDataMovementKind::Copy,
            ComputeDataMovementSource::Tensor(source),
            output,
        )
    }
    pub fn materialize(source: TensorResourceDescriptor, output: TensorDescriptor) -> Self {
        Self::new(
            ComputeDataMovementKind::Materialize,
            ComputeDataMovementSource::Tensor(source),
            output,
        )
    }
    pub fn transfer(source: TensorResourceDescriptor, output: TensorDescriptor) -> Self {
        Self::new(
            ComputeDataMovementKind::Transfer,
            ComputeDataMovementSource::Tensor(source),
            output,
        )
    }
    pub fn dtype_conversion(source: TensorResourceDescriptor, output: TensorDescriptor) -> Self {
        Self::new(
            ComputeDataMovementKind::DTypeConversion,
            ComputeDataMovementSource::Tensor(source),
            output,
        )
    }
    pub fn placement_conversion(
        source: TensorResourceDescriptor,
        output: TensorDescriptor,
    ) -> Self {
        Self::new(
            ComputeDataMovementKind::PlacementConversion,
            ComputeDataMovementSource::Tensor(source),
            output,
        )
    }
    fn new(
        kind: ComputeDataMovementKind,
        source: ComputeDataMovementSource,
        output: TensorDescriptor,
    ) -> Self {
        Self {
            kind,
            source,
            output,
            target_provider: None,
            target_device: None,
            target_group: None,
            allow_host_staging: false,
        }
    }
    pub fn with_target_provider(mut self, provider: ProviderBinding) -> Self {
        self.target_provider = Some(provider);
        self
    }
    pub fn with_target_device(mut self, device: DeviceBinding) -> Self {
        self.target_device = Some(device);
        self
    }
    pub fn with_target_group(mut self, group: AffinityGroupId) -> Self {
        self.target_group = Some(group);
        self
    }
    pub fn with_host_staging(mut self) -> Self {
        self.allow_host_staging = true;
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeDataMovementSupport {
    pub dtypes: BTreeSet<ComputeDType>,
    pub provider_specific_dtypes: BTreeSet<String>,
    pub layouts: BTreeSet<ComputeLayout>,
    pub host_encodings: BTreeSet<HostBufferEncoding>,
    pub descriptor_limits: TensorDescriptorLimits,
    pub allow_host_staging: bool,
}
impl Default for ComputeDataMovementSupport {
    fn default() -> Self {
        Self {
            dtypes: BTreeSet::new(),
            provider_specific_dtypes: BTreeSet::new(),
            layouts: BTreeSet::new(),
            host_encodings: BTreeSet::new(),
            descriptor_limits: TensorDescriptorLimits::default(),
            allow_host_staging: false,
        }
    }
}
impl ComputeDataMovementSupport {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_dtypes(mut self, dtypes: impl IntoIterator<Item = ComputeDType>) -> Self {
        self.dtypes.extend(dtypes);
        self
    }
    pub fn with_provider_specific_dtypes(
        mut self,
        dtypes: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.provider_specific_dtypes
            .extend(dtypes.into_iter().map(Into::into));
        self
    }
    pub fn with_layouts(mut self, layouts: impl IntoIterator<Item = ComputeLayout>) -> Self {
        self.layouts.extend(layouts);
        self
    }
    pub fn with_host_encodings(
        mut self,
        encodings: impl IntoIterator<Item = HostBufferEncoding>,
    ) -> Self {
        self.host_encodings.extend(encodings);
        self
    }
    pub fn with_descriptor_limits(mut self, limits: TensorDescriptorLimits) -> Self {
        self.descriptor_limits = limits;
        self
    }
    pub const fn with_host_staging(mut self) -> Self {
        self.allow_host_staging = true;
        self
    }
    fn supports(
        &self,
        _provider: &ProviderBinding,
        movement: &ComputeDataMovementDescriptor,
    ) -> Result<(), ComputeValidationError> {
        movement.output.validate(&self.descriptor_limits)?;
        self.supports_dtype(&movement.output.dtype, movement.kind)?;
        self.supports_layout(movement.output.layout.kind(), movement.kind)?;
        if let Some(source) = movement.source.tensor() {
            source.descriptor.validate(&self.descriptor_limits)?;
            self.supports_dtype(&source.descriptor.dtype, movement.kind)?;
            self.supports_layout(source.descriptor.layout.kind(), movement.kind)?;
        }
        match &movement.source {
            ComputeDataMovementSource::Host(host) => {
                if movement.kind != ComputeDataMovementKind::Upload {
                    return Err(ComputeValidationError::InvalidTransfer {
                        reason: "host buffers are valid only as upload sources".into(),
                    });
                }
                if !self.host_encodings.is_empty() && !self.host_encodings.contains(&host.encoding)
                {
                    return Err(ComputeValidationError::InvalidHostBuffer {
                        reason: format!("host encoding {:?} is not supported", host.encoding),
                    });
                }
                host.validate_for(&movement.output)?;
            }
            ComputeDataMovementSource::Tensor(source) => {
                if movement.kind == ComputeDataMovementKind::Upload {
                    return Err(ComputeValidationError::InvalidTransfer {
                        reason: "upload requires a host buffer source".into(),
                    });
                }
                if movement.kind == ComputeDataMovementKind::Download {
                    source.descriptor.byte_size()?;
                }
                if movement.kind == ComputeDataMovementKind::Materialize
                    && source.descriptor.view.is_none()
                {
                    return Err(ComputeValidationError::MaterializationRequired {
                        reason: "materialize requires a tensor view source".into(),
                    });
                }
                if movement.kind == ComputeDataMovementKind::DTypeConversion
                    && source.descriptor.dtype == movement.output.dtype
                {
                    return Err(ComputeValidationError::UnsupportedConversion {
                        reason: "dtype conversion requires a different output dtype".into(),
                    });
                }
                if movement.kind == ComputeDataMovementKind::PlacementConversion
                    && movement.target_provider.is_none()
                    && movement.target_device.is_none()
                    && movement.target_group.is_none()
                {
                    return Err(ComputeValidationError::InvalidTransfer {
                        reason: "placement conversion requires an explicit target placement".into(),
                    });
                }
                if movement.allow_host_staging && !self.allow_host_staging {
                    return Err(ComputeValidationError::InvalidTransfer {
                        reason: "host-staged data movement is not advertised by the provider"
                            .into(),
                    });
                }
            }
        }
        Ok(())
    }
    fn supports_dtype(
        &self,
        dtype: &DTypeDescriptor,
        kind: ComputeDataMovementKind,
    ) -> Result<(), ComputeValidationError> {
        match dtype {
            DTypeDescriptor::Portable(dtype) => {
                if !self.dtypes.is_empty() && !self.dtypes.contains(dtype) {
                    return Err(ComputeValidationError::UnsupportedConversion {
                        reason: format!(
                            "data movement '{}' does not support dtype {dtype:?}",
                            kind.id()
                        ),
                    });
                }
            }
            DTypeDescriptor::ProviderSpecific { id, .. } => {
                if !self.provider_specific_dtypes.contains(id) {
                    return Err(ComputeValidationError::UnsupportedConversion {
                        reason: format!(
                            "data movement '{}' does not support provider-specific dtype '{id}'",
                            kind.id()
                        ),
                    });
                }
            }
        }
        Ok(())
    }
    fn supports_layout(
        &self,
        layout: ComputeLayout,
        kind: ComputeDataMovementKind,
    ) -> Result<(), ComputeValidationError> {
        if !self.layouts.is_empty() && !self.layouts.contains(&layout) {
            return Err(ComputeValidationError::UnsupportedConversion {
                reason: format!(
                    "data movement '{}' does not support layout {layout:?}",
                    kind.id()
                ),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ComputeOperationId(String);
impl ComputeOperationId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl fmt::Display for ComputeOperationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceComputeSupport {
    pub device: DeviceId,
    pub memory_bytes: Option<u64>,
    pub operation_families: BTreeMap<ComputeOperationFamily, OperationFamilySupport>,
    pub operation_schemas: BTreeMap<ComputeOperationId, OperationSchemaSupport>,
    pub data_movement: BTreeMap<ComputeDataMovementKind, DataMovementSupport>,
}
impl DeviceComputeSupport {
    pub fn new(device: DeviceId) -> Self {
        Self {
            device,
            memory_bytes: None,
            operation_families: BTreeMap::new(),
            operation_schemas: BTreeMap::new(),
            data_movement: BTreeMap::new(),
        }
    }
    pub const fn with_memory_bytes(mut self, memory_bytes: u64) -> Self {
        self.memory_bytes = Some(memory_bytes);
        self
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProviderComputeAdvertisement {
    pub capability: ComputeCapabilitySupport,
    pub operation_families: BTreeMap<ComputeOperationFamily, OperationFamilySupport>,
    pub operation_schemas: BTreeMap<ComputeOperationId, OperationSchemaSupport>,
    pub unsupported_operation_schemas: BTreeSet<ComputeOperationId>,
    pub provider_extension_schemas: BTreeSet<ComputeOperationId>,
    pub data_movement: BTreeMap<ComputeDataMovementKind, DataMovementSupport>,
    pub devices: BTreeMap<DeviceId, DeviceComputeSupport>,
    pub diagnostics: BTreeMap<String, String>,
}
impl ProviderComputeAdvertisement {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn is_empty(&self) -> bool {
        self.capability.versions.is_empty()
            && self.operation_families.is_empty()
            && self.operation_schemas.is_empty()
            && self.unsupported_operation_schemas.is_empty()
            && self.provider_extension_schemas.is_empty()
            && self.data_movement.is_empty()
            && self.devices.is_empty()
            && self.diagnostics.is_empty()
    }
    pub fn supports_capability_version(&self, required: CapabilityVersion) -> bool {
        self.capability.versions.is_empty()
            || self
                .capability
                .versions
                .iter()
                .any(|version| version.is_compatible_with(required))
    }
    pub fn with_capability(mut self, capability: ComputeCapabilitySupport) -> Self {
        self.capability = capability;
        self
    }
    pub fn with_operation_family(mut self, support: OperationFamilySupport) -> Self {
        self.operation_families.insert(support.family, support);
        self
    }
    pub fn with_operation_schema(mut self, support: OperationSchemaSupport) -> Self {
        self.operation_schemas
            .insert(support.operation.clone(), support);
        self
    }
    pub fn with_unsupported_operation_schema(mut self, operation: ComputeOperationId) -> Self {
        self.unsupported_operation_schemas.insert(operation);
        self
    }
    pub fn with_provider_extension_schema(mut self, operation: ComputeOperationId) -> Self {
        self.provider_extension_schemas.insert(operation);
        self
    }
    pub fn with_data_movement(mut self, support: DataMovementSupport) -> Self {
        self.data_movement.insert(support.kind, support);
        self
    }
    pub fn with_device(mut self, support: DeviceComputeSupport) -> Self {
        self.devices.insert(support.device.clone(), support);
        self
    }
}

fn effective_compute_advertisement(metadata: &ProviderMetadata) -> ProviderComputeAdvertisement {
    let mut advertisement = metadata.compute_advertisement.clone();
    for capability in metadata
        .capabilities
        .iter()
        .filter(|capability| capability.id.as_str() == COMPUTE_CAPABILITY_ID)
    {
        advertisement.capability.versions.insert(capability.version);
    }
    for (family, support) in &metadata.compute_operation_support {
        advertisement
            .operation_families
            .entry(*family)
            .or_insert_with(|| {
                OperationFamilySupport::from_operation_support(*family, support.clone())
            });
    }
    for (operation, support) in &metadata.compute_operation_schema_support {
        let family = initial_compute_operation_schemas()
            .get(operation)
            .map(|schema| schema.family)
            .unwrap_or(ComputeOperationFamily::DescriptorAndView);
        advertisement
            .operation_schemas
            .entry(operation.clone())
            .or_insert_with(|| {
                OperationSchemaSupport::from_operation_support(
                    operation.clone(),
                    family,
                    support.clone(),
                )
            });
    }
    for (kind, support) in &metadata.compute_data_movement_support {
        advertisement
            .data_movement
            .entry(*kind)
            .or_insert_with(|| DataMovementSupport::from_compute_support(*kind, support.clone()));
    }
    advertisement
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeAttributeKind {
    Boolean,
    Integer,
    Float,
    String,
    DType,
    Shape,
    Axes,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ComputeOperationAttribute {
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
    DType(ComputeDType),
    Shape(ShapeDescriptor),
    Axes(Vec<u64>),
}
impl ComputeOperationAttribute {
    pub const fn kind(&self) -> ComputeAttributeKind {
        match self {
            Self::Boolean(_) => ComputeAttributeKind::Boolean,
            Self::Integer(_) => ComputeAttributeKind::Integer,
            Self::Float(_) => ComputeAttributeKind::Float,
            Self::String(_) => ComputeAttributeKind::String,
            Self::DType(_) => ComputeAttributeKind::DType,
            Self::Shape(_) => ComputeAttributeKind::Shape,
            Self::Axes(_) => ComputeAttributeKind::Axes,
        }
    }
}
impl Eq for ComputeOperationAttribute {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeOperationAttributeRule {
    pub kind: ComputeAttributeKind,
    pub required: bool,
}
impl ComputeOperationAttributeRule {
    pub const fn required(kind: ComputeAttributeKind) -> Self {
        Self {
            kind,
            required: true,
        }
    }
    pub const fn optional(kind: ComputeAttributeKind) -> Self {
        Self {
            kind,
            required: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeOperationInputRule {
    pub min_inputs: usize,
    pub max_inputs: Option<usize>,
    pub require_same_dtype: bool,
    pub allow_broadcast: bool,
    pub boolean_inputs: BTreeSet<usize>,
    pub integer_index_inputs: BTreeSet<usize>,
}
impl ComputeOperationInputRule {
    pub fn exactly(count: usize) -> Self {
        Self {
            min_inputs: count,
            max_inputs: Some(count),
            require_same_dtype: false,
            allow_broadcast: false,
            boolean_inputs: BTreeSet::new(),
            integer_index_inputs: BTreeSet::new(),
        }
    }
    pub fn at_least(count: usize) -> Self {
        Self {
            min_inputs: count,
            max_inputs: None,
            require_same_dtype: false,
            allow_broadcast: false,
            boolean_inputs: BTreeSet::new(),
            integer_index_inputs: BTreeSet::new(),
        }
    }
    pub fn with_same_dtype(mut self) -> Self {
        self.require_same_dtype = true;
        self
    }
    pub fn with_broadcast(mut self) -> Self {
        self.allow_broadcast = true;
        self
    }
    pub fn with_boolean_input(mut self, index: usize) -> Self {
        self.boolean_inputs.insert(index);
        self
    }
    pub fn with_integer_index_input(mut self, index: usize) -> Self {
        self.integer_index_inputs.insert(index);
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ComputeOutputDTypeRule {
    SameAsInput(usize),
    Boolean,
    ExplicitAttribute(String),
    IntegerIndex,
    ProviderDefined,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ComputeOutputShapeRule {
    SameAsInput(usize),
    ExplicitAttribute(String),
    BroadcastInputs,
    Reduction {
        axes_attribute: String,
        keep_dimensions_attribute: String,
    },
    MatrixMultiplication,
    BatchedMatrixMultiplication,
    Concatenation {
        axis_attribute: String,
    },
    ProviderDefined,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeOperationOutputRule {
    pub output_count: usize,
    pub dtype: ComputeOutputDTypeRule,
    pub shape: ComputeOutputShapeRule,
}
impl ComputeOperationOutputRule {
    pub fn new(
        output_count: usize,
        dtype: ComputeOutputDTypeRule,
        shape: ComputeOutputShapeRule,
    ) -> Self {
        Self {
            output_count,
            dtype,
            shape,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeOperationSchema {
    pub id: ComputeOperationId,
    pub family: ComputeOperationFamily,
    pub attributes: BTreeMap<String, ComputeOperationAttributeRule>,
    pub input_rule: ComputeOperationInputRule,
    pub output_rule: ComputeOperationOutputRule,
    pub provider_specific_semantics: bool,
}
impl ComputeOperationSchema {
    pub fn new(
        id: impl Into<String>,
        family: ComputeOperationFamily,
        input_rule: ComputeOperationInputRule,
        output_rule: ComputeOperationOutputRule,
    ) -> Self {
        Self {
            id: ComputeOperationId::new(id),
            family,
            attributes: BTreeMap::new(),
            input_rule,
            output_rule,
            provider_specific_semantics: false,
        }
    }
    pub fn with_attribute(
        mut self,
        name: impl Into<String>,
        rule: ComputeOperationAttributeRule,
    ) -> Self {
        self.attributes.insert(name.into(), rule);
        self
    }
    pub const fn with_provider_specific_semantics(mut self) -> Self {
        self.provider_specific_semantics = true;
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeOperationValidationResult {
    pub schema: ComputeOperationId,
    pub family: ComputeOperationFamily,
    pub input_count: usize,
    pub output_count: usize,
}

/// Portable operation-specific schema descriptor inside `magnetar:compute/run`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeOperationDescriptor {
    pub schema_id: Option<ComputeOperationId>,
    pub family: ComputeOperationFamily,
    pub dtype: Option<ComputeDType>,
    pub layout: Option<ComputeLayout>,
    pub precision: Option<ComputePrecision>,
    pub attributes: BTreeMap<String, ComputeOperationAttribute>,
    pub tensors: Vec<TensorDescriptor>,
}
impl ComputeOperationDescriptor {
    pub fn new(family: ComputeOperationFamily) -> Self {
        Self {
            schema_id: None,
            family,
            dtype: None,
            layout: None,
            precision: None,
            attributes: BTreeMap::new(),
            tensors: Vec::new(),
        }
    }
    pub fn from_schema(schema: &ComputeOperationSchema) -> Self {
        Self {
            schema_id: Some(schema.id.clone()),
            family: schema.family,
            dtype: None,
            layout: None,
            precision: None,
            attributes: BTreeMap::new(),
            tensors: Vec::new(),
        }
    }
    pub fn with_dtype(mut self, dtype: ComputeDType) -> Self {
        self.dtype = Some(dtype);
        self
    }
    pub fn with_layout(mut self, layout: ComputeLayout) -> Self {
        self.layout = Some(layout);
        self
    }
    pub fn with_precision(mut self, precision: ComputePrecision) -> Self {
        self.precision = Some(precision);
        self
    }
    pub fn with_attribute(
        mut self,
        name: impl Into<String>,
        value: ComputeOperationAttribute,
    ) -> Self {
        self.attributes.insert(name.into(), value);
        self
    }
    pub fn with_tensor(mut self, tensor: TensorDescriptor) -> Self {
        self.tensors.push(tensor);
        self
    }
}

pub fn initial_compute_operation_schemas() -> BTreeMap<ComputeOperationId, ComputeOperationSchema> {
    let mut schemas = BTreeMap::new();
    let same_output = || {
        ComputeOperationOutputRule::new(
            1,
            ComputeOutputDTypeRule::SameAsInput(0),
            ComputeOutputShapeRule::SameAsInput(0),
        )
    };
    for id in [
        "tensor.transpose",
        "tensor.permute",
        "tensor.slice",
        "tensor.squeeze",
        "tensor.unsqueeze",
    ] {
        insert_schema(
            &mut schemas,
            ComputeOperationSchema::new(
                id,
                ComputeOperationFamily::DescriptorAndView,
                ComputeOperationInputRule::exactly(1),
                same_output(),
            ),
        );
    }
    insert_schema(
        &mut schemas,
        ComputeOperationSchema::new(
            "tensor.reshape",
            ComputeOperationFamily::DescriptorAndView,
            ComputeOperationInputRule::exactly(1),
            ComputeOperationOutputRule::new(
                1,
                ComputeOutputDTypeRule::SameAsInput(0),
                ComputeOutputShapeRule::ExplicitAttribute("shape".into()),
            ),
        )
        .with_attribute(
            "shape",
            ComputeOperationAttributeRule::required(ComputeAttributeKind::Shape),
        ),
    );
    insert_schema(
        &mut schemas,
        ComputeOperationSchema::new(
            "tensor.broadcast",
            ComputeOperationFamily::DescriptorAndView,
            ComputeOperationInputRule::exactly(1).with_broadcast(),
            ComputeOperationOutputRule::new(
                1,
                ComputeOutputDTypeRule::SameAsInput(0),
                ComputeOutputShapeRule::ExplicitAttribute("shape".into()),
            ),
        )
        .with_attribute(
            "shape",
            ComputeOperationAttributeRule::required(ComputeAttributeKind::Shape),
        ),
    );
    for op in [
        "abs", "neg", "exp", "log", "sqrt", "recip", "sin", "cos", "tanh", "relu", "silu", "gelu",
        "erf", "floor", "ceil", "round",
    ] {
        insert_schema(
            &mut schemas,
            ComputeOperationSchema::new(
                format!("elementwise.unary.{op}"),
                ComputeOperationFamily::Elementwise,
                ComputeOperationInputRule::exactly(1),
                same_output(),
            ),
        );
    }
    for op in ["add", "sub", "mul", "div", "maximum", "minimum"] {
        insert_schema(
            &mut schemas,
            ComputeOperationSchema::new(
                format!("elementwise.binary.{op}"),
                ComputeOperationFamily::Elementwise,
                ComputeOperationInputRule::exactly(2)
                    .with_same_dtype()
                    .with_broadcast(),
                ComputeOperationOutputRule::new(
                    1,
                    ComputeOutputDTypeRule::SameAsInput(0),
                    ComputeOutputShapeRule::BroadcastInputs,
                ),
            ),
        );
    }
    for op in ["eq", "ne", "lt", "le", "gt", "ge"] {
        insert_schema(
            &mut schemas,
            ComputeOperationSchema::new(
                format!("comparison.{op}"),
                ComputeOperationFamily::ComparisonAndSelection,
                ComputeOperationInputRule::exactly(2)
                    .with_same_dtype()
                    .with_broadcast(),
                ComputeOperationOutputRule::new(
                    1,
                    ComputeOutputDTypeRule::Boolean,
                    ComputeOutputShapeRule::BroadcastInputs,
                ),
            ),
        );
    }
    insert_schema(
        &mut schemas,
        ComputeOperationSchema::new(
            "selection.where",
            ComputeOperationFamily::ComparisonAndSelection,
            ComputeOperationInputRule::exactly(3)
                .with_boolean_input(0)
                .with_broadcast(),
            ComputeOperationOutputRule::new(
                1,
                ComputeOutputDTypeRule::SameAsInput(1),
                ComputeOutputShapeRule::BroadcastInputs,
            ),
        ),
    );
    for op in ["sum", "mean", "min", "max", "argmin", "argmax"] {
        let dtype = if matches!(op, "argmin" | "argmax") {
            ComputeOutputDTypeRule::IntegerIndex
        } else {
            ComputeOutputDTypeRule::SameAsInput(0)
        };
        insert_schema(
            &mut schemas,
            ComputeOperationSchema::new(
                format!("reduction.{op}"),
                ComputeOperationFamily::Reduction,
                ComputeOperationInputRule::exactly(1),
                ComputeOperationOutputRule::new(
                    1,
                    dtype,
                    ComputeOutputShapeRule::Reduction {
                        axes_attribute: "axes".into(),
                        keep_dimensions_attribute: "keep-dimensions".into(),
                    },
                ),
            )
            .with_attribute(
                "axes",
                ComputeOperationAttributeRule::required(ComputeAttributeKind::Axes),
            )
            .with_attribute(
                "keep-dimensions",
                ComputeOperationAttributeRule::optional(ComputeAttributeKind::Boolean),
            ),
        );
    }
    for (id, shape_rule) in [
        (
            "linalg.matmul",
            ComputeOutputShapeRule::MatrixMultiplication,
        ),
        (
            "linalg.batched-matmul",
            ComputeOutputShapeRule::BatchedMatrixMultiplication,
        ),
    ] {
        insert_schema(
            &mut schemas,
            ComputeOperationSchema::new(
                id,
                ComputeOperationFamily::LinearAlgebra,
                ComputeOperationInputRule::exactly(2).with_same_dtype(),
                ComputeOperationOutputRule::new(
                    1,
                    ComputeOutputDTypeRule::SameAsInput(0),
                    shape_rule,
                ),
            )
            .with_attribute(
                "transpose-a",
                ComputeOperationAttributeRule::optional(ComputeAttributeKind::Boolean),
            )
            .with_attribute(
                "transpose-b",
                ComputeOperationAttributeRule::optional(ComputeAttributeKind::Boolean),
            )
            .with_attribute(
                "accumulation-dtype",
                ComputeOperationAttributeRule::optional(ComputeAttributeKind::DType),
            )
            .with_attribute(
                "precision",
                ComputeOperationAttributeRule::optional(ComputeAttributeKind::String),
            ),
        );
    }
    for id in ["tensor.gather", "tensor.index-select"] {
        insert_schema(
            &mut schemas,
            ComputeOperationSchema::new(
                id,
                ComputeOperationFamily::IndexingAndUpdate,
                ComputeOperationInputRule::exactly(2).with_integer_index_input(1),
                ComputeOperationOutputRule::new(
                    1,
                    ComputeOutputDTypeRule::SameAsInput(0),
                    ComputeOutputShapeRule::ProviderDefined,
                ),
            )
            .with_attribute(
                "axis",
                ComputeOperationAttributeRule::required(ComputeAttributeKind::Integer),
            ),
        );
    }
    for id in ["tensor.scatter", "tensor.scatter-add"] {
        insert_schema(
            &mut schemas,
            ComputeOperationSchema::new(
                id,
                ComputeOperationFamily::IndexingAndUpdate,
                ComputeOperationInputRule::exactly(3)
                    .with_same_dtype()
                    .with_integer_index_input(1),
                ComputeOperationOutputRule::new(
                    1,
                    ComputeOutputDTypeRule::SameAsInput(0),
                    ComputeOutputShapeRule::SameAsInput(0),
                ),
            )
            .with_attribute(
                "axis",
                ComputeOperationAttributeRule::required(ComputeAttributeKind::Integer),
            )
            .with_provider_specific_semantics(),
        );
    }
    insert_schema(
        &mut schemas,
        ComputeOperationSchema::new(
            "tensor.concat",
            ComputeOperationFamily::IndexingAndUpdate,
            ComputeOperationInputRule::at_least(1).with_same_dtype(),
            ComputeOperationOutputRule::new(
                1,
                ComputeOutputDTypeRule::SameAsInput(0),
                ComputeOutputShapeRule::Concatenation {
                    axis_attribute: "axis".into(),
                },
            ),
        )
        .with_attribute(
            "axis",
            ComputeOperationAttributeRule::required(ComputeAttributeKind::Integer),
        ),
    );
    for id in ["random.uniform", "random.normal"] {
        insert_schema(
            &mut schemas,
            ComputeOperationSchema::new(
                id,
                ComputeOperationFamily::RandomGeneration,
                ComputeOperationInputRule::exactly(0),
                ComputeOperationOutputRule::new(
                    1,
                    ComputeOutputDTypeRule::ExplicitAttribute("dtype".into()),
                    ComputeOutputShapeRule::ExplicitAttribute("shape".into()),
                ),
            )
            .with_attribute(
                "shape",
                ComputeOperationAttributeRule::required(ComputeAttributeKind::Shape),
            )
            .with_attribute(
                "dtype",
                ComputeOperationAttributeRule::required(ComputeAttributeKind::DType),
            )
            .with_attribute(
                "seed",
                ComputeOperationAttributeRule::optional(ComputeAttributeKind::Integer),
            ),
        );
    }
    schemas
}

fn insert_schema(
    schemas: &mut BTreeMap<ComputeOperationId, ComputeOperationSchema>,
    schema: ComputeOperationSchema,
) {
    schemas.insert(schema.id.clone(), schema);
}

fn validate_compute_operation_schema(
    operation: &ComputeOperationDescriptor,
) -> Result<Option<ComputeOperationValidationResult>, ComputeValidationError> {
    let Some(schema_id) = &operation.schema_id else {
        return Ok(None);
    };
    let schemas = initial_compute_operation_schemas();
    let schema = schemas
        .get(schema_id)
        .ok_or_else(|| ComputeValidationError::UnknownOperationSchema(schema_id.clone()))?;
    if schema.family != operation.family {
        return Err(ComputeValidationError::UnknownOperationFamily(
            operation.family.id().into(),
        ));
    }
    validate_operation_attributes(schema, operation)?;

    if operation.tensors.len() < schema.output_rule.output_count {
        return Err(ComputeValidationError::InvalidOutputDescriptor {
            operation: schema.id.clone(),
            reason: format!(
                "operation declares {} tensor descriptors but schema requires {} output(s)",
                operation.tensors.len(),
                schema.output_rule.output_count
            ),
        });
    }
    let input_count = operation.tensors.len() - schema.output_rule.output_count;
    validate_operation_arity(schema, input_count)?;
    let (inputs, outputs) = operation.tensors.split_at(input_count);
    validate_operation_input_rule(schema, inputs)?;
    validate_operation_output_rule(schema, operation, inputs, outputs)?;
    Ok(Some(ComputeOperationValidationResult {
        schema: schema.id.clone(),
        family: schema.family,
        input_count,
        output_count: outputs.len(),
    }))
}

fn validate_operation_attributes(
    schema: &ComputeOperationSchema,
    operation: &ComputeOperationDescriptor,
) -> Result<(), ComputeValidationError> {
    for name in operation.attributes.keys() {
        if !schema.attributes.contains_key(name) {
            return Err(ComputeValidationError::InvalidOperationAttribute {
                operation: schema.id.clone(),
                attribute: name.clone(),
                reason: "attribute is not defined by the operation schema".into(),
            });
        }
    }
    for (name, rule) in &schema.attributes {
        match operation.attributes.get(name) {
            Some(value) if value.kind() == rule.kind => {}
            Some(value) => {
                return Err(ComputeValidationError::InvalidOperationAttribute {
                    operation: schema.id.clone(),
                    attribute: name.clone(),
                    reason: format!("expected {:?}, found {:?}", rule.kind, value.kind()),
                });
            }
            None if rule.required => {
                return Err(ComputeValidationError::InvalidOperationAttribute {
                    operation: schema.id.clone(),
                    attribute: name.clone(),
                    reason: "required attribute is missing".into(),
                });
            }
            None => {}
        }
    }
    Ok(())
}

fn validate_operation_arity(
    schema: &ComputeOperationSchema,
    input_count: usize,
) -> Result<(), ComputeValidationError> {
    let too_few = input_count < schema.input_rule.min_inputs;
    let too_many = schema
        .input_rule
        .max_inputs
        .is_some_and(|max| input_count > max);
    if too_few || too_many {
        let expected = match schema.input_rule.max_inputs {
            Some(max) if max == schema.input_rule.min_inputs => max.to_string(),
            Some(max) => format!("{}..={max}", schema.input_rule.min_inputs),
            None => format!("at least {}", schema.input_rule.min_inputs),
        };
        return Err(ComputeValidationError::InvalidOperationArity {
            operation: schema.id.clone(),
            expected,
            found: input_count,
        });
    }
    Ok(())
}

fn validate_operation_input_rule(
    schema: &ComputeOperationSchema,
    inputs: &[TensorDescriptor],
) -> Result<(), ComputeValidationError> {
    for index in &schema.input_rule.boolean_inputs {
        match inputs.get(*index).map(|tensor| &tensor.dtype) {
            Some(DTypeDescriptor::Portable(ComputeDType::Boolean)) => {}
            _ => {
                return Err(ComputeValidationError::InvalidOperationAttribute {
                    operation: schema.id.clone(),
                    attribute: format!("input[{index}]"),
                    reason: "input must have boolean dtype".into(),
                });
            }
        }
    }
    for index in &schema.input_rule.integer_index_inputs {
        match inputs.get(*index).map(|tensor| &tensor.dtype) {
            Some(DTypeDescriptor::Portable(dtype)) if dtype.is_integer() => {}
            _ => {
                return Err(ComputeValidationError::InvalidOperationAttribute {
                    operation: schema.id.clone(),
                    attribute: format!("input[{index}]"),
                    reason: "index input must have an integer dtype".into(),
                });
            }
        }
    }
    if schema.input_rule.require_same_dtype {
        let Some(first) = inputs.first().map(|tensor| &tensor.dtype) else {
            return Ok(());
        };
        if inputs.iter().any(|tensor| &tensor.dtype != first) {
            return Err(ComputeValidationError::UnsupportedDType {
                family: schema.family,
                dtype: portable_dtype(first).unwrap_or(ComputeDType::UInt8),
            });
        }
    }
    if schema.input_rule.allow_broadcast && inputs.len() > 1 {
        broadcast_shape(inputs.iter().map(|tensor| &tensor.shape)).map_err(|reason| {
            ComputeValidationError::InvalidShape {
                reason: format!("operation schema '{}': {reason}", schema.id),
            }
        })?;
    }
    Ok(())
}

fn validate_operation_output_rule(
    schema: &ComputeOperationSchema,
    operation: &ComputeOperationDescriptor,
    inputs: &[TensorDescriptor],
    outputs: &[TensorDescriptor],
) -> Result<(), ComputeValidationError> {
    if outputs.len() != schema.output_rule.output_count {
        return Err(ComputeValidationError::InvalidOutputDescriptor {
            operation: schema.id.clone(),
            reason: format!(
                "expected {} output(s), found {}",
                schema.output_rule.output_count,
                outputs.len()
            ),
        });
    }
    for output in outputs {
        validate_output_dtype(schema, operation, inputs, output)?;
    }
    validate_output_shape(schema, operation, inputs, outputs)
}

fn validate_output_dtype(
    schema: &ComputeOperationSchema,
    operation: &ComputeOperationDescriptor,
    inputs: &[TensorDescriptor],
    output: &TensorDescriptor,
) -> Result<(), ComputeValidationError> {
    match &schema.output_rule.dtype {
        ComputeOutputDTypeRule::SameAsInput(index) => {
            let Some(input) = inputs.get(*index) else {
                return Ok(());
            };
            if output.dtype != input.dtype {
                return Err(ComputeValidationError::InvalidOutputDescriptor {
                    operation: schema.id.clone(),
                    reason: "output dtype must match input dtype".into(),
                });
            }
        }
        ComputeOutputDTypeRule::Boolean => {
            if output.dtype != DTypeDescriptor::Portable(ComputeDType::Boolean) {
                return Err(ComputeValidationError::InvalidOutputDescriptor {
                    operation: schema.id.clone(),
                    reason: "output dtype must be boolean".into(),
                });
            }
        }
        ComputeOutputDTypeRule::ExplicitAttribute(name) => {
            if let Some(ComputeOperationAttribute::DType(dtype)) = operation.attributes.get(name)
                && output.dtype != DTypeDescriptor::Portable(*dtype)
            {
                return Err(ComputeValidationError::InvalidOutputDescriptor {
                    operation: schema.id.clone(),
                    reason: format!("output dtype must match '{name}' attribute"),
                });
            }
        }
        ComputeOutputDTypeRule::IntegerIndex => {
            if !matches!(&output.dtype, DTypeDescriptor::Portable(dtype) if dtype.is_integer()) {
                return Err(ComputeValidationError::InvalidOutputDescriptor {
                    operation: schema.id.clone(),
                    reason: "output dtype must be an integer index dtype".into(),
                });
            }
        }
        ComputeOutputDTypeRule::ProviderDefined => {}
    }
    Ok(())
}

fn validate_output_shape(
    schema: &ComputeOperationSchema,
    operation: &ComputeOperationDescriptor,
    inputs: &[TensorDescriptor],
    outputs: &[TensorDescriptor],
) -> Result<(), ComputeValidationError> {
    let Some(output) = outputs.first() else {
        return Ok(());
    };
    let expected = match &schema.output_rule.shape {
        ComputeOutputShapeRule::SameAsInput(index) => inputs.get(*index).map(|t| t.shape.clone()),
        ComputeOutputShapeRule::ExplicitAttribute(name) => match operation.attributes.get(name) {
            Some(ComputeOperationAttribute::Shape(shape)) => Some(shape.clone()),
            _ => None,
        },
        ComputeOutputShapeRule::BroadcastInputs => {
            Some(broadcast_shape(inputs.iter().map(|t| &t.shape))?)
        }
        ComputeOutputShapeRule::Reduction {
            axes_attribute,
            keep_dimensions_attribute,
        } => Some(reduction_shape(
            &schema.id,
            inputs.first(),
            operation.attributes.get(axes_attribute),
            operation.attributes.get(keep_dimensions_attribute),
        )?),
        ComputeOutputShapeRule::MatrixMultiplication => Some(matmul_shape(&schema.id, inputs)?),
        ComputeOutputShapeRule::BatchedMatrixMultiplication => {
            Some(batched_matmul_shape(&schema.id, inputs)?)
        }
        ComputeOutputShapeRule::Concatenation { axis_attribute } => Some(concat_shape(
            &schema.id,
            inputs,
            operation.attributes.get(axis_attribute),
        )?),
        ComputeOutputShapeRule::ProviderDefined => None,
    };
    if let Some(expected) = expected
        && output.shape != expected
    {
        return Err(ComputeValidationError::InvalidOutputDescriptor {
            operation: schema.id.clone(),
            reason: format!(
                "output shape {:?} does not match expected {:?}",
                output.shape.dimensions, expected.dimensions
            ),
        });
    }
    Ok(())
}

impl ComputeDType {
    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            Self::UInt8
                | Self::SInt8
                | Self::UInt16
                | Self::SInt16
                | Self::UInt32
                | Self::SInt32
                | Self::UInt64
                | Self::SInt64
        )
    }
}

fn portable_dtype(dtype: &DTypeDescriptor) -> Option<ComputeDType> {
    match dtype {
        DTypeDescriptor::Portable(dtype) => Some(*dtype),
        DTypeDescriptor::ProviderSpecific { .. } => None,
    }
}

fn broadcast_shape<'a>(
    shapes: impl IntoIterator<Item = &'a ShapeDescriptor>,
) -> Result<ShapeDescriptor, ComputeValidationError> {
    let shapes = shapes.into_iter().collect::<Vec<_>>();
    let max_rank = shapes
        .iter()
        .map(|shape| shape.dimensions.len())
        .max()
        .unwrap_or(0);
    let mut result = vec![1_u64; max_rank];
    for shape in shapes {
        for (offset, dimension) in shape.dimensions.iter().rev().enumerate() {
            let index = max_rank - 1 - offset;
            let current = result[index];
            if current == 1 {
                result[index] = *dimension;
            } else if *dimension == 1 || current == *dimension {
                continue;
            } else {
                return Err(ComputeValidationError::InvalidShape {
                    reason: format!(
                        "dimensions {current} and {dimension} are not broadcast-compatible"
                    ),
                });
            }
        }
    }
    Ok(ShapeDescriptor::new(result))
}

fn reduction_shape(
    operation: &ComputeOperationId,
    input: Option<&TensorDescriptor>,
    axes: Option<&ComputeOperationAttribute>,
    keep_dimensions: Option<&ComputeOperationAttribute>,
) -> Result<ShapeDescriptor, ComputeValidationError> {
    let input = input.ok_or_else(|| ComputeValidationError::InvalidOperationArity {
        operation: operation.clone(),
        expected: "1".into(),
        found: 0,
    })?;
    let axes = match axes {
        Some(ComputeOperationAttribute::Axes(axes)) => axes,
        _ => {
            return Err(ComputeValidationError::InvalidOperationAttribute {
                operation: operation.clone(),
                attribute: "axes".into(),
                reason: "axes attribute is required".into(),
            });
        }
    };
    let keep = matches!(
        keep_dimensions,
        Some(ComputeOperationAttribute::Boolean(true))
    );
    let rank = input.shape.dimensions.len() as u64;
    for axis in axes {
        if *axis >= rank {
            return Err(ComputeValidationError::InvalidShape {
                reason: format!("reduction axis {axis} is outside rank {rank}"),
            });
        }
    }
    let axis_set = axes.iter().copied().collect::<BTreeSet<_>>();
    let dimensions = input
        .shape
        .dimensions
        .iter()
        .enumerate()
        .filter_map(|(index, dimension)| {
            if axis_set.contains(&(index as u64)) {
                keep.then_some(1)
            } else {
                Some(*dimension)
            }
        })
        .collect::<Vec<_>>();
    Ok(ShapeDescriptor::new(dimensions))
}

fn matmul_shape(
    operation: &ComputeOperationId,
    inputs: &[TensorDescriptor],
) -> Result<ShapeDescriptor, ComputeValidationError> {
    let [lhs, rhs] = inputs else {
        return Err(ComputeValidationError::InvalidOperationArity {
            operation: operation.clone(),
            expected: "2".into(),
            found: inputs.len(),
        });
    };
    if lhs.shape.dimensions.len() != 2 || rhs.shape.dimensions.len() != 2 {
        return Err(ComputeValidationError::InvalidShape {
            reason: "matrix multiplication requires rank-2 inputs".into(),
        });
    }
    if lhs.shape.dimensions[1] != rhs.shape.dimensions[0] {
        return Err(ComputeValidationError::InvalidShape {
            reason: "matrix multiplication inner dimensions are incompatible".into(),
        });
    }
    Ok(ShapeDescriptor::new([
        lhs.shape.dimensions[0],
        rhs.shape.dimensions[1],
    ]))
}

fn batched_matmul_shape(
    operation: &ComputeOperationId,
    inputs: &[TensorDescriptor],
) -> Result<ShapeDescriptor, ComputeValidationError> {
    let [lhs, rhs] = inputs else {
        return Err(ComputeValidationError::InvalidOperationArity {
            operation: operation.clone(),
            expected: "2".into(),
            found: inputs.len(),
        });
    };
    if lhs.shape.dimensions.len() < 3 || rhs.shape.dimensions.len() < 3 {
        return Err(ComputeValidationError::InvalidShape {
            reason: "batched matrix multiplication requires rank >= 3 inputs".into(),
        });
    }
    let lhs_rank = lhs.shape.dimensions.len();
    let rhs_rank = rhs.shape.dimensions.len();
    if lhs.shape.dimensions[lhs_rank - 1] != rhs.shape.dimensions[rhs_rank - 2] {
        return Err(ComputeValidationError::InvalidShape {
            reason: "batched matrix multiplication inner dimensions are incompatible".into(),
        });
    }
    let mut batch = broadcast_shape(
        [
            ShapeDescriptor::new(lhs.shape.dimensions[..lhs_rank - 2].to_vec()),
            ShapeDescriptor::new(rhs.shape.dimensions[..rhs_rank - 2].to_vec()),
        ]
        .iter(),
    )?
    .dimensions;
    batch.push(lhs.shape.dimensions[lhs_rank - 2]);
    batch.push(rhs.shape.dimensions[rhs_rank - 1]);
    Ok(ShapeDescriptor::new(batch))
}

fn concat_shape(
    operation: &ComputeOperationId,
    inputs: &[TensorDescriptor],
    axis: Option<&ComputeOperationAttribute>,
) -> Result<ShapeDescriptor, ComputeValidationError> {
    let Some(first) = inputs.first() else {
        return Err(ComputeValidationError::InvalidOperationArity {
            operation: operation.clone(),
            expected: "at least 1".into(),
            found: 0,
        });
    };
    let axis = match axis {
        Some(ComputeOperationAttribute::Integer(axis)) if *axis >= 0 => *axis as usize,
        _ => {
            return Err(ComputeValidationError::InvalidOperationAttribute {
                operation: operation.clone(),
                attribute: "axis".into(),
                reason: "axis must be a non-negative integer".into(),
            });
        }
    };
    if axis >= first.shape.dimensions.len() {
        return Err(ComputeValidationError::InvalidShape {
            reason: "concatenation axis is outside input rank".into(),
        });
    }
    let mut dimensions = first.shape.dimensions.clone();
    for input in &inputs[1..] {
        if input.shape.dimensions.len() != dimensions.len() {
            return Err(ComputeValidationError::InvalidShape {
                reason: "all concatenation inputs must have the same rank".into(),
            });
        }
        for (index, dimension) in input.shape.dimensions.iter().enumerate() {
            if index == axis {
                dimensions[index] = dimensions[index].checked_add(*dimension).ok_or(
                    ComputeValidationError::SizeOverflow {
                        reason: "concatenated dimension overflows u64".into(),
                    },
                )?;
            } else if dimensions[index] != *dimension {
                return Err(ComputeValidationError::InvalidShape {
                    reason: "non-concatenated dimensions must match".into(),
                });
            }
        }
    }
    Ok(ShapeDescriptor::new(dimensions))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeOperationRequest {
    pub family_id: String,
    pub dtype: Option<ComputeDType>,
    pub layout: Option<ComputeLayout>,
    pub precision: Option<ComputePrecision>,
    pub tensors: Vec<TensorDescriptor>,
}
impl ComputeOperationRequest {
    pub fn new(family_id: impl Into<String>) -> Self {
        Self {
            family_id: family_id.into(),
            dtype: None,
            layout: None,
            precision: None,
            tensors: Vec::new(),
        }
    }
    pub fn with_dtype(mut self, dtype: ComputeDType) -> Self {
        self.dtype = Some(dtype);
        self
    }
    pub fn with_layout(mut self, layout: ComputeLayout) -> Self {
        self.layout = Some(layout);
        self
    }
    pub fn with_precision(mut self, precision: ComputePrecision) -> Self {
        self.precision = Some(precision);
        self
    }
    pub fn with_tensor(mut self, tensor: TensorDescriptor) -> Self {
        self.tensors.push(tensor);
        self
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ComputeGraphId(String);
impl ComputeGraphId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl fmt::Display for ComputeGraphId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ComputeNodeId(String);
impl ComputeNodeId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl fmt::Display for ComputeNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ComputeInputId(String);
impl ComputeInputId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl fmt::Display for ComputeInputId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ComputeOutputId(String);
impl ComputeOutputId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl fmt::Display for ComputeOutputId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ComputeInputValue {
    TensorResource(TensorResourceDescriptor),
    TensorDescriptor(TensorDescriptor),
    Constant(TensorDescriptor),
}
impl ComputeInputValue {
    fn descriptor(&self) -> &TensorDescriptor {
        match self {
            Self::TensorResource(resource) => &resource.descriptor,
            Self::TensorDescriptor(descriptor) | Self::Constant(descriptor) => descriptor,
        }
    }
    fn affinity(&self) -> Option<&ResourceAffinity> {
        match self {
            Self::TensorResource(resource) => Some(&resource.affinity),
            Self::TensorDescriptor(_) | Self::Constant(_) => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeInput {
    pub id: ComputeInputId,
    pub value: ComputeInputValue,
}
impl ComputeInput {
    pub fn new(id: ComputeInputId, value: ComputeInputValue) -> Self {
        Self { id, value }
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeValueRef {
    Input(ComputeInputId),
    NodeOutput {
        node: ComputeNodeId,
        output: ComputeOutputId,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeNodeOutput {
    pub id: ComputeOutputId,
    pub descriptor: TensorDescriptor,
}
impl ComputeNodeOutput {
    pub fn new(id: ComputeOutputId, descriptor: TensorDescriptor) -> Self {
        Self { id, descriptor }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeNode {
    pub id: ComputeNodeId,
    pub operation: ComputeOperationDescriptor,
    pub inputs: Vec<ComputeValueRef>,
    pub outputs: Vec<ComputeNodeOutput>,
}
impl ComputeNode {
    pub fn new(id: ComputeNodeId, operation: ComputeOperationDescriptor) -> Self {
        Self {
            id,
            operation,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }
    pub fn with_input(mut self, input: ComputeValueRef) -> Self {
        self.inputs.push(input);
        self
    }
    pub fn with_output(mut self, output: ComputeNodeOutput) -> Self {
        self.outputs.push(output);
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeOutput {
    pub id: ComputeOutputId,
    pub source: ComputeValueRef,
}
impl ComputeOutput {
    pub fn new(id: ComputeOutputId, source: ComputeValueRef) -> Self {
        Self { id, source }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeGraph {
    pub id: ComputeGraphId,
    pub inputs: Vec<ComputeInput>,
    pub nodes: Vec<ComputeNode>,
    pub outputs: Vec<ComputeOutput>,
}
impl ComputeGraph {
    pub fn new(id: ComputeGraphId) -> Self {
        Self {
            id,
            inputs: Vec::new(),
            nodes: Vec::new(),
            outputs: Vec::new(),
        }
    }
    pub fn with_input(mut self, input: ComputeInput) -> Self {
        self.inputs.push(input);
        self
    }
    pub fn with_node(mut self, node: ComputeNode) -> Self {
        self.nodes.push(node);
        self
    }
    pub fn with_output(mut self, output: ComputeOutput) -> Self {
        self.outputs.push(output);
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComputeSubmissionState {
    Pending,
    Running,
    Completed,
    Cancelled,
    Failed,
}
impl ComputeSubmissionState {
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Cancelled | Self::Failed)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeExecutionResult {
    pub state: ComputeSubmissionState,
    pub outputs: Vec<TensorResourceDescriptor>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeSubmission {
    pub graph: ComputeGraphId,
    pub provider: ProviderBinding,
    pub affinity: ResourceAffinity,
    state: ComputeSubmissionState,
    result: Option<ComputeExecutionResult>,
}
impl ComputeSubmission {
    pub fn new(
        graph: ComputeGraphId,
        provider: ProviderBinding,
        affinity: ResourceAffinity,
    ) -> Self {
        Self {
            graph,
            provider,
            affinity,
            state: ComputeSubmissionState::Pending,
            result: None,
        }
    }
    pub fn state(&self) -> ComputeSubmissionState {
        self.state
    }
    pub fn start(&mut self) -> Result<(), ComputeValidationError> {
        if self.state != ComputeSubmissionState::Pending {
            return Err(ComputeValidationError::InvalidState {
                reason: format!("cannot start submission from {:?} state", self.state),
            });
        }
        self.state = ComputeSubmissionState::Running;
        Ok(())
    }
    pub fn complete(
        &mut self,
        outputs: Vec<TensorResourceDescriptor>,
    ) -> Result<(), ComputeValidationError> {
        if self.state.is_terminal() {
            return Err(ComputeValidationError::InvalidState {
                reason: "submission is already terminal".into(),
            });
        }
        self.state = ComputeSubmissionState::Completed;
        self.result = Some(ComputeExecutionResult {
            state: self.state,
            outputs,
        });
        Ok(())
    }
    pub fn cancel(&mut self) -> Result<(), ComputeValidationError> {
        if self.state.is_terminal() {
            return Err(ComputeValidationError::InvalidState {
                reason: "submission is already terminal".into(),
            });
        }
        self.state = ComputeSubmissionState::Cancelled;
        self.result = Some(ComputeExecutionResult {
            state: self.state,
            outputs: Vec::new(),
        });
        Ok(())
    }
    pub fn fail(&mut self) -> Result<(), ComputeValidationError> {
        if self.state.is_terminal() {
            return Err(ComputeValidationError::InvalidState {
                reason: "submission is already terminal".into(),
            });
        }
        self.state = ComputeSubmissionState::Failed;
        self.result = Some(ComputeExecutionResult {
            state: self.state,
            outputs: Vec::new(),
        });
        Ok(())
    }
    pub fn result(&self) -> Option<&ComputeExecutionResult> {
        self.result.as_ref()
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ExecutionPlanId(String);
impl ExecutionPlanId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl fmt::Display for ExecutionPlanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeExecutionPhase {
    Validation,
    Resolution,
    Planning,
    DataMovement,
    Materialization,
    MemoryAllocation,
    ProviderSubmission,
    Execution,
    Completion,
    Cancellation,
    Interruption,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeExecutionClassification {
    Transparent,
    Restartable,
    ProviderPinned,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionStepKind {
    ValidateGraph,
    ResolveProvider,
    ResolveDevice,
    ValidateCapabilityVersion,
    ValidateOperationSchema,
    ValidateDType,
    ValidateLayout,
    ValidatePrecisionPolicy,
    ValidateDeterminism,
    BindInputResource,
    BindOutputResource,
    PreserveProviderPinnedAffinity,
    PreserveDeviceBoundAffinity,
    PreserveAffinityGroup,
    RejectIncompatibleResourceChain,
    Upload,
    Download,
    Copy,
    Transfer,
    Materialize,
    AllocateMemory,
    ValidateMemory,
    SubmitToProvider,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionStep {
    pub id: String,
    pub phase: ComputeExecutionPhase,
    pub kind: ExecutionStepKind,
    pub provider: ProviderBinding,
    pub device: Option<DeviceBinding>,
    pub dependencies: Vec<String>,
}
impl ExecutionStep {
    pub fn new(
        id: impl Into<String>,
        phase: ComputeExecutionPhase,
        kind: ExecutionStepKind,
        provider: ProviderBinding,
    ) -> Self {
        Self {
            id: id.into(),
            phase,
            kind,
            provider,
            device: None,
            dependencies: Vec::new(),
        }
    }
    pub fn with_device(mut self, device: Option<DeviceBinding>) -> Self {
        self.device = device;
        self
    }
    pub fn depends_on(mut self, dependency: impl Into<String>) -> Self {
        self.dependencies.push(dependency.into());
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionInput {
    pub id: ComputeInputId,
    pub descriptor: TensorDescriptor,
    pub resource: Option<TensorResourceId>,
    pub affinity: ResourceAffinity,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionOutput {
    pub id: ComputeOutputId,
    pub descriptor: TensorDescriptor,
    pub affinity: ResourceAffinity,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionConstraint {
    ResolutionPolicy(ResolutionPolicyId),
    CapabilityVersion(CapabilityBinding),
    Provider(ProviderBinding),
    Device(DeviceBinding),
    ResourceAffinity(ResourceAffinity),
    AffinityGroup(AffinityGroupId),
    OperationSchema(ComputeOperationId),
    DType(ComputeDType),
    Layout(ComputeLayout),
    PrecisionPolicy(ComputePrecision),
    DeterministicBehavior,
    MemoryRequirement(String),
    ExplicitTransferRequired(TensorResourceId),
    ExplicitTransferRequirement(String),
    ExplicitMaterializationRequired(String),
    NoHiddenCpuStaging,
    NoImplicitProviderMigration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionDiagnostic {
    SelectedProvider(ProviderBinding),
    SelectedDevice(DeviceBinding),
    SelectedCapability(CapabilityBinding),
    ResolutionDecision(ResolutionDecision),
    RejectedProviderCandidate {
        provider: ProviderBinding,
        reason: ResolutionRejectionReason,
    },
    Memory(MemoryPlanningDiagnostic),
    TransferRequired {
        resource: TensorResourceId,
        from: ResourceAffinity,
        to: ResourceAffinity,
    },
    MaterializationRequired {
        source: String,
    },
    PolicyDecisionReason(ResolutionDecisionReason),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeExecutionPlan {
    pub id: ExecutionPlanId,
    pub graph: ComputeGraphId,
    pub provider: ProviderBinding,
    pub device: Option<DeviceBinding>,
    pub capability: CapabilityBinding,
    pub policy: ResolutionPolicyId,
    pub classification: ComputeExecutionClassification,
    pub inputs: Vec<ExecutionInput>,
    pub outputs: Vec<ExecutionOutput>,
    pub constraints: Vec<ExecutionConstraint>,
    pub steps: Vec<ExecutionStep>,
    pub memory_plan: MemoryPlan,
    pub diagnostics: Vec<ExecutionDiagnostic>,
    validated: bool,
}
impl ComputeExecutionPlan {
    pub fn is_validated(&self) -> bool {
        self.validated
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ScheduledOperationId(u64);
impl ScheduledOperationId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}
impl fmt::Display for ScheduledOperationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SchedulingPolicy {
    #[default]
    Fifo,
    Priority,
    Deadline,
    ResourceAware,
    BatchAware,
    Fairness,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SchedulingState {
    Accepted,
    Queued,
    Ready,
    Submitted,
    Running,
    Completed,
    Cancelled,
    Failed,
    Interrupted,
}
impl SchedulingState {
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Cancelled | Self::Failed | Self::Interrupted
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SchedulingDiagnostic {
    Accepted {
        operation: ScheduledOperationId,
    },
    Queued {
        operation: ScheduledOperationId,
        position: usize,
    },
    SelectedProvider(ProviderBinding),
    SelectedDevice(DeviceBinding),
    QueueTime {
        accepted_order: u64,
    },
    CancellationRequested,
    CancellationForwardedToProvider(ProviderBinding),
    TerminalState(SchedulingState),
    StableFailureReason(SchedulerErrorCode),
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SchedulerErrorCode {
    InvalidExecutionPlan,
    QueueCapacityExceeded,
    ProviderUnavailable,
    DeviceUnavailable,
    ResourceAffinityConflict,
    MemoryPlanInvalid,
    SubmissionFailed,
    CancellationUnsupported,
    CancellationFailed,
    ExecutionFailed,
    ExecutionInterrupted,
    OperationTimeout,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SchedulerError {
    InvalidExecutionPlan {
        reason: String,
    },
    QueueCapacityExceeded {
        capacity: usize,
    },
    ProviderUnavailable(ProviderBinding),
    DeviceUnavailable(DeviceBinding),
    ResourceAffinityConflict {
        reason: String,
    },
    MemoryPlanInvalid {
        reason: String,
    },
    SubmissionFailed {
        operation: ScheduledOperationId,
        reason: String,
    },
    CancellationUnsupported(ScheduledOperationId),
    CancellationFailed {
        operation: ScheduledOperationId,
        reason: String,
    },
    ExecutionFailed {
        operation: ScheduledOperationId,
        reason: String,
    },
    ExecutionInterrupted {
        operation: ScheduledOperationId,
        reason: String,
    },
    OperationTimeout(ScheduledOperationId),
}
impl SchedulerError {
    pub const fn code(&self) -> SchedulerErrorCode {
        match self {
            Self::InvalidExecutionPlan { .. } => SchedulerErrorCode::InvalidExecutionPlan,
            Self::QueueCapacityExceeded { .. } => SchedulerErrorCode::QueueCapacityExceeded,
            Self::ProviderUnavailable(_) => SchedulerErrorCode::ProviderUnavailable,
            Self::DeviceUnavailable(_) => SchedulerErrorCode::DeviceUnavailable,
            Self::ResourceAffinityConflict { .. } => SchedulerErrorCode::ResourceAffinityConflict,
            Self::MemoryPlanInvalid { .. } => SchedulerErrorCode::MemoryPlanInvalid,
            Self::SubmissionFailed { .. } => SchedulerErrorCode::SubmissionFailed,
            Self::CancellationUnsupported(_) => SchedulerErrorCode::CancellationUnsupported,
            Self::CancellationFailed { .. } => SchedulerErrorCode::CancellationFailed,
            Self::ExecutionFailed { .. } => SchedulerErrorCode::ExecutionFailed,
            Self::ExecutionInterrupted { .. } => SchedulerErrorCode::ExecutionInterrupted,
            Self::OperationTimeout(_) => SchedulerErrorCode::OperationTimeout,
        }
    }
}
impl fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidExecutionPlan { reason } => {
                write!(f, "invalid execution plan: {reason}")
            }
            Self::QueueCapacityExceeded { capacity } => {
                write!(f, "scheduler queue capacity {capacity} exceeded")
            }
            Self::ProviderUnavailable(provider) => {
                write!(f, "provider '{provider}' is unavailable before submission")
            }
            Self::DeviceUnavailable(device) => {
                write!(f, "device '{device}' is unavailable before submission")
            }
            Self::ResourceAffinityConflict { reason } => {
                write!(f, "resource affinity conflict: {reason}")
            }
            Self::MemoryPlanInvalid { reason } => write!(f, "memory plan invalid: {reason}"),
            Self::SubmissionFailed { operation, reason } => {
                write!(
                    f,
                    "scheduled operation '{operation}' submission failed: {reason}"
                )
            }
            Self::CancellationUnsupported(operation) => {
                write!(
                    f,
                    "scheduled operation '{operation}' cancellation is unsupported"
                )
            }
            Self::CancellationFailed { operation, reason } => {
                write!(
                    f,
                    "scheduled operation '{operation}' cancellation failed: {reason}"
                )
            }
            Self::ExecutionFailed { operation, reason } => {
                write!(
                    f,
                    "scheduled operation '{operation}' execution failed: {reason}"
                )
            }
            Self::ExecutionInterrupted { operation, reason } => {
                write!(
                    f,
                    "scheduled operation '{operation}' execution interrupted: {reason}"
                )
            }
            Self::OperationTimeout(operation) => {
                write!(f, "scheduled operation '{operation}' timed out")
            }
        }
    }
}
impl Error for SchedulerError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScheduledOperationResult {
    pub state: SchedulingState,
    pub outputs: Vec<ExecutionOutput>,
    pub diagnostics: Vec<SchedulingDiagnostic>,
    pub error: Option<SchedulerError>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScheduledOperation {
    pub id: ScheduledOperationId,
    pub plan: ComputeExecutionPlan,
    pub state: SchedulingState,
    pub accepted_order: u64,
    pub diagnostics: Vec<SchedulingDiagnostic>,
    pub result: Option<ScheduledOperationResult>,
}
impl ScheduledOperation {
    pub fn provider(&self) -> &ProviderBinding {
        &self.plan.provider
    }
    pub fn device(&self) -> Option<&DeviceBinding> {
        self.plan.device.as_ref()
    }
    pub fn state(&self) -> SchedulingState {
        self.state
    }
    pub fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SchedulerQueue {
    capacity: usize,
    order: VecDeque<ScheduledOperationId>,
}
impl SchedulerQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            order: VecDeque::new(),
        }
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn len(&self) -> usize {
        self.order.len()
    }
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
    fn push(&mut self, id: ScheduledOperationId) -> Result<usize, SchedulerError> {
        if self.order.len() >= self.capacity {
            return Err(SchedulerError::QueueCapacityExceeded {
                capacity: self.capacity,
            });
        }
        self.order.push_back(id);
        Ok(self.order.len() - 1)
    }
    fn pop_next(&mut self) -> Option<ScheduledOperationId> {
        self.order.pop_front()
    }
    fn remove(&mut self, id: ScheduledOperationId) -> bool {
        let Some(position) = self.order.iter().position(|candidate| *candidate == id) else {
            return false;
        };
        self.order.remove(position);
        true
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Scheduler {
    queue: SchedulerQueue,
    policy: SchedulingPolicy,
    operations: BTreeMap<ScheduledOperationId, ScheduledOperation>,
    next_order: u64,
}
impl Scheduler {
    pub fn new(policy: SchedulingPolicy, capacity: usize) -> Self {
        Self {
            queue: SchedulerQueue::new(capacity),
            policy,
            operations: BTreeMap::new(),
            next_order: 0,
        }
    }
    pub fn policy(&self) -> SchedulingPolicy {
        self.policy
    }
    pub fn queue(&self) -> &SchedulerQueue {
        &self.queue
    }
    pub fn operation(&self, id: ScheduledOperationId) -> Option<&ScheduledOperation> {
        self.operations.get(&id)
    }
    pub fn schedule(
        &mut self,
        runtime: &Runtime,
        plan: ComputeExecutionPlan,
    ) -> Result<ScheduledOperationId, SchedulerError> {
        runtime.validate_scheduler_plan(&plan).map_err(|error| {
            SchedulerError::InvalidExecutionPlan {
                reason: error.to_string(),
            }
        })?;
        let id = next_scheduled_operation_id();
        let accepted_order = self.next_order;
        self.next_order += 1;
        let mut operation = ScheduledOperation {
            id,
            plan,
            state: SchedulingState::Accepted,
            accepted_order,
            diagnostics: vec![
                SchedulingDiagnostic::Accepted { operation: id },
                SchedulingDiagnostic::QueueTime { accepted_order },
            ],
            result: None,
        };
        let position = match self.queue.push(id) {
            Ok(position) => position,
            Err(error) => {
                operation
                    .diagnostics
                    .push(SchedulingDiagnostic::StableFailureReason(error.code()));
                return Err(error);
            }
        };
        operation.state = SchedulingState::Queued;
        operation.diagnostics.push(SchedulingDiagnostic::Queued {
            operation: id,
            position,
        });
        operation
            .diagnostics
            .push(SchedulingDiagnostic::SelectedProvider(
                operation.plan.provider.clone(),
            ));
        if let Some(device) = &operation.plan.device {
            operation
                .diagnostics
                .push(SchedulingDiagnostic::SelectedDevice(device.clone()));
        }
        self.operations.insert(id, operation);
        Ok(id)
    }
    pub fn submit_next(
        &mut self,
        runtime: &Runtime,
    ) -> Result<Option<ScheduledOperationId>, SchedulerError> {
        let Some(id) = self.queue.pop_next() else {
            return Ok(None);
        };
        let provider = self
            .operations
            .get(&id)
            .map(|operation| operation.plan.provider.clone())
            .ok_or_else(|| SchedulerError::SubmissionFailed {
                operation: id,
                reason: "operation is not registered".into(),
            })?;
        if runtime
            .providers()
            .provider(provider.as_str())
            .map(Provider::health)
            .unwrap_or(ProviderHealth::Unavailable)
            == ProviderHealth::Unavailable
        {
            self.interrupt_operation(id, "selected Provider is unavailable before submission");
            return Err(SchedulerError::ProviderUnavailable(provider));
        }
        if let Some(device) = self
            .operations
            .get(&id)
            .and_then(ScheduledOperation::device)
            && runtime
                .device(device.id())
                .map(Device::availability)
                .unwrap_or(DeviceAvailability::Unavailable)
                == DeviceAvailability::Unavailable
        {
            let device = device.clone();
            self.interrupt_operation(id, "selected Device is unavailable before submission");
            return Err(SchedulerError::DeviceUnavailable(device));
        }
        let operation = self
            .operations
            .get_mut(&id)
            .expect("operation checked above");
        operation.state = SchedulingState::Ready;
        operation.state = SchedulingState::Submitted;
        operation.state = SchedulingState::Running;
        Ok(Some(id))
    }
    pub fn complete(&mut self, id: ScheduledOperationId) -> Result<(), SchedulerError> {
        let operation = self.operation_mut(id)?;
        if operation.state.is_terminal() {
            return Ok(());
        }
        operation.state = SchedulingState::Completed;
        operation
            .diagnostics
            .push(SchedulingDiagnostic::TerminalState(
                SchedulingState::Completed,
            ));
        operation.result = Some(ScheduledOperationResult {
            state: SchedulingState::Completed,
            outputs: operation.plan.outputs.clone(),
            diagnostics: operation.diagnostics.clone(),
            error: None,
        });
        Ok(())
    }
    pub fn fail(
        &mut self,
        id: ScheduledOperationId,
        reason: impl Into<String>,
    ) -> Result<(), SchedulerError> {
        let reason = reason.into();
        let error = SchedulerError::ExecutionFailed {
            operation: id,
            reason: reason.clone(),
        };
        let operation = self.operation_mut(id)?;
        if operation.state.is_terminal() {
            return Ok(());
        }
        operation.state = SchedulingState::Failed;
        operation
            .diagnostics
            .push(SchedulingDiagnostic::StableFailureReason(error.code()));
        operation
            .diagnostics
            .push(SchedulingDiagnostic::TerminalState(SchedulingState::Failed));
        operation.result = Some(ScheduledOperationResult {
            state: SchedulingState::Failed,
            outputs: Vec::new(),
            diagnostics: operation.diagnostics.clone(),
            error: Some(error),
        });
        Ok(())
    }
    pub fn cancel(&mut self, id: ScheduledOperationId) -> Result<(), SchedulerError> {
        let state = self.operation_mut(id)?.state;
        match state {
            SchedulingState::Accepted | SchedulingState::Queued | SchedulingState::Ready => {
                self.queue.remove(id);
                let operation = self.operation_mut(id)?;
                operation.state = SchedulingState::Cancelled;
                operation
                    .diagnostics
                    .push(SchedulingDiagnostic::CancellationRequested);
                operation
                    .diagnostics
                    .push(SchedulingDiagnostic::TerminalState(
                        SchedulingState::Cancelled,
                    ));
                operation.result = Some(ScheduledOperationResult {
                    state: SchedulingState::Cancelled,
                    outputs: Vec::new(),
                    diagnostics: operation.diagnostics.clone(),
                    error: None,
                });
                Ok(())
            }
            SchedulingState::Submitted | SchedulingState::Running => {
                let operation = self.operation_mut(id)?;
                operation
                    .diagnostics
                    .push(SchedulingDiagnostic::CancellationRequested);
                operation
                    .diagnostics
                    .push(SchedulingDiagnostic::CancellationForwardedToProvider(
                        operation.plan.provider.clone(),
                    ));
                Err(SchedulerError::CancellationUnsupported(id))
            }
            SchedulingState::Completed
            | SchedulingState::Cancelled
            | SchedulingState::Failed
            | SchedulingState::Interrupted => Ok(()),
        }
    }
    pub fn result(&self, id: ScheduledOperationId) -> Option<&ScheduledOperationResult> {
        self.operations.get(&id)?.result.as_ref()
    }
    fn operation_mut(
        &mut self,
        id: ScheduledOperationId,
    ) -> Result<&mut ScheduledOperation, SchedulerError> {
        self.operations
            .get_mut(&id)
            .ok_or_else(|| SchedulerError::InvalidExecutionPlan {
                reason: format!("scheduled operation '{id}' is unknown"),
            })
    }
    fn interrupt_operation(&mut self, id: ScheduledOperationId, reason: impl Into<String>) {
        if let Some(operation) = self.operations.get_mut(&id) {
            let error = SchedulerError::ExecutionInterrupted {
                operation: id,
                reason: reason.into(),
            };
            operation.state = SchedulingState::Interrupted;
            operation
                .diagnostics
                .push(SchedulingDiagnostic::StableFailureReason(error.code()));
            operation
                .diagnostics
                .push(SchedulingDiagnostic::TerminalState(
                    SchedulingState::Interrupted,
                ));
            operation.result = Some(ScheduledOperationResult {
                state: SchedulingState::Interrupted,
                outputs: Vec::new(),
                diagnostics: operation.diagnostics.clone(),
                error: Some(error),
            });
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ComputePlanningError {
    PlanningFailed {
        reason: String,
    },
    NoCompatibleProvider {
        capability: CapabilityBinding,
    },
    NoCompatibleDevice {
        provider: ProviderBinding,
    },
    PolicyRejectedProvider {
        capability: CapabilityBinding,
        policy: ResolutionPolicyId,
    },
    UnsupportedOperation(ComputeOperationId),
    UnsupportedDType(ComputeDType),
    UnsupportedLayout(ComputeLayout),
    UnsupportedPrecisionPolicy(ComputePrecision),
    IncompatibleResourceAffinity(AffinityError),
    UnresolvedAffinityGroup(AffinityGroupId),
    MemoryPlanFailed(MemoryPlanningError),
    DataMovementRequired {
        resource: TensorResourceId,
    },
    UnsupportedTransfer {
        reason: String,
    },
    MaterializationRequired {
        source: String,
    },
    ProviderUnavailable(ProviderBinding),
    DeviceUnavailable(DeviceBinding),
    InvalidExecutionPlan {
        reason: String,
    },
}
impl fmt::Display for ComputePlanningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PlanningFailed { reason } => {
                write!(f, "compute execution planning failed: {reason}")
            }
            Self::NoCompatibleProvider { capability } => {
                write!(f, "no compatible provider for capability '{capability}'")
            }
            Self::NoCompatibleDevice { provider } => {
                write!(f, "no compatible device for provider '{provider}'")
            }
            Self::PolicyRejectedProvider { capability, policy } => write!(
                f,
                "resolution policy '{policy}' rejected all providers for capability '{capability}'"
            ),
            Self::UnsupportedOperation(operation) => {
                write!(f, "unsupported operation schema '{operation}'")
            }
            Self::UnsupportedDType(dtype) => write!(f, "unsupported dtype {dtype:?}"),
            Self::UnsupportedLayout(layout) => write!(f, "unsupported layout {layout:?}"),
            Self::UnsupportedPrecisionPolicy(precision) => {
                write!(f, "unsupported precision policy {precision:?}")
            }
            Self::IncompatibleResourceAffinity(error) => {
                write!(f, "incompatible execution resource affinity: {error}")
            }
            Self::UnresolvedAffinityGroup(group) => {
                write!(f, "unresolved affinity group '{group}'")
            }
            Self::MemoryPlanFailed(error) => write!(f, "{error}"),
            Self::DataMovementRequired { resource } => {
                write!(
                    f,
                    "explicit data movement required for resource '{resource}'"
                )
            }
            Self::UnsupportedTransfer { reason } => {
                write!(f, "unsupported execution transfer: {reason}")
            }
            Self::MaterializationRequired { source } => {
                write!(f, "explicit materialization required for '{source}'")
            }
            Self::ProviderUnavailable(provider) => {
                write!(f, "provider '{provider}' is unavailable")
            }
            Self::DeviceUnavailable(device) => write!(f, "device '{device}' is unavailable"),
            Self::InvalidExecutionPlan { reason } => {
                write!(f, "invalid compute execution plan: {reason}")
            }
        }
    }
}
impl Error for ComputePlanningError {}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MemoryRegionKind {
    GraphInput,
    GraphOutput,
    Intermediate,
    Temporary,
    Materialization,
    Transfer,
    HostStaging,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryRequirement {
    pub id: String,
    pub region: MemoryRegionKind,
    pub byte_size: u64,
    pub affinity: ResourceAffinity,
    pub reusable: bool,
}
impl MemoryRequirement {
    pub fn new(
        id: impl Into<String>,
        region: MemoryRegionKind,
        byte_size: u64,
        affinity: ResourceAffinity,
    ) -> Self {
        Self {
            id: id.into(),
            region,
            byte_size,
            affinity,
            reusable: false,
        }
    }
    pub const fn reusable(mut self) -> Self {
        self.reusable = true;
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorLifetime {
    pub id: String,
    pub first_step: usize,
    pub last_step: usize,
    pub byte_size: u64,
    pub affinity: ResourceAffinity,
}
impl TensorLifetime {
    pub fn overlaps(&self, other: &Self) -> bool {
        self.first_step <= other.last_step && other.first_step <= self.last_step
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BufferLifetime {
    pub id: String,
    pub source: String,
    pub first_step: usize,
    pub last_step: usize,
    pub byte_size: u64,
    pub affinity: ResourceAffinity,
    pub reuses: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MemoryPressureReport {
    pub estimated_required_bytes: u64,
    pub estimated_peak_bytes: u64,
    pub selected_provider: Option<ProviderBinding>,
    pub selected_device: Option<DeviceBinding>,
    pub rejected_device_limit: Option<u64>,
    pub materialization_cost_bytes: u64,
    pub transfer_buffer_cost_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemoryPlanningDecision {
    Allocate { requirement: String },
    Reuse { requirement: String, buffer: String },
    PreservePinnedResource { resource: TensorResourceId },
    RequireMaterialization { requirement: String },
    RequireTransfer { requirement: String },
    AccountHostStaging { requirement: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemoryPlanningDiagnostic {
    EstimatedRequirement {
        requirement: String,
        bytes: u64,
    },
    PeakBytes {
        bytes: u64,
    },
    ProviderLimit {
        provider: ProviderBinding,
        max_bytes: u64,
    },
    DeviceLimit {
        device: DeviceBinding,
        max_bytes: u64,
    },
    MaterializationCost {
        bytes: u64,
    },
    TransferCost {
        bytes: u64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryPlan {
    pub provider: ProviderBinding,
    pub graph: Option<ComputeGraphId>,
    pub requirements: Vec<MemoryRequirement>,
    pub tensor_lifetimes: Vec<TensorLifetime>,
    pub buffer_lifetimes: Vec<BufferLifetime>,
    pub pressure: MemoryPressureReport,
    pub decisions: Vec<MemoryPlanningDecision>,
    pub diagnostics: Vec<MemoryPlanningDiagnostic>,
    pub output_affinity: ResourceAffinity,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemoryPlanningError {
    MemoryPlanningFailed {
        reason: String,
        report: MemoryPressureReport,
    },
    OutOfMemory {
        required: u64,
        limit: u64,
        report: MemoryPressureReport,
    },
    ResourceExhausted {
        reason: String,
        report: MemoryPressureReport,
    },
    SizeOverflow {
        reason: String,
        report: MemoryPressureReport,
    },
    IncompatibleResourceAffinity(AffinityError),
    UnsupportedLayout {
        layout: ComputeLayout,
        report: MemoryPressureReport,
    },
    MaterializationRequired {
        reason: String,
        report: MemoryPressureReport,
    },
    TransferRequired {
        reason: String,
        report: MemoryPressureReport,
    },
    ProviderMemoryLimitExceeded {
        provider: ProviderBinding,
        required: u64,
        limit: u64,
        report: MemoryPressureReport,
    },
    DeviceMemoryLimitExceeded {
        device: DeviceBinding,
        required: u64,
        limit: u64,
        report: MemoryPressureReport,
    },
}
impl fmt::Display for MemoryPlanningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MemoryPlanningFailed { reason, .. } => {
                write!(f, "memory planning failed: {reason}")
            }
            Self::OutOfMemory {
                required, limit, ..
            } => write!(
                f,
                "memory planning requires {required} bytes but only {limit} bytes are available"
            ),
            Self::ResourceExhausted { reason, .. } => write!(f, "resource exhausted: {reason}"),
            Self::SizeOverflow { reason, .. } => write!(f, "memory size overflow: {reason}"),
            Self::IncompatibleResourceAffinity(error) => {
                write!(f, "incompatible memory resource affinity: {error}")
            }
            Self::UnsupportedLayout { layout, .. } => {
                write!(f, "memory planning does not support layout {layout:?}")
            }
            Self::MaterializationRequired { reason, .. } => {
                write!(f, "memory planning requires materialization: {reason}")
            }
            Self::TransferRequired { reason, .. } => {
                write!(f, "memory planning requires explicit transfer: {reason}")
            }
            Self::ProviderMemoryLimitExceeded {
                provider,
                required,
                limit,
                ..
            } => write!(
                f,
                "provider '{provider}' memory limit exceeded: required {required}, limit {limit}"
            ),
            Self::DeviceMemoryLimitExceeded {
                device,
                required,
                limit,
                ..
            } => write!(
                f,
                "device '{device}' memory limit exceeded: required {required}, limit {limit}"
            ),
        }
    }
}
impl Error for MemoryPlanningError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeGraphValidationReport {
    pub provider: ProviderBinding,
    pub graph: ComputeGraphId,
    pub node_count: usize,
    pub input_count: usize,
    pub output_count: usize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeErrorPhase {
    Validation,
    Resolution,
    AffinityValidation,
    Planning,
    DataMovement,
    Materialization,
    MemoryPlanning,
    Submission,
    Execution,
    Cancellation,
    Completion,
    Interruption,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeErrorSeverity {
    Recoverable,
    Terminal,
    Fatal,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RecoveryHint {
    NotRetryable,
    RetryBeforeState,
    RestartableWithReplay,
    ExplicitTransferRequired,
    ExplicitMaterializationRequired,
    ProviderPinned,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeErrorCode {
    InvalidTensorDescriptor,
    InvalidShape,
    InvalidDType,
    InvalidLayout,
    InvalidOperationAttribute,
    InvalidOperationArity,
    InvalidOutputDescriptor,
    SizeOverflow,
    InvalidGraph,
    CyclicGraph,
    MissingInput,
    MissingOutput,
    NoCompatibleProvider,
    NoCompatibleDevice,
    PolicyRejectedProvider,
    ProviderUnavailable,
    DeviceUnavailable,
    CapabilityVersionMismatch,
    UnsupportedOperation,
    UnsupportedOperationFamily,
    UnsupportedDType,
    UnsupportedLayout,
    UnsupportedDataMovement,
    IncompatibleResourceAffinity,
    ProviderPinnedResource,
    DeviceBoundResource,
    ArtifactFingerprintMismatch,
    AffinityGroupMismatch,
    ExecutionFailed,
    ExecutionInterrupted,
    ExecutionCancelled,
    OperationTimeout,
    PlanningFailed,
    InvalidExecutionPlan,
    DataMovementRequired,
    UnsupportedTransfer,
    MemoryPlanningFailed,
    OutOfMemory,
    ResourceExhausted,
    ProviderMemoryLimitExceeded,
    DeviceMemoryLimitExceeded,
    InvalidHostBuffer,
    InvalidTransfer,
    UnsupportedConversion,
    MaterializationRequired,
    InvalidState,
    Internal,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ComputeDiagnostic {
    pub provider: Option<ProviderBinding>,
    pub device: Option<DeviceBinding>,
    pub capability: Option<CapabilityBinding>,
    pub operation_family: Option<ComputeOperationFamily>,
    pub rejected_candidates: Vec<ProviderBinding>,
    pub backend_message: Option<String>,
    pub debug_trace_id: Option<String>,
}
impl ComputeDiagnostic {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_provider(mut self, provider: ProviderBinding) -> Self {
        self.provider = Some(provider);
        self
    }
    pub fn with_device(mut self, device: DeviceBinding) -> Self {
        self.device = Some(device);
        self
    }
    pub fn with_capability(mut self, capability: CapabilityBinding) -> Self {
        self.capability = Some(capability);
        self
    }
    pub fn with_operation_family(mut self, family: ComputeOperationFamily) -> Self {
        self.operation_family = Some(family);
        self
    }
    pub fn with_rejected_candidate(mut self, provider: ProviderBinding) -> Self {
        self.rejected_candidates.push(provider);
        self
    }
    pub fn with_backend_message(mut self, message: impl AsRef<str>) -> Self {
        self.backend_message = Some(redact_backend_diagnostic(message.as_ref()));
        self
    }
    pub fn with_debug_trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.debug_trace_id = Some(trace_id.into());
        self
    }
}

fn redact_backend_diagnostic(message: &str) -> String {
    let contains_native_handle = message.contains("0x") || message.contains("handle=");
    let contains_path = message.contains('\\') || message.contains('/');
    if contains_native_handle || contains_path {
        "[redacted backend diagnostic]".into()
    } else {
        message.into()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeError {
    pub code: ComputeErrorCode,
    pub phase: ComputeErrorPhase,
    pub severity: ComputeErrorSeverity,
    pub message: String,
    pub diagnostics: Vec<ComputeDiagnostic>,
    pub recovery_hints: Vec<RecoveryHint>,
}
impl ComputeError {
    pub fn new(
        code: ComputeErrorCode,
        phase: ComputeErrorPhase,
        severity: ComputeErrorSeverity,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code,
            phase,
            severity,
            message: message.into(),
            diagnostics: Vec::new(),
            recovery_hints: Vec::new(),
        }
    }
    pub fn with_diagnostic(mut self, diagnostic: ComputeDiagnostic) -> Self {
        self.diagnostics.push(diagnostic);
        self
    }
    pub fn with_recovery_hint(mut self, hint: RecoveryHint) -> Self {
        if !self.recovery_hints.contains(&hint) {
            self.recovery_hints.push(hint);
        }
        self
    }
    pub fn validation(code: ComputeErrorCode, message: impl Into<String>) -> Self {
        Self::new(
            code,
            ComputeErrorPhase::Validation,
            ComputeErrorSeverity::Terminal,
            message,
        )
        .with_recovery_hint(RecoveryHint::NotRetryable)
    }
}
impl fmt::Display for ComputeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "compute error {:?} during {:?}: {}",
            self.code, self.phase, self.message
        )
    }
}
impl Error for ComputeError {}

fn ensure_non_empty_id(kind: &str, value: &str) -> Result<(), ComputeValidationError> {
    if value.trim().is_empty() {
        return Err(ComputeValidationError::InvalidGraph {
            reason: format!("{kind} identifier must not be empty"),
        });
    }
    Ok(())
}

fn insert_unique<T: Ord + Clone + fmt::Display>(
    ids: &mut BTreeSet<T>,
    kind: &str,
    id: &T,
) -> Result<(), ComputeValidationError> {
    if !ids.insert(id.clone()) {
        return Err(ComputeValidationError::InvalidGraph {
            reason: format!("duplicate {kind} identifier '{id}'"),
        });
    }
    Ok(())
}

fn resolve_compute_value_descriptor<'a>(
    current_node: Option<&ComputeNodeId>,
    value: &ComputeValueRef,
    input_descriptors: &'a BTreeMap<ComputeInputId, TensorDescriptor>,
    output_descriptors: &'a BTreeMap<(ComputeNodeId, ComputeOutputId), TensorDescriptor>,
    completed_nodes: &BTreeSet<ComputeNodeId>,
) -> Result<&'a TensorDescriptor, ComputeValidationError> {
    match value {
        ComputeValueRef::Input(input) => {
            input_descriptors
                .get(input)
                .ok_or_else(|| ComputeValidationError::MissingInput {
                    input: input.clone(),
                })
        }
        ComputeValueRef::NodeOutput { node, output } => {
            if !completed_nodes.contains(node) {
                return Err(ComputeValidationError::CyclicGraph {
                    node: current_node.cloned().unwrap_or_else(|| node.clone()),
                    depends_on: node.clone(),
                });
            }
            output_descriptors
                .get(&(node.clone(), output.clone()))
                .ok_or_else(|| ComputeValidationError::MissingOutput {
                    node: node.clone(),
                    output: output.clone(),
                })
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ComputeValidationError {
    UnknownOperationFamily(String),
    UnknownOperationSchema(ComputeOperationId),
    InvalidGraph {
        reason: String,
    },
    MissingInput {
        input: ComputeInputId,
    },
    MissingOutput {
        node: ComputeNodeId,
        output: ComputeOutputId,
    },
    CyclicGraph {
        node: ComputeNodeId,
        depends_on: ComputeNodeId,
    },
    InvalidState {
        reason: String,
    },
    InvalidOperationAttribute {
        operation: ComputeOperationId,
        attribute: String,
        reason: String,
    },
    InvalidOperationArity {
        operation: ComputeOperationId,
        expected: String,
        found: usize,
    },
    InvalidOutputDescriptor {
        operation: ComputeOperationId,
        reason: String,
    },
    InvalidShape {
        reason: String,
    },
    InvalidLayout {
        reason: String,
    },
    SizeOverflow {
        reason: String,
    },
    UnsupportedOperationFamily {
        provider: ProviderBinding,
        family: ComputeOperationFamily,
    },
    UnsupportedOperationSchema {
        provider: ProviderBinding,
        operation: ComputeOperationId,
    },
    UnsupportedAdvertisement {
        provider: ProviderBinding,
        reason: String,
    },
    UnsupportedDType {
        family: ComputeOperationFamily,
        dtype: ComputeDType,
    },
    UnsupportedProviderDType {
        family: ComputeOperationFamily,
        dtype: String,
    },
    UnsupportedLayout {
        family: ComputeOperationFamily,
        layout: ComputeLayout,
    },
    UnsupportedPrecision {
        family: ComputeOperationFamily,
        precision: ComputePrecision,
    },
    UnsupportedDataMovement {
        provider: ProviderBinding,
        kind: ComputeDataMovementKind,
    },
    InvalidHostBuffer {
        reason: String,
    },
    InvalidTransfer {
        reason: String,
    },
    UnsupportedConversion {
        reason: String,
    },
    MaterializationRequired {
        reason: String,
    },
    MemoryPlanning(MemoryPlanningError),
    ProviderUnavailable(ProviderBinding),
    IncompatibleResourceAffinity(AffinityError),
}
impl fmt::Display for ComputeValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownOperationFamily(family) => {
                write!(f, "unknown compute operation family '{family}'")
            }
            Self::UnknownOperationSchema(operation) => {
                write!(f, "unknown compute operation schema '{operation}'")
            }
            Self::InvalidGraph { reason } => write!(f, "invalid compute graph: {reason}"),
            Self::MissingInput { input } => {
                write!(f, "compute graph references missing input '{input}'")
            }
            Self::MissingOutput { node, output } => write!(
                f,
                "compute graph references missing output '{output}' on node '{node}'"
            ),
            Self::CyclicGraph { node, depends_on } => write!(
                f,
                "compute graph node '{node}' depends on future or cyclic node '{depends_on}'"
            ),
            Self::InvalidState { reason } => {
                write!(f, "invalid compute submission state: {reason}")
            }
            Self::InvalidOperationAttribute {
                operation,
                attribute,
                reason,
            } => write!(
                f,
                "invalid attribute '{attribute}' for operation schema '{operation}': {reason}"
            ),
            Self::InvalidOperationArity {
                operation,
                expected,
                found,
            } => write!(
                f,
                "invalid arity for operation schema '{operation}': expected {expected}, found {found}"
            ),
            Self::InvalidOutputDescriptor { operation, reason } => write!(
                f,
                "invalid output descriptor for operation schema '{operation}': {reason}"
            ),
            Self::InvalidShape { reason } => write!(f, "invalid tensor shape: {reason}"),
            Self::InvalidLayout { reason } => write!(f, "invalid tensor layout: {reason}"),
            Self::SizeOverflow { reason } => write!(f, "invalid tensor size: {reason}"),
            Self::UnsupportedOperationFamily { provider, family } => write!(
                f,
                "provider '{provider}' does not support compute operation family '{}'",
                family.id()
            ),
            Self::UnsupportedOperationSchema {
                provider,
                operation,
            } => write!(
                f,
                "provider '{provider}' does not support compute operation schema '{operation}'"
            ),
            Self::UnsupportedAdvertisement { provider, reason } => {
                write!(
                    f,
                    "provider '{provider}' compute advertisement is unsupported: {reason}"
                )
            }
            Self::UnsupportedDType { family, dtype } => write!(
                f,
                "compute operation family '{}' does not support dtype {dtype:?}",
                family.id()
            ),
            Self::UnsupportedProviderDType { family, dtype } => write!(
                f,
                "compute operation family '{}' does not support provider-specific dtype '{dtype}'",
                family.id()
            ),
            Self::UnsupportedLayout { family, layout } => write!(
                f,
                "compute operation family '{}' does not support layout {layout:?}",
                family.id()
            ),
            Self::UnsupportedPrecision { family, precision } => write!(
                f,
                "compute operation family '{}' does not support precision {precision:?}",
                family.id()
            ),
            Self::UnsupportedDataMovement { provider, kind } => write!(
                f,
                "provider '{provider}' does not support compute data movement '{}'",
                kind.id()
            ),
            Self::InvalidHostBuffer { reason } => {
                write!(f, "invalid host buffer: {reason}")
            }
            Self::InvalidTransfer { reason } => {
                write!(f, "invalid compute data transfer: {reason}")
            }
            Self::UnsupportedConversion { reason } => {
                write!(f, "unsupported compute data conversion: {reason}")
            }
            Self::MaterializationRequired { reason } => {
                write!(f, "materialization required: {reason}")
            }
            Self::MemoryPlanning(error) => {
                write!(f, "{error}")
            }
            Self::ProviderUnavailable(provider) => {
                write!(f, "provider '{provider}' is unavailable")
            }
            Self::IncompatibleResourceAffinity(error) => {
                write!(f, "incompatible tensor resource affinity: {error}")
            }
        }
    }
}
impl Error for ComputeValidationError {}

impl From<ComputeValidationError> for ComputeError {
    fn from(error: ComputeValidationError) -> Self {
        let message = error.to_string();
        match error {
            ComputeValidationError::UnknownOperationFamily(family) => {
                ComputeError::validation(ComputeErrorCode::UnsupportedOperationFamily, message)
                    .with_diagnostic(
                        ComputeDiagnostic::new()
                            .with_backend_message(format!("unknown operation family: {family}")),
                    )
            }
            ComputeValidationError::UnknownOperationSchema(operation) => {
                ComputeError::validation(ComputeErrorCode::UnsupportedOperation, message)
                    .with_diagnostic(
                        ComputeDiagnostic::new()
                            .with_backend_message(format!("unknown operation schema: {operation}")),
                    )
            }
            ComputeValidationError::InvalidGraph { .. } => {
                ComputeError::validation(ComputeErrorCode::InvalidGraph, message)
            }
            ComputeValidationError::MissingInput { .. } => {
                ComputeError::validation(ComputeErrorCode::MissingInput, message)
            }
            ComputeValidationError::MissingOutput { .. } => {
                ComputeError::validation(ComputeErrorCode::MissingOutput, message)
            }
            ComputeValidationError::CyclicGraph { .. } => {
                ComputeError::validation(ComputeErrorCode::CyclicGraph, message)
            }
            ComputeValidationError::InvalidState { .. } => ComputeError::new(
                ComputeErrorCode::InvalidState,
                ComputeErrorPhase::Submission,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputeValidationError::InvalidOperationAttribute { .. } => {
                ComputeError::validation(ComputeErrorCode::InvalidOperationAttribute, message)
            }
            ComputeValidationError::InvalidOperationArity { .. } => {
                ComputeError::validation(ComputeErrorCode::InvalidOperationArity, message)
            }
            ComputeValidationError::InvalidOutputDescriptor { .. } => {
                ComputeError::validation(ComputeErrorCode::InvalidOutputDescriptor, message)
            }
            ComputeValidationError::InvalidShape { .. } => {
                ComputeError::validation(ComputeErrorCode::InvalidShape, message)
            }
            ComputeValidationError::InvalidLayout { .. } => {
                ComputeError::validation(ComputeErrorCode::InvalidLayout, message)
            }
            ComputeValidationError::SizeOverflow { .. } => {
                ComputeError::validation(ComputeErrorCode::SizeOverflow, message)
            }
            ComputeValidationError::UnsupportedOperationFamily { provider, family } => {
                ComputeError::validation(ComputeErrorCode::UnsupportedOperationFamily, message)
                    .with_diagnostic(
                        ComputeDiagnostic::new()
                            .with_provider(provider)
                            .with_operation_family(family),
                    )
            }
            ComputeValidationError::UnsupportedOperationSchema {
                provider,
                operation,
            } => ComputeError::validation(ComputeErrorCode::UnsupportedOperation, message)
                .with_diagnostic(
                    ComputeDiagnostic::new()
                        .with_provider(provider)
                        .with_backend_message(format!("unsupported operation schema: {operation}")),
                ),
            ComputeValidationError::UnsupportedAdvertisement { provider, reason } => {
                ComputeError::validation(ComputeErrorCode::NoCompatibleProvider, message)
                    .with_diagnostic(
                        ComputeDiagnostic::new()
                            .with_provider(provider)
                            .with_backend_message(reason),
                    )
            }
            ComputeValidationError::UnsupportedDType { family, .. }
            | ComputeValidationError::UnsupportedProviderDType { family, .. } => {
                ComputeError::validation(ComputeErrorCode::UnsupportedDType, message)
                    .with_diagnostic(ComputeDiagnostic::new().with_operation_family(family))
            }
            ComputeValidationError::UnsupportedLayout { family, layout: _ } => {
                ComputeError::validation(ComputeErrorCode::UnsupportedLayout, message)
                    .with_diagnostic(ComputeDiagnostic::new().with_operation_family(family))
            }
            ComputeValidationError::UnsupportedPrecision { family, .. } => {
                ComputeError::validation(ComputeErrorCode::UnsupportedOperation, message)
                    .with_diagnostic(ComputeDiagnostic::new().with_operation_family(family))
            }
            ComputeValidationError::UnsupportedDataMovement { provider, .. } => {
                ComputeError::validation(ComputeErrorCode::UnsupportedDataMovement, message)
                    .with_diagnostic(ComputeDiagnostic::new().with_provider(provider))
                    .with_recovery_hint(RecoveryHint::ExplicitTransferRequired)
            }
            ComputeValidationError::InvalidHostBuffer { .. } => {
                ComputeError::validation(ComputeErrorCode::InvalidHostBuffer, message)
            }
            ComputeValidationError::InvalidTransfer { .. } => {
                ComputeError::validation(ComputeErrorCode::InvalidTransfer, message)
                    .with_recovery_hint(RecoveryHint::ExplicitTransferRequired)
            }
            ComputeValidationError::UnsupportedConversion { .. } => {
                ComputeError::validation(ComputeErrorCode::UnsupportedConversion, message)
            }
            ComputeValidationError::MaterializationRequired { .. } => {
                ComputeError::validation(ComputeErrorCode::MaterializationRequired, message)
                    .with_recovery_hint(RecoveryHint::ExplicitMaterializationRequired)
            }
            ComputeValidationError::MemoryPlanning(error) => ComputeError::from(error),
            ComputeValidationError::ProviderUnavailable(provider) => ComputeError::new(
                ComputeErrorCode::ProviderUnavailable,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_provider(provider))
            .with_recovery_hint(RecoveryHint::RetryBeforeState),
            ComputeValidationError::IncompatibleResourceAffinity(error) => {
                ComputeError::from(error)
            }
        }
    }
}

impl From<ComputePlanningError> for ComputeError {
    fn from(error: ComputePlanningError) -> Self {
        let message = error.to_string();
        match error {
            ComputePlanningError::PlanningFailed { .. } => ComputeError::new(
                ComputeErrorCode::PlanningFailed,
                ComputeErrorPhase::Planning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputePlanningError::NoCompatibleProvider { capability } => ComputeError::new(
                ComputeErrorCode::NoCompatibleProvider,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_capability(capability))
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputePlanningError::NoCompatibleDevice { provider } => ComputeError::new(
                ComputeErrorCode::NoCompatibleDevice,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_provider(provider))
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputePlanningError::PolicyRejectedProvider { capability, .. } => ComputeError::new(
                ComputeErrorCode::PolicyRejectedProvider,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_capability(capability))
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputePlanningError::UnsupportedOperation(operation) => ComputeError::new(
                ComputeErrorCode::UnsupportedOperation,
                ComputeErrorPhase::Planning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(
                ComputeDiagnostic::new()
                    .with_backend_message(format!("unsupported operation schema: {operation}")),
            )
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputePlanningError::UnsupportedDType(_) => ComputeError::new(
                ComputeErrorCode::UnsupportedDType,
                ComputeErrorPhase::Planning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputePlanningError::UnsupportedLayout(_) => ComputeError::new(
                ComputeErrorCode::UnsupportedLayout,
                ComputeErrorPhase::Planning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputePlanningError::UnsupportedPrecisionPolicy(_) => ComputeError::new(
                ComputeErrorCode::UnsupportedOperation,
                ComputeErrorPhase::Planning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputePlanningError::IncompatibleResourceAffinity(error) => ComputeError::from(error),
            ComputePlanningError::UnresolvedAffinityGroup(_) => ComputeError::new(
                ComputeErrorCode::AffinityGroupMismatch,
                ComputeErrorPhase::Planning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::NotRetryable),
            ComputePlanningError::MemoryPlanFailed(error) => ComputeError::from(error),
            ComputePlanningError::DataMovementRequired { .. } => ComputeError::new(
                ComputeErrorCode::DataMovementRequired,
                ComputeErrorPhase::DataMovement,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::ExplicitTransferRequired),
            ComputePlanningError::UnsupportedTransfer { .. } => ComputeError::new(
                ComputeErrorCode::UnsupportedTransfer,
                ComputeErrorPhase::DataMovement,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::ExplicitTransferRequired),
            ComputePlanningError::MaterializationRequired { .. } => ComputeError::new(
                ComputeErrorCode::MaterializationRequired,
                ComputeErrorPhase::Materialization,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::ExplicitMaterializationRequired),
            ComputePlanningError::ProviderUnavailable(provider) => ComputeError::new(
                ComputeErrorCode::ProviderUnavailable,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_provider(provider))
            .with_recovery_hint(RecoveryHint::RetryBeforeState),
            ComputePlanningError::DeviceUnavailable(device) => ComputeError::new(
                ComputeErrorCode::DeviceUnavailable,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_device(device))
            .with_recovery_hint(RecoveryHint::RetryBeforeState),
            ComputePlanningError::InvalidExecutionPlan { .. } => ComputeError::new(
                ComputeErrorCode::InvalidExecutionPlan,
                ComputeErrorPhase::Planning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_recovery_hint(RecoveryHint::NotRetryable),
        }
    }
}

impl From<MemoryPlanningError> for ComputeError {
    fn from(error: MemoryPlanningError) -> Self {
        let message = error.to_string();
        match error {
            MemoryPlanningError::MemoryPlanningFailed { report, .. } => ComputeError::new(
                ComputeErrorCode::MemoryPlanningFailed,
                ComputeErrorPhase::MemoryPlanning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(memory_pressure_diagnostic(&report))
            .with_recovery_hint(RecoveryHint::NotRetryable),
            MemoryPlanningError::OutOfMemory { report, .. } => ComputeError::new(
                ComputeErrorCode::OutOfMemory,
                ComputeErrorPhase::MemoryPlanning,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(memory_pressure_diagnostic(&report)),
            MemoryPlanningError::ResourceExhausted { report, .. } => ComputeError::new(
                ComputeErrorCode::ResourceExhausted,
                ComputeErrorPhase::MemoryPlanning,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(memory_pressure_diagnostic(&report)),
            MemoryPlanningError::SizeOverflow { report, .. } => ComputeError::new(
                ComputeErrorCode::SizeOverflow,
                ComputeErrorPhase::MemoryPlanning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(memory_pressure_diagnostic(&report)),
            MemoryPlanningError::IncompatibleResourceAffinity(error) => ComputeError::from(error),
            MemoryPlanningError::UnsupportedLayout { report, .. } => ComputeError::new(
                ComputeErrorCode::UnsupportedLayout,
                ComputeErrorPhase::MemoryPlanning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(memory_pressure_diagnostic(&report)),
            MemoryPlanningError::MaterializationRequired { report, .. } => ComputeError::new(
                ComputeErrorCode::MaterializationRequired,
                ComputeErrorPhase::MemoryPlanning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(memory_pressure_diagnostic(&report))
            .with_recovery_hint(RecoveryHint::ExplicitMaterializationRequired),
            MemoryPlanningError::TransferRequired { report, .. } => ComputeError::new(
                ComputeErrorCode::InvalidTransfer,
                ComputeErrorPhase::MemoryPlanning,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(memory_pressure_diagnostic(&report))
            .with_recovery_hint(RecoveryHint::ExplicitTransferRequired),
            MemoryPlanningError::ProviderMemoryLimitExceeded { report, .. } => ComputeError::new(
                ComputeErrorCode::ProviderMemoryLimitExceeded,
                ComputeErrorPhase::MemoryPlanning,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(memory_pressure_diagnostic(&report)),
            MemoryPlanningError::DeviceMemoryLimitExceeded { report, .. } => ComputeError::new(
                ComputeErrorCode::DeviceMemoryLimitExceeded,
                ComputeErrorPhase::MemoryPlanning,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(memory_pressure_diagnostic(&report)),
        }
    }
}

fn memory_pressure_diagnostic(report: &MemoryPressureReport) -> ComputeDiagnostic {
    let mut diagnostic = ComputeDiagnostic::new();
    if let Some(provider) = &report.selected_provider {
        diagnostic = diagnostic.with_provider(provider.clone());
    }
    if let Some(device) = &report.selected_device {
        diagnostic = diagnostic.with_device(device.clone());
    }
    diagnostic.with_backend_message(format!(
        "memory pressure: required={} peak={} materialization={} transfer={}",
        report.estimated_required_bytes,
        report.estimated_peak_bytes,
        report.materialization_cost_bytes,
        report.transfer_buffer_cost_bytes
    ))
}

impl MemoryPlan {
    fn new(
        provider: ProviderBinding,
        graph: Option<ComputeGraphId>,
        execution_context: ExecutionContextId,
    ) -> Self {
        let output_affinity = ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(provider.clone())
            .with_capability(CapabilityBinding::new(
                CapabilityId::new(COMPUTE_CAPABILITY_ID),
                COMPUTE_CAPABILITY_VERSION,
            ))
            .with_execution_context(execution_context);
        Self {
            provider: provider.clone(),
            graph,
            requirements: Vec::new(),
            tensor_lifetimes: Vec::new(),
            buffer_lifetimes: Vec::new(),
            pressure: MemoryPressureReport {
                selected_provider: Some(provider),
                ..MemoryPressureReport::default()
            },
            decisions: Vec::new(),
            diagnostics: Vec::new(),
            output_affinity,
        }
    }
    fn add_requirement(
        &mut self,
        requirement: MemoryRequirement,
    ) -> Result<(), MemoryPlanningError> {
        self.pressure.estimated_required_bytes = self
            .pressure
            .estimated_required_bytes
            .checked_add(requirement.byte_size)
            .ok_or_else(|| MemoryPlanningError::SizeOverflow {
                reason: "total memory requirements overflow u64".into(),
                report: self.pressure.clone(),
            })?;
        self.pressure.estimated_peak_bytes = self
            .pressure
            .estimated_peak_bytes
            .checked_add(requirement.byte_size)
            .ok_or_else(|| MemoryPlanningError::SizeOverflow {
                reason: "peak memory requirements overflow u64".into(),
                report: self.pressure.clone(),
            })?;
        self.diagnostics
            .push(MemoryPlanningDiagnostic::EstimatedRequirement {
                requirement: requirement.id.clone(),
                bytes: requirement.byte_size,
            });
        self.diagnostics.push(MemoryPlanningDiagnostic::PeakBytes {
            bytes: self.pressure.estimated_peak_bytes,
        });
        self.requirements.push(requirement);
        Ok(())
    }
    fn find_reusable_buffer(&self, lifetime: &TensorLifetime) -> Option<String> {
        self.buffer_lifetimes
            .iter()
            .find(|buffer| {
                buffer.byte_size >= lifetime.byte_size
                    && buffer.last_step < lifetime.first_step
                    && buffer.affinity.validate_with(&lifetime.affinity).is_ok()
            })
            .map(|buffer| buffer.id.clone())
    }
}

fn provider_memory_limit(metadata: &ProviderMetadata) -> u64 {
    let advertisement = effective_compute_advertisement(metadata);
    advertisement
        .operation_families
        .values()
        .map(|support| support.shapes.descriptor_limits.max_bytes)
        .chain(
            advertisement
                .operation_schemas
                .values()
                .map(|support| support.shapes.descriptor_limits.max_bytes),
        )
        .chain(
            advertisement
                .data_movement
                .values()
                .map(|support| support.shapes.descriptor_limits.max_bytes),
        )
        .min()
        .unwrap_or(u64::MAX)
}

fn memory_bytes(
    descriptor: &TensorDescriptor,
    report: &MemoryPressureReport,
) -> Result<u64, MemoryPlanningError> {
    descriptor
        .byte_size()
        .map_err(|error| MemoryPlanningError::SizeOverflow {
            reason: error.to_string(),
            report: report.clone(),
        })
}

fn memory_error_from_compute_validation(error: ComputeValidationError) -> MemoryPlanningError {
    match error {
        ComputeValidationError::SizeOverflow { reason } => MemoryPlanningError::SizeOverflow {
            reason,
            report: MemoryPressureReport::default(),
        },
        ComputeValidationError::IncompatibleResourceAffinity(error) => {
            MemoryPlanningError::IncompatibleResourceAffinity(error)
        }
        other => MemoryPlanningError::MemoryPlanningFailed {
            reason: other.to_string(),
            report: MemoryPressureReport::default(),
        },
    }
}

fn last_use_for_input(graph: &ComputeGraph, input: &ComputeInputId) -> Option<usize> {
    let mut last = None;
    for (index, node) in graph.nodes.iter().enumerate() {
        if node
            .inputs
            .iter()
            .any(|value| matches!(value, ComputeValueRef::Input(id) if id == input))
        {
            last = Some(index + 1);
        }
    }
    if graph
        .outputs
        .iter()
        .any(|output| matches!(&output.source, ComputeValueRef::Input(id) if id == input))
    {
        last = Some(graph.nodes.len() + 1);
    }
    last
}

fn last_use_for_node_output(
    graph: &ComputeGraph,
    node: &ComputeNodeId,
    output: &ComputeOutputId,
) -> Option<usize> {
    let mut last = None;
    for (index, candidate) in graph.nodes.iter().enumerate() {
        if candidate.inputs.iter().any(|value| {
            matches!(
                value,
                ComputeValueRef::NodeOutput {
                    node: candidate_node,
                    output: candidate_output,
                } if candidate_node == node && candidate_output == output
            )
        }) {
            last = Some(index + 1);
        }
    }
    if graph_output_uses(graph, node, output) {
        last = Some(graph.nodes.len() + 1);
    }
    last
}

fn graph_output_uses(graph: &ComputeGraph, node: &ComputeNodeId, output: &ComputeOutputId) -> bool {
    graph.outputs.iter().any(|graph_output| {
        matches!(
            &graph_output.source,
            ComputeValueRef::NodeOutput {
                node: candidate_node,
                output: candidate_output,
            } if candidate_node == node && candidate_output == output
        )
    })
}

fn planning_error_from_affinity(error: AffinityError) -> ComputePlanningError {
    match error {
        AffinityError::NoCompatibleProvider(capability) => {
            ComputePlanningError::NoCompatibleProvider { capability }
        }
        AffinityError::PolicyRejectedProvider { capability, policy } => {
            ComputePlanningError::PolicyRejectedProvider { capability, policy }
        }
        AffinityError::BoundProviderUnavailable(provider) => {
            ComputePlanningError::ProviderUnavailable(provider)
        }
        AffinityError::BoundDeviceUnavailable(device) => {
            ComputePlanningError::DeviceUnavailable(device)
        }
        other => ComputePlanningError::IncompatibleResourceAffinity(other),
    }
}

fn planning_error_from_validation(error: ComputeValidationError) -> ComputePlanningError {
    match error {
        ComputeValidationError::UnknownOperationSchema(operation)
        | ComputeValidationError::UnsupportedOperationSchema { operation, .. } => {
            ComputePlanningError::UnsupportedOperation(operation)
        }
        ComputeValidationError::UnsupportedDType { dtype, .. } => {
            ComputePlanningError::UnsupportedDType(dtype)
        }
        ComputeValidationError::UnsupportedLayout { layout, .. } => {
            ComputePlanningError::UnsupportedLayout(layout)
        }
        ComputeValidationError::UnsupportedPrecision { precision, .. } => {
            ComputePlanningError::UnsupportedPrecisionPolicy(precision)
        }
        ComputeValidationError::UnsupportedDataMovement { kind, .. } => {
            ComputePlanningError::UnsupportedTransfer {
                reason: format!("provider does not advertise '{}'", kind.id()),
            }
        }
        ComputeValidationError::MaterializationRequired { reason } => {
            ComputePlanningError::MaterializationRequired { source: reason }
        }
        ComputeValidationError::MemoryPlanning(error) => {
            ComputePlanningError::MemoryPlanFailed(error)
        }
        ComputeValidationError::ProviderUnavailable(provider) => {
            ComputePlanningError::ProviderUnavailable(provider)
        }
        ComputeValidationError::IncompatibleResourceAffinity(error) => {
            ComputePlanningError::IncompatibleResourceAffinity(error)
        }
        other => ComputePlanningError::PlanningFailed {
            reason: other.to_string(),
        },
    }
}

fn execution_plan_id(graph: &ComputeGraphId, provider: &ProviderBinding) -> ExecutionPlanId {
    ExecutionPlanId::new(format!("plan:{graph}:{provider}"))
}

fn classify_execution_plan(inputs: &[ExecutionInput]) -> ComputeExecutionClassification {
    if inputs.iter().any(|input| {
        input.resource.is_some() && input.affinity.fallback() == FallbackClass::ProviderPinned
    }) {
        ComputeExecutionClassification::ProviderPinned
    } else if inputs.iter().any(|input| {
        input.resource.is_some() && input.affinity.fallback() == FallbackClass::Restartable
    }) {
        ComputeExecutionClassification::Restartable
    } else {
        ComputeExecutionClassification::Transparent
    }
}

fn execution_step_kind_from_memory_decision(
    decision: &MemoryPlanningDecision,
) -> ExecutionStepKind {
    match decision {
        MemoryPlanningDecision::Allocate { .. } | MemoryPlanningDecision::Reuse { .. } => {
            ExecutionStepKind::AllocateMemory
        }
        MemoryPlanningDecision::PreservePinnedResource { .. } => {
            ExecutionStepKind::PreserveProviderPinnedAffinity
        }
        MemoryPlanningDecision::RequireMaterialization { .. } => ExecutionStepKind::Materialize,
        MemoryPlanningDecision::RequireTransfer { .. } => ExecutionStepKind::Transfer,
        MemoryPlanningDecision::AccountHostStaging { .. } => ExecutionStepKind::Transfer,
    }
}

fn execution_phase_from_step_kind(kind: &ExecutionStepKind) -> ComputeExecutionPhase {
    match kind {
        ExecutionStepKind::Upload
        | ExecutionStepKind::Download
        | ExecutionStepKind::Copy
        | ExecutionStepKind::Transfer => ComputeExecutionPhase::DataMovement,
        ExecutionStepKind::Materialize => ComputeExecutionPhase::Materialization,
        ExecutionStepKind::AllocateMemory | ExecutionStepKind::ValidateMemory => {
            ComputeExecutionPhase::MemoryAllocation
        }
        ExecutionStepKind::ResolveProvider | ExecutionStepKind::ResolveDevice => {
            ComputeExecutionPhase::Resolution
        }
        ExecutionStepKind::SubmitToProvider => ComputeExecutionPhase::ProviderSubmission,
        _ => ComputeExecutionPhase::Validation,
    }
}

/// Returns the canonical hardware-independent Compute capability declaration.
///
/// Providers add this value to their metadata to advertise support for the
/// `magnetar:compute/run@1.1.0` WIT contract.
pub fn compute_capability() -> Capability {
    Capability::new(
        CapabilityId::new(COMPUTE_CAPABILITY_ID),
        COMPUTE_CAPABILITY_VERSION,
        CapabilityDescriptor::new("coarse provider-owned graph execution").with_contract(
            WitInterface::new(
                COMPUTE_WIT_INTERFACE,
                COMPUTE_CAPABILITY_VERSION.to_string(),
            ),
        ),
    )
}

fn parse_capability_version(value: &str) -> Result<CapabilityVersion, ProviderError> {
    let mut segments = value.split('.');
    let parse = |segment: Option<&str>| {
        segment
            .ok_or_else(|| ProviderError::InvalidCapabilityVersion(value.into()))?
            .parse::<u64>()
            .map_err(|_| ProviderError::InvalidCapabilityVersion(value.into()))
    };
    let version = CapabilityVersion::new(
        parse(segments.next())?,
        parse(segments.next())?,
        parse(segments.next())?,
    );
    if segments.next().is_some() {
        return Err(ProviderError::InvalidCapabilityVersion(value.into()));
    }
    Ok(version)
}

/// General extension contract; a provider may register any supported contribution.
pub trait Provider: Send + Sync {
    fn metadata(&self) -> ProviderMetadata;
    fn register(&self, registry: &mut ProviderRegistry) -> Result<(), ProviderError>;
    fn health(&self) -> ProviderHealth {
        ProviderHealth::Healthy
    }
    /// Discovers the execution devices owned by this provider.
    fn devices(&self) -> Vec<Arc<dyn Device>> {
        Vec::new()
    }
    fn initialize(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    fn shutdown(&self) -> Result<(), ProviderError> {
        Ok(())
    }
}

/// Receives provider contributions.
#[derive(Clone, Default)]
pub struct ProviderRegistry {
    backends: BTreeMap<String, Arc<dyn Backend>>,
    devices: BTreeMap<DeviceId, Arc<dyn Device>>,
    device_providers: BTreeMap<DeviceId, String>,
    capabilities: BTreeMap<CapabilityId, BTreeMap<CapabilityVersion, Capability>>,
    capability_providers: BTreeMap<(CapabilityId, CapabilityVersion), BTreeSet<String>>,
}
impl ProviderRegistry {
    pub fn register_devices(
        &mut self,
        provider: &str,
        devices: impl IntoIterator<Item = Arc<dyn Device>>,
    ) -> Result<(), ProviderError> {
        let devices = devices.into_iter().collect::<Vec<_>>();
        let mut ids = BTreeSet::new();
        for device in &devices {
            let metadata = device.metadata();
            if metadata.id.as_str().trim().is_empty() {
                return Err(ProviderError::InvalidDevice(
                    "identifier must not be empty".into(),
                ));
            }
            if metadata.provider != provider {
                return Err(ProviderError::DeviceProviderMismatch {
                    device: metadata.id.clone(),
                    expected: provider.into(),
                    found: metadata.provider.clone(),
                });
            }
            if self.devices.contains_key(&metadata.id) || !ids.insert(metadata.id.clone()) {
                return Err(ProviderError::DeviceAlreadyRegistered(metadata.id.clone()));
            }
        }
        for device in devices {
            let metadata = device.metadata();
            self.device_providers
                .insert(metadata.id.clone(), provider.into());
            self.devices.insert(metadata.id.clone(), device);
        }
        Ok(())
    }
    pub fn device(&self, id: &DeviceId) -> Option<&dyn Device> {
        self.devices.get(id).map(AsRef::as_ref)
    }
    pub fn devices(&self) -> impl Iterator<Item = &dyn Device> {
        self.devices.values().map(AsRef::as_ref)
    }
    pub fn provider_for_device(&self, id: &DeviceId) -> Option<&str> {
        self.device_providers.get(id).map(String::as_str)
    }
    pub fn devices_for_provider<'a>(
        &'a self,
        provider: &'a str,
    ) -> impl Iterator<Item = &'a dyn Device> {
        self.devices.iter().filter_map(move |(id, device)| {
            (self.provider_for_device(id) == Some(provider)).then_some(device.as_ref())
        })
    }
    pub fn register_capabilities(
        &mut self,
        provider: &str,
        capabilities: impl IntoIterator<Item = Capability>,
    ) -> Result<(), ProviderError> {
        for capability in capabilities {
            self.register_capability(capability.clone())?;
            self.capability_providers
                .entry((capability.id, capability.version))
                .or_default()
                .insert(provider.into());
        }
        Ok(())
    }
    pub fn register_capability(&mut self, capability: Capability) -> Result<(), ProviderError> {
        if capability.id.as_str().trim().is_empty() {
            return Err(ProviderError::InvalidCapability(
                "identifier must not be empty".into(),
            ));
        }
        if capability.descriptor.contracts.is_empty() {
            return Err(ProviderError::InvalidCapability(format!(
                "capability '{}' must declare at least one WIT contract",
                capability.id
            )));
        }
        let versions = self.capabilities.entry(capability.id.clone()).or_default();
        match versions.get(&capability.version) {
            Some(existing) if existing != &capability => {
                Err(ProviderError::ConflictingCapability {
                    id: capability.id,
                    version: capability.version,
                })
            }
            Some(_) => Ok(()),
            None => {
                versions.insert(capability.version, capability);
                Ok(())
            }
        }
    }
    pub fn register_backend(&mut self, backend: Arc<dyn Backend>) -> Result<(), ProviderError> {
        let name = backend.name().to_owned();
        if self.backends.contains_key(&name) {
            return Err(ProviderError::BackendAlreadyRegistered(name));
        }
        self.backends.insert(name, backend);
        Ok(())
    }
    pub fn backend(&self, name: &str) -> Option<&dyn Backend> {
        self.backends.get(name).map(AsRef::as_ref)
    }
    pub fn backend_names(&self) -> impl Iterator<Item = &str> {
        self.backends.keys().map(String::as_str)
    }
    pub fn capabilities(&self) -> impl Iterator<Item = &Capability> {
        self.capabilities
            .values()
            .flat_map(|versions| versions.values())
    }
    pub fn capability(&self, id: &CapabilityId, version: CapabilityVersion) -> Option<&Capability> {
        self.capabilities.get(id)?.get(&version)
    }
    /// Returns the latest registered version that satisfies `required`.
    pub fn resolve_capability(
        &self,
        id: &CapabilityId,
        required: CapabilityVersion,
    ) -> Option<&Capability> {
        self.capabilities
            .get(id)?
            .iter()
            .rev()
            .find_map(|(_, capability)| {
                capability
                    .version
                    .is_compatible_with(required)
                    .then_some(capability)
            })
    }
    /// Returns one exact Capability only when `provider` advertises it.
    pub fn capability_for_provider(
        &self,
        id: &CapabilityId,
        version: CapabilityVersion,
        provider: &str,
    ) -> Option<&Capability> {
        let capability = self.capability(id, version)?;
        self.providers_for(capability)
            .any(|candidate| candidate == provider)
            .then_some(capability)
    }
    /// Returns the latest compatible version advertised by one Provider.
    pub fn resolve_capability_for_provider(
        &self,
        id: &CapabilityId,
        required: CapabilityVersion,
        provider: &str,
    ) -> Option<&Capability> {
        self.capabilities
            .get(id)?
            .iter()
            .rev()
            .find_map(|(_, capability)| {
                (capability.version.is_compatible_with(required)
                    && self
                        .providers_for(capability)
                        .any(|candidate| candidate == provider))
                .then_some(capability)
            })
    }
    pub fn validate_dependencies(&self) -> Result<(), ProviderError> {
        for capability in self.capabilities() {
            for (dependency, version) in &capability.descriptor.dependencies {
                if self.resolve_capability(dependency, *version).is_none() {
                    return Err(ProviderError::MissingCapabilityDependency {
                        capability: capability.id.clone(),
                        dependency: dependency.clone(),
                        required: *version,
                    });
                }
            }
        }
        Ok(())
    }
    /// Returns implementations for an exact, resolved capability version in deterministic order.
    pub fn providers_for(&self, capability: &Capability) -> impl Iterator<Item = &str> {
        self.capability_providers
            .get(&(capability.id.clone(), capability.version))
            .into_iter()
            .flat_map(|providers| providers.iter().map(String::as_str))
    }
    pub fn has_capability(&self, capability: &Capability) -> bool {
        self.capability(&capability.id, capability.version)
            .is_some()
    }
    fn clear(&mut self) {
        self.backends.clear();
        self.devices.clear();
        self.device_providers.clear();
        self.capabilities.clear();
        self.capability_providers.clear();
    }
}

#[derive(Default)]
pub struct ProviderLoader {
    registry: ProviderRegistry,
    providers: BTreeMap<String, Arc<dyn Provider>>,
    order: Vec<String>,
    /* declared last: unloaded after provider drops */ libraries: Vec<Library>,
}
impl ProviderLoader {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn registry(&self) -> &ProviderRegistry {
        &self.registry
    }
    pub fn registry_mut(&mut self) -> &mut ProviderRegistry {
        &mut self.registry
    }
    pub fn provider(&self, name: &str) -> Option<&dyn Provider> {
        self.providers.get(name).map(AsRef::as_ref)
    }
    pub fn provider_names(&self) -> impl Iterator<Item = &str> {
        self.providers.keys().map(String::as_str)
    }
    pub fn try_resolve_providers(
        &self,
        capability: &Capability,
    ) -> Result<Vec<&dyn Provider>, ProviderError> {
        self.registry.validate_dependencies()?;
        let Some(resolved) = self
            .registry
            .resolve_capability(&capability.id, capability.version)
        else {
            return Ok(Vec::new());
        };
        Ok(self
            .registry
            .providers_for(resolved)
            .filter_map(|name| self.provider(name))
            .collect())
    }
    pub fn resolve_providers(&self, capability: &Capability) -> Vec<&dyn Provider> {
        self.try_resolve_providers(capability).unwrap_or_default()
    }
    pub fn try_resolve_providers_with_policy(
        &self,
        capability: &Capability,
        policy: BuiltInResolutionPolicy,
    ) -> Result<Vec<&dyn Provider>, ProviderError> {
        let candidates = self.candidates_for_capability(capability)?;
        let context = ResolutionContext {
            requested_capability: capability.id.clone(),
            requested_version: capability.version,
            candidates,
            affinity: None,
            fallback: FallbackClass::Transparent,
            execution_phase: ExecutionPhase::BeforeResourceCreation,
            replayable_input: true,
        };
        let decision = policy.decide(&context);
        let Some(selected) = decision.selected_provider.as_ref() else {
            let requested = CapabilityBinding::new(capability.id.clone(), capability.version);
            return if context.candidates.is_empty() {
                Err(ProviderError::NoCompatibleProvider(requested))
            } else {
                Err(ProviderError::PolicyRejectedProvider {
                    capability: requested,
                    policy: decision.policy_id,
                })
            };
        };
        let mut names = vec![selected.as_str().to_owned()];
        for candidate in &context.candidates {
            if candidate.provider.as_str() != selected.as_str()
                && !decision
                    .rejected_candidates
                    .iter()
                    .any(|rejected| rejected.provider == candidate.provider)
            {
                names.push(candidate.provider.as_str().to_owned());
            }
        }
        Ok(names
            .into_iter()
            .filter_map(|name| self.provider(&name))
            .collect())
    }
    fn candidates_for_capability(
        &self,
        capability: &Capability,
    ) -> Result<Vec<ResolutionCandidate>, ProviderError> {
        self.registry.validate_dependencies()?;
        let Some(resolved) = self
            .registry
            .resolve_capability(&capability.id, capability.version)
        else {
            return Ok(Vec::new());
        };
        Ok(self
            .registry
            .providers_for(resolved)
            .filter_map(|name| {
                let provider = self.provider(name)?;
                Some(self.resolution_candidate(provider, resolved, true))
            })
            .collect())
    }
    fn resolution_candidate(
        &self,
        provider: &dyn Provider,
        capability: &Capability,
        affinity_compatible: bool,
    ) -> ResolutionCandidate {
        let provider_name = provider.metadata().name;
        let device = self
            .registry
            .devices_for_provider(&provider_name)
            .find(|device| {
                let capabilities = &device.metadata().execution_capabilities;
                capabilities.is_empty() || capabilities.contains(&capability.id)
            })
            .map(|device| {
                (
                    DeviceBinding::new(device.id().clone()),
                    device.availability(),
                )
            });
        ResolutionCandidate {
            provider: ProviderBinding::new(provider_name),
            capability: CapabilityBinding::new(capability.id.clone(), capability.version),
            device: device.as_ref().map(|(binding, _)| binding.clone()),
            provider_health: provider.health(),
            device_availability: device
                .map(|(_, availability)| availability)
                .unwrap_or(DeviceAvailability::Available),
            affinity_compatible,
            priority: 0,
        }
    }
    fn resolve_with_constraints<'a>(
        &'a self,
        requested: &Capability,
        constraints: &AffinityConstraints,
        policy: BuiltInResolutionPolicy,
        execution_phase: ExecutionPhase,
        replayable_input: bool,
    ) -> Result<(&'a dyn Provider, &'a Capability, ResolutionDecision), AffinityError> {
        let affinity = constraints.affinity();
        let mut bound_provider = affinity.provider().cloned();

        if let Some(device) = affinity.device() {
            let owner = self
                .registry
                .provider_for_device(device.id())
                .map(ProviderBinding::new)
                .ok_or_else(|| AffinityError::BoundDeviceUnavailable(device.clone()))?;
            if let Some(provider) = &bound_provider
                && provider != &owner
            {
                return Err(AffinityError::DeviceProviderMismatch {
                    device: device.clone(),
                    provider: provider.clone(),
                    owner,
                });
            }
            bound_provider = Some(owner);
        }

        if let Some(provider_binding) = bound_provider {
            let provider = self
                .provider(provider_binding.as_str())
                .ok_or_else(|| AffinityError::BoundProviderUnavailable(provider_binding.clone()))?;
            let requested_binding = affinity
                .capability(&requested.id)
                .cloned()
                .unwrap_or_else(|| CapabilityBinding::new(requested.id.clone(), requested.version));
            if let Some(bound) = affinity.capability(&requested.id)
                && !bound.version.is_compatible_with(requested.version)
            {
                return Err(AffinityError::CapabilityMismatch {
                    id: requested.id.clone(),
                    expected: requested.version,
                    found: bound.version,
                });
            }
            for binding in affinity.capabilities() {
                if self
                    .registry
                    .capability_for_provider(
                        binding.id(),
                        binding.version(),
                        provider_binding.as_str(),
                    )
                    .is_none()
                {
                    return Err(AffinityError::ProviderDoesNotImplementCapability {
                        provider: provider_binding.clone(),
                        capability: binding.clone(),
                    });
                }
            }
            let capability = if let Some(bound) = affinity.capability(&requested.id) {
                self.registry.capability_for_provider(
                    &requested.id,
                    bound.version,
                    provider_binding.as_str(),
                )
            } else {
                self.registry.resolve_capability_for_provider(
                    &requested.id,
                    requested.version,
                    provider_binding.as_str(),
                )
            }
            .ok_or(AffinityError::ProviderDoesNotImplementCapability {
                provider: provider_binding,
                capability: requested_binding.clone(),
            })?;
            let context = ResolutionContext {
                requested_capability: requested.id.clone(),
                requested_version: requested.version,
                candidates: vec![self.resolution_candidate(provider, capability, true)],
                affinity: Some(affinity.clone()),
                fallback: FallbackClass::Transparent,
                execution_phase,
                replayable_input,
            };
            let mut decision = policy.decide(&context);
            if decision.selected_provider.is_none() {
                return Err(AffinityError::PolicyRejectedProvider {
                    capability: requested_binding,
                    policy: decision.policy_id,
                });
            }
            decision.reason = ResolutionDecisionReason::PreservedAffinity;
            return Ok((provider, capability, decision));
        }

        let requested_binding = affinity
            .capability(&requested.id)
            .cloned()
            .unwrap_or_else(|| CapabilityBinding::new(requested.id.clone(), requested.version));
        let capability = if let Some(bound) = affinity.capability(&requested.id) {
            if !bound.version.is_compatible_with(requested.version) {
                return Err(AffinityError::CapabilityMismatch {
                    id: requested.id.clone(),
                    expected: requested.version,
                    found: bound.version,
                });
            }
            self.registry.capability(&requested.id, bound.version)
        } else {
            self.registry
                .resolve_capability(&requested.id, requested.version)
        }
        .ok_or_else(|| AffinityError::NoCompatibleProvider(requested_binding.clone()))?;

        let candidates = self
            .registry
            .providers_for(capability)
            .filter_map(|name| {
                let provider = self.provider(name)?;
                let affinity_compatible = affinity.capabilities().all(|binding| {
                    self.registry
                        .capability_for_provider(binding.id(), binding.version(), name)
                        .is_some()
                });
                Some(self.resolution_candidate(provider, capability, affinity_compatible))
            })
            .collect::<Vec<_>>();
        let context = ResolutionContext {
            requested_capability: requested.id.clone(),
            requested_version: requested.version,
            candidates,
            affinity: Some(affinity.clone()),
            fallback: affinity.fallback(),
            execution_phase,
            replayable_input,
        };
        let decision = policy.decide(&context);
        let Some(selected_provider) = decision.selected_provider.as_ref() else {
            return if context.candidates.is_empty() {
                Err(AffinityError::NoCompatibleProvider(requested_binding))
            } else {
                Err(AffinityError::PolicyRejectedProvider {
                    capability: requested_binding,
                    policy: decision.policy_id,
                })
            };
        };
        let provider = self
            .provider(selected_provider.as_str())
            .ok_or_else(|| AffinityError::BoundProviderUnavailable(selected_provider.clone()))?;
        Ok((provider, capability, decision))
    }
    pub fn register_provider(&mut self, provider: Arc<dyn Provider>) -> Result<(), ProviderError> {
        let metadata = provider.metadata();
        if metadata.api_version != PROVIDER_API_VERSION {
            return Err(ProviderError::IncompatibleApiVersion {
                provider: metadata.name,
                expected: PROVIDER_API_VERSION,
                found: metadata.api_version,
            });
        }
        if self.providers.contains_key(&metadata.name) {
            return Err(ProviderError::ProviderAlreadyRegistered(metadata.name));
        }
        let previous = self.registry.clone();
        if let Err(error) = provider
            .register(&mut self.registry)
            .and_then(|()| provider.initialize())
            .and_then(|()| {
                self.registry
                    .register_devices(&metadata.name, provider.devices())
            })
        {
            self.registry = previous;
            return Err(error);
        }
        if let Err(error) = self
            .registry
            .register_capabilities(&metadata.name, metadata.capabilities)
        {
            self.registry = previous;
            return Err(error);
        }
        self.order.push(metadata.name.clone());
        self.providers.insert(metadata.name, provider);
        Ok(())
    }
    /// Registers a provider without allowing one failed extension to abort runtime startup.
    pub fn register_provider_isolated(&mut self, provider: Arc<dyn Provider>) -> bool {
        self.register_provider(provider).is_ok()
    }
    pub fn discover(
        paths: impl IntoIterator<Item = impl AsRef<Path>>,
    ) -> Result<Vec<PathBuf>, ProviderError> {
        let mut found = BTreeSet::new();
        for dir in paths {
            let dir = dir.as_ref();
            for entry in std::fs::read_dir(dir).map_err(|source| ProviderError::Discovery {
                path: dir.into(),
                source,
            })? {
                let path = entry
                    .map_err(|source| ProviderError::Discovery {
                        path: dir.into(),
                        source,
                    })?
                    .path();
                if path.is_file()
                    && matches!(
                        path.extension().and_then(|x| x.to_str()),
                        Some("dll" | "dylib" | "so")
                    )
                {
                    found.insert(path);
                }
            }
        }
        Ok(found.into_iter().collect())
    }
    /// Loads a compatible Rust library exporting `magnetar_provider_create`.
    ///
    /// # Safety
    ///
    /// The library must be trusted, remain loaded while its provider is used,
    /// and export a factory with the exact declared Rust ABI and contract.
    pub unsafe fn load_dynamic(&mut self, path: impl AsRef<Path>) -> Result<(), ProviderError> {
        let path = path.as_ref();
        let library = unsafe { Library::new(path) }.map_err(|e| ProviderError::Load {
            path: path.into(),
            message: e.to_string(),
        })?;
        type Factory = unsafe fn() -> Box<dyn Provider>;
        let factory =
            unsafe { library.get::<Factory>(b"magnetar_provider_create") }.map_err(|e| {
                ProviderError::Load {
                    path: path.into(),
                    message: e.to_string(),
                }
            })?;
        self.register_provider(Arc::from(unsafe { factory() }))?;
        self.libraries.push(library);
        Ok(())
    }
    /// Discovers and loads every compatible provider library in the given paths.
    ///
    /// # Safety
    ///
    /// Every discovered dynamic library must satisfy the safety requirements of
    /// [`Self::load_dynamic`].
    pub unsafe fn discover_and_load(
        &mut self,
        paths: impl IntoIterator<Item = impl AsRef<Path>>,
    ) -> Result<(), ProviderError> {
        for path in Self::discover(paths)? {
            unsafe { self.load_dynamic(path) }?;
        }
        Ok(())
    }
    pub fn shutdown(&mut self) -> Result<(), ProviderError> {
        let mut first = None;
        for name in self.order.iter().rev() {
            if let Some(provider) = self.providers.get(name)
                && let Err(error) = provider.shutdown()
            {
                first.get_or_insert(error);
            }
        }
        self.order.clear();
        self.providers.clear();
        self.registry.clear();
        self.libraries.clear();
        first.map_or(Ok(()), Err)
    }
}

static NEXT_EXECUTION_CONTEXT_ID: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);
static NEXT_AFFINITY_GROUP_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
static NEXT_SCHEDULED_OPERATION_ID: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);

fn next_execution_context_id() -> ExecutionContextId {
    ExecutionContextId::new(
        NEXT_EXECUTION_CONTEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    )
}
fn next_affinity_group_id() -> AffinityGroupId {
    AffinityGroupId::new(NEXT_AFFINITY_GROUP_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}
fn next_scheduled_operation_id() -> ScheduledOperationId {
    ScheduledOperationId::new(
        NEXT_SCHEDULED_OPERATION_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    )
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RuntimeConfig {
    pub preferred_backend: Option<String>,
    pub resolution_policy: BuiltInResolutionPolicy,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionContext {
    id: ExecutionContextId,
    config: RuntimeConfig,
    backend_name: Option<String>,
}
impl Default for ExecutionContext {
    fn default() -> Self {
        Self {
            id: next_execution_context_id(),
            config: RuntimeConfig::default(),
            backend_name: None,
        }
    }
}
impl ExecutionContext {
    pub const fn id(&self) -> ExecutionContextId {
        self.id
    }
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }
    pub fn backend_name(&self) -> Option<&str> {
        self.backend_name.as_deref()
    }
}
#[derive(Default)]
pub struct RuntimeBuilder {
    config: RuntimeConfig,
    backends: Vec<Arc<dyn Backend>>,
    providers: Vec<Arc<dyn Provider>>,
}
impl RuntimeBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn config(mut self, x: RuntimeConfig) -> Self {
        self.config = x;
        self
    }
    pub fn register_backend(mut self, x: Arc<dyn Backend>) -> Self {
        self.backends.push(x);
        self
    }
    pub fn register_provider(mut self, x: Arc<dyn Provider>) -> Self {
        self.providers.push(x);
        self
    }
    pub fn build(self) -> Result<Runtime, ProviderError> {
        let mut providers = ProviderLoader::new();
        for x in self.backends {
            providers.registry_mut().register_backend(x)?;
        }
        for x in self.providers {
            providers.register_provider_isolated(x);
        }
        let mut runtime = Runtime {
            context: ExecutionContext {
                id: next_execution_context_id(),
                config: self.config,
                backend_name: None,
            },
            providers,
            initialized: true,
        };
        if let Some(x) = runtime.context.config.preferred_backend.clone() {
            runtime.select_backend(&x)?;
        }
        Ok(runtime)
    }
}
pub struct Runtime {
    context: ExecutionContext,
    providers: ProviderLoader,
    initialized: bool,
}
impl Runtime {
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::new()
    }
    pub fn initialize(config: RuntimeConfig) -> Self {
        Self::builder()
            .config(config)
            .build()
            .expect("valid configuration")
    }
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
    pub fn context(&self) -> &ExecutionContext {
        &self.context
    }
    pub fn providers(&self) -> &ProviderLoader {
        &self.providers
    }
    pub fn register_backend(&mut self, x: Arc<dyn Backend>) -> Result<(), ProviderError> {
        self.providers.registry_mut().register_backend(x)
    }
    pub fn register_provider(&mut self, x: Arc<dyn Provider>) -> Result<(), ProviderError> {
        self.providers.register_provider(x)
    }
    /// Returns every registered execution target in deterministic ID order.
    pub fn devices(&self) -> impl Iterator<Item = &dyn Device> {
        self.providers.registry().devices()
    }
    pub fn device(&self, id: &DeviceId) -> Option<&dyn Device> {
        self.providers.registry().device(id)
    }
    /// Resolves all compatible providers, ordered for deterministic fallback.
    pub fn resolve_providers(&self, capability: &Capability) -> Vec<&dyn Provider> {
        self.try_resolve_providers(capability).unwrap_or_default()
    }
    /// Resolves providers while reporting invalid capability dependencies.
    pub fn try_resolve_providers(
        &self,
        capability: &Capability,
    ) -> Result<Vec<&dyn Provider>, ProviderError> {
        self.providers
            .try_resolve_providers_with_policy(capability, self.context.config.resolution_policy)
    }
    /// Resolves one coherent Provider for resources that already carry state.
    ///
    /// A group is created only when `dependencies` is non-empty and no input
    /// group exists. Independent resources therefore remain shareable until a
    /// dependent operation creates a grouped resource from them.
    pub fn resolve_with_affinity<'a>(
        &'a self,
        capability: &Capability,
        dependencies: &[&ResourceAffinity],
        fallback: FallbackClass,
    ) -> Result<AffinityResolution<'a>, AffinityError> {
        if !self.initialized {
            return Err(AffinityError::RuntimeNotInitialized);
        }
        let mut constraints =
            AffinityConstraints::try_from_affinities(dependencies.iter().copied())?;
        constraints.require_fallback(fallback);
        constraints.merge(
            &ResourceAffinity::new(FallbackClass::Transparent)
                .with_execution_context(self.context.id),
        )?;
        if !dependencies.is_empty() && constraints.affinity().group().is_none() {
            constraints.merge(
                &ResourceAffinity::new(FallbackClass::Transparent)
                    .with_group(next_affinity_group_id()),
            )?;
        }

        let (provider, selected, decision) = self.providers.resolve_with_constraints(
            capability,
            &constraints,
            self.context.config.resolution_policy,
            ExecutionPhase::BeforeResourceCreation,
            true,
        )?;
        let provider_binding = ProviderBinding::new(provider.metadata().name);
        let affinity = constraints
            .into_affinity()
            .with_provider(provider_binding)
            .with_capability(CapabilityBinding::new(
                selected.id.clone(),
                selected.version,
            ));
        Ok(AffinityResolution {
            provider,
            capability: selected,
            affinity,
            decision,
        })
    }
    pub fn resolve_with_affinity_at_phase<'a>(
        &'a self,
        capability: &Capability,
        dependencies: &[&ResourceAffinity],
        fallback: FallbackClass,
        execution_phase: ExecutionPhase,
        replayable_input: bool,
    ) -> Result<AffinityResolution<'a>, AffinityError> {
        if !self.initialized {
            return Err(AffinityError::RuntimeNotInitialized);
        }
        let mut constraints =
            AffinityConstraints::try_from_affinities(dependencies.iter().copied())?;
        constraints.require_fallback(fallback);
        constraints.merge(
            &ResourceAffinity::new(FallbackClass::Transparent)
                .with_execution_context(self.context.id),
        )?;
        if !dependencies.is_empty() && constraints.affinity().group().is_none() {
            constraints.merge(
                &ResourceAffinity::new(FallbackClass::Transparent)
                    .with_group(next_affinity_group_id()),
            )?;
        }

        let (provider, selected, decision) = self.providers.resolve_with_constraints(
            capability,
            &constraints,
            self.context.config.resolution_policy,
            execution_phase,
            replayable_input,
        )?;
        let provider_binding = ProviderBinding::new(provider.metadata().name);
        let affinity = constraints
            .into_affinity()
            .with_provider(provider_binding)
            .with_capability(CapabilityBinding::new(
                selected.id.clone(),
                selected.version,
            ));
        Ok(AffinityResolution {
            provider,
            capability: selected,
            affinity,
            decision,
        })
    }
    /// Resolves providers for a component's WIT import without exposing a
    /// provider dependency to the component itself.
    pub fn resolve_component_import(
        &self,
        interface: &WitInterface,
    ) -> Result<Vec<&dyn Provider>, ProviderError> {
        self.providers.try_resolve_providers_with_policy(
            &Capability::from_wit(interface.clone())?,
            self.context.config.resolution_policy,
        )
    }
    pub fn validate_compute_operation_requests(
        &self,
        provider: &str,
        operations: &[ComputeOperationRequest],
    ) -> Result<Vec<ComputeOperationDescriptor>, ComputeValidationError> {
        let descriptors = operations
            .iter()
            .map(|operation| {
                let family =
                    ComputeOperationFamily::from_id(&operation.family_id).ok_or_else(|| {
                        ComputeValidationError::UnknownOperationFamily(operation.family_id.clone())
                    })?;
                Ok(ComputeOperationDescriptor {
                    schema_id: None,
                    family,
                    dtype: operation.dtype,
                    layout: operation.layout,
                    precision: operation.precision,
                    attributes: BTreeMap::new(),
                    tensors: operation.tensors.clone(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.validate_compute_operations(provider, &descriptors)?;
        Ok(descriptors)
    }
    pub fn validate_compute_operations(
        &self,
        provider: &str,
        operations: &[ComputeOperationDescriptor],
    ) -> Result<(), ComputeValidationError> {
        let metadata = self
            .providers
            .provider(provider)
            .map(Provider::metadata)
            .ok_or_else(|| {
                ComputeValidationError::ProviderUnavailable(ProviderBinding::new(provider))
            })?;
        let advertisement = effective_compute_advertisement(&metadata);
        if !advertisement.supports_capability_version(COMPUTE_CAPABILITY_VERSION) {
            return Err(ComputeValidationError::UnsupportedAdvertisement {
                provider: ProviderBinding::new(&metadata.name),
                reason: format!(
                    "provider does not advertise compatible '{}' version {}",
                    COMPUTE_CAPABILITY_ID, COMPUTE_CAPABILITY_VERSION
                ),
            });
        }
        for operation in operations {
            if let Some(schema_id) = &operation.schema_id
                && advertisement
                    .unsupported_operation_schemas
                    .contains(schema_id)
            {
                return Err(ComputeValidationError::UnsupportedOperationSchema {
                    provider: ProviderBinding::new(&metadata.name),
                    operation: schema_id.clone(),
                });
            }
            let schema_result = validate_compute_operation_schema(operation)?;
            let support = if let Some(schema_id) = &operation.schema_id {
                advertisement
                    .operation_schemas
                    .get(schema_id)
                    .map(OperationSchemaSupport::operation_support)
                    .or_else(|| {
                        advertisement
                            .operation_families
                            .get(&operation.family)
                            .map(OperationFamilySupport::operation_support)
                    })
                    .ok_or_else(|| ComputeValidationError::UnsupportedOperationFamily {
                        provider: ProviderBinding::new(&metadata.name),
                        family: operation.family,
                    })?
            } else {
                advertisement
                    .operation_families
                    .get(&operation.family)
                    .map(OperationFamilySupport::operation_support)
                    .ok_or_else(|| ComputeValidationError::UnsupportedOperationFamily {
                        provider: ProviderBinding::new(&metadata.name),
                        family: operation.family,
                    })?
            };
            support.supports(operation)?;
            drop(schema_result);
        }
        Ok(())
    }
    pub fn validate_compute_tensor_resources(
        &self,
        provider: &str,
        resources: &[TensorResourceDescriptor],
    ) -> Result<(), ComputeValidationError> {
        let target = ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(ProviderBinding::new(provider))
            .with_capability(CapabilityBinding::new(
                CapabilityId::new(COMPUTE_CAPABILITY_ID),
                COMPUTE_CAPABILITY_VERSION,
            ));
        for resource in resources {
            target
                .validate_with(&resource.affinity)
                .map_err(ComputeValidationError::IncompatibleResourceAffinity)?;
        }
        Ok(())
    }
    pub fn validate_compute_data_movement(
        &self,
        provider: &str,
        movements: &[ComputeDataMovementDescriptor],
    ) -> Result<(), ComputeValidationError> {
        let metadata = self
            .providers
            .provider(provider)
            .map(Provider::metadata)
            .ok_or_else(|| {
                ComputeValidationError::ProviderUnavailable(ProviderBinding::new(provider))
            })?;
        let provider_binding = ProviderBinding::new(&metadata.name);
        let advertisement = effective_compute_advertisement(&metadata);
        let target = ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(provider_binding.clone())
            .with_capability(CapabilityBinding::new(
                CapabilityId::new(COMPUTE_CAPABILITY_ID),
                COMPUTE_CAPABILITY_VERSION,
            ));
        for movement in movements {
            let support = advertisement
                .data_movement
                .get(&movement.kind)
                .map(DataMovementSupport::movement_support)
                .ok_or_else(|| ComputeValidationError::UnsupportedDataMovement {
                    provider: provider_binding.clone(),
                    kind: movement.kind,
                })?;
            support.supports(&provider_binding, movement)?;
            if let Some(source) = movement.source.tensor() {
                let permits_explicit_replacement = matches!(
                    movement.kind,
                    ComputeDataMovementKind::Transfer
                        | ComputeDataMovementKind::PlacementConversion
                );
                if !permits_explicit_replacement
                    || source
                        .affinity
                        .provider()
                        .is_none_or(|source_provider| source_provider.as_str() == provider)
                {
                    target
                        .validate_with(&source.affinity)
                        .map_err(ComputeValidationError::IncompatibleResourceAffinity)?;
                }
                if let Some(target_device) = &movement.target_device {
                    match source.affinity.device() {
                        Some(source_device) if source_device == target_device => {}
                        _ if matches!(
                            movement.kind,
                            ComputeDataMovementKind::Transfer
                                | ComputeDataMovementKind::PlacementConversion
                        ) => {}
                        _ => {
                            return Err(ComputeValidationError::InvalidTransfer {
                                reason:
                                    "device changes require explicit transfer or placement conversion"
                                        .into(),
                            });
                        }
                    }
                }
                if let Some(target_group) = movement.target_group {
                    match source.affinity.group() {
                        Some(source_group) if source_group == target_group => {}
                        _ if matches!(
                            movement.kind,
                            ComputeDataMovementKind::Transfer
                                | ComputeDataMovementKind::PlacementConversion
                        ) => {}
                        _ => {
                            return Err(ComputeValidationError::InvalidTransfer {
                                reason:
                                    "affinity group changes require explicit transfer or placement conversion"
                                        .into(),
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }
    pub fn plan_compute_data_movement_memory(
        &self,
        provider: &str,
        movements: &[ComputeDataMovementDescriptor],
    ) -> Result<MemoryPlan, MemoryPlanningError> {
        let metadata = self
            .providers
            .provider(provider)
            .map(Provider::metadata)
            .ok_or_else(|| MemoryPlanningError::MemoryPlanningFailed {
                reason: format!("provider '{provider}' is unavailable"),
                report: MemoryPressureReport::default(),
            })?;
        let provider_binding = ProviderBinding::new(&metadata.name);
        let mut plan = MemoryPlan::new(provider_binding.clone(), None, self.context.id);
        let advertisement = effective_compute_advertisement(&metadata);

        for (index, movement) in movements.iter().enumerate() {
            let support = advertisement
                .data_movement
                .get(&movement.kind)
                .ok_or_else(|| MemoryPlanningError::TransferRequired {
                    reason: format!("provider does not advertise '{}'", movement.kind.id()),
                    report: plan.pressure.clone(),
                })?;
            if movement.allow_host_staging && !support.allow_host_staging {
                return Err(MemoryPlanningError::TransferRequired {
                    reason: "host staging must be explicit and advertised".into(),
                    report: plan.pressure.clone(),
                });
            }
            let output_bytes = memory_bytes(&movement.output, &plan.pressure)?;
            let mut affinity = ResourceAffinity::new(FallbackClass::ProviderPinned)
                .with_provider(
                    movement
                        .target_provider
                        .clone()
                        .unwrap_or_else(|| provider_binding.clone()),
                )
                .with_capability(CapabilityBinding::new(
                    CapabilityId::new(COMPUTE_CAPABILITY_ID),
                    COMPUTE_CAPABILITY_VERSION,
                ))
                .with_execution_context(self.context.id);
            if let Some(device) = &movement.target_device {
                affinity = affinity.with_device(device.clone());
            }
            let requirement_id = format!("movement:{index}:{}", movement.kind.id());
            let region = match movement.kind {
                ComputeDataMovementKind::Materialize => MemoryRegionKind::Materialization,
                ComputeDataMovementKind::Transfer => MemoryRegionKind::Transfer,
                ComputeDataMovementKind::Upload
                | ComputeDataMovementKind::Download
                | ComputeDataMovementKind::Copy
                | ComputeDataMovementKind::DTypeConversion
                | ComputeDataMovementKind::PlacementConversion => MemoryRegionKind::Transfer,
            };
            plan.add_requirement(MemoryRequirement::new(
                requirement_id.clone(),
                region,
                output_bytes,
                affinity.clone(),
            ))?;
            plan.decisions.push(match movement.kind {
                ComputeDataMovementKind::Materialize => {
                    plan.pressure.materialization_cost_bytes = plan
                        .pressure
                        .materialization_cost_bytes
                        .checked_add(output_bytes)
                        .ok_or_else(|| MemoryPlanningError::SizeOverflow {
                            reason: "materialization memory cost overflows u64".into(),
                            report: plan.pressure.clone(),
                        })?;
                    MemoryPlanningDecision::RequireMaterialization {
                        requirement: requirement_id.clone(),
                    }
                }
                _ if movement.allow_host_staging => {
                    plan.pressure.transfer_buffer_cost_bytes = plan
                        .pressure
                        .transfer_buffer_cost_bytes
                        .checked_add(output_bytes)
                        .ok_or_else(|| MemoryPlanningError::SizeOverflow {
                            reason: "transfer memory cost overflows u64".into(),
                            report: plan.pressure.clone(),
                        })?;
                    MemoryPlanningDecision::AccountHostStaging {
                        requirement: requirement_id.clone(),
                    }
                }
                _ => MemoryPlanningDecision::RequireTransfer {
                    requirement: requirement_id.clone(),
                },
            });
            plan.tensor_lifetimes.push(TensorLifetime {
                id: requirement_id,
                first_step: index,
                last_step: index,
                byte_size: output_bytes,
                affinity,
            });
        }
        self.validate_memory_plan(&metadata, &mut plan)?;
        Ok(plan)
    }
    pub fn plan_compute_graph_memory(
        &self,
        provider: &str,
        graph: &ComputeGraph,
    ) -> Result<MemoryPlan, MemoryPlanningError> {
        let metadata = self
            .providers
            .provider(provider)
            .map(Provider::metadata)
            .ok_or_else(|| MemoryPlanningError::MemoryPlanningFailed {
                reason: format!("provider '{provider}' is unavailable"),
                report: MemoryPressureReport::default(),
            })?;
        let provider_binding = ProviderBinding::new(&metadata.name);
        let mut plan = MemoryPlan::new(
            provider_binding.clone(),
            Some(graph.id.clone()),
            self.context.id,
        );
        let mut input_descriptors = BTreeMap::new();
        let mut input_affinities = BTreeMap::new();
        let mut output_descriptors = BTreeMap::new();
        let mut completed_nodes = BTreeSet::new();
        let graph_end = graph.nodes.len() + 1;
        let target = ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(provider_binding.clone())
            .with_capability(CapabilityBinding::new(
                CapabilityId::new(COMPUTE_CAPABILITY_ID),
                COMPUTE_CAPABILITY_VERSION,
            ))
            .with_execution_context(self.context.id);

        for input in &graph.inputs {
            input_descriptors.insert(input.id.clone(), input.value.descriptor().clone());
            let affinity = input
                .value
                .affinity()
                .cloned()
                .unwrap_or_else(|| target.clone());
            if let ComputeInputValue::TensorResource(resource) = &input.value {
                if let Some(source_provider) = resource.affinity.provider()
                    && source_provider.as_str() != provider
                {
                    return Err(MemoryPlanningError::TransferRequired {
                        reason: format!(
                            "input resource '{}' is bound to provider '{source_provider}'",
                            resource.id
                        ),
                        report: plan.pressure.clone(),
                    });
                }
                if resource.affinity.fallback() == FallbackClass::ProviderPinned {
                    plan.decisions
                        .push(MemoryPlanningDecision::PreservePinnedResource {
                            resource: resource.id.clone(),
                        });
                }
            }
            let bytes = memory_bytes(input.value.descriptor(), &plan.pressure)?;
            let id = format!("input:{}", input.id);
            plan.add_requirement(MemoryRequirement::new(
                id.clone(),
                MemoryRegionKind::GraphInput,
                bytes,
                affinity.clone(),
            ))?;
            plan.tensor_lifetimes.push(TensorLifetime {
                id,
                first_step: 0,
                last_step: last_use_for_input(graph, &input.id).unwrap_or(graph_end),
                byte_size: bytes,
                affinity: affinity.clone(),
            });
            input_affinities.insert(input.id.clone(), affinity);
        }

        for (node_index, node) in graph.nodes.iter().enumerate() {
            let step = node_index + 1;
            for input in &node.inputs {
                let descriptor = resolve_compute_value_descriptor(
                    Some(&node.id),
                    input,
                    &input_descriptors,
                    &output_descriptors,
                    &completed_nodes,
                )
                .map_err(memory_error_from_compute_validation)?;
                if descriptor.view.is_some() && descriptor.layout.kind() != ComputeLayout::Dense {
                    let bytes = memory_bytes(descriptor, &plan.pressure)?;
                    let requirement_id = format!("materialize:{}:{step}", node.id);
                    plan.add_requirement(MemoryRequirement::new(
                        requirement_id.clone(),
                        MemoryRegionKind::Materialization,
                        bytes,
                        target.clone(),
                    ))?;
                    plan.pressure.materialization_cost_bytes = plan
                        .pressure
                        .materialization_cost_bytes
                        .checked_add(bytes)
                        .ok_or_else(|| MemoryPlanningError::SizeOverflow {
                            reason: "materialization memory cost overflows u64".into(),
                            report: plan.pressure.clone(),
                        })?;
                    plan.decisions
                        .push(MemoryPlanningDecision::RequireMaterialization {
                            requirement: requirement_id,
                        });
                }
            }
            for output in &node.outputs {
                let bytes = memory_bytes(&output.descriptor, &plan.pressure)?;
                let id = format!("node:{}:{}", node.id, output.id);
                let lifetime = TensorLifetime {
                    id: id.clone(),
                    first_step: step,
                    last_step: last_use_for_node_output(graph, &node.id, &output.id)
                        .unwrap_or(step),
                    byte_size: bytes,
                    affinity: target.clone(),
                };
                let reuses = plan.find_reusable_buffer(&lifetime);
                let mut requirement = MemoryRequirement::new(
                    id.clone(),
                    if graph_output_uses(graph, &node.id, &output.id) {
                        MemoryRegionKind::GraphOutput
                    } else {
                        MemoryRegionKind::Intermediate
                    },
                    bytes,
                    target.clone(),
                );
                if !graph_output_uses(graph, &node.id, &output.id) {
                    requirement = requirement.reusable();
                }
                plan.add_requirement(requirement)?;
                plan.buffer_lifetimes.push(BufferLifetime {
                    id: format!("buffer:{id}"),
                    source: id.clone(),
                    first_step: lifetime.first_step,
                    last_step: lifetime.last_step,
                    byte_size: bytes,
                    affinity: target.clone(),
                    reuses: reuses.clone(),
                });
                plan.decisions.push(match reuses {
                    Some(buffer) => MemoryPlanningDecision::Reuse {
                        requirement: id.clone(),
                        buffer,
                    },
                    None => MemoryPlanningDecision::Allocate {
                        requirement: id.clone(),
                    },
                });
                output_descriptors.insert(
                    (node.id.clone(), output.id.clone()),
                    output.descriptor.clone(),
                );
                plan.tensor_lifetimes.push(lifetime);
            }
            completed_nodes.insert(node.id.clone());
        }

        for output in &graph.outputs {
            match &output.source {
                ComputeValueRef::Input(input) => {
                    if let Some(affinity) = input_affinities.get(input) {
                        plan.output_affinity = affinity.clone();
                    }
                }
                ComputeValueRef::NodeOutput { .. } => {
                    plan.output_affinity = target.clone();
                }
            }
        }
        self.validate_memory_plan(&metadata, &mut plan)?;
        Ok(plan)
    }
    pub fn validate_memory_plan(
        &self,
        metadata: &ProviderMetadata,
        plan: &mut MemoryPlan,
    ) -> Result<(), MemoryPlanningError> {
        let provider_limit = provider_memory_limit(metadata);
        if provider_limit != u64::MAX {
            plan.diagnostics
                .push(MemoryPlanningDiagnostic::ProviderLimit {
                    provider: plan.provider.clone(),
                    max_bytes: provider_limit,
                });
        }
        for requirement in &plan.requirements {
            if requirement.byte_size > provider_limit {
                return Err(MemoryPlanningError::ProviderMemoryLimitExceeded {
                    provider: plan.provider.clone(),
                    required: requirement.byte_size,
                    limit: provider_limit,
                    report: plan.pressure.clone(),
                });
            }
        }
        let selected_device = plan
            .requirements
            .iter()
            .find_map(|requirement| requirement.affinity.device().cloned())
            .or_else(|| {
                self.providers
                    .registry()
                    .devices_for_provider(plan.provider.as_str())
                    .find(|device| device.metadata().memory_capacity > 0)
                    .map(|device| DeviceBinding::new(device.id().clone()))
            });
        if let Some(device) = selected_device {
            if let Some(runtime_device) = self.device(device.id()) {
                let limit = runtime_device.metadata().memory_capacity;
                if limit > 0 {
                    plan.pressure.selected_device = Some(device.clone());
                    plan.diagnostics
                        .push(MemoryPlanningDiagnostic::DeviceLimit {
                            device: device.clone(),
                            max_bytes: limit,
                        });
                    if plan.pressure.estimated_peak_bytes > limit {
                        plan.pressure.rejected_device_limit = Some(limit);
                        return Err(MemoryPlanningError::DeviceMemoryLimitExceeded {
                            device,
                            required: plan.pressure.estimated_peak_bytes,
                            limit,
                            report: plan.pressure.clone(),
                        });
                    }
                }
            }
        }
        Ok(())
    }
    pub fn plan_compute_execution(
        &self,
        graph: &ComputeGraph,
    ) -> Result<ComputeExecutionPlan, ComputePlanningError> {
        if !self.initialized {
            return Err(ComputePlanningError::PlanningFailed {
                reason: "runtime is not initialized".into(),
            });
        }

        let dependencies = graph
            .inputs
            .iter()
            .filter_map(|input| input.value.affinity())
            .collect::<Vec<_>>();
        let mut constraints = AffinityConstraints::try_from_affinities(dependencies)
            .map_err(ComputePlanningError::IncompatibleResourceAffinity)?;
        constraints.require_fallback(FallbackClass::ProviderPinned);
        constraints
            .merge(
                &ResourceAffinity::new(FallbackClass::ProviderPinned)
                    .with_capability(CapabilityBinding::new(
                        CapabilityId::new(COMPUTE_CAPABILITY_ID),
                        COMPUTE_CAPABILITY_VERSION,
                    ))
                    .with_execution_context(self.context.id),
            )
            .map_err(ComputePlanningError::IncompatibleResourceAffinity)?;
        if !graph.inputs.is_empty() && constraints.affinity().group().is_none() {
            constraints
                .merge(
                    &ResourceAffinity::new(FallbackClass::Transparent)
                        .with_group(next_affinity_group_id()),
                )
                .map_err(ComputePlanningError::IncompatibleResourceAffinity)?;
        }

        let compute = compute_capability();
        let (provider, capability, decision) = self
            .providers
            .resolve_with_constraints(
                &compute,
                &constraints,
                self.context.config.resolution_policy,
                ExecutionPhase::BeforeResourceCreation,
                true,
            )
            .map_err(planning_error_from_affinity)?;
        let metadata = provider.metadata();
        let provider_binding = ProviderBinding::new(&metadata.name);
        let selected_device = decision.selected_device.clone().or_else(|| {
            constraints.affinity().device().cloned().or_else(|| {
                self.providers
                    .registry()
                    .devices_for_provider(provider_binding.as_str())
                    .find(|device| device.availability() != DeviceAvailability::Unavailable)
                    .map(|device| DeviceBinding::new(device.id().clone()))
            })
        });
        if let Some(device) = &selected_device
            && self.device(device.id()).is_none()
        {
            return Err(ComputePlanningError::DeviceUnavailable(device.clone()));
        }

        self.validate_compute_graph(provider_binding.as_str(), graph)
            .map_err(planning_error_from_validation)?;
        let memory_plan = self
            .plan_compute_graph_memory(provider_binding.as_str(), graph)
            .map_err(ComputePlanningError::MemoryPlanFailed)?;

        let target_affinity = ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(provider_binding.clone())
            .with_capability(CapabilityBinding::new(
                capability.id.clone(),
                capability.version,
            ))
            .with_execution_context(self.context.id);
        let mut input_descriptors = BTreeMap::new();
        let mut output_descriptors = BTreeMap::new();
        let mut completed_nodes = BTreeSet::new();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut constraints_out = vec![
            ExecutionConstraint::ResolutionPolicy(decision.policy_id.clone()),
            ExecutionConstraint::Provider(provider_binding.clone()),
            ExecutionConstraint::CapabilityVersion(CapabilityBinding::new(
                capability.id.clone(),
                capability.version,
            )),
            ExecutionConstraint::NoHiddenCpuStaging,
            ExecutionConstraint::NoImplicitProviderMigration,
            ExecutionConstraint::DeterministicBehavior,
        ];
        if let Some(device) = &selected_device {
            constraints_out.push(ExecutionConstraint::Device(device.clone()));
        }

        for input in &graph.inputs {
            let descriptor = input.value.descriptor().clone();
            input_descriptors.insert(input.id.clone(), descriptor.clone());
            let affinity = input
                .value
                .affinity()
                .cloned()
                .unwrap_or_else(|| target_affinity.clone());
            if let Some(group) = affinity.group() {
                constraints_out.push(ExecutionConstraint::AffinityGroup(group));
            }
            constraints_out.push(ExecutionConstraint::ResourceAffinity(affinity.clone()));
            let resource = match &input.value {
                ComputeInputValue::TensorResource(resource) => Some(resource.id.clone()),
                ComputeInputValue::TensorDescriptor(_) | ComputeInputValue::Constant(_) => None,
            };
            inputs.push(ExecutionInput {
                id: input.id.clone(),
                descriptor,
                resource,
                affinity,
            });
        }

        for node in &graph.nodes {
            if let Some(schema_id) = &node.operation.schema_id {
                constraints_out.push(ExecutionConstraint::OperationSchema(schema_id.clone()));
            }
            if let Some(dtype) = node.operation.dtype {
                constraints_out.push(ExecutionConstraint::DType(dtype));
            }
            if let Some(layout) = node.operation.layout {
                constraints_out.push(ExecutionConstraint::Layout(layout));
            }
            if let Some(precision) = node.operation.precision {
                constraints_out.push(ExecutionConstraint::PrecisionPolicy(precision));
            }
            for output in &node.outputs {
                output_descriptors.insert(
                    (node.id.clone(), output.id.clone()),
                    output.descriptor.clone(),
                );
            }
            completed_nodes.insert(node.id.clone());
        }

        for requirement in &memory_plan.requirements {
            constraints_out.push(ExecutionConstraint::MemoryRequirement(
                requirement.id.clone(),
            ));
        }
        for decision in &memory_plan.decisions {
            match decision {
                MemoryPlanningDecision::RequireTransfer { requirement }
                | MemoryPlanningDecision::AccountHostStaging { requirement } => {
                    constraints_out.push(ExecutionConstraint::ExplicitTransferRequirement(
                        requirement.clone(),
                    ));
                }
                MemoryPlanningDecision::RequireMaterialization { requirement } => {
                    constraints_out.push(ExecutionConstraint::ExplicitMaterializationRequired(
                        requirement.clone(),
                    ));
                }
                MemoryPlanningDecision::PreservePinnedResource { .. } => {}
                MemoryPlanningDecision::Allocate { .. } | MemoryPlanningDecision::Reuse { .. } => {}
            }
        }

        for output in &graph.outputs {
            let descriptor = resolve_compute_value_descriptor(
                None,
                &output.source,
                &input_descriptors,
                &output_descriptors,
                &completed_nodes,
            )
            .map_err(planning_error_from_validation)?
            .clone();
            let affinity = match &output.source {
                ComputeValueRef::Input(input) => inputs
                    .iter()
                    .find(|candidate| &candidate.id == input)
                    .map(|input| input.affinity.clone())
                    .unwrap_or_else(|| target_affinity.clone()),
                ComputeValueRef::NodeOutput { .. } => memory_plan.output_affinity.clone(),
            };
            outputs.push(ExecutionOutput {
                id: output.id.clone(),
                descriptor,
                affinity,
            });
        }

        let mut steps = vec![
            ExecutionStep::new(
                "validate:graph",
                ComputeExecutionPhase::Validation,
                ExecutionStepKind::ValidateGraph,
                provider_binding.clone(),
            )
            .with_device(selected_device.clone()),
            ExecutionStep::new(
                "resolve:provider",
                ComputeExecutionPhase::Resolution,
                ExecutionStepKind::ResolveProvider,
                provider_binding.clone(),
            )
            .with_device(selected_device.clone())
            .depends_on("validate:graph"),
        ];
        if selected_device.is_some() {
            steps.push(
                ExecutionStep::new(
                    "resolve:device",
                    ComputeExecutionPhase::Resolution,
                    ExecutionStepKind::ResolveDevice,
                    provider_binding.clone(),
                )
                .with_device(selected_device.clone())
                .depends_on("resolve:provider"),
            );
        }
        for input in &inputs {
            let kind = if input.affinity.fallback() == FallbackClass::ProviderPinned {
                ExecutionStepKind::PreserveProviderPinnedAffinity
            } else if input.affinity.device().is_some() {
                ExecutionStepKind::PreserveDeviceBoundAffinity
            } else if input.affinity.group().is_some() {
                ExecutionStepKind::PreserveAffinityGroup
            } else {
                ExecutionStepKind::BindInputResource
            };
            steps.push(
                ExecutionStep::new(
                    format!("bind:input:{}", input.id),
                    ComputeExecutionPhase::Planning,
                    kind,
                    provider_binding.clone(),
                )
                .with_device(selected_device.clone())
                .depends_on("resolve:provider"),
            );
        }
        for decision in &memory_plan.decisions {
            let kind = execution_step_kind_from_memory_decision(decision);
            steps.push(
                ExecutionStep::new(
                    format!("memory:{decision:?}"),
                    execution_phase_from_step_kind(&kind),
                    kind,
                    provider_binding.clone(),
                )
                .with_device(selected_device.clone())
                .depends_on("resolve:provider"),
            );
        }
        steps.push(
            ExecutionStep::new(
                "validate:memory",
                ComputeExecutionPhase::MemoryAllocation,
                ExecutionStepKind::ValidateMemory,
                provider_binding.clone(),
            )
            .with_device(selected_device.clone())
            .depends_on("resolve:provider"),
        );
        for output in &outputs {
            steps.push(
                ExecutionStep::new(
                    format!("bind:output:{}", output.id),
                    ComputeExecutionPhase::Planning,
                    ExecutionStepKind::BindOutputResource,
                    provider_binding.clone(),
                )
                .with_device(selected_device.clone())
                .depends_on("validate:memory"),
            );
        }
        steps.push(
            ExecutionStep::new(
                "submit:provider",
                ComputeExecutionPhase::ProviderSubmission,
                ExecutionStepKind::SubmitToProvider,
                provider_binding.clone(),
            )
            .with_device(selected_device.clone())
            .depends_on("validate:memory"),
        );

        let mut diagnostics = vec![
            ExecutionDiagnostic::SelectedProvider(provider_binding.clone()),
            ExecutionDiagnostic::SelectedCapability(CapabilityBinding::new(
                capability.id.clone(),
                capability.version,
            )),
            ExecutionDiagnostic::PolicyDecisionReason(decision.reason.clone()),
            ExecutionDiagnostic::ResolutionDecision(decision.clone()),
        ];
        if let Some(device) = &selected_device {
            diagnostics.push(ExecutionDiagnostic::SelectedDevice(device.clone()));
        }
        diagnostics.extend(decision.rejected_candidates.iter().map(|rejection| {
            ExecutionDiagnostic::RejectedProviderCandidate {
                provider: rejection.provider.clone(),
                reason: rejection.reason.clone(),
            }
        }));
        diagnostics.extend(
            memory_plan
                .diagnostics
                .iter()
                .cloned()
                .map(ExecutionDiagnostic::Memory),
        );

        let mut plan = ComputeExecutionPlan {
            id: execution_plan_id(&graph.id, &provider_binding),
            graph: graph.id.clone(),
            provider: provider_binding,
            device: selected_device,
            capability: CapabilityBinding::new(capability.id.clone(), capability.version),
            policy: decision.policy_id,
            classification: classify_execution_plan(&inputs),
            inputs,
            outputs,
            constraints: constraints_out,
            steps,
            memory_plan,
            diagnostics,
            validated: false,
        };
        self.validate_compute_execution_plan(&plan)?;
        plan.validated = true;
        Ok(plan)
    }
    pub fn validate_compute_execution_plan(
        &self,
        plan: &ComputeExecutionPlan,
    ) -> Result<(), ComputePlanningError> {
        if plan.graph
            != plan.memory_plan.graph.clone().ok_or_else(|| {
                ComputePlanningError::InvalidExecutionPlan {
                    reason: "execution plan memory plan is not tied to a graph".into(),
                }
            })?
        {
            return Err(ComputePlanningError::InvalidExecutionPlan {
                reason: "execution plan graph does not match memory plan graph".into(),
            });
        }
        if self.providers.provider(plan.provider.as_str()).is_none() {
            return Err(ComputePlanningError::ProviderUnavailable(
                plan.provider.clone(),
            ));
        }
        if let Some(device) = &plan.device
            && self.device(device.id()).is_none()
        {
            return Err(ComputePlanningError::DeviceUnavailable(device.clone()));
        }
        let step_ids = plan
            .steps
            .iter()
            .map(|step| step.id.as_str())
            .collect::<BTreeSet<_>>();
        for step in &plan.steps {
            for dependency in &step.dependencies {
                if !step_ids.contains(dependency.as_str()) {
                    return Err(ComputePlanningError::InvalidExecutionPlan {
                        reason: format!(
                            "step '{}' has unresolved dependency '{}'",
                            step.id, dependency
                        ),
                    });
                }
            }
            if step.provider != plan.provider {
                return Err(ComputePlanningError::InvalidExecutionPlan {
                    reason: format!(
                        "step '{}' migrates provider from '{}' to '{}'",
                        step.id, plan.provider, step.provider
                    ),
                });
            }
        }
        if plan
            .inputs
            .iter()
            .any(|input| input.resource.is_some() && input.affinity.provider().is_none())
        {
            return Err(ComputePlanningError::InvalidExecutionPlan {
                reason: "tensor resource inputs must retain Provider affinity".into(),
            });
        }
        Ok(())
    }
    pub fn scheduler(&self, capacity: usize) -> Scheduler {
        Scheduler::new(SchedulingPolicy::Fifo, capacity)
    }
    pub fn validate_scheduler_plan(
        &self,
        plan: &ComputeExecutionPlan,
    ) -> Result<(), ComputePlanningError> {
        if !plan.is_validated() {
            return Err(ComputePlanningError::InvalidExecutionPlan {
                reason: "scheduler accepts only validated execution plans".into(),
            });
        }
        self.validate_compute_execution_plan(plan)?;
        if plan.constraints.iter().any(|constraint| {
            matches!(constraint, ExecutionConstraint::NoImplicitProviderMigration)
        }) == false
        {
            return Err(ComputePlanningError::InvalidExecutionPlan {
                reason: "execution plan must forbid implicit Provider migration".into(),
            });
        }
        Ok(())
    }
    pub fn schedule_compute_execution(
        &self,
        scheduler: &mut Scheduler,
        plan: ComputeExecutionPlan,
    ) -> Result<ScheduledOperationId, SchedulerError> {
        scheduler.schedule(self, plan)
    }
    pub fn validate_compute_graph(
        &self,
        provider: &str,
        graph: &ComputeGraph,
    ) -> Result<ComputeGraphValidationReport, ComputeValidationError> {
        ensure_non_empty_id("graph", graph.id.as_str())?;

        let mut input_ids = BTreeSet::new();
        let mut input_descriptors = BTreeMap::new();
        let mut resource_affinities = Vec::new();
        for input in &graph.inputs {
            ensure_non_empty_id("input", input.id.as_str())?;
            insert_unique(&mut input_ids, "input", &input.id)?;
            input
                .value
                .descriptor()
                .validate(&TensorDescriptorLimits::default())?;
            input_descriptors.insert(input.id.clone(), input.value.descriptor().clone());
            if let Some(affinity) = input.value.affinity() {
                resource_affinities.push(affinity);
            }
        }

        let mut node_ids = BTreeSet::new();
        let mut completed_nodes = BTreeSet::new();
        let mut output_descriptors = BTreeMap::new();
        let mut operations = Vec::new();
        for node in &graph.nodes {
            ensure_non_empty_id("node", node.id.as_str())?;
            insert_unique(&mut node_ids, "node", &node.id)?;

            let mut node_output_ids = BTreeSet::new();
            for output in &node.outputs {
                ensure_non_empty_id("node output", output.id.as_str())?;
                insert_unique(&mut node_output_ids, "node output", &output.id)?;
                output
                    .descriptor
                    .validate(&TensorDescriptorLimits::default())?;
                output_descriptors.insert(
                    (node.id.clone(), output.id.clone()),
                    output.descriptor.clone(),
                );
            }

            let mut operation = node.operation.clone();
            for input in &node.inputs {
                operation.tensors.push(
                    resolve_compute_value_descriptor(
                        Some(&node.id),
                        input,
                        &input_descriptors,
                        &output_descriptors,
                        &completed_nodes,
                    )?
                    .clone(),
                );
            }
            operation
                .tensors
                .extend(node.outputs.iter().map(|output| output.descriptor.clone()));
            operations.push(operation);
            completed_nodes.insert(node.id.clone());
        }

        let mut graph_output_ids = BTreeSet::new();
        for output in &graph.outputs {
            ensure_non_empty_id("graph output", output.id.as_str())?;
            insert_unique(&mut graph_output_ids, "graph output", &output.id)?;
            resolve_compute_value_descriptor(
                None,
                &output.source,
                &input_descriptors,
                &output_descriptors,
                &completed_nodes,
            )?;
        }

        self.validate_compute_operations(provider, &operations)?;
        self.plan_compute_graph_memory(provider, graph)
            .map_err(ComputeValidationError::MemoryPlanning)?;
        if !resource_affinities.is_empty() {
            let resources = graph
                .inputs
                .iter()
                .filter_map(|input| match &input.value {
                    ComputeInputValue::TensorResource(resource) => Some(resource.clone()),
                    ComputeInputValue::TensorDescriptor(_) | ComputeInputValue::Constant(_) => None,
                })
                .collect::<Vec<_>>();
            self.validate_compute_tensor_resources(provider, &resources)?;
        }

        Ok(ComputeGraphValidationReport {
            provider: ProviderBinding::new(provider),
            graph: graph.id.clone(),
            node_count: graph.nodes.len(),
            input_count: graph.inputs.len(),
            output_count: graph.outputs.len(),
        })
    }
    pub fn submit_validated_compute_graph(
        &self,
        provider: &str,
        graph: &ComputeGraph,
    ) -> Result<ComputeSubmission, ComputeValidationError> {
        self.validate_compute_graph(provider, graph)?;
        let dependencies = graph
            .inputs
            .iter()
            .filter_map(|input| input.value.affinity())
            .collect::<Vec<_>>();
        let mut constraints = AffinityConstraints::try_from_affinities(dependencies)
            .map_err(ComputeValidationError::IncompatibleResourceAffinity)?;
        constraints.require_fallback(FallbackClass::ProviderPinned);
        constraints
            .merge(
                &ResourceAffinity::new(FallbackClass::ProviderPinned)
                    .with_provider(ProviderBinding::new(provider))
                    .with_capability(CapabilityBinding::new(
                        CapabilityId::new(COMPUTE_CAPABILITY_ID),
                        COMPUTE_CAPABILITY_VERSION,
                    ))
                    .with_execution_context(self.context.id),
            )
            .map_err(ComputeValidationError::IncompatibleResourceAffinity)?;
        if constraints.affinity().group().is_none() {
            constraints
                .merge(
                    &ResourceAffinity::new(FallbackClass::Transparent)
                        .with_group(next_affinity_group_id()),
                )
                .map_err(ComputeValidationError::IncompatibleResourceAffinity)?;
        }
        let affinity = constraints.into_affinity();
        Ok(ComputeSubmission::new(
            graph.id.clone(),
            ProviderBinding::new(provider),
            affinity,
        ))
    }
    pub fn wrap_compute_outputs(
        &self,
        submission: &ComputeSubmission,
        outputs: impl IntoIterator<Item = (TensorResourceId, TensorDescriptor)>,
    ) -> Vec<TensorResourceDescriptor> {
        outputs
            .into_iter()
            .map(|(id, descriptor)| {
                TensorResourceDescriptor::new(id, descriptor, submission.affinity.clone())
            })
            .collect()
    }
    pub fn wrap_compute_data_movement_output(
        &self,
        provider: &str,
        movement: &ComputeDataMovementDescriptor,
        id: TensorResourceId,
    ) -> Result<TensorResourceDescriptor, ComputeValidationError> {
        self.validate_compute_data_movement(provider, std::slice::from_ref(movement))?;
        let mut affinity = ResourceAffinity::new(FallbackClass::ProviderPinned)
            .with_provider(
                movement
                    .target_provider
                    .clone()
                    .unwrap_or_else(|| ProviderBinding::new(provider)),
            )
            .with_capability(CapabilityBinding::new(
                CapabilityId::new(COMPUTE_CAPABILITY_ID),
                COMPUTE_CAPABILITY_VERSION,
            ))
            .with_execution_context(self.context.id);
        if let Some(device) = &movement.target_device {
            affinity = affinity.with_device(device.clone());
        }
        if let Some(group) = movement.target_group {
            affinity = affinity.with_group(group);
        } else if let Some(source) = movement.source.tensor()
            && let Some(group) = source.affinity.group()
        {
            affinity = affinity.with_group(group);
        }
        Ok(TensorResourceDescriptor::new(
            id,
            movement.output.clone(),
            affinity,
        ))
    }
    pub fn select_backend(&mut self, name: &str) -> Result<(), ProviderError> {
        if self.providers.registry().backend(name).is_none() {
            return Err(ProviderError::BackendNotFound(name.into()));
        }
        self.context.backend_name = Some(name.into());
        Ok(())
    }
    pub fn selected_backend(&self) -> Option<&dyn Backend> {
        self.context
            .backend_name
            .as_deref()
            .and_then(|x| self.providers.registry().backend(x))
    }
    pub fn shutdown(&mut self) {
        let _ = self.providers.shutdown();
        self.context.backend_name = None;
        self.initialized = false;
    }
}

#[derive(Debug)]
pub enum ProviderError {
    ProviderAlreadyRegistered(String),
    BackendAlreadyRegistered(String),
    BackendNotFound(String),
    DeviceAlreadyRegistered(DeviceId),
    DeviceProviderMismatch {
        device: DeviceId,
        expected: String,
        found: String,
    },
    InvalidDevice(String),
    IncompatibleApiVersion {
        provider: String,
        expected: u32,
        found: u32,
    },
    InvalidCapability(String),
    InvalidCapabilityVersion(String),
    ConflictingCapability {
        id: CapabilityId,
        version: CapabilityVersion,
    },
    MissingCapabilityDependency {
        capability: CapabilityId,
        dependency: CapabilityId,
        required: CapabilityVersion,
    },
    NoCompatibleProvider(CapabilityBinding),
    PolicyRejectedProvider {
        capability: CapabilityBinding,
        policy: ResolutionPolicyId,
    },
    Discovery {
        path: PathBuf,
        source: std::io::Error,
    },
    Load {
        path: PathBuf,
        message: String,
    },
    Registration(String),
    Lifecycle(String),
}
impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProviderAlreadyRegistered(x) => write!(f, "provider '{x}' is already registered"),
            Self::BackendAlreadyRegistered(x) => write!(f, "backend '{x}' is already registered"),
            Self::BackendNotFound(x) => write!(f, "backend '{x}' is not registered"),
            Self::DeviceAlreadyRegistered(id) => write!(f, "device '{id}' is already registered"),
            Self::DeviceProviderMismatch {
                device,
                expected,
                found,
            } => write!(
                f,
                "device '{device}' is owned by provider '{found}', not registering provider '{expected}'"
            ),
            Self::InvalidDevice(message) => write!(f, "invalid device: {message}"),
            Self::IncompatibleApiVersion {
                provider,
                expected,
                found,
            } => write!(
                f,
                "provider '{provider}' targets API {found}, but Magnetar supports API {expected}"
            ),
            Self::InvalidCapability(message) => write!(f, "invalid capability: {message}"),
            Self::InvalidCapabilityVersion(version) => {
                write!(f, "invalid capability semantic version '{version}'")
            }
            Self::ConflictingCapability { id, version } => {
                write!(
                    f,
                    "capability '{id}@{version}' has a conflicting definition"
                )
            }
            Self::MissingCapabilityDependency {
                capability,
                dependency,
                required,
            } => write!(
                f,
                "capability '{capability}' requires unavailable dependency '{dependency}@{required}'"
            ),
            Self::NoCompatibleProvider(capability) => {
                write!(
                    f,
                    "no Provider implements compatible capability '{capability}'"
                )
            }
            Self::PolicyRejectedProvider { capability, policy } => write!(
                f,
                "resolution policy '{policy}' rejected every Provider for capability '{capability}'"
            ),
            Self::Discovery { path, source } => write!(
                f,
                "could not discover providers in '{}': {source}",
                path.display()
            ),
            Self::Load { path, message } => {
                write!(f, "could not load provider '{}': {message}", path.display())
            }
            Self::Registration(x) => write!(f, "provider registration failed: {x}"),
            Self::Lifecycle(x) => write!(f, "provider lifecycle operation failed: {x}"),
        }
    }
}
impl Error for ProviderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        if let Self::Discovery { source, .. } = self {
            Some(source)
        } else {
            None
        }
    }
}

impl From<ProviderError> for ComputeError {
    fn from(error: ProviderError) -> Self {
        let message = error.to_string();
        match error {
            ProviderError::NoCompatibleProvider(capability) => ComputeError::new(
                ComputeErrorCode::NoCompatibleProvider,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_capability(capability)),
            ProviderError::PolicyRejectedProvider {
                capability,
                policy: _,
            } => ComputeError::new(
                ComputeErrorCode::PolicyRejectedProvider,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_capability(capability)),
            ProviderError::BackendNotFound(provider) => ComputeError::new(
                ComputeErrorCode::ProviderUnavailable,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_provider(ProviderBinding::new(provider)))
            .with_recovery_hint(RecoveryHint::RetryBeforeState),
            ProviderError::DeviceAlreadyRegistered(device) => ComputeError::new(
                ComputeErrorCode::DeviceUnavailable,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_device(DeviceBinding::new(device))),
            ProviderError::DeviceProviderMismatch {
                device,
                expected,
                found,
            } => ComputeError::new(
                ComputeErrorCode::DeviceBoundResource,
                ComputeErrorPhase::AffinityValidation,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(
                ComputeDiagnostic::new()
                    .with_device(DeviceBinding::new(device))
                    .with_provider(ProviderBinding::new(found))
                    .with_rejected_candidate(ProviderBinding::new(expected)),
            ),
            ProviderError::IncompatibleApiVersion { provider, .. } => ComputeError::new(
                ComputeErrorCode::ProviderUnavailable,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(
                ComputeDiagnostic::new().with_provider(ProviderBinding::new(provider)),
            ),
            ProviderError::InvalidCapabilityVersion(version) => ComputeError::new(
                ComputeErrorCode::CapabilityVersionMismatch,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_backend_message(version)),
            ProviderError::Lifecycle(message_text) => ComputeError::new(
                ComputeErrorCode::ExecutionInterrupted,
                ComputeErrorPhase::Interruption,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_backend_message(message_text))
            .with_recovery_hint(RecoveryHint::RetryBeforeState),
            ProviderError::Load {
                path: _,
                message: backend_message,
            } => ComputeError::new(
                ComputeErrorCode::ProviderUnavailable,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Recoverable,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_backend_message(backend_message))
            .with_recovery_hint(RecoveryHint::RetryBeforeState),
            other => ComputeError::new(
                ComputeErrorCode::Internal,
                ComputeErrorPhase::Resolution,
                ComputeErrorSeverity::Terminal,
                message,
            )
            .with_diagnostic(ComputeDiagnostic::new().with_backend_message(other.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    };
    struct TestBackend;
    impl Backend for TestBackend {
        fn name(&self) -> &str {
            "test"
        }
        fn devices(&self) -> Vec<Arc<dyn Device>> {
            vec![]
        }
    }
    struct TestProvider {
        metadata: ProviderMetadata,
        initialized: AtomicBool,
        shut_down: AtomicBool,
        backend: bool,
        fail_initialization: bool,
        health: ProviderHealth,
        devices: Vec<Arc<dyn Device>>,
    }
    impl TestProvider {
        fn new(name: &str) -> Self {
            Self {
                metadata: ProviderMetadata::new(name, "1", "test", "test"),
                initialized: AtomicBool::new(false),
                shut_down: AtomicBool::new(false),
                backend: false,
                fail_initialization: false,
                health: ProviderHealth::Healthy,
                devices: Vec::new(),
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
    impl Provider for TestProvider {
        fn metadata(&self) -> ProviderMetadata {
            self.metadata.clone()
        }
        fn register(&self, r: &mut ProviderRegistry) -> Result<(), ProviderError> {
            if self.backend {
                r.register_backend(Arc::new(TestBackend))
            } else {
                Ok(())
            }
        }
        fn health(&self) -> ProviderHealth {
            self.health
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
        assert!(
            !CapabilityVersion::new(0, 1, 1).is_compatible_with(CapabilityVersion::new(0, 1, 0))
        );
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
            BTreeSet::from([WitInterface::new(COMPUTE_WIT_INTERFACE, "1.1.0")])
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
    fn compute_wit_defines_the_stabilized_run_surface() {
        let wit = include_str!("../wit/compute.wit");
        assert!(wit.contains("package magnetar:compute@1.1.0;"));
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
                ComputeCapabilitySupport::default()
                    .with_versions([CapabilityVersion::new(0, 9, 0)]),
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

        let mut unsupported =
            provider_with_capabilities("unsupported-schema", [compute_capability()]);
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
            .validate_compute_data_movement("portable-compute", &[upload.clone()])
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
        let transfer = ComputeDataMovementDescriptor::transfer(foreign, descriptor.clone())
            .with_target_provider(ProviderBinding::new("portable-compute"));
        runtime
            .validate_compute_data_movement("portable-compute", &[transfer])
            .unwrap();

        let materialized = TensorResourceDescriptor::new(
            TensorResourceId::new("materialized"),
            descriptor.clone(),
            ResourceAffinity::new(FallbackClass::ProviderPinned)
                .with_provider(ProviderBinding::new("portable-compute")),
        );
        let invalid_materialize =
            ComputeDataMovementDescriptor::materialize(materialized, descriptor);
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
        let movement = ComputeDataMovementDescriptor::transfer(source, descriptor)
            .with_target_provider(ProviderBinding::new("movement-compute"))
            .with_host_staging();

        let plan = runtime
            .plan_compute_data_movement_memory("movement-compute", &[movement])
            .unwrap();

        assert_eq!(plan.pressure.transfer_buffer_cost_bytes, 16);
        assert!(plan.decisions.iter().any(|decision| {
            matches!(decision, MemoryPlanningDecision::AccountHostStaging { .. })
        }));
    }
    #[test]
    fn provider_compute_advertisement_drives_data_movement_validation() {
        let mut provider =
            provider_with_capabilities("advertised-movement", [compute_capability()]);
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
        let mut unavailable =
            provider_with_capabilities("portable-compute", [compute_capability()]);
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

        let mut device_provider =
            provider_with_capabilities("device-limited", [compute_capability()]);
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
                .resolve_component_import(&WitInterface::new(COMPUTE_WIT_INTERFACE, "1.0.0",))
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
            runtime.resolve_component_import(&WitInterface::new(COMPUTE_WIT_INTERFACE, "1.1.0")),
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
                preferred_backend: None,
                resolution_policy: BuiltInResolutionPolicy::Availability,
            })
            .register_provider(Arc::new(degraded))
            .register_provider(Arc::new(healthy))
            .build()
            .unwrap();

        let providers =
            runtime.resolve_component_import(&WitInterface::new(COMPUTE_WIT_INTERFACE, "1.1.0"));

        assert_eq!(providers.unwrap()[0].metadata().name, "z-healthy");
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
    fn backend_provider_and_shutdown() {
        let mut p = TestProvider::new("backend");
        p.backend = true;
        let p = Arc::new(p);
        let mut m = ProviderLoader::new();
        m.register_provider(p.clone()).unwrap();
        assert!(m.registry().backend("test").is_some());
        m.shutdown().unwrap();
        assert!(p.shut_down.load(Ordering::SeqCst));
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
        let capability_a =
            capability_binding("magnetar:compute/run", CapabilityVersion::new(1, 1, 0));
        let capability_b =
            capability_binding("magnetar:tokenize/run", CapabilityVersion::new(1, 0, 0));
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

    struct TestComponent {
        metadata: ComponentMetadata,
        events: Arc<Mutex<Vec<String>>>,
    }
    impl TestComponent {
        fn new(name: &str, events: Arc<Mutex<Vec<String>>>) -> Self {
            Self {
                metadata: ComponentMetadata::new(name, "1", "test component"),
                events,
            }
        }
        fn event(&self, event: &str) {
            self.events
                .lock()
                .unwrap()
                .push(format!("{}:{event}", self.metadata.name));
        }
    }
    impl Component for TestComponent {
        fn metadata(&self) -> ComponentMetadata {
            self.metadata.clone()
        }
        fn instantiate(&mut self) -> Result<(), ComponentError> {
            self.event("instantiate");
            Ok(())
        }
        fn start(&mut self) -> Result<(), ComponentError> {
            self.event("start");
            Ok(())
        }
        fn stop(&mut self) -> Result<(), ComponentError> {
            self.event("stop");
            Ok(())
        }
        fn destroy(&mut self) -> Result<(), ComponentError> {
            self.event("destroy");
            Ok(())
        }
    }

    #[test]
    fn component_lifecycle_resolves_dependencies_and_stops_in_reverse_order() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let first = TestComponent::new("first", events.clone());
        let mut second = TestComponent::new("second", events.clone());
        second.metadata.dependencies.insert("first".into());
        let mut manager = ComponentManager::new();
        manager.register_component(Box::new(second)).unwrap();
        manager.register_component(Box::new(first)).unwrap();

        manager.start_all().unwrap();
        assert_eq!(
            manager.component_state("first"),
            Some(ComponentState::Started)
        );
        manager.shutdown();
        assert_eq!(
            *events.lock().unwrap(),
            vec![
                "first:instantiate",
                "first:start",
                "second:instantiate",
                "second:start",
                "second:stop",
                "first:stop",
                "first:destroy",
                "second:destroy",
            ]
        );
    }

    #[test]
    fn component_contracts_require_host_or_component_exports() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut component = TestComponent::new("consumer", events);
        let interface = WitInterface::new("magnetar:runtime/run", "1.0.0");
        component.metadata.imports.insert(interface.clone());
        let mut manager = ComponentManager::new();
        manager.register_component(Box::new(component)).unwrap();
        assert!(matches!(
            manager.validate_interfaces(),
            Err(ComponentError::MissingInterface { .. })
        ));
        manager.provide_interface(interface);
        manager.validate_interfaces().unwrap();
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
}
