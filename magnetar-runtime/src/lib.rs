//! Hardware-agnostic runtime contracts and provider support for Magnetar.

use libloading::Library;
use std::{
    collections::{BTreeMap, BTreeSet},
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
    pub compute_operation_support: BTreeMap<ComputeOperationFamily, ComputeOperationSupport>,
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
            compute_operation_support: BTreeMap::new(),
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

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeLayout {
    Dense,
    Strided,
    ProviderOpaque,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputePrecision {
    Exact,
    Default,
    Reduced,
    Mixed,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ComputeOperationSupport {
    pub dtypes: BTreeSet<ComputeDType>,
    pub layouts: BTreeSet<ComputeLayout>,
    pub precision_modes: BTreeSet<ComputePrecision>,
}
impl ComputeOperationSupport {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_dtypes(mut self, dtypes: impl IntoIterator<Item = ComputeDType>) -> Self {
        self.dtypes.extend(dtypes);
        self
    }
    pub fn with_layouts(mut self, layouts: impl IntoIterator<Item = ComputeLayout>) -> Self {
        self.layouts.extend(layouts);
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
}

/// Placeholder for future operation-specific schemas inside `magnetar:compute/run`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeOperationDescriptor {
    pub family: ComputeOperationFamily,
    pub dtype: Option<ComputeDType>,
    pub layout: Option<ComputeLayout>,
    pub precision: Option<ComputePrecision>,
}
impl ComputeOperationDescriptor {
    pub fn new(family: ComputeOperationFamily) -> Self {
        Self {
            family,
            dtype: None,
            layout: None,
            precision: None,
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
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeOperationRequest {
    pub family_id: String,
    pub dtype: Option<ComputeDType>,
    pub layout: Option<ComputeLayout>,
    pub precision: Option<ComputePrecision>,
}
impl ComputeOperationRequest {
    pub fn new(family_id: impl Into<String>) -> Self {
        Self {
            family_id: family_id.into(),
            dtype: None,
            layout: None,
            precision: None,
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
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ComputeValidationError {
    UnknownOperationFamily(String),
    UnsupportedOperationFamily {
        provider: ProviderBinding,
        family: ComputeOperationFamily,
    },
    UnsupportedDType {
        family: ComputeOperationFamily,
        dtype: ComputeDType,
    },
    UnsupportedLayout {
        family: ComputeOperationFamily,
        layout: ComputeLayout,
    },
    UnsupportedPrecision {
        family: ComputeOperationFamily,
        precision: ComputePrecision,
    },
    ProviderUnavailable(ProviderBinding),
}
impl fmt::Display for ComputeValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownOperationFamily(family) => {
                write!(f, "unknown compute operation family '{family}'")
            }
            Self::UnsupportedOperationFamily { provider, family } => write!(
                f,
                "provider '{provider}' does not support compute operation family '{}'",
                family.id()
            ),
            Self::UnsupportedDType { family, dtype } => write!(
                f,
                "compute operation family '{}' does not support dtype {dtype:?}",
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
            Self::ProviderUnavailable(provider) => {
                write!(f, "provider '{provider}' is unavailable")
            }
        }
    }
}
impl Error for ComputeValidationError {}

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

fn next_execution_context_id() -> ExecutionContextId {
    ExecutionContextId::new(
        NEXT_EXECUTION_CONTEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    )
}
fn next_affinity_group_id() -> AffinityGroupId {
    AffinityGroupId::new(NEXT_AFFINITY_GROUP_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
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
                    family,
                    dtype: operation.dtype,
                    layout: operation.layout,
                    precision: operation.precision,
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
        for operation in operations {
            let support = metadata
                .compute_operation_support
                .get(&operation.family)
                .ok_or(ComputeValidationError::UnsupportedOperationFamily {
                    provider: ProviderBinding::new(&metadata.name),
                    family: operation.family,
                })?;
            support.supports(operation)?;
        }
        Ok(())
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
        assert!(wit.contains("record operation-descriptor"));
        assert!(wit.contains("unsupported-operation-family"));
        assert!(wit.contains("unsupported-element-type"));
        assert!(wit.contains("unsupported-layout"));
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
