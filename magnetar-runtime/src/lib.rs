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
pub trait Device: Send + Sync {
    fn id(&self) -> &DeviceId;
    fn device_type(&self) -> DeviceType;
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
    capabilities: BTreeMap<CapabilityId, BTreeMap<CapabilityVersion, Capability>>,
    capability_providers: BTreeMap<(CapabilityId, CapabilityVersion), BTreeSet<String>>,
}
impl ProviderRegistry {
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

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RuntimeConfig {
    pub preferred_backend: Option<String>,
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExecutionContext {
    config: RuntimeConfig,
    backend_name: Option<String>,
}
impl ExecutionContext {
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
    /// Resolves all compatible providers, ordered for deterministic fallback.
    pub fn resolve_providers(&self, capability: &Capability) -> Vec<&dyn Provider> {
        self.providers.resolve_providers(capability)
    }
    /// Resolves providers while reporting invalid capability dependencies.
    pub fn try_resolve_providers(
        &self,
        capability: &Capability,
    ) -> Result<Vec<&dyn Provider>, ProviderError> {
        self.providers.try_resolve_providers(capability)
    }
    /// Resolves providers for a component's WIT import without exposing a
    /// provider dependency to the component itself.
    pub fn resolve_component_import(
        &self,
        interface: &WitInterface,
    ) -> Result<Vec<&dyn Provider>, ProviderError> {
        self.try_resolve_providers(&Capability::from_wit(interface.clone())?)
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
    }
    impl TestProvider {
        fn new(name: &str) -> Self {
            Self {
                metadata: ProviderMetadata::new(name, "1", "test", "test"),
                initialized: AtomicBool::new(false),
                shut_down: AtomicBool::new(false),
                backend: false,
                fail_initialization: false,
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
        fn initialize(&self) -> Result<(), ProviderError> {
            if self.fail_initialization {
                return Err(ProviderError::Lifecycle("unavailable".into()));
            }
            self.initialized.store(true, Ordering::SeqCst);
            Ok(())
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
