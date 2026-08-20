//! Hardware-agnostic runtime contracts and plugin support for Magnetar.

use libloading::Library;
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
    path::{Path, PathBuf},
    sync::Arc,
};

pub const PLUGIN_API_VERSION: u32 = 1;

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

/// A hardware execution contribution. It is distinct from [`Plugin`].
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    fn devices(&self) -> Vec<Arc<dyn Device>>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PluginMetadata {
    pub name: String,
    pub version: String,
    pub vendor: String,
    pub api_version: u32,
    pub description: String,
    pub capabilities: PluginCapabilities,
}
impl PluginMetadata {
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
            api_version: PLUGIN_API_VERSION,
            description: description.into(),
            capabilities: PluginCapabilities::default(),
        }
    }
}

/// Categories reserved for plugins. Backends are the only implemented category today.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PluginCapabilities {
    pub backend: bool,
    pub model_loader: bool,
    pub kernel_provider: bool,
    pub compiler_pass: bool,
    pub scheduler_extension: bool,
    pub telemetry_provider: bool,
}

/// General extension contract; a plugin may register any supported contribution.
pub trait Plugin: Send + Sync {
    fn metadata(&self) -> PluginMetadata;
    fn register(&self, registry: &mut Registry) -> Result<(), PluginError>;
    fn initialize(&self) -> Result<(), PluginError> {
        Ok(())
    }
    fn shutdown(&self) -> Result<(), PluginError> {
        Ok(())
    }
}

/// Receives plugin contributions.
#[derive(Clone, Default)]
pub struct Registry {
    backends: BTreeMap<String, Arc<dyn Backend>>,
}
impl Registry {
    pub fn register_backend(&mut self, backend: Arc<dyn Backend>) -> Result<(), PluginError> {
        let name = backend.name().to_owned();
        if self.backends.contains_key(&name) {
            return Err(PluginError::BackendAlreadyRegistered(name));
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
    fn clear(&mut self) {
        self.backends.clear();
    }
}

#[derive(Default)]
pub struct PluginManager {
    registry: Registry,
    plugins: BTreeMap<String, Arc<dyn Plugin>>,
    order: Vec<String>,
    /* declared last: unloaded after plugin drops */ libraries: Vec<Library>,
}
impl PluginManager {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn registry(&self) -> &Registry {
        &self.registry
    }
    pub fn registry_mut(&mut self) -> &mut Registry {
        &mut self.registry
    }
    pub fn plugin(&self, name: &str) -> Option<&dyn Plugin> {
        self.plugins.get(name).map(AsRef::as_ref)
    }
    pub fn plugin_names(&self) -> impl Iterator<Item = &str> {
        self.plugins.keys().map(String::as_str)
    }
    pub fn register_plugin(&mut self, plugin: Arc<dyn Plugin>) -> Result<(), PluginError> {
        let metadata = plugin.metadata();
        if metadata.api_version != PLUGIN_API_VERSION {
            return Err(PluginError::IncompatibleApiVersion {
                plugin: metadata.name,
                expected: PLUGIN_API_VERSION,
                found: metadata.api_version,
            });
        }
        if self.plugins.contains_key(&metadata.name) {
            return Err(PluginError::PluginAlreadyRegistered(metadata.name));
        }
        let previous = self.registry.clone();
        if let Err(error) = plugin
            .register(&mut self.registry)
            .and_then(|()| plugin.initialize())
        {
            self.registry = previous;
            return Err(error);
        }
        self.order.push(metadata.name.clone());
        self.plugins.insert(metadata.name, plugin);
        Ok(())
    }
    pub fn discover(
        paths: impl IntoIterator<Item = impl AsRef<Path>>,
    ) -> Result<Vec<PathBuf>, PluginError> {
        let mut found = BTreeSet::new();
        for dir in paths {
            let dir = dir.as_ref();
            for entry in std::fs::read_dir(dir).map_err(|source| PluginError::Discovery {
                path: dir.into(),
                source,
            })? {
                let path = entry
                    .map_err(|source| PluginError::Discovery {
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
    /// Loads a compatible Rust library exporting `magnetar_plugin_create`.
    pub unsafe fn load_dynamic(&mut self, path: impl AsRef<Path>) -> Result<(), PluginError> {
        let path = path.as_ref();
        let library = unsafe { Library::new(path) }.map_err(|e| PluginError::Load {
            path: path.into(),
            message: e.to_string(),
        })?;
        type Factory = unsafe fn() -> Box<dyn Plugin>;
        let factory =
            unsafe { library.get::<Factory>(b"magnetar_plugin_create") }.map_err(|e| {
                PluginError::Load {
                    path: path.into(),
                    message: e.to_string(),
                }
            })?;
        self.register_plugin(Arc::from(unsafe { factory() }))?;
        self.libraries.push(library);
        Ok(())
    }
    pub unsafe fn discover_and_load(
        &mut self,
        paths: impl IntoIterator<Item = impl AsRef<Path>>,
    ) -> Result<(), PluginError> {
        for path in Self::discover(paths)? {
            unsafe { self.load_dynamic(path) }?;
        }
        Ok(())
    }
    pub fn shutdown(&mut self) -> Result<(), PluginError> {
        let mut first = None;
        for name in self.order.iter().rev() {
            if let Some(plugin) = self.plugins.get(name) {
                if let Err(error) = plugin.shutdown() {
                    first.get_or_insert(error);
                }
            }
        }
        self.order.clear();
        self.plugins.clear();
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
    plugins: Vec<Arc<dyn Plugin>>,
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
    pub fn register_plugin(mut self, x: Arc<dyn Plugin>) -> Self {
        self.plugins.push(x);
        self
    }
    pub fn build(self) -> Result<Runtime, PluginError> {
        let mut plugins = PluginManager::new();
        for x in self.backends {
            plugins.registry_mut().register_backend(x)?;
        }
        for x in self.plugins {
            plugins.register_plugin(x)?;
        }
        let mut runtime = Runtime {
            context: ExecutionContext {
                config: self.config,
                backend_name: None,
            },
            plugins,
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
    plugins: PluginManager,
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
    pub fn plugins(&self) -> &PluginManager {
        &self.plugins
    }
    pub fn register_backend(&mut self, x: Arc<dyn Backend>) -> Result<(), PluginError> {
        self.plugins.registry_mut().register_backend(x)
    }
    pub fn register_plugin(&mut self, x: Arc<dyn Plugin>) -> Result<(), PluginError> {
        self.plugins.register_plugin(x)
    }
    pub fn select_backend(&mut self, name: &str) -> Result<(), PluginError> {
        if self.plugins.registry().backend(name).is_none() {
            return Err(PluginError::BackendNotFound(name.into()));
        }
        self.context.backend_name = Some(name.into());
        Ok(())
    }
    pub fn selected_backend(&self) -> Option<&dyn Backend> {
        self.context
            .backend_name
            .as_deref()
            .and_then(|x| self.plugins.registry().backend(x))
    }
    pub fn shutdown(&mut self) {
        let _ = self.plugins.shutdown();
        self.context.backend_name = None;
        self.initialized = false;
    }
}

#[derive(Debug)]
pub enum PluginError {
    PluginAlreadyRegistered(String),
    BackendAlreadyRegistered(String),
    BackendNotFound(String),
    IncompatibleApiVersion {
        plugin: String,
        expected: u32,
        found: u32,
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
impl fmt::Display for PluginError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PluginAlreadyRegistered(x) => write!(f, "plugin '{x}' is already registered"),
            Self::BackendAlreadyRegistered(x) => write!(f, "backend '{x}' is already registered"),
            Self::BackendNotFound(x) => write!(f, "backend '{x}' is not registered"),
            Self::IncompatibleApiVersion {
                plugin,
                expected,
                found,
            } => write!(
                f,
                "plugin '{plugin}' targets API {found}, but Magnetar supports API {expected}"
            ),
            Self::Discovery { path, source } => write!(
                f,
                "could not discover plugins in '{}': {source}",
                path.display()
            ),
            Self::Load { path, message } => {
                write!(f, "could not load plugin '{}': {message}", path.display())
            }
            Self::Registration(x) => write!(f, "plugin registration failed: {x}"),
            Self::Lifecycle(x) => write!(f, "plugin lifecycle operation failed: {x}"),
        }
    }
}
impl Error for PluginError {
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
    struct TestPlugin {
        metadata: PluginMetadata,
        initialized: AtomicBool,
        shut_down: AtomicBool,
        backend: bool,
    }
    impl TestPlugin {
        fn new(name: &str) -> Self {
            Self {
                metadata: PluginMetadata::new(name, "1", "test", "test"),
                initialized: AtomicBool::new(false),
                shut_down: AtomicBool::new(false),
                backend: false,
            }
        }
    }
    impl Plugin for TestPlugin {
        fn metadata(&self) -> PluginMetadata {
            self.metadata.clone()
        }
        fn register(&self, r: &mut Registry) -> Result<(), PluginError> {
            if self.backend {
                r.register_backend(Arc::new(TestBackend))
            } else {
                Ok(())
            }
        }
        fn initialize(&self) -> Result<(), PluginError> {
            self.initialized.store(true, Ordering::SeqCst);
            Ok(())
        }
        fn shutdown(&self) -> Result<(), PluginError> {
            self.shut_down.store(true, Ordering::SeqCst);
            Ok(())
        }
    }
    #[test]
    fn load_valid_plugin() {
        let p = Arc::new(TestPlugin::new("valid"));
        let mut m = PluginManager::new();
        m.register_plugin(p.clone()).unwrap();
        assert!(m.plugin("valid").is_some());
        assert!(p.initialized.load(Ordering::SeqCst));
    }
    #[test]
    fn reject_incompatible() {
        let mut p = TestPlugin::new("old");
        p.metadata.api_version += 1;
        assert!(matches!(
            PluginManager::new().register_plugin(Arc::new(p)),
            Err(PluginError::IncompatibleApiVersion { .. })
        ));
    }
    #[test]
    fn reject_duplicate() {
        let mut m = PluginManager::new();
        m.register_plugin(Arc::new(TestPlugin::new("same")))
            .unwrap();
        assert!(matches!(
            m.register_plugin(Arc::new(TestPlugin::new("same"))),
            Err(PluginError::PluginAlreadyRegistered(_))
        ));
    }
    #[test]
    fn backend_plugin_and_shutdown() {
        let mut p = TestPlugin::new("backend");
        p.backend = true;
        let p = Arc::new(p);
        let mut m = PluginManager::new();
        m.register_plugin(p.clone()).unwrap();
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
