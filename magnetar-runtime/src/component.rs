use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
    path::{Path, PathBuf},
};
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
