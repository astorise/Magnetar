//! Hardware-agnostic runtime contracts for magnetar.
//!
//! ```
//! use magnetar_runtime::{Runtime, RuntimeConfig};
//!
//! let mut runtime = Runtime::initialize(RuntimeConfig::default());
//! assert!(runtime.is_initialized());
//! runtime.shutdown();
//! ```

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

/// Identifies a device within a backend.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DeviceId(String);

impl DeviceId {
    /// Creates an identifier from an implementation-defined value.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Returns the identifier value.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DeviceId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Categorizes a device without coupling the runtime to a concrete backend.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DeviceType {
    Cpu,
    Gpu,
    Npu,
    Tpu,
    Other,
}

/// A compute device exposed by a backend.
pub trait Device: Send + Sync {
    fn id(&self) -> &DeviceId;
    fn device_type(&self) -> DeviceType;
}

/// A hardware implementation that can expose compute devices to the runtime.
pub trait Backend: Send + Sync {
    /// Stable backend identifier used for registration and selection.
    fn name(&self) -> &str;

    /// Devices available through this backend.
    fn devices(&self) -> Vec<Arc<dyn Device>>;
}

/// Runtime configuration that is independent of any backend implementation.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RuntimeConfig {
    /// Optional backend selected when the runtime is built.
    pub preferred_backend: Option<String>,
}

/// Immutable runtime information passed to future execution operations.
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

/// Creates a [`Runtime`] and registers its optional backends.
#[derive(Default)]
pub struct RuntimeBuilder {
    config: RuntimeConfig,
    backends: Vec<Arc<dyn Backend>>,
}

impl RuntimeBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn config(mut self, config: RuntimeConfig) -> Self {
        self.config = config;
        self
    }

    pub fn register_backend(mut self, backend: Arc<dyn Backend>) -> Self {
        self.backends.push(backend);
        self
    }

    /// Initializes the runtime. No registered backend is required.
    pub fn build(self) -> Result<Runtime, RuntimeError> {
        let mut runtime = Runtime {
            context: ExecutionContext {
                config: self.config,
                backend_name: None,
            },
            backends: BTreeMap::new(),
            initialized: true,
        };

        for backend in self.backends {
            runtime.register_backend(backend)?;
        }

        if let Some(name) = runtime.context.config.preferred_backend.clone() {
            runtime.select_backend(&name)?;
        }

        Ok(runtime)
    }
}

/// The entry point for backend-independent execution.
pub struct Runtime {
    context: ExecutionContext,
    backends: BTreeMap<String, Arc<dyn Backend>>,
    initialized: bool,
}

impl Runtime {
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::new()
    }

    /// Initializes a runtime without registering a backend.
    pub fn initialize(config: RuntimeConfig) -> Self {
        RuntimeBuilder::new()
            .config(config)
            .build()
            .expect("a backend-independent runtime configuration is always valid")
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub fn context(&self) -> &ExecutionContext {
        &self.context
    }

    pub fn register_backend(&mut self, backend: Arc<dyn Backend>) -> Result<(), RuntimeError> {
        let name = backend.name().to_owned();
        if self.backends.contains_key(&name) {
            return Err(RuntimeError::BackendAlreadyRegistered(name));
        }
        self.backends.insert(name, backend);
        Ok(())
    }

    pub fn select_backend(&mut self, name: &str) -> Result<(), RuntimeError> {
        if !self.backends.contains_key(name) {
            return Err(RuntimeError::BackendNotFound(name.to_owned()));
        }
        self.context.backend_name = Some(name.to_owned());
        Ok(())
    }

    pub fn selected_backend(&self) -> Option<&dyn Backend> {
        self.context
            .backend_name
            .as_ref()
            .and_then(|name| self.backends.get(name))
            .map(AsRef::as_ref)
    }

    /// Releases registered backends and marks this runtime as shut down.
    pub fn shutdown(&mut self) {
        self.backends.clear();
        self.context.backend_name = None;
        self.initialized = false;
    }
}

/// Errors caused by backend registration or selection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RuntimeError {
    BackendAlreadyRegistered(String),
    BackendNotFound(String),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendAlreadyRegistered(name) => {
                write!(formatter, "backend '{name}' is already registered")
            }
            Self::BackendNotFound(name) => write!(formatter, "backend '{name}' is not registered"),
        }
    }
}

impl Error for RuntimeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initializes_without_a_backend() {
        let runtime = Runtime::initialize(RuntimeConfig::default());
        assert!(runtime.is_initialized());
        assert!(runtime.selected_backend().is_none());
    }

    #[test]
    fn shutdown_releases_backends() {
        let mut runtime = Runtime::initialize(RuntimeConfig::default());
        runtime.shutdown();
        assert!(!runtime.is_initialized());
        assert!(runtime.selected_backend().is_none());
    }
}
