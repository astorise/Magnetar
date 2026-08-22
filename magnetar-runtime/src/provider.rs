use crate::*;
use libloading::Library;
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
    path::{Path, PathBuf},
    sync::Arc,
};

pub const PROVIDER_API_VERSION: u32 = 1;
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
/// Trusted native extension contract for provider-owned execution capabilities.
pub trait Provider: Send + Sync {
    fn metadata(&self) -> ProviderMetadata;
    fn register(&self, registry: &mut ProviderRegistry) -> Result<(), ProviderError>;
    fn health(&self) -> ProviderHealth {
        ProviderHealth::Available
    }
    fn health_report(&self) -> ProviderHealthReport {
        ProviderHealthReport::new(ProviderBinding::new(self.metadata().name), self.health())
    }
    fn capability_health(&self, capability: &CapabilityBinding) -> Option<CapabilityHealth> {
        Some(CapabilityHealth::new(
            ProviderBinding::new(self.metadata().name),
            capability.clone(),
            self.health(),
        ))
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
    fn execution_api(&self) -> Option<&dyn ProviderExecutionApi> {
        None
    }
}

/// Native Runtime-to-Provider execution boundary for validated planned work.
pub trait ProviderExecutionApi: Send + Sync {
    fn submit(
        &self,
        request: ProviderExecutionRequest,
    ) -> Result<ProviderExecutionHandle, ProviderExecutionError>;
    fn status(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionStatus, ProviderExecutionError>;
    fn cancel(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderCancellationOutcome, ProviderExecutionError>;
    fn complete(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionResult, ProviderExecutionError>;
    fn release(&self, handle: ProviderExecutionHandle) -> Result<(), ProviderExecutionError>;
}

/// Receives provider contributions.
#[derive(Clone, Default)]
pub struct ProviderRegistry {
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
    pub(crate) fn candidates_for_capability(
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
        let capability_binding = CapabilityBinding::new(capability.id.clone(), capability.version);
        let capability_health = provider
            .capability_health(&capability_binding)
            .map(|health| health.state);
        ResolutionCandidate {
            provider: ProviderBinding::new(provider_name),
            capability: capability_binding,
            device: device.as_ref().map(|(binding, _)| binding.clone()),
            provider_health: provider.health(),
            capability_health,
            device_availability: device
                .map(|(_, availability)| availability)
                .unwrap_or(DeviceAvailability::Available),
            affinity_compatible,
            priority: 0,
        }
    }
    pub(crate) fn resolve_with_constraints<'a>(
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
                && !bound.version().is_compatible_with(requested.version)
            {
                return Err(AffinityError::CapabilityMismatch {
                    id: requested.id.clone(),
                    expected: requested.version,
                    found: bound.version(),
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
                    bound.version(),
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
            if !bound.version().is_compatible_with(requested.version) {
                return Err(AffinityError::CapabilityMismatch {
                    id: requested.id.clone(),
                    expected: requested.version,
                    found: bound.version(),
                });
            }
            self.registry.capability(&requested.id, bound.version())
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
#[derive(Debug)]
pub enum ProviderError {
    ProviderAlreadyRegistered(String),
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
