//! Provider loading.
//!
//! Loads Provider libraries from the filesystem under a loading policy
//! (built-in, dynamic library, test, development) and registers what they
//! export. Filesystem access lives here, not in the Provider contract the
//! Providers implement.

use crate::affinity::*;
use crate::capability::*;
use crate::provider_api::*;
use crate::provider_status::ProviderError;
use crate::resolution::{
    BuiltInResolutionPolicy, ResolutionCandidate, ResolutionContext, ResolutionDecision,
    ResolutionDecisionReason, ResolutionPolicy,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::Arc,
};

pub const PROVIDER_ABI_FACTORY_SYMBOL_V1: &str = "magnetar_provider_v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderLoadingMode {
    BuiltIn,
    DynamicLibrary,
    TestProvider,
    DevelopmentProvider,
}

impl ProviderLoadingMode {
    pub const fn is_dynamic(self) -> bool {
        matches!(self, Self::DynamicLibrary | Self::DevelopmentProvider)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderLoadingPolicy {
    pub mode: ProviderLoadingMode,
    pub allowed_paths: Vec<PathBuf>,
    pub development_mode: bool,
}

impl ProviderLoadingPolicy {
    pub fn dynamic_library(allowed_paths: impl IntoIterator<Item = impl Into<PathBuf>>) -> Self {
        Self {
            mode: ProviderLoadingMode::DynamicLibrary,
            allowed_paths: allowed_paths.into_iter().map(Into::into).collect(),
            development_mode: false,
        }
    }

    pub fn development(allowed_paths: impl IntoIterator<Item = impl Into<PathBuf>>) -> Self {
        Self {
            mode: ProviderLoadingMode::DevelopmentProvider,
            allowed_paths: allowed_paths.into_iter().map(Into::into).collect(),
            development_mode: true,
        }
    }

    pub fn allows(&self, path: &Path) -> bool {
        self.mode.is_dynamic()
            && !self.allowed_paths.is_empty()
            && self.allowed_paths.iter().any(|root| path.starts_with(root))
    }
}

impl Default for ProviderLoadingPolicy {
    fn default() -> Self {
        Self {
            mode: ProviderLoadingMode::DynamicLibrary,
            allowed_paths: Vec::new(),
            development_mode: false,
        }
    }
}

#[derive(Default)]
pub struct ProviderLoader {
    registry: ProviderRegistry,
    providers: BTreeMap<String, Arc<dyn Provider>>,
    order: Vec<String>,
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
    pub(crate) fn resolution_candidate(
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
        let provider_status = provider.status_snapshot();
        ResolutionCandidate {
            provider: ProviderBinding::new(provider_name),
            capability: capability_binding,
            device: device.as_ref().map(|(binding, _)| binding.clone()),
            provider_health: provider.health(),
            provider_status,
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
    /// Loads a dynamic provider library through the stable Provider ABI.
    ///
    /// # Safety
    ///
    /// Dynamic providers are trusted native code. Callers must only pass
    /// libraries allowed by Runtime policy.
    pub unsafe fn load_dynamic(&mut self, path: impl AsRef<Path>) -> Result<(), ProviderError> {
        unsafe { self.load_dynamic_with_policy(path, &ProviderLoadingPolicy::default()) }
    }

    /// Loads a dynamic provider library only when policy allows the path.
    ///
    /// The current Runtime intentionally rejects the legacy Rust trait-object
    /// factory shape. Full C-compatible descriptor loading is defined by policy
    /// but not implemented in this runtime revision.
    ///
    /// # Safety
    ///
    /// Dynamic providers are trusted native code. Callers must only pass
    /// libraries allowed by Runtime policy.
    pub unsafe fn load_dynamic_with_policy(
        &mut self,
        path: impl AsRef<Path>,
        policy: &ProviderLoadingPolicy,
    ) -> Result<(), ProviderError> {
        let path = path.as_ref();
        if !policy.allows(path) {
            return Err(ProviderError::ProviderPathDenied { path: path.into() });
        }
        Err(ProviderError::UnsupportedDynamicAbi {
            path: path.into(),
            expected_symbol: PROVIDER_ABI_FACTORY_SYMBOL_V1,
            expected_descriptor: ProviderAbiDescriptor::current(),
        })
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
        first.map_or(Ok(()), Err)
    }
}

#[cfg(test)]
mod tests;
