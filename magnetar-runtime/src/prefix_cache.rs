//! Runtime-owned Prefix Cache contracts.
//!
//! Prefix Cache is the index and policy layer above sealed KV cache state. It
//! owns opaque entry identity, token-derived fingerprints, compatibility
//! checks, sharing/privacy policy, lifecycle, memory accounting, resource
//! affinity, eviction, invalidation, and redacted observations. It never stores
//! raw prompt text or exposes raw token sequences or raw KV cache contents by
//! default.

use crate::{
    DeviceBinding, FallbackClass, GenerationModelReference, InferenceSessionId, KvCache, KvCacheId,
    KvCacheLifecycleState, MemoryAllocationClass, MemoryAllocationId, MemoryAllocationLifetime,
    MemoryAllocationOwner, MemoryAllocationRequest, MemoryDTypeRelation, MemoryManager,
    MemoryPlacement, PrefixFingerprint, ProviderBinding, ResourceAffinity, TokenId, TokenizerId,
};
use std::{collections::BTreeMap, error::Error, fmt};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PrefixCacheEntryId(String);

impl PrefixCacheEntryId {
    pub fn new(value: impl Into<String>) -> Result<Self, PrefixCacheError> {
        let value = value.into();
        validate_prefix_cache_identity(&value)?;
        Ok(Self(value))
    }

    pub(crate) fn runtime_issued(sequence: u64) -> Self {
        Self(format!("prefix-cache-{sequence}"))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PrefixCacheEntryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixCacheFingerprint {
    value: PrefixFingerprint,
    token_length: u32,
}

impl PrefixCacheFingerprint {
    pub fn from_validated_tokens(
        token_ids: &[TokenId],
        model_key: &str,
        tokenizer: &TokenizerId,
    ) -> Self {
        Self {
            value: PrefixFingerprint::from_tokens(token_ids, model_key, tokenizer),
            token_length: token_ids.len() as u32,
        }
    }

    pub fn value(&self) -> &PrefixFingerprint {
        &self.value
    }

    pub const fn token_length(&self) -> u32 {
        self.token_length
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PrefixCacheLifecycleState {
    Creating,
    Ready,
    Sealed,
    Active,
    Stale,
    Invalid,
    Evicting,
    Evicted,
    Released,
    Failed,
}

impl PrefixCacheLifecycleState {
    pub const fn reusable(self) -> bool {
        matches!(self, Self::Ready | Self::Sealed | Self::Active)
    }

    pub const fn allows_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Creating, Self::Ready)
                | (Self::Creating, Self::Failed)
                | (Self::Ready, Self::Sealed)
                | (Self::Ready, Self::Active)
                | (Self::Ready, Self::Stale)
                | (Self::Ready, Self::Invalid)
                | (Self::Ready, Self::Evicting)
                | (Self::Ready, Self::Released)
                | (Self::Sealed, Self::Active)
                | (Self::Sealed, Self::Stale)
                | (Self::Sealed, Self::Invalid)
                | (Self::Sealed, Self::Evicting)
                | (Self::Sealed, Self::Released)
                | (Self::Active, Self::Sealed)
                | (Self::Active, Self::Stale)
                | (Self::Active, Self::Invalid)
                | (Self::Active, Self::Released)
                | (Self::Stale, Self::Ready)
                | (Self::Stale, Self::Invalid)
                | (Self::Stale, Self::Evicting)
                | (Self::Invalid, Self::Released)
                | (Self::Evicting, Self::Evicted)
                | (Self::Evicting, Self::Released)
                | (Self::Evicted, Self::Released)
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PrefixCacheScope {
    Operation,
    Session,
    ModelInstance,
    Runtime,
    Tenant,
    Private,
    Shared,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefixCacheSharingPolicy {
    PrivateOnly,
    SessionLocal,
    Tenant,
    Runtime,
    SharedAuthorized,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixCachePrivacyPolicy {
    pub raw_prompt_storage_allowed: bool,
    pub raw_prompt_logging_allowed: bool,
    pub raw_token_export_allowed: bool,
    pub raw_kv_cache_export_allowed: bool,
    pub redacted_diagnostics: bool,
}

impl Default for PrefixCachePrivacyPolicy {
    fn default() -> Self {
        Self {
            raw_prompt_storage_allowed: false,
            raw_prompt_logging_allowed: false,
            raw_token_export_allowed: false,
            raw_kv_cache_export_allowed: false,
            redacted_diagnostics: true,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixCachePolicy {
    pub enabled: bool,
    pub allow_partial_reuse: bool,
    pub require_sealed_kv_cache_for_sharing: bool,
    pub sharing: PrefixCacheSharingPolicy,
    pub privacy: PrefixCachePrivacyPolicy,
    pub max_memory_bytes: Option<u64>,
    pub max_prefix_tokens: Option<u32>,
    pub ttl_millis: Option<u64>,
    pub idle_ttl_millis: Option<u64>,
    pub persist_after_session_close: bool,
}

impl Default for PrefixCachePolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            allow_partial_reuse: false,
            require_sealed_kv_cache_for_sharing: true,
            sharing: PrefixCacheSharingPolicy::PrivateOnly,
            privacy: PrefixCachePrivacyPolicy::default(),
            max_memory_bytes: None,
            max_prefix_tokens: None,
            ttl_millis: None,
            idle_ttl_millis: None,
            persist_after_session_close: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixCacheCompatibility {
    pub model: GenerationModelReference,
    pub model_revision: Option<String>,
    pub tokenizer: TokenizerId,
    pub tokenizer_revision: Option<String>,
    pub template: Option<String>,
    pub position_encoding: Option<String>,
    pub attention_implementation: Option<String>,
}

impl PrefixCacheCompatibility {
    pub fn new(model: GenerationModelReference, tokenizer: TokenizerId) -> Self {
        Self {
            model,
            model_revision: None,
            tokenizer,
            tokenizer_revision: None,
            template: None,
            position_encoding: None,
            attention_implementation: None,
        }
    }

    pub fn validate_reuse(&self, requested: &Self) -> Result<(), PrefixCacheError> {
        if self.model != requested.model || self.model_revision != requested.model_revision {
            return Err(PrefixCacheError::PrefixModelMismatch);
        }
        if self.tokenizer != requested.tokenizer
            || self.tokenizer_revision != requested.tokenizer_revision
        {
            return Err(PrefixCacheError::PrefixTokenizerMismatch);
        }
        if self.template != requested.template {
            return Err(PrefixCacheError::PrefixTemplateMismatch);
        }
        if self.position_encoding != requested.position_encoding {
            return Err(PrefixCacheError::PrefixPositionMismatch);
        }
        if self.attention_implementation != requested.attention_implementation {
            return Err(PrefixCacheError::PrefixEntryIncompatible);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixCacheBackingKvCache {
    pub cache: KvCacheId,
    pub lifecycle: KvCacheLifecycleState,
    pub affinity: ResourceAffinity,
    pub provider: Option<ProviderBinding>,
    pub device: Option<DeviceBinding>,
    pub memory_allocation: Option<MemoryAllocationId>,
}

impl PrefixCacheBackingKvCache {
    pub fn from_kv_cache(cache: &KvCache) -> Self {
        Self {
            cache: cache.id.clone(),
            lifecycle: cache.lifecycle,
            affinity: cache.residency.affinity.clone(),
            provider: cache.residency.provider.clone(),
            device: cache.residency.device.clone(),
            memory_allocation: cache.residency.memory_allocation,
        }
    }

    pub fn validate_reusable(&self) -> Result<(), PrefixCacheError> {
        match self.lifecycle {
            KvCacheLifecycleState::Sealed => Ok(()),
            KvCacheLifecycleState::Evicted => Err(PrefixCacheError::PrefixEvicted),
            KvCacheLifecycleState::Invalid => Err(PrefixCacheError::PrefixBackingCacheInvalid),
            KvCacheLifecycleState::Released => Err(PrefixCacheError::PrefixBackingCacheMissing),
            _ => Err(PrefixCacheError::PrefixEntryIncompatible),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixCacheEntry {
    pub id: PrefixCacheEntryId,
    pub lifecycle: PrefixCacheLifecycleState,
    pub fingerprint: PrefixCacheFingerprint,
    pub prefix_token_length: u32,
    pub compatibility: PrefixCacheCompatibility,
    pub session: Option<InferenceSessionId>,
    pub owner: Option<String>,
    pub tenant: Option<String>,
    pub scope: PrefixCacheScope,
    pub sharing: PrefixCacheSharingPolicy,
    pub privacy: PrefixCachePrivacyPolicy,
    pub backing_kv_cache: PrefixCacheBackingKvCache,
    pub memory_estimate_bytes: u64,
    pub memory_allocation: Option<MemoryAllocationId>,
    pub position_start: u64,
    pub position_end_exclusive: u64,
    pub created_at_millis: u64,
    pub last_used_at_millis: u64,
    pub hit_count: u64,
    pub eviction_priority: u8,
}

impl PrefixCacheEntry {
    pub fn new(
        id: PrefixCacheEntryId,
        fingerprint: PrefixCacheFingerprint,
        compatibility: PrefixCacheCompatibility,
        backing_kv_cache: PrefixCacheBackingKvCache,
    ) -> Self {
        Self {
            id,
            lifecycle: PrefixCacheLifecycleState::Creating,
            prefix_token_length: fingerprint.token_length(),
            fingerprint,
            compatibility,
            session: None,
            owner: None,
            tenant: None,
            scope: PrefixCacheScope::Private,
            sharing: PrefixCacheSharingPolicy::PrivateOnly,
            privacy: PrefixCachePrivacyPolicy::default(),
            backing_kv_cache,
            memory_estimate_bytes: 256,
            memory_allocation: None,
            position_start: 0,
            position_end_exclusive: 0,
            created_at_millis: 0,
            last_used_at_millis: 0,
            hit_count: 0,
            eviction_priority: 0,
        }
    }

    pub fn with_session(mut self, session: InferenceSessionId) -> Self {
        self.session = Some(session);
        self.scope = PrefixCacheScope::Session;
        self.sharing = PrefixCacheSharingPolicy::SessionLocal;
        self
    }

    pub fn with_policy(mut self, policy: &PrefixCachePolicy) -> Self {
        self.sharing = policy.sharing;
        self.privacy = policy.privacy.clone();
        self
    }

    pub fn transition_to(
        &mut self,
        next: PrefixCacheLifecycleState,
    ) -> Result<(), PrefixCacheError> {
        if self.lifecycle.allows_transition_to(next) {
            self.lifecycle = next;
            Ok(())
        } else {
            Err(PrefixCacheError::PrefixInvalid)
        }
    }

    pub fn allocation_request(&self) -> MemoryAllocationRequest {
        let mut request = MemoryAllocationRequest::new(
            MemoryAllocationClass::PrefixCache,
            self.memory_estimate_bytes,
            MemoryPlacement::HostOrdinary,
            self.session
                .as_ref()
                .map(|session| MemoryAllocationOwner::Session(session.as_str().into()))
                .unwrap_or(MemoryAllocationOwner::Runtime),
        )
        .with_affinity(ResourceAffinity::new(FallbackClass::Transparent))
        .with_dtype_relation(MemoryDTypeRelation::new(
            crate::DTypeDescriptor::portable(crate::ComputeDType::UInt8),
            crate::DTypeDescriptor::portable(crate::ComputeDType::UInt8),
        ));
        request.lifetime = match self.scope {
            PrefixCacheScope::Operation => MemoryAllocationLifetime::Operation,
            PrefixCacheScope::Session => MemoryAllocationLifetime::Session,
            PrefixCacheScope::ModelInstance
            | PrefixCacheScope::Runtime
            | PrefixCacheScope::Tenant
            | PrefixCacheScope::Private
            | PrefixCacheScope::Shared => MemoryAllocationLifetime::Runtime,
        };
        request
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixCacheLookupRequest {
    pub fingerprint: PrefixCacheFingerprint,
    pub compatibility: PrefixCacheCompatibility,
    pub requested_prefix_token_length: u32,
    pub session: Option<InferenceSessionId>,
    pub owner: Option<String>,
    pub tenant: Option<String>,
    pub affinity: Option<ResourceAffinity>,
    pub allow_partial: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefixCacheMatchKind {
    Miss,
    ExactPrefixHit,
    PartialPrefixHit,
    IncompatibleHit,
    PolicyDeniedHit,
    StaleHit,
    EvictedHit,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixCacheLookupResult {
    pub kind: PrefixCacheMatchKind,
    pub entry: Option<PrefixCacheEntryId>,
    pub reusable_prefix_token_length: u32,
    pub error: Option<PrefixCacheError>,
}

impl PrefixCacheLookupResult {
    pub fn miss() -> Self {
        Self {
            kind: PrefixCacheMatchKind::Miss,
            entry: None,
            reusable_prefix_token_length: 0,
            error: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefixCacheObservationKind {
    Lookup,
    Hit,
    Miss,
    PartialHit,
    PolicyDeniedHit,
    IncompatibleHit,
    EntryCreated,
    EntrySealed,
    EntryReused,
    EntryInvalidated,
    EntryEvicted,
    BackingKvCacheMissing,
    SharingDenied,
    PrivacyRedaction,
    MemoryPressureEviction,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixCacheObservation {
    pub kind: PrefixCacheObservationKind,
    pub entry: Option<PrefixCacheEntryId>,
    pub message: String,
    pub prefix_token_length: Option<u32>,
    pub raw_prompt_available: bool,
    pub raw_token_sequence_available: bool,
    pub raw_kv_cache_available: bool,
}

impl PrefixCacheObservation {
    pub fn redacted(
        kind: PrefixCacheObservationKind,
        entry: Option<PrefixCacheEntryId>,
        message: impl Into<String>,
        prefix_token_length: Option<u32>,
    ) -> Self {
        Self {
            kind,
            entry,
            message: message.into(),
            prefix_token_length,
            raw_prompt_available: false,
            raw_token_sequence_available: false,
            raw_kv_cache_available: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixCacheManager {
    next_id: u64,
    entries: BTreeMap<PrefixCacheEntryId, PrefixCacheEntry>,
    observations: Vec<PrefixCacheObservation>,
}

impl Default for PrefixCacheManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PrefixCacheManager {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            entries: BTreeMap::new(),
            observations: Vec::new(),
        }
    }

    pub fn entries(&self) -> impl Iterator<Item = &PrefixCacheEntry> {
        self.entries.values()
    }

    pub fn observations(&self) -> &[PrefixCacheObservation] {
        &self.observations
    }

    pub fn entry(&self, id: &PrefixCacheEntryId) -> Result<&PrefixCacheEntry, PrefixCacheError> {
        self.entries
            .get(id)
            .ok_or(PrefixCacheError::PrefixEntryNotFound)
    }

    pub fn entry_mut(
        &mut self,
        id: &PrefixCacheEntryId,
    ) -> Result<&mut PrefixCacheEntry, PrefixCacheError> {
        self.entries
            .get_mut(id)
            .ok_or(PrefixCacheError::PrefixEntryNotFound)
    }

    pub fn create(
        &mut self,
        mut entry: PrefixCacheEntry,
        policy: &PrefixCachePolicy,
    ) -> Result<PrefixCacheEntryId, PrefixCacheError> {
        if !policy.enabled {
            return Err(PrefixCacheError::PrefixCacheDisabled);
        }
        if let Some(max_tokens) = policy.max_prefix_tokens
            && entry.prefix_token_length > max_tokens
        {
            return Err(PrefixCacheError::PrefixPolicyDenied);
        }
        if let Some(max_bytes) = policy.max_memory_bytes
            && entry.memory_estimate_bytes > max_bytes
        {
            return Err(PrefixCacheError::PrefixMemoryPressure);
        }
        if policy.require_sealed_kv_cache_for_sharing
            && !matches!(
                entry.backing_kv_cache.lifecycle,
                KvCacheLifecycleState::Sealed
            )
            && !matches!(entry.sharing, PrefixCacheSharingPolicy::PrivateOnly)
        {
            return Err(PrefixCacheError::PrefixEntryIncompatible);
        }
        let id = PrefixCacheEntryId::runtime_issued(self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        entry.id = id.clone();
        entry.lifecycle = PrefixCacheLifecycleState::Ready;
        self.entries.insert(id.clone(), entry);
        self.observe(
            PrefixCacheObservationKind::EntryCreated,
            Some(id.clone()),
            "prefix cache entry created",
            None,
        );
        Ok(id)
    }

    pub fn allocate_memory(
        &mut self,
        id: &PrefixCacheEntryId,
        memory: &mut MemoryManager,
    ) -> Result<MemoryAllocationId, PrefixCacheError> {
        let request = self.entry(id)?.allocation_request();
        let allocation = memory
            .allocate(request)
            .map_err(|_| PrefixCacheError::PrefixAllocationFailed)?;
        self.entry_mut(id)?.memory_allocation = Some(allocation.id);
        Ok(allocation.id)
    }

    pub fn lookup(&mut self, request: &PrefixCacheLookupRequest) -> PrefixCacheLookupResult {
        self.observe(
            PrefixCacheObservationKind::Lookup,
            None,
            "prefix cache lookup",
            Some(request.requested_prefix_token_length),
        );
        let mut partial_candidate = None;
        for (id, entry) in &self.entries {
            if entry.fingerprint != request.fingerprint {
                if request.allow_partial
                    && entry.prefix_token_length < request.requested_prefix_token_length
                    && entry.compatibility == request.compatibility
                {
                    partial_candidate = Some(id.clone());
                }
                continue;
            }
            let result = self.validate_entry_for_request(id, request);
            self.observe_lookup_result(&result);
            return result;
        }
        if let Some(id) = partial_candidate {
            let result = self.validate_entry_for_request(&id, request);
            let result = if result.kind == PrefixCacheMatchKind::ExactPrefixHit {
                PrefixCacheLookupResult {
                    kind: PrefixCacheMatchKind::PartialPrefixHit,
                    ..result
                }
            } else {
                result
            };
            self.observe_lookup_result(&result);
            return result;
        }
        let result = PrefixCacheLookupResult::miss();
        self.observe_lookup_result(&result);
        result
    }

    pub fn validate_reuse(
        &mut self,
        id: &PrefixCacheEntryId,
        request: &PrefixCacheLookupRequest,
    ) -> Result<(), PrefixCacheError> {
        let result = self.validate_entry_for_request(id, request);
        self.observe_lookup_result(&result);
        match result.error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    pub fn mark_backing_kv_cache_state(
        &mut self,
        cache: &KvCacheId,
        lifecycle: KvCacheLifecycleState,
    ) -> Vec<PrefixCacheEntryId> {
        let ids = self
            .entries
            .values()
            .filter(|entry| &entry.backing_kv_cache.cache == cache)
            .map(|entry| entry.id.clone())
            .collect::<Vec<_>>();
        for id in &ids {
            if let Some(entry) = self.entries.get_mut(id) {
                entry.backing_kv_cache.lifecycle = lifecycle;
                entry.lifecycle = match lifecycle {
                    KvCacheLifecycleState::Evicted => PrefixCacheLifecycleState::Evicted,
                    KvCacheLifecycleState::Invalid => PrefixCacheLifecycleState::Invalid,
                    KvCacheLifecycleState::Released => PrefixCacheLifecycleState::Released,
                    _ => PrefixCacheLifecycleState::Stale,
                };
            }
            self.observe(
                match lifecycle {
                    KvCacheLifecycleState::Evicted => PrefixCacheObservationKind::EntryEvicted,
                    KvCacheLifecycleState::Released => {
                        PrefixCacheObservationKind::BackingKvCacheMissing
                    }
                    _ => PrefixCacheObservationKind::EntryInvalidated,
                },
                Some(id.clone()),
                "prefix cache backing kv cache changed",
                None,
            );
        }
        ids
    }

    pub fn release_session_entries(
        &mut self,
        session: &InferenceSessionId,
        persist_after_session_close: bool,
    ) -> Vec<PrefixCacheEntryId> {
        let ids = self
            .entries
            .values()
            .filter(|entry| entry.session.as_ref() == Some(session))
            .map(|entry| entry.id.clone())
            .collect::<Vec<_>>();
        for id in &ids {
            if let Some(entry) = self.entries.get_mut(id)
                && !persist_after_session_close
            {
                entry.lifecycle = PrefixCacheLifecycleState::Released;
            }
        }
        ids
    }

    pub fn evict(&mut self, id: &PrefixCacheEntryId) -> Result<(), PrefixCacheError> {
        self.entry_mut(id)?
            .transition_to(PrefixCacheLifecycleState::Evicting)?;
        self.observe(
            PrefixCacheObservationKind::EntryEvicted,
            Some(id.clone()),
            "prefix cache entry evicting",
            None,
        );
        self.entry_mut(id)?
            .transition_to(PrefixCacheLifecycleState::Evicted)?;
        Ok(())
    }

    pub fn invalidate(&mut self, id: &PrefixCacheEntryId) -> Result<(), PrefixCacheError> {
        self.entry_mut(id)?.lifecycle = PrefixCacheLifecycleState::Invalid;
        self.observe(
            PrefixCacheObservationKind::EntryInvalidated,
            Some(id.clone()),
            "prefix cache entry invalidated",
            None,
        );
        Ok(())
    }

    pub fn release(&mut self, id: &PrefixCacheEntryId) -> Result<(), PrefixCacheError> {
        self.entry_mut(id)?.lifecycle = PrefixCacheLifecycleState::Released;
        Ok(())
    }

    fn validate_entry_for_request(
        &self,
        id: &PrefixCacheEntryId,
        request: &PrefixCacheLookupRequest,
    ) -> PrefixCacheLookupResult {
        let Ok(entry) = self.entry(id) else {
            return PrefixCacheLookupResult {
                kind: PrefixCacheMatchKind::Miss,
                entry: Some(id.clone()),
                reusable_prefix_token_length: 0,
                error: Some(PrefixCacheError::PrefixEntryNotFound),
            };
        };
        if entry.lifecycle == PrefixCacheLifecycleState::Evicted {
            return non_reusable(
                id,
                PrefixCacheMatchKind::EvictedHit,
                PrefixCacheError::PrefixEvicted,
            );
        }
        if entry.lifecycle == PrefixCacheLifecycleState::Stale {
            return non_reusable(
                id,
                PrefixCacheMatchKind::StaleHit,
                PrefixCacheError::PrefixStale,
            );
        }
        if !entry.lifecycle.reusable() {
            return non_reusable(
                id,
                PrefixCacheMatchKind::IncompatibleHit,
                PrefixCacheError::PrefixInvalid,
            );
        }
        if let Err(error) = entry.compatibility.validate_reuse(&request.compatibility) {
            return non_reusable(id, PrefixCacheMatchKind::IncompatibleHit, error);
        }
        if let Err(error) = entry.backing_kv_cache.validate_reusable() {
            return non_reusable(id, PrefixCacheMatchKind::IncompatibleHit, error);
        }
        if let Some(affinity) = &request.affinity
            && entry
                .backing_kv_cache
                .affinity
                .validate_with(affinity)
                .is_err()
        {
            return non_reusable(
                id,
                PrefixCacheMatchKind::IncompatibleHit,
                PrefixCacheError::PrefixResourceAffinityConflict,
            );
        }
        if !sharing_allowed(entry, request) {
            return non_reusable(
                id,
                PrefixCacheMatchKind::PolicyDeniedHit,
                PrefixCacheError::PrefixSharingDenied,
            );
        }
        let kind = if entry.prefix_token_length == request.requested_prefix_token_length {
            PrefixCacheMatchKind::ExactPrefixHit
        } else if request.allow_partial
            && entry.prefix_token_length < request.requested_prefix_token_length
        {
            PrefixCacheMatchKind::PartialPrefixHit
        } else {
            PrefixCacheMatchKind::IncompatibleHit
        };
        let error = (kind == PrefixCacheMatchKind::IncompatibleHit)
            .then_some(PrefixCacheError::PrefixPositionMismatch);
        PrefixCacheLookupResult {
            kind,
            entry: Some(id.clone()),
            reusable_prefix_token_length: if error.is_none() {
                entry.prefix_token_length
            } else {
                0
            },
            error,
        }
    }

    fn observe_lookup_result(&mut self, result: &PrefixCacheLookupResult) {
        let kind = match result.kind {
            PrefixCacheMatchKind::Miss => PrefixCacheObservationKind::Miss,
            PrefixCacheMatchKind::ExactPrefixHit => PrefixCacheObservationKind::Hit,
            PrefixCacheMatchKind::PartialPrefixHit => PrefixCacheObservationKind::PartialHit,
            PrefixCacheMatchKind::IncompatibleHit => PrefixCacheObservationKind::IncompatibleHit,
            PrefixCacheMatchKind::PolicyDeniedHit => PrefixCacheObservationKind::PolicyDeniedHit,
            PrefixCacheMatchKind::StaleHit => PrefixCacheObservationKind::IncompatibleHit,
            PrefixCacheMatchKind::EvictedHit => PrefixCacheObservationKind::EntryEvicted,
        };
        self.observe(
            kind,
            result.entry.clone(),
            "prefix cache lookup result",
            Some(result.reusable_prefix_token_length),
        );
    }

    fn observe(
        &mut self,
        kind: PrefixCacheObservationKind,
        entry: Option<PrefixCacheEntryId>,
        message: impl Into<String>,
        prefix_token_length: Option<u32>,
    ) {
        self.observations.push(PrefixCacheObservation::redacted(
            kind,
            entry,
            message,
            prefix_token_length,
        ));
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PrefixCacheError {
    PrefixCacheDisabled,
    PrefixCacheUnavailable,
    PrefixEntryNotFound,
    PrefixEntryIncompatible,
    PrefixFingerprintMismatch,
    PrefixModelMismatch,
    PrefixTokenizerMismatch,
    PrefixTemplateMismatch,
    PrefixPositionMismatch,
    PrefixPolicyDenied,
    PrefixSharingDenied,
    PrefixPrivacyDenied,
    PrefixStale,
    PrefixInvalid,
    PrefixEvicted,
    PrefixBackingCacheMissing,
    PrefixBackingCacheInvalid,
    PrefixResourceAffinityConflict,
    PrefixMovementRequired,
    PrefixMovementUnsupported,
    PrefixMemoryPressure,
    PrefixAllocationFailed,
    PrefixBrowserFeatureUnsupported,
    PrefixInternal,
}

impl fmt::Display for PrefixCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::PrefixCacheDisabled => "prefix cache disabled",
            Self::PrefixCacheUnavailable => "prefix cache unavailable",
            Self::PrefixEntryNotFound => "prefix entry not found",
            Self::PrefixEntryIncompatible => "prefix entry incompatible",
            Self::PrefixFingerprintMismatch => "prefix fingerprint mismatch",
            Self::PrefixModelMismatch => "prefix model mismatch",
            Self::PrefixTokenizerMismatch => "prefix tokenizer mismatch",
            Self::PrefixTemplateMismatch => "prefix template mismatch",
            Self::PrefixPositionMismatch => "prefix position mismatch",
            Self::PrefixPolicyDenied => "prefix policy denied",
            Self::PrefixSharingDenied => "prefix sharing denied",
            Self::PrefixPrivacyDenied => "prefix privacy denied",
            Self::PrefixStale => "prefix stale",
            Self::PrefixInvalid => "prefix invalid",
            Self::PrefixEvicted => "prefix evicted",
            Self::PrefixBackingCacheMissing => "prefix backing cache missing",
            Self::PrefixBackingCacheInvalid => "prefix backing cache invalid",
            Self::PrefixResourceAffinityConflict => "prefix resource affinity conflict",
            Self::PrefixMovementRequired => "prefix movement required",
            Self::PrefixMovementUnsupported => "prefix movement unsupported",
            Self::PrefixMemoryPressure => "prefix memory pressure",
            Self::PrefixAllocationFailed => "prefix allocation failed",
            Self::PrefixBrowserFeatureUnsupported => "prefix browser feature unsupported",
            Self::PrefixInternal => "prefix internal error",
        })
    }
}

impl Error for PrefixCacheError {}

fn non_reusable(
    id: &PrefixCacheEntryId,
    kind: PrefixCacheMatchKind,
    error: PrefixCacheError,
) -> PrefixCacheLookupResult {
    PrefixCacheLookupResult {
        kind,
        entry: Some(id.clone()),
        reusable_prefix_token_length: 0,
        error: Some(error),
    }
}

fn sharing_allowed(entry: &PrefixCacheEntry, request: &PrefixCacheLookupRequest) -> bool {
    match entry.sharing {
        PrefixCacheSharingPolicy::PrivateOnly => entry.owner == request.owner,
        PrefixCacheSharingPolicy::SessionLocal => entry.session == request.session,
        PrefixCacheSharingPolicy::Tenant => {
            entry.tenant.is_some() && entry.tenant == request.tenant
        }
        PrefixCacheSharingPolicy::Runtime => true,
        PrefixCacheSharingPolicy::SharedAuthorized => {
            entry.owner == request.owner || entry.tenant == request.tenant
        }
    }
}

fn validate_prefix_cache_identity(value: &str) -> Result<(), PrefixCacheError> {
    if value.trim().is_empty()
        || value.contains('/')
        || value.contains('\\')
        || value.contains(':')
        || value.contains("provider")
        || value.contains("device")
        || value.contains("0x")
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return Err(PrefixCacheError::PrefixPolicyDenied);
    }
    Ok(())
}
