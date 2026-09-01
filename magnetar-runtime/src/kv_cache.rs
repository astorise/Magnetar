//! Runtime-owned KV cache contracts.
//!
//! A KV cache is mutable inference state produced during prefill and decode.
//! It is distinct from sessions, model artifacts, provider handles, scheduler
//! state, and prefix-cache indexes. Runtime owns identity, lifecycle,
//! compatibility, memory accounting, policy, affinity, and redacted
//! observability for this state.

use crate::model_instance::ModelInstanceId;
use crate::{
    ComputeDType, DTypeDescriptor, DeviceBinding, FallbackClass, GenerationModelReference,
    InferenceSessionId, MemoryAllocationClass, MemoryAllocationId, MemoryAllocationLifetime,
    MemoryAllocationOwner, MemoryAllocationRequest, MemoryDTypeRelation, MemoryManager,
    MemoryPlacement, ProviderBinding, ResourceAffinity, TensorResourceId, TokenId, TokenizerId,
};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, error::Error, fmt};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KvCacheId(String);

impl KvCacheId {
    pub fn new(value: impl Into<String>) -> Result<Self, KvCacheError> {
        let value = value.into();
        validate_kv_cache_identity(&value)?;
        Ok(Self(value))
    }

    pub(crate) fn runtime_issued(sequence: u64) -> Self {
        Self(format!("kv-cache-{sequence}"))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for KvCacheId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KvCacheScope {
    Operation,
    Session,
    ModelInstance,
    PrefixCache,
    BatchSlot,
    RuntimeCache,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KvCacheLifecycleState {
    Allocating,
    Empty,
    Prefilling,
    Ready,
    Active,
    Sealed,
    Evicting,
    Evicted,
    Invalid,
    Released,
    Failed,
}

impl KvCacheLifecycleState {
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Evicted | Self::Invalid | Self::Released | Self::Failed
        )
    }

    pub const fn allows_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Allocating, Self::Empty)
                | (Self::Allocating, Self::Prefilling)
                | (Self::Allocating, Self::Failed)
                | (Self::Empty, Self::Prefilling)
                | (Self::Empty, Self::Evicting)
                | (Self::Empty, Self::Released)
                | (Self::Prefilling, Self::Ready)
                | (Self::Prefilling, Self::Invalid)
                | (Self::Prefilling, Self::Released)
                | (Self::Prefilling, Self::Failed)
                | (Self::Ready, Self::Active)
                | (Self::Ready, Self::Sealed)
                | (Self::Ready, Self::Evicting)
                | (Self::Ready, Self::Invalid)
                | (Self::Ready, Self::Released)
                | (Self::Active, Self::Ready)
                | (Self::Active, Self::Sealed)
                | (Self::Active, Self::Invalid)
                | (Self::Active, Self::Released)
                | (Self::Sealed, Self::Evicting)
                | (Self::Sealed, Self::Invalid)
                | (Self::Sealed, Self::Released)
                | (Self::Evicting, Self::Evicted)
                | (Self::Evicting, Self::Released)
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KvCacheLayoutFormat {
    Contiguous,
    Paged,
    BlockBased,
    ProviderOpaque,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvCachePageMetadata {
    pub page_size_tokens: u32,
    pub page_count: u32,
    pub occupied_pages: u32,
    pub reusable_free_pages: u32,
    pub prefix_shared_pages: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvCacheQuantization {
    pub mode: String,
    pub group_size: Option<u32>,
    pub scale_dtype: Option<DTypeDescriptor>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvCacheLayoutMetadata {
    pub layer_count: u32,
    pub head_count: u32,
    pub key_head_count: u32,
    pub value_head_count: u32,
    pub head_dimension: u32,
    pub token_capacity: u32,
    pub current_token_length: u32,
    pub batch_dimension: u32,
    pub sequence_dimension: u32,
    pub block_size_tokens: Option<u32>,
    pub page: Option<KvCachePageMetadata>,
    pub storage_dtype: DTypeDescriptor,
    pub compute_dtype: DTypeDescriptor,
    pub layout: KvCacheLayoutFormat,
    pub position_start: u64,
    pub position_end_exclusive: u64,
    pub quantization: Option<KvCacheQuantization>,
}

impl KvCacheLayoutMetadata {
    pub fn contiguous(
        layer_count: u32,
        head_count: u32,
        head_dimension: u32,
        token_capacity: u32,
        dtype: ComputeDType,
    ) -> Self {
        Self {
            layer_count,
            head_count,
            key_head_count: head_count,
            value_head_count: head_count,
            head_dimension,
            token_capacity,
            current_token_length: 0,
            batch_dimension: 1,
            sequence_dimension: token_capacity,
            block_size_tokens: None,
            page: None,
            storage_dtype: DTypeDescriptor::portable(dtype),
            compute_dtype: DTypeDescriptor::portable(dtype),
            layout: KvCacheLayoutFormat::Contiguous,
            position_start: 0,
            position_end_exclusive: 0,
            quantization: None,
        }
    }

    pub fn with_paged_metadata(mut self, page: KvCachePageMetadata) -> Self {
        self.layout = KvCacheLayoutFormat::Paged;
        self.page = Some(page);
        self
    }

    pub fn with_storage_dtype(mut self, storage_dtype: DTypeDescriptor) -> Self {
        self.storage_dtype = storage_dtype;
        self
    }

    pub fn with_compute_dtype(mut self, compute_dtype: DTypeDescriptor) -> Self {
        self.compute_dtype = compute_dtype;
        self
    }

    pub fn with_quantization(mut self, quantization: KvCacheQuantization) -> Self {
        self.quantization = Some(quantization);
        self
    }

    pub fn append_tokens(&mut self, count: u32) -> Result<(), KvCacheError> {
        let next = self
            .current_token_length
            .checked_add(count)
            .ok_or(KvCacheError::CacheCapacityExceeded)?;
        if next > self.token_capacity {
            return Err(KvCacheError::CacheCapacityExceeded);
        }
        self.current_token_length = next;
        self.position_end_exclusive = self.position_start.saturating_add(next as u64);
        Ok(())
    }

    pub fn estimated_storage_bytes(&self) -> Result<u64, KvCacheError> {
        let elements = [
            self.layer_count as u64,
            self.key_head_count.saturating_add(self.value_head_count) as u64,
            self.head_dimension as u64,
            self.token_capacity as u64,
            self.batch_dimension as u64,
        ]
        .into_iter()
        .try_fold(1u64, |total, value| {
            total.checked_mul(value).ok_or(KvCacheError::CacheInternal)
        })?;
        elements
            .checked_mul(self.storage_dtype.size_bytes())
            .ok_or(KvCacheError::CacheInternal)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixFingerprint(String);

impl PrefixFingerprint {
    pub fn from_tokens(token_ids: &[TokenId], model_key: &str, tokenizer: &TokenizerId) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"magnetar-kv-prefix-v1");
        hasher.update(model_key.as_bytes());
        hasher.update(tokenizer.as_str().as_bytes());
        for token_id in token_ids {
            hasher.update(token_id.to_le_bytes());
        }
        let digest = hasher.finalize();
        let hex = digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        Self(format!("sha256:{hex}"))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PrefixFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvCacheCompatibility {
    pub model: GenerationModelReference,
    pub model_architecture: Option<String>,
    pub model_revision: Option<String>,
    pub tokenizer: TokenizerId,
    pub tokenizer_vocabulary_size: u32,
    pub prefix_fingerprint: Option<PrefixFingerprint>,
    pub position_encoding: Option<String>,
    pub attention_implementation: Option<String>,
    pub quantization_mode: Option<String>,
}

impl KvCacheCompatibility {
    pub fn new(model: GenerationModelReference, tokenizer: TokenizerId) -> Self {
        Self {
            model,
            model_architecture: None,
            model_revision: None,
            tokenizer,
            tokenizer_vocabulary_size: 0,
            prefix_fingerprint: None,
            position_encoding: None,
            attention_implementation: None,
            quantization_mode: None,
        }
    }

    pub fn with_prefix_fingerprint(mut self, fingerprint: PrefixFingerprint) -> Self {
        self.prefix_fingerprint = Some(fingerprint);
        self
    }

    pub fn validate_reuse(&self, requested: &Self) -> Result<(), KvCacheError> {
        if self.model != requested.model {
            return Err(KvCacheError::CacheModelMismatch);
        }
        if self.tokenizer != requested.tokenizer {
            return Err(KvCacheError::CacheTokenizerMismatch);
        }
        if self.prefix_fingerprint != requested.prefix_fingerprint {
            return Err(KvCacheError::CachePromptMismatch);
        }
        if self.position_encoding != requested.position_encoding {
            return Err(KvCacheError::CachePositionMismatch);
        }
        if self.quantization_mode != requested.quantization_mode {
            return Err(KvCacheError::CacheDTypeMismatch);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvCacheSharingPolicy {
    Deny,
    AllowReadOnlySealed,
    AllowWithinSession,
    PolicyControlled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvCacheRetentionPolicy {
    ReleaseOnOperationEnd,
    ReleaseOnSessionClose,
    RetainWhileRuntimeAllows,
    RetainForPrefixReuse,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvCachePolicy {
    pub enabled: bool,
    pub max_cache_tokens: Option<u32>,
    pub max_cache_memory_bytes: Option<u64>,
    pub sharing: KvCacheSharingPolicy,
    pub retention: KvCacheRetentionPolicy,
    pub prefix_reuse_allowed: bool,
    pub privacy_redaction_required: bool,
}

impl Default for KvCachePolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            max_cache_tokens: None,
            max_cache_memory_bytes: None,
            sharing: KvCacheSharingPolicy::Deny,
            retention: KvCacheRetentionPolicy::ReleaseOnSessionClose,
            prefix_reuse_allowed: false,
            privacy_redaction_required: true,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderKvCacheResource {
    pub provider: ProviderBinding,
    pub handle_kind: String,
    pub release_required: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvCacheResidency {
    pub placement: MemoryPlacement,
    pub affinity: ResourceAffinity,
    pub provider: Option<ProviderBinding>,
    pub device: Option<DeviceBinding>,
    pub memory_allocation: Option<MemoryAllocationId>,
    pub provider_resource: Option<ProviderKvCacheResource>,
    pub memory_pressure: Option<crate::MemoryPressureLevel>,
}

impl KvCacheResidency {
    pub fn host() -> Self {
        Self {
            placement: MemoryPlacement::HostOrdinary,
            affinity: ResourceAffinity::new(FallbackClass::Transparent),
            provider: None,
            device: None,
            memory_allocation: None,
            provider_resource: None,
            memory_pressure: None,
        }
    }

    pub fn provider_owned(provider: ProviderBinding) -> Self {
        let affinity =
            ResourceAffinity::new(FallbackClass::ProviderPinned).with_provider(provider.clone());
        Self {
            placement: MemoryPlacement::ProviderOwnedOpaque(provider.clone()),
            affinity,
            provider: Some(provider),
            device: None,
            memory_allocation: None,
            provider_resource: None,
            memory_pressure: None,
        }
    }

    pub fn with_device(mut self, device: DeviceBinding) -> Self {
        self.affinity = self.affinity.with_device(device.clone());
        self.device = Some(device);
        self
    }

    pub fn with_allocation(mut self, allocation: MemoryAllocationId) -> Self {
        self.memory_allocation = Some(allocation);
        self
    }

    pub fn validate_placement(&self, requested: &ResourceAffinity) -> Result<(), KvCacheError> {
        self.affinity.validate_with(requested).map_err(|_| {
            if self.affinity.provider() != requested.provider() {
                KvCacheError::CacheProviderMismatch
            } else if self.affinity.device() != requested.device() {
                KvCacheError::CacheDeviceMismatch
            } else {
                KvCacheError::CacheIncompatible
            }
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvCache {
    pub id: KvCacheId,
    pub scope: KvCacheScope,
    pub lifecycle: KvCacheLifecycleState,
    pub session: Option<InferenceSessionId>,
    pub compatibility: KvCacheCompatibility,
    pub layout: KvCacheLayoutMetadata,
    pub residency: KvCacheResidency,
    pub policy: KvCachePolicy,
    pub created_at_millis: u64,
    pub last_access_millis: u64,
    pub idle_ttl_millis: Option<u64>,
    pub total_ttl_millis: Option<u64>,
    /// Per-layer K/V tensor resource bindings for this cache's *committed*
    /// data (see `KvLayerResourceBinding`), keyed by layer index. Runtime-
    /// owned: the actual bytes live in the bound Provider's storage,
    /// addressed by `TensorResourceId`, not in any executor-private map.
    /// Empty until the first successful prefill commit.
    pub layer_resources: BTreeMap<u32, KvLayerResourceBinding>,
}

/// One layer's committed K/V tensor resource identities plus the
/// `MemoryManager` allocations accounting for their current byte size, so a
/// later commit (decode append) or cache release can find and release the
/// allocation it is about to replace or free.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvLayerResourceBinding {
    pub k: TensorResourceId,
    pub v: TensorResourceId,
    pub k_allocation: MemoryAllocationId,
    pub v_allocation: MemoryAllocationId,
}

impl KvCache {
    pub fn new(
        id: KvCacheId,
        scope: KvCacheScope,
        compatibility: KvCacheCompatibility,
        layout: KvCacheLayoutMetadata,
    ) -> Self {
        Self {
            id,
            scope,
            lifecycle: KvCacheLifecycleState::Allocating,
            session: None,
            compatibility,
            layout,
            residency: KvCacheResidency::host(),
            policy: KvCachePolicy::default(),
            created_at_millis: 0,
            last_access_millis: 0,
            idle_ttl_millis: None,
            total_ttl_millis: None,
            layer_resources: BTreeMap::new(),
        }
    }

    pub fn with_session(mut self, session: InferenceSessionId) -> Self {
        self.session = Some(session);
        self
    }

    pub fn with_residency(mut self, residency: KvCacheResidency) -> Self {
        self.residency = residency;
        self
    }

    pub fn with_policy(mut self, policy: KvCachePolicy) -> Self {
        self.policy = policy;
        self
    }

    pub fn transition_to(&mut self, next: KvCacheLifecycleState) -> Result<(), KvCacheError> {
        if self.lifecycle.allows_transition_to(next) {
            self.lifecycle = next;
            Ok(())
        } else {
            Err(KvCacheError::CacheInvalid)
        }
    }

    pub fn validate_reuse(
        &self,
        requested: &KvCacheCompatibility,
        affinity: Option<&ResourceAffinity>,
    ) -> Result<(), KvCacheError> {
        match self.lifecycle {
            KvCacheLifecycleState::Ready | KvCacheLifecycleState::Active => {}
            KvCacheLifecycleState::Sealed => {
                if self.policy.sharing == KvCacheSharingPolicy::Deny {
                    return Err(KvCacheError::CacheSharingDenied);
                }
            }
            KvCacheLifecycleState::Evicted => return Err(KvCacheError::CacheEvicted),
            KvCacheLifecycleState::Released => return Err(KvCacheError::CacheReleased),
            KvCacheLifecycleState::Invalid => return Err(KvCacheError::CacheInvalid),
            _ => return Err(KvCacheError::CacheIncompatible),
        }
        self.compatibility.validate_reuse(requested)?;
        if let Some(affinity) = affinity {
            self.residency.validate_placement(affinity)?;
        }
        Ok(())
    }

    pub fn append_tokens(&mut self, count: u32) -> Result<(), KvCacheError> {
        if self.lifecycle == KvCacheLifecycleState::Sealed {
            return Err(KvCacheError::CacheSealed);
        }
        if self.lifecycle.is_terminal() {
            return Err(KvCacheError::CacheInvalid);
        }
        self.layout.append_tokens(count)
    }

    pub fn seal(&mut self) -> Result<(), KvCacheError> {
        self.transition_to(KvCacheLifecycleState::Sealed)
    }

    pub fn allocation_request(&self) -> Result<MemoryAllocationRequest, KvCacheError> {
        let mut request = MemoryAllocationRequest::new(
            MemoryAllocationClass::KvCache,
            self.layout.estimated_storage_bytes()?,
            self.residency.placement.clone(),
            self.session
                .as_ref()
                .map(|session| MemoryAllocationOwner::Session(session.as_str().into()))
                .unwrap_or(MemoryAllocationOwner::Runtime),
        )
        .with_affinity(self.residency.affinity.clone())
        .with_dtype_relation(MemoryDTypeRelation::new(
            self.layout.storage_dtype.clone(),
            self.layout.compute_dtype.clone(),
        ));
        request.lifetime = match self.scope {
            KvCacheScope::Operation | KvCacheScope::BatchSlot => {
                MemoryAllocationLifetime::Operation
            }
            KvCacheScope::Session => MemoryAllocationLifetime::Session,
            KvCacheScope::ModelInstance
            | KvCacheScope::PrefixCache
            | KvCacheScope::RuntimeCache => MemoryAllocationLifetime::Runtime,
        };
        Ok(request)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvCacheObservationKind {
    AllocationRequested,
    AllocationCompleted,
    AllocationFailed,
    PrefillStarted,
    PrefillCompleted,
    DecodeAppend,
    Hit,
    Miss,
    CompatibilityFailed,
    Sealed,
    Evicting,
    Evicted,
    Invalidated,
    Released,
    MemoryPressure,
    MovementRequired,
    SharingDenied,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvCacheObservation {
    pub kind: KvCacheObservationKind,
    pub cache: Option<KvCacheId>,
    pub message: String,
    pub raw_prompt_available: bool,
    pub raw_cache_available: bool,
    pub raw_provider_handle_available: bool,
}

impl KvCacheObservation {
    pub fn redacted(
        kind: KvCacheObservationKind,
        cache: Option<KvCacheId>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            cache,
            message: message.into(),
            raw_prompt_available: false,
            raw_cache_available: false,
            raw_provider_handle_available: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvCacheManager {
    next_id: u64,
    caches: BTreeMap<KvCacheId, KvCache>,
    observations: Vec<KvCacheObservation>,
}

impl Default for KvCacheManager {
    fn default() -> Self {
        Self::new()
    }
}

impl KvCacheManager {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            caches: BTreeMap::new(),
            observations: Vec::new(),
        }
    }

    pub fn caches(&self) -> impl Iterator<Item = &KvCache> {
        self.caches.values()
    }

    pub fn observations(&self) -> &[KvCacheObservation] {
        &self.observations
    }

    pub fn cache(&self, id: &KvCacheId) -> Result<&KvCache, KvCacheError> {
        self.caches.get(id).ok_or(KvCacheError::CacheNotFound)
    }

    pub fn cache_mut(&mut self, id: &KvCacheId) -> Result<&mut KvCache, KvCacheError> {
        self.caches.get_mut(id).ok_or(KvCacheError::CacheNotFound)
    }

    pub fn create(&mut self, mut cache: KvCache) -> Result<KvCacheId, KvCacheError> {
        if !cache.policy.enabled {
            return Err(KvCacheError::CacheAdmissionDenied);
        }
        if let Some(max_tokens) = cache.policy.max_cache_tokens
            && cache.layout.token_capacity > max_tokens
        {
            return Err(KvCacheError::CacheCapacityExceeded);
        }
        let estimated = cache.layout.estimated_storage_bytes()?;
        if let Some(max_bytes) = cache.policy.max_cache_memory_bytes
            && estimated > max_bytes
        {
            return Err(KvCacheError::CacheAdmissionDenied);
        }
        let id = KvCacheId::runtime_issued(self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        cache.id = id.clone();
        cache.lifecycle = KvCacheLifecycleState::Empty;
        self.caches.insert(id.clone(), cache);
        self.observe(
            KvCacheObservationKind::AllocationRequested,
            Some(id.clone()),
            "kv cache allocation requested",
        );
        Ok(id)
    }

    pub fn allocate_memory(
        &mut self,
        id: &KvCacheId,
        memory: &mut MemoryManager,
    ) -> Result<MemoryAllocationId, KvCacheError> {
        let request = self.cache(id)?.allocation_request()?;
        let allocation = memory
            .allocate(request)
            .map_err(|_| KvCacheError::CacheAllocationFailed)?;
        self.cache_mut(id)?.residency.memory_allocation = Some(allocation.id);
        self.observe(
            KvCacheObservationKind::AllocationCompleted,
            Some(id.clone()),
            "kv cache allocation completed",
        );
        Ok(allocation.id)
    }

    pub fn validate_reuse(
        &mut self,
        id: &KvCacheId,
        requested: &KvCacheCompatibility,
        affinity: Option<&ResourceAffinity>,
    ) -> Result<(), KvCacheError> {
        let result = self.cache(id)?.validate_reuse(requested, affinity);
        self.observe(
            match result {
                Ok(()) => KvCacheObservationKind::Hit,
                Err(KvCacheError::CacheSharingDenied) => KvCacheObservationKind::SharingDenied,
                Err(KvCacheError::CacheProviderMismatch | KvCacheError::CacheDeviceMismatch) => {
                    KvCacheObservationKind::MovementRequired
                }
                Err(_) => KvCacheObservationKind::CompatibilityFailed,
            },
            Some(id.clone()),
            if result.is_ok() {
                "kv cache reuse accepted"
            } else {
                "kv cache reuse rejected"
            },
        );
        result
    }

    pub fn prefill_completed(&mut self, id: &KvCacheId, tokens: u32) -> Result<(), KvCacheError> {
        {
            let cache = self.cache_mut(id)?;
            if cache.lifecycle == KvCacheLifecycleState::Empty {
                cache.transition_to(KvCacheLifecycleState::Prefilling)?;
            }
            cache.append_tokens(tokens)?;
            cache.transition_to(KvCacheLifecycleState::Ready)?;
        }
        self.observe(
            KvCacheObservationKind::PrefillCompleted,
            Some(id.clone()),
            "kv cache prefill completed",
        );
        Ok(())
    }

    pub fn decode_append(&mut self, id: &KvCacheId, tokens: u32) -> Result<(), KvCacheError> {
        self.cache_mut(id)?.append_tokens(tokens)?;
        self.observe(
            KvCacheObservationKind::DecodeAppend,
            Some(id.clone()),
            "kv cache decode append",
        );
        Ok(())
    }

    pub fn seal(&mut self, id: &KvCacheId) -> Result<(), KvCacheError> {
        self.cache_mut(id)?.seal()?;
        self.observe(
            KvCacheObservationKind::Sealed,
            Some(id.clone()),
            "kv cache sealed",
        );
        Ok(())
    }

    pub fn evict(&mut self, id: &KvCacheId) -> Result<(), KvCacheError> {
        self.cache_mut(id)?
            .transition_to(KvCacheLifecycleState::Evicting)?;
        self.observe(
            KvCacheObservationKind::Evicting,
            Some(id.clone()),
            "kv cache evicting",
        );
        self.cache_mut(id)?
            .transition_to(KvCacheLifecycleState::Evicted)?;
        self.observe(
            KvCacheObservationKind::Evicted,
            Some(id.clone()),
            "kv cache evicted",
        );
        Ok(())
    }

    pub fn invalidate(&mut self, id: &KvCacheId) -> Result<(), KvCacheError> {
        self.cache_mut(id)?.lifecycle = KvCacheLifecycleState::Invalid;
        self.observe(
            KvCacheObservationKind::Invalidated,
            Some(id.clone()),
            "kv cache invalidated",
        );
        Ok(())
    }

    pub fn release(&mut self, id: &KvCacheId) -> Result<(), KvCacheError> {
        self.cache_mut(id)?.lifecycle = KvCacheLifecycleState::Released;
        self.observe(
            KvCacheObservationKind::Released,
            Some(id.clone()),
            "kv cache released",
        );
        Ok(())
    }

    pub fn release_session_caches(
        &mut self,
        session: &InferenceSessionId,
    ) -> Result<Vec<KvCacheId>, KvCacheError> {
        let ids = self
            .caches
            .values()
            .filter(|cache| cache.session.as_ref() == Some(session))
            .map(|cache| cache.id.clone())
            .collect::<Vec<_>>();
        for id in &ids {
            self.release(id)?;
        }
        Ok(ids)
    }

    pub fn release_model_instance_caches(
        &mut self,
        instance: &ModelInstanceId,
    ) -> Result<Vec<KvCacheId>, KvCacheError> {
        let model = GenerationModelReference::ModelInstance(instance.clone());
        let ids = self
            .caches
            .values()
            .filter(|cache| cache.compatibility.model == model)
            .map(|cache| cache.id.clone())
            .collect::<Vec<_>>();
        for id in &ids {
            self.release(id)?;
        }
        Ok(ids)
    }

    fn observe(
        &mut self,
        kind: KvCacheObservationKind,
        cache: Option<KvCacheId>,
        message: impl Into<String>,
    ) {
        self.observations
            .push(KvCacheObservation::redacted(kind, cache, message));
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KvCacheError {
    CacheAllocationFailed,
    CacheAdmissionDenied,
    CacheNotFound,
    CacheIncompatible,
    CacheInvalid,
    CacheEvicted,
    CacheReleased,
    CacheCapacityExceeded,
    CachePositionMismatch,
    CachePromptMismatch,
    CacheModelMismatch,
    CacheTokenizerMismatch,
    CacheDTypeMismatch,
    CacheLayoutMismatch,
    CacheProviderMismatch,
    CacheDeviceMismatch,
    CacheMovementRequired,
    CacheMovementUnsupported,
    CacheSharingDenied,
    CacheSealed,
    CacheMutationDenied,
    CacheMemoryPressure,
    CacheProviderFailure,
    CacheDeviceUnavailable,
    CacheCancelled,
    CacheInternal,
}

impl fmt::Display for KvCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::CacheAllocationFailed => "kv cache allocation failed",
            Self::CacheAdmissionDenied => "kv cache admission denied",
            Self::CacheNotFound => "kv cache not found",
            Self::CacheIncompatible => "kv cache incompatible",
            Self::CacheInvalid => "kv cache invalid",
            Self::CacheEvicted => "kv cache evicted",
            Self::CacheReleased => "kv cache released",
            Self::CacheCapacityExceeded => "kv cache capacity exceeded",
            Self::CachePositionMismatch => "kv cache position mismatch",
            Self::CachePromptMismatch => "kv cache prompt mismatch",
            Self::CacheModelMismatch => "kv cache model mismatch",
            Self::CacheTokenizerMismatch => "kv cache tokenizer mismatch",
            Self::CacheDTypeMismatch => "kv cache dtype mismatch",
            Self::CacheLayoutMismatch => "kv cache layout mismatch",
            Self::CacheProviderMismatch => "kv cache provider mismatch",
            Self::CacheDeviceMismatch => "kv cache device mismatch",
            Self::CacheMovementRequired => "kv cache movement required",
            Self::CacheMovementUnsupported => "kv cache movement unsupported",
            Self::CacheSharingDenied => "kv cache sharing denied",
            Self::CacheSealed => "kv cache sealed",
            Self::CacheMutationDenied => "kv cache mutation denied",
            Self::CacheMemoryPressure => "kv cache memory pressure",
            Self::CacheProviderFailure => "kv cache provider failure",
            Self::CacheDeviceUnavailable => "kv cache device unavailable",
            Self::CacheCancelled => "kv cache cancelled",
            Self::CacheInternal => "kv cache internal error",
        })
    }
}

impl Error for KvCacheError {}

fn validate_kv_cache_identity(value: &str) -> Result<(), KvCacheError> {
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
        return Err(KvCacheError::CacheAdmissionDenied);
    }
    Ok(())
}
