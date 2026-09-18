//! Conformance tests for this crate's roadmap/release-process
//! contracts. Moved out of `magnetar-runtime/src/tests.rs` when the
//! roadmap modules themselves moved to this crate (#62) -- unchanged
//! otherwise, still the same conformance checks that ran inside
//! magnetar-runtime before.

use super::*;
use magnetar_runtime::*;
use std::collections::{BTreeMap, BTreeSet};

// ---------------------------------------------------------------------
// model_source_cache_roadmap
// ---------------------------------------------------------------------

fn model_source_cache_probe_artifact_id(name: &str) -> ModelArtifactId {
    let digest = ModelDigest::sha256(name.as_bytes());
    ModelArtifactId::new(
        ModelArtifactKind::ModelBundle,
        ModelName::new(name).unwrap(),
        ModelRevision::new("v1").unwrap(),
        digest,
    )
}

#[test]
fn model_source_cache_roadmap_source_kinds_cover_seven_categories() {
    assert_eq!(MODEL_SOURCE_KINDS.len(), 7);
    let mut ids: BTreeSet<&str> = BTreeSet::new();
    for kind in MODEL_SOURCE_KINDS {
        assert!(
            ids.insert(kind.id()),
            "duplicate source kind id {}",
            kind.id()
        );
        assert!(
            !kind.grants_trust(),
            "source kind {} must not grant trust",
            kind.id()
        );
    }
}

#[test]
fn model_source_cache_roadmap_source_kind_normalizes_from_artifact_source() {
    assert_eq!(
        ModelSourceKind::from_artifact_source(&ModelArtifactSource::LocalCache("x".into())),
        ModelSourceKind::LocalCache
    );
    assert_eq!(
        ModelSourceKind::from_artifact_source(&ModelArtifactSource::ClientProvided("x".into())),
        ModelSourceKind::ClientProvidedArtifact
    );
    assert_eq!(
        ModelSourceKind::from_artifact_source(&ModelArtifactSource::LocalPath("/models".into())),
        ModelSourceKind::LocalDirectorySource
    );
    assert_eq!(
        ModelSourceKind::from_artifact_source(&ModelArtifactSource::Registry("x".into())),
        ModelSourceKind::ExternalRegistrySource
    );
    assert_eq!(
        ModelSourceKind::from_artifact_source(&ModelArtifactSource::Oci("x".into())),
        ModelSourceKind::ExternalRegistrySource
    );
    assert_eq!(
        ModelSourceKind::from_artifact_source(&ModelArtifactSource::HuggingFace("x".into())),
        ModelSourceKind::ModelHubSource
    );
    assert_eq!(
        ModelSourceKind::from_artifact_source(&ModelArtifactSource::Tachyon("x".into())),
        ModelSourceKind::TachyonProvidedSource
    );
}

#[test]
fn model_source_cache_roadmap_source_kind_normalizes_from_resolution_source() {
    assert_eq!(
        ModelSourceKind::from_resolution_source(ModelResolutionSource::DevelopmentFixture),
        Some(ModelSourceKind::DevelopmentFixture)
    );
    assert_eq!(
        ModelSourceKind::from_resolution_source(ModelResolutionSource::ClientProvidedArtifact),
        Some(ModelSourceKind::ClientProvidedArtifact)
    );
    assert_eq!(
        ModelSourceKind::from_resolution_source(ModelResolutionSource::TrustedCache),
        Some(ModelSourceKind::LocalCache)
    );
    assert_eq!(
        ModelSourceKind::from_resolution_source(ModelResolutionSource::FutureExternalSource),
        Some(ModelSourceKind::ExternalRegistrySource)
    );
    assert_eq!(
        ModelSourceKind::from_resolution_source(ModelResolutionSource::FutureTachyonSource),
        Some(ModelSourceKind::TachyonProvidedSource)
    );
    assert_eq!(
        ModelSourceKind::from_resolution_source(ModelResolutionSource::LocalRegistry),
        None
    );
}

#[test]
fn model_source_cache_roadmap_development_fixture_denied_in_production_by_default() {
    assert!(validate_development_fixture_source(true, false).is_err());
    assert!(validate_development_fixture_source(true, true).is_ok());
    assert!(validate_development_fixture_source(false, false).is_ok());
}

#[test]
fn model_source_cache_roadmap_development_fixture_still_uses_real_trust_store() {
    let store = ModelTrustStore::default();
    let manifest = ModelManifest {
        schema_version: MODEL_ARTIFACT_SCHEMA_VERSION,
        id: model_source_cache_probe_artifact_id("fixture-probe"),
        architecture: ModelArchitecture::new("probe", "probe"),
        parts: BTreeMap::new(),
        storage_dtype: None,
        compute_dtype: None,
        supported_compute_dtypes: BTreeSet::new(),
        tensors: Vec::new(),
        tokenizer: None,
        tokenizer_config: None,
        chat_template: None,
        prompt_template: None,
        generation: None,
        quantization: None,
        shards: Vec::new(),
        runtime_features: BTreeSet::new(),
        memory_features: BTreeSet::new(),
        provider_capabilities: Vec::new(),
        component: None,
        license: None,
        provenance: None,
        signatures: Vec::new(),
        source: None,
        architecture_config: None,
    };
    let decision = development_fixture_requires_explicit_trust_evaluation(&store, &manifest);
    assert_eq!(decision.status(), ModelTrustStatus::Unknown);
}

#[test]
fn model_source_cache_roadmap_client_provided_source_requires_authorization() {
    assert!(validate_client_provided_source(false).is_err());
    assert!(validate_client_provided_source(true).is_ok());
}

#[test]
fn model_source_cache_roadmap_cache_key_is_digest_based() {
    let id = model_source_cache_probe_artifact_id("qwen-local");
    let key = CacheKey::from_artifact(&id);
    assert_eq!(key.as_str(), id.digest.value.as_str());
}

#[test]
fn model_source_cache_roadmap_cache_entry_ref_redacts_path_by_default() {
    let key = CacheKey::from_artifact(&model_source_cache_probe_artifact_id("qwen-local"));
    let entry_ref = CacheEntryRef::new(key, "/var/cache/magnetar/qwen-local");
    assert_eq!(entry_ref.redacted_path(false), None);
    assert_eq!(
        entry_ref.redacted_path(true).as_deref(),
        Some("/var/cache/magnetar/qwen-local")
    );
}

#[test]
fn model_source_cache_roadmap_local_directory_source_denies_unauthorized_access() {
    let source = ModelArtifactSource::LocalPath("/models/qwen".into());
    assert!(validate_local_directory_source(&source, false).is_err());
    assert!(validate_local_directory_source(&source, true).is_ok());
}

#[test]
fn model_source_cache_roadmap_remote_source_policy_denies_by_default() {
    let policy = SourcePolicy::default();
    assert!(
        validate_remote_source_policy(ModelSourceKind::ExternalRegistrySource, &policy).is_err()
    );
    assert!(validate_remote_source_policy(ModelSourceKind::ModelHubSource, &policy).is_err());
    assert!(
        validate_remote_source_policy(ModelSourceKind::TachyonProvidedSource, &policy).is_err()
    );
    assert!(validate_remote_source_policy(ModelSourceKind::LocalCache, &policy).is_ok());
}

#[test]
fn model_source_cache_roadmap_tachyon_source_requires_explicit_policy_flag_even_if_listed() {
    let mut policy = SourcePolicy::default();
    policy
        .allowed_kinds
        .insert(ModelSourceKind::TachyonProvidedSource);
    assert!(!policy.allow_tachyon_provided_sources);
    assert!(!policy.allows_kind(ModelSourceKind::TachyonProvidedSource));
    policy.allow_tachyon_provided_sources = true;
    assert!(policy.allows_kind(ModelSourceKind::TachyonProvidedSource));
}

#[test]
fn model_source_cache_roadmap_model_ref_resolution_rejects_zero_and_multiple_candidates() {
    let none = resolve_model_ref_candidates("qwen", Vec::new());
    assert!(matches!(
        none,
        Err(ModelSourceCacheRoadmapError::ModelSourceNotFound { .. })
    ));

    let one = resolve_model_ref_candidates(
        "qwen",
        vec![ModelRefResolutionOutcome::SourceCandidate {
            kind: ModelSourceKind::LocalCache,
            reference: "qwen".into(),
        }],
    );
    assert!(one.is_ok());

    let many = resolve_model_ref_candidates(
        "qwen",
        vec![
            ModelRefResolutionOutcome::SourceCandidate {
                kind: ModelSourceKind::LocalCache,
                reference: "qwen-a".into(),
            },
            ModelRefResolutionOutcome::SourceCandidate {
                kind: ModelSourceKind::DevelopmentFixture,
                reference: "qwen-b".into(),
            },
        ],
    );
    assert!(matches!(
        many,
        Err(ModelSourceCacheRoadmapError::ModelSourceAmbiguous { .. })
    ));
}

#[test]
fn model_source_cache_roadmap_alias_missing_and_ambiguous_errors() {
    let alias = ModelAlias::new("qwen").unwrap();
    let mut table = ModelAliasTable::new();
    assert!(matches!(
        table.resolve(&alias),
        Err(ModelSourceCacheRoadmapError::ModelAliasNotFound { .. })
    ));

    table.register(alias.clone(), "qwen-a");
    assert_eq!(table.resolve(&alias).unwrap(), "qwen-a");

    table.register(alias.clone(), "qwen-b");
    assert!(matches!(
        table.resolve(&alias),
        Err(ModelSourceCacheRoadmapError::ModelAliasAmbiguous { .. })
    ));
}

#[test]
fn model_source_cache_roadmap_alias_rejects_empty_name() {
    assert!(ModelAlias::new("").is_err());
    assert!(ModelAlias::new("qwen").is_ok());
}

#[test]
fn model_source_cache_roadmap_artifact_identity_coverage_tracks_present_fields() {
    let mut manifest = ModelManifest {
        schema_version: MODEL_ARTIFACT_SCHEMA_VERSION,
        id: model_source_cache_probe_artifact_id("qwen-local"),
        architecture: ModelArchitecture::new("qwen", "qwen"),
        parts: BTreeMap::new(),
        storage_dtype: None,
        compute_dtype: None,
        supported_compute_dtypes: BTreeSet::new(),
        tensors: Vec::new(),
        tokenizer: Some("tokenizer-id".into()),
        tokenizer_config: None,
        chat_template: None,
        prompt_template: None,
        generation: None,
        quantization: None,
        shards: Vec::new(),
        runtime_features: BTreeSet::new(),
        memory_features: BTreeSet::new(),
        provider_capabilities: Vec::new(),
        component: None,
        license: None,
        provenance: None,
        signatures: Vec::new(),
        source: Some(ModelArtifactSource::LocalCache("qwen".into())),
        architecture_config: None,
    };
    let coverage = ArtifactIdentityCoverage::from_manifest(&manifest);
    assert!(coverage.content_digest);
    assert!(coverage.tokenizer_reference);
    assert!(coverage.source_annotation);
    assert!(coverage.version_metadata);
    assert!(coverage.covers_required_fields());

    manifest.tokenizer = None;
    manifest.source = None;
    let coverage = ArtifactIdentityCoverage::from_manifest(&manifest);
    assert!(!coverage.tokenizer_reference);
    assert!(!coverage.source_annotation);
}

#[test]
fn model_source_cache_roadmap_same_name_different_digest_remains_distinct() {
    let left = model_source_cache_probe_artifact_id("qwen-local");
    let mut right = model_source_cache_probe_artifact_id("qwen-local");
    right.digest = ModelDigest::sha256(b"a-different-payload");
    assert!(artifacts_are_distinct_despite_same_name(&left, &right));
}

#[test]
fn model_source_cache_roadmap_cache_lifecycle_states_cover_thirteen_categories() {
    assert_eq!(CACHE_LIFECYCLE_STATES.len(), 13);
    let mut ready_count = 0;
    for state in CACHE_LIFECYCLE_STATES {
        assert!(!state.id().is_empty());
        if state.is_loadable() {
            ready_count += 1;
        }
    }
    assert_eq!(ready_count, 1, "exactly Ready should be loadable");
}

#[test]
fn model_source_cache_roadmap_rejects_every_non_ready_lifecycle_state_for_loading() {
    for state in CACHE_LIFECYCLE_STATES {
        let outcome = reject_non_ready_cache_entry_for_loading(*state);
        if matches!(state, CacheLifecycleState::Ready) {
            assert!(outcome.is_ok(), "Ready must be loadable");
        } else {
            assert!(outcome.is_err(), "{state:?} must not be loadable");
        }
    }
}

#[test]
fn model_source_cache_roadmap_partial_and_corrupt_entries_map_to_specific_errors() {
    assert!(matches!(
        reject_non_ready_cache_entry_for_loading(CacheLifecycleState::Partial),
        Err(ModelSourceCacheRoadmapError::ModelCachePartialEntry { .. })
    ));
    assert!(matches!(
        reject_non_ready_cache_entry_for_loading(CacheLifecycleState::Corrupt),
        Err(ModelSourceCacheRoadmapError::ModelCacheEntryCorrupt { .. })
    ));
    assert!(matches!(
        reject_non_ready_cache_entry_for_loading(CacheLifecycleState::Untrusted),
        Err(ModelSourceCacheRoadmapError::ModelCacheEntryUntrusted { .. })
    ));
    assert!(matches!(
        reject_non_ready_cache_entry_for_loading(CacheLifecycleState::Revoked),
        Err(ModelSourceCacheRoadmapError::ModelCacheEntryRevoked { .. })
    ));
}

#[test]
fn model_source_cache_roadmap_cache_entry_metadata_defaults_to_discovered_and_unpinned() {
    let entry = CacheEntryMetadata::new(
        model_source_cache_probe_artifact_id("qwen-local"),
        ModelSourceKind::LocalCache,
    );
    assert_eq!(entry.lifecycle, CacheLifecycleState::Discovered);
    assert!(!entry.pinned);
    assert_eq!(entry.trust_status, ModelTrustStatus::Unknown);
    assert_eq!(entry.integrity_status, CacheIntegrityStatus::Unchecked);
    assert_eq!(entry.validation_status, CacheValidationStatus::Unvalidated);
    assert_eq!(entry.key(), CacheKey::from_artifact(&entry.identity));
}

#[test]
fn model_source_cache_roadmap_cache_trust_re_evaluates_and_revocation_wins() {
    let digest = ModelDigest::sha256(b"trusted-model");
    let store = ModelTrustStore::default().trust_digest(digest.value.clone());
    let mut manifest_id = model_source_cache_probe_artifact_id("trusted-model");
    manifest_id.digest = digest;
    let manifest = ModelManifest {
        schema_version: MODEL_ARTIFACT_SCHEMA_VERSION,
        id: manifest_id,
        architecture: ModelArchitecture::new("probe", "probe"),
        parts: BTreeMap::new(),
        storage_dtype: None,
        compute_dtype: None,
        supported_compute_dtypes: BTreeSet::new(),
        tensors: Vec::new(),
        tokenizer: None,
        tokenizer_config: None,
        chat_template: None,
        prompt_template: None,
        generation: None,
        quantization: None,
        shards: Vec::new(),
        runtime_features: BTreeSet::new(),
        memory_features: BTreeSet::new(),
        provider_capabilities: Vec::new(),
        component: None,
        license: None,
        provenance: None,
        signatures: Vec::new(),
        source: None,
        architecture_config: None,
    };
    let trusted = evaluate_cache_trust(&store, &manifest, false);
    assert_eq!(trusted.status(), ModelTrustStatus::Trusted);
    let revoked = evaluate_cache_trust(&store, &manifest, true);
    assert_eq!(revoked.status(), ModelTrustStatus::Revoked);
}

#[test]
fn model_source_cache_roadmap_cache_integrity_detects_digest_mismatch() {
    let declared = ModelDigest::sha256(b"declared-bytes");
    let matching = ModelDigest::sha256(b"declared-bytes");
    let mismatched = ModelDigest::sha256(b"other-bytes");
    assert!(validate_cache_integrity(&declared, &matching).is_ok());
    assert!(matches!(
        validate_cache_integrity(&declared, &mismatched),
        Err(ModelSourceCacheRoadmapError::ModelCacheIntegrityFailed { .. })
    ));
}

#[test]
fn model_source_cache_roadmap_cache_shard_integrity_composes_shard_verify_bytes() {
    let bytes = b"shard-bytes";
    let shard = ModelShard {
        id: ModelShardId::new("shard-0").unwrap(),
        digest: ModelDigest::sha256(bytes),
        size_bytes: bytes.len() as u64,
        order: 0,
    };
    assert!(validate_cache_shard_integrity(&shard, bytes).is_ok());
    assert!(matches!(
        validate_cache_shard_integrity(&shard, b"wrong-bytes"),
        Err(ModelSourceCacheRoadmapError::ModelCacheEntryCorrupt { .. })
    ));
}

#[test]
fn model_source_cache_roadmap_mutation_requires_policy_and_denies_eviction_with_active_reference() {
    assert!(authorize_cache_mutation(CacheMutationKind::Insert, false, 0).is_err());
    assert!(authorize_cache_mutation(CacheMutationKind::Insert, true, 0).is_ok());
    assert!(matches!(
        authorize_cache_mutation(CacheMutationKind::Evict, true, 1),
        Err(ModelSourceCacheRoadmapError::ModelCacheActiveReference { .. })
    ));
    assert!(matches!(
        authorize_cache_mutation(CacheMutationKind::Evict, false, 0),
        Err(ModelSourceCacheRoadmapError::ModelCacheEvictionDenied { .. })
    ));
    assert!(authorize_cache_mutation(CacheMutationKind::Evict, true, 0).is_ok());
}

#[test]
fn model_source_cache_roadmap_eviction_respects_pin_and_active_reference_and_orders_by_last_used() {
    let mut old_entry = CacheEntryMetadata::new(
        model_source_cache_probe_artifact_id("qwen-old"),
        ModelSourceKind::LocalCache,
    );
    old_entry.lifecycle = CacheLifecycleState::Ready;
    old_entry.last_used_unix_seconds = 10;

    let mut new_entry = CacheEntryMetadata::new(
        model_source_cache_probe_artifact_id("qwen-new"),
        ModelSourceKind::LocalCache,
    );
    new_entry.lifecycle = CacheLifecycleState::Ready;
    new_entry.last_used_unix_seconds = 20;

    let mut pinned_entry = CacheEntryMetadata::new(
        model_source_cache_probe_artifact_id("qwen-pinned"),
        ModelSourceKind::LocalCache,
    );
    pinned_entry.lifecycle = CacheLifecycleState::Ready;
    pin_entry(&mut pinned_entry);

    let mut active_entry = CacheEntryMetadata::new(
        model_source_cache_probe_artifact_id("qwen-active"),
        ModelSourceKind::LocalCache,
    );
    active_entry.lifecycle = CacheLifecycleState::Ready;

    let candidates = vec![
        EvictionCandidate {
            entry: new_entry,
            active_instance_refs: 0,
        },
        EvictionCandidate {
            entry: old_entry,
            active_instance_refs: 0,
        },
        EvictionCandidate {
            entry: pinned_entry,
            active_instance_refs: 0,
        },
        EvictionCandidate {
            entry: active_entry,
            active_instance_refs: 1,
        },
    ];

    let selected = select_eviction_candidates(&candidates);
    assert_eq!(selected.len(), 2);
    assert_eq!(selected[0].identity.name.as_str(), "qwen-old");
    assert_eq!(selected[1].identity.name.as_str(), "qwen-new");
}

#[test]
fn model_source_cache_roadmap_pin_and_unpin_toggle_evictability() {
    let mut entry = CacheEntryMetadata::new(
        model_source_cache_probe_artifact_id("qwen-local"),
        ModelSourceKind::LocalCache,
    );
    entry.lifecycle = CacheLifecycleState::Ready;
    let candidate = EvictionCandidate {
        entry: entry.clone(),
        active_instance_refs: 0,
    };
    assert!(is_evictable(&candidate));

    pin_entry(&mut entry);
    let pinned_candidate = EvictionCandidate {
        entry: entry.clone(),
        active_instance_refs: 0,
    };
    assert!(!is_evictable(&pinned_candidate));

    unpin_entry(&mut entry);
    let unpinned_candidate = EvictionCandidate {
        entry,
        active_instance_refs: 0,
    };
    assert!(is_evictable(&unpinned_candidate));
}

#[test]
fn model_source_cache_roadmap_in_flight_lifecycle_states_are_never_evictable() {
    for state in [
        CacheLifecycleState::Resolving,
        CacheLifecycleState::Fetching,
        CacheLifecycleState::Normalizing,
        CacheLifecycleState::Validating,
        CacheLifecycleState::Evicting,
    ] {
        let mut entry = CacheEntryMetadata::new(
            model_source_cache_probe_artifact_id("qwen-in-flight"),
            ModelSourceKind::LocalCache,
        );
        entry.lifecycle = state;
        let candidate = EvictionCandidate {
            entry,
            active_instance_refs: 0,
        };
        assert!(!is_evictable(&candidate), "{state:?} must not be evictable");
    }
}

#[test]
fn model_source_cache_roadmap_offline_mode_allows_only_local_sources() {
    assert!(validate_offline_source(ModelSourceKind::LocalCache, true).is_ok());
    assert!(validate_offline_source(ModelSourceKind::ClientProvidedArtifact, true).is_ok());
    assert!(validate_offline_source(ModelSourceKind::DevelopmentFixture, true).is_ok());
    assert!(validate_offline_source(ModelSourceKind::LocalDirectorySource, true).is_err());
    assert!(validate_offline_source(ModelSourceKind::ExternalRegistrySource, true).is_err());
    assert!(validate_offline_source(ModelSourceKind::ModelHubSource, true).is_err());
    assert!(validate_offline_source(ModelSourceKind::TachyonProvidedSource, true).is_err());
    // Online mode never restricts by source kind.
    assert!(validate_offline_source(ModelSourceKind::ExternalRegistrySource, false).is_ok());
}

#[test]
fn model_source_cache_roadmap_rejects_credential_shaped_metadata_keys() {
    let mut annotations = BTreeMap::new();
    annotations.insert("model_family".to_string(), "qwen".to_string());
    assert!(reject_credential_in_metadata(&annotations).is_ok());

    for key in ["registry_token", "api_key", "auth_secret", "bearer_header"] {
        let mut annotations = BTreeMap::new();
        annotations.insert(key.to_string(), "value".to_string());
        assert!(
            reject_credential_in_metadata(&annotations).is_err(),
            "key '{key}' should be rejected"
        );
    }
}

#[test]
fn model_source_cache_roadmap_source_policy_defaults_deny_remote_and_allow_offline_kinds() {
    let policy = SourcePolicy::default();
    for kind in [
        ModelSourceKind::DevelopmentFixture,
        ModelSourceKind::ClientProvidedArtifact,
        ModelSourceKind::LocalCache,
    ] {
        assert!(
            policy.allows_kind(kind),
            "{kind:?} should be allowed by default"
        );
    }
    for kind in [
        ModelSourceKind::LocalDirectorySource,
        ModelSourceKind::ExternalRegistrySource,
        ModelSourceKind::ModelHubSource,
        ModelSourceKind::TachyonProvidedSource,
    ] {
        assert!(
            !policy.allows_kind(kind),
            "{kind:?} should be denied by default"
        );
    }
}

#[test]
fn model_source_cache_roadmap_source_policy_enforces_size_limit() {
    let policy = SourcePolicy {
        max_artifact_size_bytes: Some(1024),
        ..SourcePolicy::default()
    };
    assert!(policy.validate_size(512).is_ok());
    assert!(policy.validate_size(2048).is_err());
}

#[test]
fn model_source_cache_roadmap_license_requires_explicit_policy_validation() {
    let license = ModelLicenseMetadata {
        identifier: "apache-2.0".into(),
        url: None,
        usage_restrictions: Vec::new(),
    };
    assert!(validate_license_policy(&license, false, true).is_err());
    assert!(validate_license_policy(&license, true, false).is_err());
    assert!(validate_license_policy(&license, true, true).is_ok());
}

#[test]
fn model_source_cache_roadmap_format_normalization_requires_metadata_and_lifecycle() {
    let mut entry = CacheEntryMetadata::new(
        model_source_cache_probe_artifact_id("qwen-local"),
        ModelSourceKind::LocalCache,
    );
    assert!(!cache_entry_ready_for_format_normalization(&entry));
    entry.format_metadata = Some("safetensors".into());
    assert!(!cache_entry_ready_for_format_normalization(&entry));
    entry.lifecycle = CacheLifecycleState::Ready;
    assert!(cache_entry_ready_for_format_normalization(&entry));
}

#[test]
fn model_source_cache_roadmap_adapter_cache_entry_requires_reference_and_compatibility() {
    let mut entry = CacheEntryMetadata::new(
        model_source_cache_probe_artifact_id("lora-local"),
        ModelSourceKind::LocalCache,
    );
    let compatibility = AdapterBaseModelCompatibility {
        model_name: ModelName::new("qwen").unwrap(),
        model_revision: ModelRevision::new("v1").unwrap(),
        model_artifact: None,
        tokenizer: None,
        architecture: AdapterArchitectureCompatibility {
            family: "qwen".into(),
            implementation: "qwen".into(),
            hidden_size: None,
            layer_count: None,
            position_encoding: None,
            target_modules: BTreeSet::new(),
            supported_storage_dtypes: BTreeSet::new(),
            supported_compute_dtypes: BTreeSet::new(),
            supported_quantization_formats: BTreeSet::new(),
        },
    };
    assert!(validate_adapter_cache_entry(&entry, &compatibility).is_err());
    entry.adapters.push(AdapterArtifactId::new(
        AdapterName::new("lora").unwrap(),
        AdapterRevision::new("v1").unwrap(),
        AdapterDigest::parse(format!("sha256:{}", "1".repeat(64))).unwrap(),
    ));
    assert!(validate_adapter_cache_entry(&entry, &compatibility).is_ok());
}

#[test]
fn model_source_cache_roadmap_tokenizer_cache_entry_requires_reference_and_compatibility() {
    let mut entry = CacheEntryMetadata::new(
        model_source_cache_probe_artifact_id("qwen-local"),
        ModelSourceKind::LocalCache,
    );
    assert!(validate_tokenizer_cache_entry(&entry, true).is_err());
    entry.tokenizer = Some(TokenizerArtifactId::new("qwen-tokenizer").unwrap());
    assert!(validate_tokenizer_cache_entry(&entry, false).is_err());
    assert!(validate_tokenizer_cache_entry(&entry, true).is_ok());
}

#[test]
fn model_source_cache_roadmap_cache_presence_never_implies_memory_residency() {
    assert!(!cache_presence_implies_memory_residency());
}

#[test]
fn model_source_cache_roadmap_diagnostic_digest_prefix_is_short_and_never_full() {
    let digest = ModelDigest::sha256(b"some-model-bytes");
    let prefix = ModelSourceCacheDiagnostic::digest_prefix_from(&digest);
    assert!(prefix.len() <= 15);
    assert_ne!(prefix, digest.value);
}

#[test]
fn model_source_cache_roadmap_diagnostic_redacts_policy_denial_reason() {
    let redacted = ModelSourceCacheDiagnostic::redact_policy_denial_reason(
        "denied for path /var/cache/magnetar/model",
    );
    assert!(!redacted.contains('/'));
}

#[test]
fn model_source_cache_roadmap_error_display_is_non_empty_for_every_variant() {
    let errors = vec![
        ModelSourceCacheRoadmapError::ModelSourceUnsupported { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelSourceInvalid { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelSourceAmbiguous {
            reference: "r".into(),
        },
        ModelSourceCacheRoadmapError::ModelSourcePolicyDenied { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelSourceNetworkDenied { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelSourceAuthenticationFailed { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelSourceNotFound {
            reference: "r".into(),
        },
        ModelSourceCacheRoadmapError::ModelSourceOfflineUnavailable { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelCacheUnavailable { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelCacheMiss { key: "k".into() },
        ModelSourceCacheRoadmapError::ModelCacheEntryInvalid { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelCacheEntryCorrupt { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelCacheEntryUntrusted { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelCacheEntryRevoked { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelCacheIntegrityFailed { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelCacheInsertDenied { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelCacheEvictionDenied { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelCacheActiveReference { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelCachePartialEntry { reason: "r".into() },
        ModelSourceCacheRoadmapError::ModelAliasNotFound { alias: "a".into() },
        ModelSourceCacheRoadmapError::ModelAliasAmbiguous { alias: "a".into() },
        ModelSourceCacheRoadmapError::InternalModelSourceCacheError { reason: "r".into() },
    ];
    let mut ids: BTreeSet<&str> = BTreeSet::new();
    for error in &errors {
        assert!(!error.to_string().is_empty());
        assert!(ids.insert(error.id()), "duplicate error id {}", error.id());
    }
    assert_eq!(ids.len(), 22);
}

#[test]
fn model_source_cache_roadmap_observation_redacts_metadata() {
    let observation =
        ModelSourceCacheRoadmapObservation::new(ModelSourceCacheRoadmapObservationKind::CacheHit)
            .with_artifact("qwen-local")
            .with_redacted_metadata("path", "/var/cache/magnetar/qwen-local");
    assert_eq!(observation.artifact.as_deref(), Some("qwen-local"));
    assert!(
        !observation
            .redacted_metadata
            .get("path")
            .unwrap()
            .contains('/')
    );
}

#[test]
fn model_source_cache_roadmap_conformance_report_is_conformant() {
    let report = run_model_source_cache_roadmap_conformance();
    assert!(!report.results.is_empty());
    for result in &report.results {
        assert!(
            result.passed,
            "{} failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
    assert!(report.is_conformant());
}

// ---------------------------------------------------------------------
// server_api_roadmap
// ---------------------------------------------------------------------

fn server_api_roadmap_tokenizer_reference() -> GenerationTokenizerReference {
    let metadata = TokenizerMetadata {
        id: TokenizerId::new("server-api-roadmap-test").unwrap(),
        artifact: TokenizerArtifactId::new("server-api-roadmap-test-artifact").unwrap(),
        digest: ModelDigest::sha256(b"server-api-roadmap-test"),
        family: TokenizerFamily::new("fixture").unwrap(),
        revision: TokenizerRevision::new("1.0.0").unwrap(),
        vocabulary_size: 256,
        added_token_count: 2,
        token_id_range: TokenIdRange::new(1, 300),
        model_max_length: Some(64),
        special_tokens: vec![SpecialToken::new(SpecialTokenKind::Eos, "<eos>", 299)],
        additional_special_tokens: Vec::new(),
        byte_fallback: false,
        normalization: None,
        pre_tokenizer: None,
        supports_offsets: true,
        supports_token_type_ids: false,
        supports_browser: true,
    };
    GenerationTokenizerReference {
        tokenizer_id: metadata.id.clone(),
        metadata,
    }
}

fn server_api_roadmap_generation_request(streaming: bool) -> ServerGenerationRequest {
    ServerGenerationRequest {
        model_or_session: ServerModelOrSessionRef::Model(ModelRef::new("fixture-model").unwrap()),
        prompt: PromptInput::PlainText("hello".into()),
        parameters: GenerationParameters::greedy(),
        max_new_tokens: 4,
        max_total_tokens: Some(32),
        stop_conditions: StopConditions::default(),
        streaming,
        cache_policy: KvCachePolicy {
            enabled: false,
            max_cache_tokens: None,
            max_cache_memory_bytes: None,
            sharing: KvCacheSharingPolicy::Deny,
            retention: KvCacheRetentionPolicy::ReleaseOnSessionClose,
            prefix_reuse_allowed: false,
            privacy_redaction_required: true,
        },
        adapter_policy: None,
        timeout_millis: Some(5_000),
        correlation_id: Some(CorrelationId::new("fixture-correlation")),
    }
}

fn server_api_roadmap_runtime_context() -> ServerGenerationRuntimeContext {
    ServerGenerationRuntimeContext {
        request_id: GenerationRequestId::new("fixture-request").unwrap(),
        model: GenerationModelReference::LoadedModelContext("fixture-model-context".into()),
        tokenizer: server_api_roadmap_tokenizer_reference(),
        input_token_ids: vec![2, 3, 4],
        model_context_length: Some(64),
        trace_id: None,
    }
}

fn server_api_roadmap_session_creation_request(
    allowed_capabilities: BTreeSet<String>,
) -> SessionCreationRequest {
    let tokenizer = server_api_roadmap_tokenizer_reference();
    SessionCreationRequest {
        model: GenerationModelReference::LoadedModelContext("fixture-model-context".into()),
        tokenizer,
        generation_defaults: GenerationParameters::default(),
        policy: SessionPolicy::default(),
        memory: SessionMemoryBudget::default(),
        allowed_capabilities,
        correlation_id: None,
        created_at_millis: 0,
    }
}

// -- Endpoint scope --

#[test]
fn server_api_roadmap_endpoints_are_illustrative_and_complete() {
    assert_eq!(SERVER_API_ENDPOINTS.len(), 10);
    let ids: BTreeSet<&str> = SERVER_API_ENDPOINTS.iter().map(|e| e.id()).collect();
    for expected in [
        "health",
        "readiness",
        "models-list",
        "model-inspect",
        "session-create",
        "session-close",
        "generate",
        "generate-stream",
        "cancel",
        "diagnostics",
    ] {
        assert!(ids.contains(expected), "missing endpoint id '{expected}'");
    }
    for endpoint in SERVER_API_ENDPOINTS {
        assert!(endpoint.is_illustrative());
    }
}

// -- Health and readiness --

#[test]
fn server_api_roadmap_health_does_not_imply_readiness_or_model_availability() {
    let health = ServerHealthStatus::alive();
    let readiness = ServerReadinessStatus::not_ready("no model loaded");
    assert!(health.alive);
    assert!(!readiness.ready);
    assert!(healthy_but_not_ready_is_representable(&health, &readiness));
}

#[test]
fn server_api_roadmap_readiness_is_redacted() {
    let readiness = ServerReadinessStatus::not_ready("provider handle=0xdeadbeef unavailable");
    let summary = readiness.model_registry_state_summary.unwrap();
    assert!(!summary.contains("0xdeadbeef"));
}

#[test]
fn server_api_roadmap_health_and_readiness_are_structurally_independent_types() {
    // No conversion exists in either direction; both are constructed
    // independently.
    let health = ServerHealthStatus::not_alive("process exiting");
    let readiness = ServerReadinessStatus::ready();
    assert!(!health.alive);
    assert!(readiness.ready);
}

// -- Model endpoints --

#[test]
fn server_api_roadmap_model_load_requires_complete_loading_proof() {
    let incomplete = validate_model_endpoint_request(
        ServerModelEndpointOperation::RequestModelLoad,
        &ModelEndpointLoadingProof::deny_by_default(),
    );
    assert!(matches!(
        incomplete,
        Err(ServerApiRoadmapError::ServerModelLoadFailed { .. })
    ));

    let complete = validate_model_endpoint_request(
        ServerModelEndpointOperation::RequestModelLoad,
        &ModelEndpointLoadingProof {
            source_validated: true,
            cache_validated: true,
            artifact_validated: true,
            model_loading_validated: true,
            trust_validated: true,
            integrity_validated: true,
            compatibility_validated: true,
            policy_validated: true,
        },
    );
    assert!(complete.is_ok());
}

#[test]
fn server_api_roadmap_model_unload_requires_complete_loading_proof() {
    let incomplete = validate_model_endpoint_request(
        ServerModelEndpointOperation::RequestModelUnload,
        &ModelEndpointLoadingProof::deny_by_default(),
    );
    assert!(incomplete.is_err());
}

#[test]
fn server_api_roadmap_read_only_model_endpoints_skip_loading_proof() {
    for operation in [
        ServerModelEndpointOperation::ListKnownModels,
        ServerModelEndpointOperation::InspectModelMetadata,
        ServerModelEndpointOperation::InspectLoadedInstance,
    ] {
        assert!(!operation.requires_loading_proof());
        let outcome = validate_model_endpoint_request(
            operation,
            &ModelEndpointLoadingProof::deny_by_default(),
        );
        assert!(outcome.is_ok());
    }
}

#[test]
fn server_api_roadmap_rejects_arbitrary_local_model_path() {
    let source = ModelArtifactSource::LocalPath(std::path::PathBuf::from("/models/qwen"));
    let outcome = reject_server_arbitrary_model_path(&source, false);
    assert!(matches!(
        outcome,
        Err(ServerApiRoadmapError::ServerSourcePolicyDenied { .. })
    ));
}

#[test]
fn server_api_roadmap_allows_authorized_local_model_path() {
    let source = ModelArtifactSource::LocalPath(std::path::PathBuf::from("/models/qwen"));
    assert!(reject_server_arbitrary_model_path(&source, true).is_ok());
}

// -- Session endpoints --

#[test]
fn server_api_roadmap_session_rejects_cli_owned_authority_capabilities() {
    for capability in [
        "workspace",
        "git",
        "shell",
        "secrets",
        "tool-call",
        "filesystem",
    ] {
        let outcome = reject_server_session_owned_authority(capability);
        assert!(
            matches!(
                outcome,
                Err(ServerApiRoadmapError::ServerBoundaryViolation { .. })
            ),
            "capability '{capability}' should be rejected"
        );

        let mut capabilities = BTreeSet::new();
        capabilities.insert(capability.to_string());
        let request =
            ServerSessionRequest::new(server_api_roadmap_session_creation_request(capabilities));
        assert!(
            request.is_err(),
            "capability '{capability}' should reject session creation"
        );
    }
}

#[test]
fn server_api_roadmap_session_accepts_inference_scoped_capability() {
    assert!(reject_server_session_owned_authority("generation").is_ok());
    let mut capabilities = BTreeSet::new();
    capabilities.insert("generation".to_string());
    let request =
        ServerSessionRequest::new(server_api_roadmap_session_creation_request(capabilities));
    assert!(request.is_ok());
}

#[test]
fn server_api_roadmap_connection_state_is_separate_from_session_state() {
    let session = InferenceSessionId::new("session-1").unwrap();
    let connection = ServerConnectionState {
        connection: ServerConnectionId::new("conn-1"),
        bound_session: Some(session.clone()),
    };
    let (connection_id, policy, bound_session) =
        server_disconnect_policy(&connection, ServerDisconnectPolicy::CancelActiveGeneration);
    assert_eq!(connection_id.as_str(), "conn-1");
    assert_eq!(policy, ServerDisconnectPolicy::CancelActiveGeneration);
    assert_eq!(bound_session, Some(session));
}

// -- Generation endpoint --

#[test]
fn server_api_roadmap_generation_request_builds_valid_runtime_generation_request() {
    let request = server_api_roadmap_generation_request(false);
    let context = server_api_roadmap_runtime_context();
    let built = build_runtime_generation_request(&request, context).unwrap();
    assert_eq!(built.max_new_tokens, 4);
    assert_eq!(built.streaming, StreamingMode::Disabled);
    assert!(built.session.is_none());
}

#[test]
fn server_api_roadmap_generation_request_targeting_session_omits_model_and_sets_session() {
    let mut request = server_api_roadmap_generation_request(true);
    let session = InferenceSessionId::new("session-1").unwrap();
    request.model_or_session = ServerModelOrSessionRef::Session(session.clone());
    let context = server_api_roadmap_runtime_context();
    let built = build_runtime_generation_request(&request, context).unwrap();
    assert_eq!(built.session, Some(session));
    assert_eq!(built.streaming, StreamingMode::TokenIds);
}

#[test]
fn server_api_roadmap_generation_endpoint_rejects_tool_execution_from_generated_output() {
    let handling = ServerGeneratedTextHandling {
        text: "git push --force".into(),
        executed_as_tool_call: true,
    };
    let outcome = reject_tool_execution_from_generated_output(&handling);
    assert!(matches!(
        outcome,
        Err(ServerApiRoadmapError::ServerBoundaryViolation { .. })
    ));
}

#[test]
fn server_api_roadmap_generation_endpoint_allows_generated_text_that_is_not_executed() {
    let handling = ServerGeneratedTextHandling {
        text: "rm -rf /".into(),
        executed_as_tool_call: false,
    };
    assert!(reject_tool_execution_from_generated_output(&handling).is_ok());
}

// -- Streaming endpoint --

#[test]
fn server_api_roadmap_streaming_preserves_event_ordering() {
    let source = [
        GenerationEventKind::PrefillCompleted,
        GenerationEventKind::DecodeStarted,
        GenerationEventKind::TokenGenerated,
    ];
    let forwarded = [
        GenerationEventKind::PrefillCompleted,
        GenerationEventKind::TokenGenerated,
    ];
    assert!(validate_stream_event_ordering(&source, &forwarded).is_ok());
}

#[test]
fn server_api_roadmap_streaming_rejects_reordered_events() {
    let source = [
        GenerationEventKind::PrefillCompleted,
        GenerationEventKind::TokenGenerated,
    ];
    let forwarded = [
        GenerationEventKind::TokenGenerated,
        GenerationEventKind::PrefillCompleted,
    ];
    let outcome = validate_stream_event_ordering(&source, &forwarded);
    assert!(matches!(
        outcome,
        Err(ServerApiRoadmapError::ServerStreamInterrupted { .. })
    ));
}

#[test]
fn server_api_roadmap_streaming_rejects_raw_payload_kinds_by_default() {
    for payload_kind in SERVER_STREAM_FORBIDDEN_PAYLOAD_KINDS {
        let outcome = reject_raw_stream_payload(payload_kind);
        assert!(
            matches!(
                outcome,
                Err(ServerApiRoadmapError::ServerStreamUnavailable { .. })
            ),
            "payload kind '{payload_kind}' should be rejected"
        );
    }
    assert!(reject_raw_stream_payload("token-id").is_ok());
}

#[test]
fn server_api_roadmap_stream_event_carries_only_redacted_metadata() {
    let event = ServerStreamEvent::new(GenerationEventKind::TokenGenerated)
        .with_redacted_metadata("path", "/var/cache/magnetar/model");
    assert!(!event.redacted_metadata.get("path").unwrap().contains('/'));
}

// -- Cancellation endpoint --

#[test]
fn server_api_roadmap_cancellation_calls_runtime_cancellation() {
    let token = CancellationToken::new(GenerationRequestId::new("cancel-fixture").unwrap());
    let outcome =
        server_cancellation_calls_runtime_cancellation(&token, CancellationStage::Decode, false);
    assert_eq!(outcome, CancellationOutcome::Cancelled);
}

#[test]
fn server_api_roadmap_cancellation_reports_limitation_for_provider_execution_stage() {
    let token = CancellationToken::new(GenerationRequestId::new("cancel-fixture-2").unwrap());
    let outcome = server_cancellation_calls_runtime_cancellation(
        &token,
        CancellationStage::ProviderExecution,
        false,
    );
    assert!(matches!(
        outcome,
        CancellationOutcome::LimitationReported { .. }
    ));
}

// -- Diagnostics endpoint --

#[test]
fn server_api_roadmap_diagnostics_summary_is_redacted() {
    let runtime_diagnostics = RuntimeDiagnostics {
        model_instance_count: 1,
        ready_model_instance_count: 1,
        active_session_count: 2,
        provider_count: 1,
        device_count: 1,
        kernel_advertisement_count: 3,
        memory_pressure: MemoryPressureLevel::Low,
        kv_cache_count: 0,
        prefix_cache_entry_count: 0,
        model_resolution_status: None,
        model_loading_status: None,
        operator_missing_count: 0,
        tokenizer_compatible: None,
        queued_admission_count: 0,
        redacted: true,
    };
    let health = ServerHealthStatus::alive();
    let readiness = ServerReadinessStatus::ready();
    let summary = server_diagnostics_summary(
        &runtime_diagnostics,
        &health,
        &readiness,
        Some("provider handle=0xdeadbeef ready"),
        4,
        1,
        vec!["server-generation-failed".into()],
    );
    assert!(summary.redacted);
    assert!(
        !summary
            .provider_readiness_summary
            .as_deref()
            .unwrap()
            .contains("0xdeadbeef")
    );
    assert_eq!(summary.active_session_count, 2);
}

// -- OpenAI-compatible facade placeholder --

#[test]
fn server_api_roadmap_openai_facade_rejects_unsupported_field_per_policy() {
    let outcome = handle_openai_unsupported_field(
        OpenAiCompatibilityPolicy::RejectUnsupportedField,
        "logprobs",
    );
    assert!(matches!(
        outcome,
        Err(ServerApiRoadmapError::ServerRequestInvalid { .. })
    ));
}

#[test]
fn server_api_roadmap_openai_facade_ignores_unsupported_field_per_policy() {
    let outcome = handle_openai_unsupported_field(
        OpenAiCompatibilityPolicy::IgnoreUnsupportedField,
        "logprobs",
    );
    assert!(outcome.is_ok());
}

#[test]
fn server_api_roadmap_openai_facade_rejects_tool_call_execution() {
    let outcome = reject_openai_tool_call_execution(true, true);
    assert!(matches!(
        outcome,
        Err(ServerApiRoadmapError::ServerBoundaryViolation { .. })
    ));
    assert!(reject_openai_tool_call_execution(true, false).is_ok());
}

#[test]
fn server_api_roadmap_openai_facade_maps_to_generation_api_request() {
    let request = server_api_roadmap_generation_request(false);
    let context = server_api_roadmap_runtime_context();
    let core = build_runtime_generation_request(&request, context).unwrap();
    let mapped =
        openai_facade_maps_to_generation_api_request(core, SessionRedactionPolicy::RedactRawInputs);
    assert_eq!(mapped.privacy, SessionRedactionPolicy::RedactRawInputs);
}

// -- Authentication boundary --

#[test]
fn server_api_roadmap_authenticated_request_carries_no_credential_type() {
    let authenticated = AuthenticatedServerRequest::from_authenticated(true);
    assert!(authenticated.is_ok());
}

#[test]
fn server_api_roadmap_authentication_required_when_not_authenticated() {
    let outcome = AuthenticatedServerRequest::from_authenticated(false);
    assert!(matches!(
        outcome,
        Err(ServerApiRoadmapError::ServerAuthenticationRequired)
    ));
}

#[test]
fn server_api_roadmap_rejects_credential_in_diagnostics() {
    let mut metadata = BTreeMap::new();
    metadata.insert("api_key".to_string(), "secret-value".to_string());
    let outcome = reject_credential_in_server_diagnostics(&metadata);
    assert!(matches!(
        outcome,
        Err(ServerApiRoadmapError::ServerAuthenticationFailed { .. })
    ));
}

#[test]
fn server_api_roadmap_redacts_diagnostic_message() {
    let redacted = redact_server_diagnostic("provider handle=0xdeadbeef failed");
    assert!(!redacted.contains("0xdeadbeef"));
}

// -- Authorization boundary --

#[test]
fn server_api_roadmap_authorization_denied_when_server_denies() {
    let decision = ServerAuthorizationDecision {
        scope: ServerAuthorizationScope::GenerationLimits,
        server_authorized: false,
    };
    let outcome = authorize_server_request(&decision, true);
    assert!(matches!(
        outcome,
        Err(ServerApiRoadmapError::ServerAuthorizationDenied { .. })
    ));
}

#[test]
fn server_api_roadmap_authorization_does_not_bypass_runtime_policy() {
    let decision = ServerAuthorizationDecision {
        scope: ServerAuthorizationScope::GenerationLimits,
        server_authorized: true,
    };
    let outcome = authorize_server_request(&decision, false);
    assert!(matches!(
        outcome,
        Err(ServerApiRoadmapError::ServerAuthorizationDenied { .. })
    ));
}

#[test]
fn server_api_roadmap_authorization_succeeds_when_both_allow() {
    let decision = ServerAuthorizationDecision {
        scope: ServerAuthorizationScope::GenerationLimits,
        server_authorized: true,
    };
    assert!(authorize_server_request(&decision, true).is_ok());
}

// -- Admission and rate policy --

#[test]
fn server_api_roadmap_admission_denied_by_default() {
    let outcome = evaluate_server_admission(
        &ServerAdmissionLimits::deny_by_default(),
        &ServerAdmissionState::default(),
    );
    assert!(matches!(
        outcome,
        Err(ServerApiRoadmapError::ServerAdmissionRejected { .. })
    ));
}

#[test]
fn server_api_roadmap_admission_allows_within_configured_limits() {
    let limits = ServerAdmissionLimits {
        max_concurrent_requests: 4,
        max_queued_requests: 4,
        max_tokens_per_request: 1024,
        max_sessions: 4,
        max_loaded_models: 2,
        memory_budget_bytes: 1_000_000,
        max_streaming_connections: 4,
        max_request_body_bytes: 1_000_000,
        max_prompt_bytes: 100_000,
        max_source_cache_operations: 4,
    };
    let state = ServerAdmissionState {
        concurrent_requests: 1,
        queued_requests: 0,
        requested_tokens: 128,
        active_sessions: 1,
        loaded_models: 1,
        memory_used_bytes: 1_000,
        streaming_connections: 0,
        request_body_bytes: 512,
        prompt_bytes: 256,
        source_cache_operations_in_flight: 0,
    };
    assert!(evaluate_server_admission(&limits, &state).is_ok());
}

// -- Source and cache boundary --

#[test]
fn server_api_roadmap_rejects_arbitrary_download_during_generation() {
    let outcome = reject_arbitrary_download_during_generation("https://example.com/model.gguf");
    assert!(matches!(
        outcome,
        Err(ServerApiRoadmapError::ServerSourcePolicyDenied { .. })
    ));
}

#[test]
fn server_api_roadmap_allows_non_network_model_reference_during_generation() {
    assert!(reject_arbitrary_download_during_generation("qwen-local").is_ok());
}

// -- Filesystem boundary --

#[test]
fn server_api_roadmap_rejects_arbitrary_filesystem_path() {
    let outcome = reject_arbitrary_filesystem_path("/etc/passwd", false);
    assert!(matches!(
        outcome,
        Err(ServerApiRoadmapError::ServerBoundaryViolation { .. })
    ));
}

#[test]
fn server_api_roadmap_allows_authorized_filesystem_path() {
    assert!(reject_arbitrary_filesystem_path("/models/qwen/model.gguf", true).is_ok());
}

// -- Tool/Shell/Git boundary --

#[test]
fn server_api_roadmap_rejects_tool_shell_git_capabilities() {
    for capability in ["tool-call", "shell", "process", "git"] {
        let outcome = reject_server_tool_shell_git_execution(capability);
        assert!(
            matches!(
                outcome,
                Err(ServerApiRoadmapError::ServerBoundaryViolation { .. })
            ),
            "capability '{capability}' should be rejected"
        );
    }
}

#[test]
fn server_api_roadmap_allows_ordinary_inference_capability() {
    assert!(reject_server_tool_shell_git_execution("generation").is_ok());
}

// -- Error model --

#[test]
fn server_api_roadmap_error_id_matches_exact_kebab_string_for_every_variant() {
    let cases: Vec<(ServerApiRoadmapError, &str)> = vec![
        (
            ServerApiRoadmapError::ServerApiUnavailable { reason: "x".into() },
            "server-api-unavailable",
        ),
        (
            ServerApiRoadmapError::ServerRequestInvalid { reason: "x".into() },
            "server-request-invalid",
        ),
        (
            ServerApiRoadmapError::ServerRequestTooLarge { reason: "x".into() },
            "server-request-too-large",
        ),
        (
            ServerApiRoadmapError::ServerAuthenticationRequired,
            "server-authentication-required",
        ),
        (
            ServerApiRoadmapError::ServerAuthenticationFailed { reason: "x".into() },
            "server-authentication-failed",
        ),
        (
            ServerApiRoadmapError::ServerAuthorizationDenied { scope: "x".into() },
            "server-authorization-denied",
        ),
        (
            ServerApiRoadmapError::ServerRateLimited { reason: "x".into() },
            "server-rate-limited",
        ),
        (
            ServerApiRoadmapError::ServerAdmissionRejected { reason: "x".into() },
            "server-admission-rejected",
        ),
        (
            ServerApiRoadmapError::ServerStreamUnavailable { reason: "x".into() },
            "server-stream-unavailable",
        ),
        (
            ServerApiRoadmapError::ServerStreamInterrupted { reason: "x".into() },
            "server-stream-interrupted",
        ),
        (
            ServerApiRoadmapError::ServerCancellationFailed { reason: "x".into() },
            "server-cancellation-failed",
        ),
        (
            ServerApiRoadmapError::ServerModelNotFound { model: "x".into() },
            "server-model-not-found",
        ),
        (
            ServerApiRoadmapError::ServerModelLoadFailed {
                reason: "x".into(),
                runtime_cause: None,
            },
            "server-model-load-failed",
        ),
        (
            ServerApiRoadmapError::ServerSessionNotFound {
                session: "x".into(),
            },
            "server-session-not-found",
        ),
        (
            ServerApiRoadmapError::ServerGenerationFailed {
                reason: "x".into(),
                runtime_cause: None,
            },
            "server-generation-failed",
        ),
        (
            ServerApiRoadmapError::ServerDiagnosticsRedacted,
            "server-diagnostics-redacted",
        ),
        (
            ServerApiRoadmapError::ServerSourcePolicyDenied { reason: "x".into() },
            "server-source-policy-denied",
        ),
        (
            ServerApiRoadmapError::ServerCachePolicyDenied { reason: "x".into() },
            "server-cache-policy-denied",
        ),
        (
            ServerApiRoadmapError::ServerBoundaryViolation {
                capability: "x".into(),
            },
            "server-boundary-violation",
        ),
        (
            ServerApiRoadmapError::InternalServerApiError { reason: "x".into() },
            "internal-server-api-error",
        ),
    ];
    assert_eq!(cases.len(), 20, "expected exactly 20 error categories");
    for (error, expected_id) in cases {
        assert_eq!(error.id(), expected_id);
        assert!(
            !error.to_string().is_empty(),
            "{expected_id} rendered empty"
        );
    }
}

#[test]
fn server_api_roadmap_error_preserves_wrapped_runtime_cause() {
    let source = InferenceApiError::ModelLoadingFailed {
        reason: "example".into(),
    };
    let wrapped = ServerApiRoadmapError::model_load_failed_from_runtime(source.clone());
    assert_eq!(wrapped.runtime_cause(), Some(&source));
    assert!(wrapped.to_string().contains("example"));
}

#[test]
fn server_api_roadmap_generation_failed_from_runtime_preserves_cause() {
    let source = InferenceApiError::GenerationFailed {
        reason: "boom".into(),
    };
    let wrapped = ServerApiRoadmapError::generation_failed_from_runtime(source.clone());
    assert_eq!(wrapped.runtime_cause(), Some(&source));
    assert_eq!(wrapped.id(), "server-generation-failed");
}

#[test]
fn server_api_roadmap_error_without_runtime_cause_returns_none() {
    let error = ServerApiRoadmapError::ServerRequestInvalid {
        reason: "bad".into(),
    };
    assert_eq!(error.runtime_cause(), None);
}

// -- Observability --

#[test]
fn server_api_roadmap_observation_redacts_metadata() {
    let observation =
        ServerApiRoadmapObservation::new(ServerApiRoadmapObservationKind::RequestReceived)
            .with_endpoint("generate")
            .with_redacted_metadata("path", "/var/cache/magnetar/model");
    assert_eq!(observation.endpoint.as_deref(), Some("generate"));
    assert!(
        !observation
            .redacted_metadata
            .get("path")
            .unwrap()
            .contains('/')
    );
}

#[test]
fn server_api_roadmap_observation_kinds_cover_all_18_categories() {
    let kinds = [
        ServerApiRoadmapObservationKind::ServerStarted,
        ServerApiRoadmapObservationKind::ServerStopped,
        ServerApiRoadmapObservationKind::RequestReceived,
        ServerApiRoadmapObservationKind::RequestRejected,
        ServerApiRoadmapObservationKind::RequestAuthorized,
        ServerApiRoadmapObservationKind::RuntimeRequestSubmitted,
        ServerApiRoadmapObservationKind::StreamOpened,
        ServerApiRoadmapObservationKind::StreamClosed,
        ServerApiRoadmapObservationKind::StreamInterrupted,
        ServerApiRoadmapObservationKind::GenerationCompleted,
        ServerApiRoadmapObservationKind::GenerationFailed,
        ServerApiRoadmapObservationKind::CancellationRequested,
        ServerApiRoadmapObservationKind::DiagnosticsRequested,
        ServerApiRoadmapObservationKind::ModelEndpointUsed,
        ServerApiRoadmapObservationKind::SessionEndpointUsed,
        ServerApiRoadmapObservationKind::RateLimitHit,
        ServerApiRoadmapObservationKind::AdmissionRejected,
        ServerApiRoadmapObservationKind::BoundaryViolationDetected,
    ];
    assert_eq!(kinds.len(), 18);
    let unique: BTreeSet<ServerApiRoadmapObservationKind> = kinds.into_iter().collect();
    assert_eq!(unique.len(), 18, "observation kinds must be distinct");
}

// -- Conformance --

#[test]
fn server_api_roadmap_conformance_report_is_conformant() {
    let report = run_server_api_roadmap_conformance();
    assert!(!report.results.is_empty());
    for result in &report.results {
        assert!(
            result.passed,
            "{} failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
    assert!(report.is_conformant());
}

#[test]
fn server_api_roadmap_version_constant_is_set() {
    assert_eq!(SERVER_API_ROADMAP_VERSION, "0.1.0");
}

// ---------------------------------------------------------------------
// release_packaging
// ---------------------------------------------------------------------

#[test]
fn release_version_displays_as_semver() {
    assert_eq!(ReleaseVersion::new(0, 1, 0).to_string(), "0.1.0");
    assert!(ReleaseVersion::new(0, 1, 0).is_pre_1_0());
    assert!(!ReleaseVersion::new(1, 0, 0).is_pre_1_0());
}

#[test]
fn breaking_change_is_rejected_in_patch_release() {
    let outcome = evaluate_version_bump(
        ReleaseVersion::new(0, 1, 0),
        ReleaseVersion::new(0, 1, 1),
        true,
        true,
    );
    assert!(matches!(
        outcome,
        Err(ReleasePackagingError::BreakingChangeInPatchRelease { .. })
    ));
}

#[test]
fn undocumented_breaking_change_is_rejected() {
    let outcome = evaluate_version_bump(
        ReleaseVersion::new(0, 1, 0),
        ReleaseVersion::new(0, 2, 0),
        true,
        false,
    );
    assert!(matches!(
        outcome,
        Err(ReleasePackagingError::UndocumentedBreakingChange { .. })
    ));
}

#[test]
fn documented_breaking_change_is_allowed_pre_1_0_in_minor_bump() {
    let outcome = evaluate_version_bump(
        ReleaseVersion::new(0, 1, 0),
        ReleaseVersion::new(0, 2, 0),
        true,
        true,
    );
    assert_eq!(outcome, Ok(ReleaseVersionBumpKind::Minor));
}

#[test]
fn crate_dependency_across_independent_versions_requires_documentation() {
    let dependent = CrateVersionMetadata {
        crate_name: "magnetar-cli".into(),
        version: ReleaseVersion::new(0, 1, 0),
        shares_workspace_version: false,
    };
    let dependency = CrateVersionMetadata {
        crate_name: "magnetar-runtime".into(),
        version: ReleaseVersion::new(0, 1, 0),
        shares_workspace_version: false,
    };
    assert!(validate_crate_dependency_compatibility(&dependent, &dependency, None).is_err());
    assert!(
        validate_crate_dependency_compatibility(&dependent, &dependency, Some("compatible"))
            .is_ok()
    );
}

#[test]
fn shared_workspace_version_crates_do_not_require_documentation() {
    let dependent = CrateVersionMetadata {
        crate_name: "magnetar-cli".into(),
        version: ReleaseVersion::new(0, 1, 0),
        shares_workspace_version: true,
    };
    let dependency = CrateVersionMetadata {
        crate_name: "magnetar-runtime".into(),
        version: ReleaseVersion::new(0, 1, 0),
        shares_workspace_version: true,
    };
    assert!(validate_crate_dependency_compatibility(&dependent, &dependency, None).is_ok());
}

#[test]
fn release_binary_version_report_includes_all_fields() {
    let report = build_release_binary_version_report(
        ReleaseVersion::new(0, 1, 0),
        vec!["reference-cpu-provider".into()],
        "release",
        Some("abc1234".into()),
    );
    assert_eq!(report.binary_version, "0.1.0");
    assert_eq!(report.runtime_crate_version, MAGNETAR_RUNTIME_VERSION);
    assert_eq!(
        report.openspec_baseline_version,
        RELEASE_PACKAGING_POLICY_VERSION
    );
    assert_eq!(report.wit_contract_versions.len(), 2);
    assert_eq!(report.enabled_feature_flags, vec!["reference-cpu-provider"]);
    assert_eq!(report.build_profile, "release");
    assert_eq!(report.commit_hash.as_deref(), Some("abc1234"));
    assert!(report.conformance_suite_version.is_some());
}

#[test]
fn required_wit_version_bump_matches_change_kind() {
    assert_eq!(
        required_wit_version_bump(WitVersionChangeKind::Breaking),
        ReleaseVersionBumpKind::Major
    );
    assert_eq!(
        required_wit_version_bump(WitVersionChangeKind::Additive),
        ReleaseVersionBumpKind::Minor
    );
    assert_eq!(
        required_wit_version_bump(WitVersionChangeKind::DocumentationOnly),
        ReleaseVersionBumpKind::Patch
    );
}

#[test]
fn breaking_wit_change_requires_major_bump() {
    assert!(
        validate_wit_version_bump(
            WitVersionChangeKind::Breaking,
            ReleaseVersionBumpKind::Minor,
            "magnetar:compute",
        )
        .is_err()
    );
    assert!(
        validate_wit_version_bump(
            WitVersionChangeKind::Breaking,
            ReleaseVersionBumpKind::Major,
            "magnetar:compute",
        )
        .is_ok()
    );
}

#[test]
fn supported_wit_version_matrix_lists_declared_interfaces() {
    let matrix = SupportedWitVersionMatrix::from_interfaces(&release_wit_contract_versions());
    assert_eq!(
        matrix.supported.get("magnetar:compute").map(String::as_str),
        Some("2.0.0")
    );
    assert_eq!(
        matrix
            .supported
            .get("magnetar:observability")
            .map(String::as_str),
        Some("1.0.0")
    );
}

#[test]
fn openspec_baseline_declaration_requires_accepted_changes_and_status() {
    let empty = OpenSpecBaselineDeclaration::default();
    assert!(empty.validate().is_err());

    let complete = OpenSpecBaselineDeclaration {
        accepted_changes: vec!["define-release-packaging-and-versioning-policy".into()],
        validation_status: Some("valid".into()),
        ..Default::default()
    };
    assert!(complete.validate().is_ok());
}

#[test]
fn freeze_denies_semantic_change_but_allows_documentation_clarification() {
    assert!(
        reject_change_after_freeze(
            ReleaseFreezeState::Frozen,
            ReleaseFreezeChangeKind::SemanticContractChange,
        )
        .is_err()
    );
    assert!(
        reject_change_after_freeze(
            ReleaseFreezeState::Frozen,
            ReleaseFreezeChangeKind::DocumentationClarification,
        )
        .is_ok()
    );
    assert!(
        reject_change_after_freeze(
            ReleaseFreezeState::Open,
            ReleaseFreezeChangeKind::SemanticContractChange,
        )
        .is_ok()
    );
}

#[test]
fn experimental_feature_flag_cannot_be_enabled_by_default() {
    let flag = ReleaseFeatureFlag {
        name: "webgpu-provider".into(),
        class: ReleaseFeatureFlagClass::Experimental,
        enabled_by_default: true,
    };
    assert!(reject_experimental_flag_enabled_by_default(&flag).is_err());

    let disabled = ReleaseFeatureFlag {
        enabled_by_default: false,
        ..flag
    };
    assert!(reject_experimental_flag_enabled_by_default(&disabled).is_ok());
}

#[test]
fn only_reference_cpu_provider_is_required_for_v0_1() {
    let flags = provider_feature_flags();
    assert_eq!(flags.len(), 7);
    assert!(validate_provider_feature_flags_for_v0_1(&flags).is_ok());

    let mut bad_flags = flags.clone();
    bad_flags[1].enabled_by_default = true; // optimized-cpu-provider
    assert!(validate_provider_feature_flags_for_v0_1(&bad_flags).is_err());
}

#[test]
fn component_engine_flags_are_disabled_by_default() {
    for flag in component_engine_feature_flags() {
        assert!(!flag.enabled_by_default);
    }
}

#[test]
fn browser_target_never_requires_wasmtime() {
    let browser = ReleasePlatformTarget {
        triple: "wasm32-unknown-unknown".into(),
        required_by_ci: false,
        check_only: true,
        is_browser_like: true,
    };
    assert!(
        reject_wasmtime_required_for_browser(&browser, &["wasmtime-component-engine"]).is_err()
    );
    assert!(reject_wasmtime_required_for_browser(&browser, &[]).is_ok());
}

#[test]
fn native_target_may_require_wasmtime() {
    let native = ReleasePlatformTarget {
        triple: "x86_64-unknown-linux-gnu".into(),
        required_by_ci: true,
        check_only: false,
        is_browser_like: false,
    };
    assert!(reject_wasmtime_required_for_browser(&native, &["wasmtime-component-engine"]).is_ok());
}

#[test]
fn release_platform_targets_include_ci_required_and_check_only_entries() {
    let targets = release_platform_targets();
    assert!(
        targets
            .iter()
            .any(|target| target.required_by_ci && !target.check_only)
    );
    assert!(
        targets
            .iter()
            .any(|target| target.check_only && !target.required_by_ci)
    );
}

#[test]
fn unsupported_targets_are_reported() {
    let supported = release_platform_targets();
    let candidates = ["x86_64-unknown-linux-gnu", "riscv64gc-unknown-linux-gnu"];
    let unsupported = unsupported_targets(&supported, &candidates);
    assert_eq!(unsupported, vec!["riscv64gc-unknown-linux-gnu"]);
}

#[test]
fn release_artifact_manifest_requires_every_kind_present_or_not_applicable() {
    let manifest = ReleaseArtifactManifest::default();
    assert!(manifest.validate().is_err());

    let mut manifest = ReleaseArtifactManifest::default();
    for kind in RELEASE_ARTIFACT_KINDS {
        manifest.set(*kind, ReleaseArtifactStatus::NotApplicable);
    }
    assert!(manifest.validate().is_ok());
}

#[test]
fn artifact_checksum_rejects_empty_digest() {
    assert!(ArtifactChecksum::new("magnetar-cli", ChecksumAlgorithm::Sha256, "").is_err());
    assert!(ArtifactChecksum::new("magnetar-cli", ChecksumAlgorithm::Sha256, "deadbeef").is_ok());
}

#[test]
fn changelog_must_be_non_empty() {
    assert!(ReleaseChangelog::default().validate().is_err());
    let changelog = ReleaseChangelog {
        entries: vec![ChangelogEntry {
            kind: ChangelogEntryKind::AddedContract,
            description: "release packaging policy".into(),
        }],
    };
    assert!(changelog.validate().is_ok());
}

#[test]
fn compatibility_matrix_requires_every_dimension_declared() {
    let mut matrix = ReleaseCompatibilityMatrix::default();
    assert!(matrix.validate().is_err());
    for dimension in COMPATIBILITY_DIMENSIONS {
        matrix.set(*dimension, CompatibilityStatus::StableForBaseline);
    }
    assert!(matrix.validate().is_ok());
}

#[test]
fn v0_1_compatibility_matrix_marks_provider_abi_unstable() {
    let matrix = v0_1_compatibility_matrix();
    assert!(matrix.validate().is_ok());
    assert_eq!(
        matrix.status.get(compatibility_dimension_id(
            CompatibilityDimension::ProviderAbi
        )),
        Some(&CompatibilityStatus::Unstable)
    );
    assert_eq!(
        matrix.status.get(compatibility_dimension_id(
            CompatibilityDimension::RustPublicApi
        )),
        Some(&CompatibilityStatus::StableForBaseline)
    );
}

#[test]
fn release_public_api_denies_raw_handle_surfaces() {
    for surface in [
        "raw-provider-handle",
        "raw-device-handle",
        "raw-kernel-handle",
        "raw-tensor-pointer",
        "raw-memory-pointer",
        "raw-kv-cache",
        "raw-model-weight",
    ] {
        assert!(reject_release_public_api_handle_exposure(surface).is_err());
    }
    assert!(reject_release_public_api_handle_exposure("generation").is_ok());
}

#[test]
fn release_conformance_versions_reuse_existing_suite_constants() {
    let versions = ReleaseConformanceVersions::default();
    assert_eq!(
        versions.provider_conformance_suite_version,
        PROVIDER_CONFORMANCE_SUITE_VERSION
    );
    assert_eq!(
        versions.first_operator_scope_conformance_version,
        FIRST_OPERATOR_SCOPE_VERSION
    );
    assert_eq!(
        versions.qwen_baseline_conformance_version,
        QWEN_BASELINE_CONTRACT_VERSION.to_string()
    );
    assert_eq!(versions.e2e_local_conformance_version, E2E_SUITE_VERSION);
}

#[test]
fn release_may_publish_stable_requires_every_gate_present_and_passed() {
    let missing: Vec<ReleaseGateResult> = REQUIRED_RELEASE_GATES[..3]
        .iter()
        .map(|gate| ReleaseGateResult {
            gate: *gate,
            passed: true,
        })
        .collect();
    assert!(matches!(
        release_may_publish_stable(&missing),
        Err(ReleasePackagingError::ReleaseGateMissing { .. })
    ));

    let complete: Vec<ReleaseGateResult> = REQUIRED_RELEASE_GATES
        .iter()
        .map(|gate| ReleaseGateResult {
            gate: *gate,
            passed: true,
        })
        .collect();
    assert!(release_may_publish_stable(&complete).is_ok());

    let mut failing = complete.clone();
    failing[0].passed = false;
    assert!(matches!(
        release_may_publish_stable(&failing),
        Err(ReleasePackagingError::ReleaseGateFailed { .. })
    ));
}

#[test]
fn release_candidate_tags_are_never_stable() {
    assert!(!ReleaseCandidateTag::Alpha.is_stable());
    assert!(!ReleaseCandidateTag::Beta.is_stable());
    assert!(!ReleaseCandidateTag::Rc(1).is_stable());
    assert_eq!(ReleaseCandidateTag::Rc(1).to_string(), "-rc.1");
}

#[test]
fn release_candidate_manifest_requires_frozen_baseline_and_conformance_report() {
    let incomplete = ReleaseCandidateManifest {
        tag: ReleaseCandidateTag::Rc(1),
        frozen_openspec_baseline: false,
        conformance_report_included: true,
        known_failures: Vec::new(),
        release_notes_draft: true,
    };
    assert!(incomplete.validate().is_err());

    let complete = ReleaseCandidateManifest {
        frozen_openspec_baseline: true,
        ..incomplete
    };
    assert!(complete.validate().is_ok());
}

#[test]
fn failed_candidate_may_be_tagged_pre_release() {
    let failing = vec![ReleaseGateResult {
        gate: ReleaseGate::Formatting,
        passed: false,
    }];
    let tag = allow_failed_candidate_as_pre_release(&failing, ReleaseCandidateTag::Rc(2));
    assert_eq!(tag, Ok(ReleaseCandidateTag::Rc(2)));
}

#[test]
fn build_metadata_redacts_secret_shaped_keys_and_local_paths() {
    assert_eq!(
        redact_build_metadata("GITHUB_TOKEN", "ghp_example"),
        "[redacted build metadata]"
    );
    assert_eq!(
        redact_build_metadata("workspace_root", "/home/user/project"),
        "[redacted backend diagnostic]"
    );
    assert_eq!(
        redact_build_metadata("target_triple", "x86_64-unknown-linux-gnu"),
        "x86_64-unknown-linux-gnu"
    );
}

#[test]
fn documentation_checklist_requires_known_limitations() {
    let empty = ReleaseDocumentationChecklist::default();
    assert!(empty.validate().is_err());
    let documented = ReleaseDocumentationChecklist {
        known_limitations: true,
        ..empty
    };
    assert!(documented.validate().is_ok());
}

#[test]
fn deferred_roadmap_features_are_never_included_baseline() {
    for feature in [
        "cuda",
        "metal",
        "openvino",
        "qnn",
        "webgpu",
        "server-api-implementation",
    ] {
        assert_eq!(
            classify_publishing_boundary(feature),
            PublishingBoundaryCategory::DeferredRoadmap
        );
        assert!(reject_roadmap_feature_as_guarantee(feature, true).is_err());
        assert!(reject_roadmap_feature_as_guarantee(feature, false).is_ok());
    }
    assert_eq!(
        classify_publishing_boundary("reference-cpu-provider"),
        PublishingBoundaryCategory::IncludedBaseline
    );
    assert!(reject_roadmap_feature_as_guarantee("reference-cpu-provider", true).is_ok());
}

#[test]
fn release_packaging_conformance_report_is_conformant() {
    let report = run_release_packaging_conformance();
    assert!(!report.results.is_empty());
    for result in &report.results {
        assert!(
            result.passed,
            "{} failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
    assert!(report.is_conformant());
}

#[test]
fn release_packaging_policy_version_constant_is_set() {
    assert_eq!(RELEASE_PACKAGING_POLICY_VERSION, "0.1.0");
}

// ---------------------------------------------------------------------
// release_security
// ---------------------------------------------------------------------

#[test]
fn release_security_policy_version_constant_is_set() {
    assert_eq!(RELEASE_SECURITY_POLICY_VERSION, "0.1.0");
}

#[test]
fn hardened_claim_is_rejected_for_excluded_feature_but_allowed_for_baseline() {
    for feature in RELEASE_SECURITY_SCOPE_EXCLUDED_FROM_HARDENED_CLAIMS {
        assert!(reject_hardened_security_claim_for_excluded_feature(feature, true).is_err());
        assert!(reject_hardened_security_claim_for_excluded_feature(feature, false).is_ok());
    }
    assert!(
        reject_hardened_security_claim_for_excluded_feature("reference-cpu-provider", true).is_ok()
    );
    assert!(!RELEASE_SECURITY_SCOPE_INCLUDED.is_empty());
}

#[test]
fn dependency_audit_blocks_stable_release_on_unmitigated_critical_advisory() {
    let mut report = DependencyAuditReport {
        advisories: vec![DependencyAdvisory {
            crate_name: "example-crate".into(),
            advisory_id: "RUSTSEC-0000-0000".into(),
            severity: DependencyAdvisorySeverity::Critical,
            mitigated: false,
            mitigation: None,
        }],
        ..Default::default()
    };
    assert!(matches!(
        report.validate_for_stable_release(),
        Err(ReleaseSecurityError::CriticalAdvisoryUnmitigated { .. })
    ));

    report.advisories[0].mitigated = true;
    report.advisories[0].mitigation = Some("vendored patch".into());
    assert!(report.validate_for_stable_release().is_ok());

    report.advisories[0].severity = DependencyAdvisorySeverity::Low;
    report.advisories[0].mitigated = false;
    assert!(report.validate_for_stable_release().is_ok());
}

#[test]
fn license_audit_blocks_stable_release_on_unapproved_incompatible_or_unknown_license() {
    for status in [
        LicenseAuditStatus::Incompatible,
        LicenseAuditStatus::Unknown,
    ] {
        let report = LicenseAuditReport {
            licenses: vec![DependencyLicense {
                crate_name: "example-crate".into(),
                spdx: None,
                status,
                exception_approved: false,
            }],
            ..Default::default()
        };
        assert!(matches!(
            report.validate_for_stable_release(),
            Err(ReleaseSecurityError::IncompatibleLicenseUnapproved { .. })
        ));
    }

    let approved = LicenseAuditReport {
        licenses: vec![DependencyLicense {
            crate_name: "example-crate".into(),
            spdx: None,
            status: LicenseAuditStatus::Unknown,
            exception_approved: true,
        }],
        ..Default::default()
    };
    assert!(approved.validate_for_stable_release().is_ok());

    let missing_metadata_not_blocking = LicenseAuditReport {
        licenses: vec![DependencyLicense {
            crate_name: "example-crate".into(),
            spdx: None,
            status: LicenseAuditStatus::MissingMetadata,
            exception_approved: false,
        }],
        ..Default::default()
    };
    assert!(
        missing_metadata_not_blocking
            .validate_for_stable_release()
            .is_ok()
    );
}

#[test]
fn sbom_manifest_requires_generation_or_documented_limitation() {
    let missing = SbomManifest {
        availability: SbomAvailability::Missing,
        ..Default::default()
    };
    assert!(missing.validate().is_err());

    let undocumented_placeholder = SbomManifest {
        availability: SbomAvailability::PlaceholderDocumented,
        ..Default::default()
    };
    assert!(undocumented_placeholder.validate().is_err());

    let documented = SbomManifest {
        availability: SbomAvailability::PlaceholderDocumented,
        limitation_note: Some("SBOM generation is not implemented for v0.1".into()),
        ..Default::default()
    };
    assert!(documented.validate().is_ok());

    let generated = SbomManifest {
        availability: SbomAvailability::Generated,
        entries: vec![SbomEntry {
            package_name: "magnetar-runtime".into(),
            package_version: "0.1.0".into(),
            licenses: vec!["MIT".into()],
            source_repository: None,
        }],
        build_target: Some("x86_64-unknown-linux-gnu".into()),
        feature_flags: vec!["reference-cpu-provider".into()],
        ..Default::default()
    };
    assert!(generated.validate().is_ok());
    assert_eq!(generated.feature_flags, vec!["reference-cpu-provider"]);
}

#[test]
fn checksum_mismatch_against_final_artifact_is_rejected() {
    let checksum =
        ArtifactChecksum::new("magnetar-cli", ChecksumAlgorithm::Sha256, "deadbeef").unwrap();
    assert!(matches!(
        verify_checksum_matches_final_artifact(&checksum, "other-digest"),
        Err(ReleaseSecurityError::ChecksumMismatch { .. })
    ));
    assert!(verify_checksum_matches_final_artifact(&checksum, "deadbeef").is_ok());
}

#[test]
fn signature_absence_must_be_documented() {
    assert!(matches!(
        validate_signature_status(SignatureStatus::NotImplementedUndocumented),
        Err(ReleaseSecurityError::SignatureAbsenceUndocumented)
    ));
    assert!(validate_signature_status(SignatureStatus::NotImplementedDocumented).is_ok());
    assert!(validate_signature_status(SignatureStatus::Implemented).is_ok());
}

#[test]
fn provenance_rejects_secret_or_path_shaped_fields() {
    let leaky_path = ReleaseProvenance {
        build_target: Some("/home/user/workspace".into()),
        ..Default::default()
    };
    assert!(matches!(
        leaky_path.validate(),
        Err(ReleaseSecurityError::ProvenanceContainsSecretOrLocalPath { .. })
    ));

    let clean = ReleaseProvenance {
        source_commit: Some("abc1234".into()),
        release_tag: Some("v0.1.0".into()),
        build_target: Some("x86_64-unknown-linux-gnu".into()),
        ..Default::default()
    };
    assert!(clean.validate().is_ok());
}

#[test]
fn reproducibility_report_requires_status_or_documented_limitations() {
    assert!(ReproducibilityReport::default().validate().is_err());

    let undocumented_partial = ReproducibilityReport {
        status: ReproducibilityStatus::PartiallyReproducible,
        limitations: Vec::new(),
    };
    assert!(undocumented_partial.validate().is_err());

    let documented_partial = ReproducibilityReport {
        status: ReproducibilityStatus::PartiallyReproducible,
        limitations: vec!["build timestamps are not normalized".into()],
    };
    assert!(documented_partial.validate().is_ok());

    let full = ReproducibilityReport {
        status: ReproducibilityStatus::FullyReproducible,
        limitations: Vec::new(),
    };
    assert!(full.validate().is_ok());
}

#[test]
fn lockfile_policy_requires_checked_in_state_and_reviewed_drift() {
    let not_checked_in = LockfileState::default();
    assert!(matches!(
        reject_unreviewed_lockfile_drift(&not_checked_in),
        Err(ReleaseSecurityError::LockfileNotCheckedIn)
    ));

    let unreviewed_drift = LockfileState {
        checked_in: true,
        digest: Some("digest".into()),
        drift_detected: true,
        drift_reviewed: false,
    };
    assert!(matches!(
        reject_unreviewed_lockfile_drift(&unreviewed_drift),
        Err(ReleaseSecurityError::LockfileDriftUnreviewed)
    ));

    let reviewed_drift = LockfileState {
        drift_reviewed: true,
        ..unreviewed_drift
    };
    assert!(reject_unreviewed_lockfile_drift(&reviewed_drift).is_ok());
}

#[test]
fn unexpected_unreviewed_build_script_is_flagged() {
    let unexpected = BuildScriptReview {
        crate_name: "example-native-sys".into(),
        has_build_script: true,
        unexpected: true,
        reviewed: false,
        native_build_documented: false,
    };
    assert!(matches!(
        flag_unexpected_build_script(&unexpected),
        Err(ReleaseSecurityError::UnexpectedBuildScriptUnreviewed { .. })
    ));

    let reviewed = BuildScriptReview {
        reviewed: true,
        ..unexpected
    };
    assert!(flag_unexpected_build_script(&reviewed).is_ok());

    let expected = BuildScriptReview {
        crate_name: "example-native-sys".into(),
        has_build_script: true,
        unexpected: false,
        reviewed: false,
        native_build_documented: true,
    };
    assert!(flag_unexpected_build_script(&expected).is_ok());
}

#[test]
fn secret_scan_targets_cover_all_eight_locations_and_block_on_detection() {
    assert_eq!(SECRET_SCAN_TARGETS.len(), 8);

    let detected = SecretScanReport {
        findings: vec![SecretScanFinding {
            target: SecretScanTarget::BuildMetadata,
            detected: true,
            location: Some("build.env".into()),
        }],
    };
    assert!(matches!(
        detected.validate_for_stable_release(),
        Err(ReleaseSecurityError::SecretDetected { .. })
    ));

    let clean = SecretScanReport {
        findings: vec![SecretScanFinding {
            target: SecretScanTarget::BuildMetadata,
            detected: false,
            location: None,
        }],
    };
    assert!(clean.validate_for_stable_release().is_ok());
}

#[test]
fn artifact_integrity_requires_every_check_to_pass() {
    assert!(ArtifactIntegrityStatus::default().validate().is_err());

    let complete = ArtifactIntegrityStatus {
        source_state_clean_or_ci_controlled: true,
        release_tag_matches_source: true,
        openspec_report_matches_baseline: true,
        conformance_reports_match_commit: true,
        checksums_match_final_artifacts: true,
    };
    assert!(complete.validate().is_ok());

    let mut partial = complete;
    partial.checksums_match_final_artifacts = false;
    assert!(matches!(
        partial.validate(),
        Err(ReleaseSecurityError::ArtifactIntegrityFailed { .. })
    ));
}

#[test]
fn redaction_gate_rejects_sensitive_content_and_native_handles() {
    assert_eq!(REDACTION_CATEGORIES.len(), 13);
    assert!(matches!(
        validate_redaction_gate("raw prompt: what is the secret?"),
        Err(ReleaseSecurityError::RedactionGateFailed { .. })
    ));
    assert!(validate_redaction_gate("provider handle=0xdeadbeef").is_err());
    assert!(validate_redaction_gate("/home/user/model.bin").is_err());
    assert!(validate_redaction_gate("generation completed in 12ms").is_ok());
}

#[test]
fn dynamic_provider_loading_requires_review_or_explicit_status() {
    assert!(matches!(
        validate_dynamic_provider_loading_status(
            ProviderLoadingMode::DynamicLibrary,
            DynamicProviderLoadingStatus::StableUnreviewed,
        ),
        Err(ReleaseSecurityError::DynamicProviderLoadingUnreviewed)
    ));
    for status in [
        DynamicProviderLoadingStatus::Disabled,
        DynamicProviderLoadingStatus::Experimental,
        DynamicProviderLoadingStatus::MarkedUnstable,
        DynamicProviderLoadingStatus::SecurityReviewed,
    ] {
        assert!(
            validate_dynamic_provider_loading_status(ProviderLoadingMode::DynamicLibrary, status)
                .is_ok()
        );
    }
    assert!(
        validate_dynamic_provider_loading_status(
            ProviderLoadingMode::BuiltIn,
            DynamicProviderLoadingStatus::StableUnreviewed,
        )
        .is_ok()
    );
}

#[test]
fn provider_registration_alone_does_not_imply_trust() {
    assert!(matches!(
        reject_provider_registration_implies_trust(ProviderTrustSignalSource::RegistrationOnly),
        Err(ReleaseSecurityError::ProviderRegistrationTrustImplied)
    ));
    assert!(
        reject_provider_registration_implies_trust(ProviderTrustSignalSource::ConfiguredPolicy)
            .is_ok()
    );
}

#[test]
fn release_native_handle_exposure_is_denied_across_all_provider_families() {
    for surface in [
        "raw-provider-handle",
        "raw-device-handle",
        "raw-kernel-handle",
        "raw-tensor-pointer",
        "raw-memory-pointer",
        "raw-kv-cache",
        "raw-model-weight",
        "cuda-stream",
        "cuda-device-pointer",
        "metal-buffer",
        "metal-command-queue",
        "openvino-compiled-graph",
        "qnn-native-handle",
        "raw-cpu-allocation-pointer",
    ] {
        assert!(
            reject_release_native_handle_exposure(surface).is_err(),
            "expected surface '{surface}' to be denied"
        );
    }
    assert!(reject_release_native_handle_exposure("generation").is_ok());
}

#[test]
fn component_release_execution_trust_requires_trusted_status_and_signature_in_production() {
    let untrusted = ComponentTrustDecision::new(ComponentTrustStatus::Rejected, "rejected fixture");
    assert!(matches!(
        validate_component_release_execution_trust(&untrusted, true, true, false),
        Err(ReleaseSecurityError::ComponentArtifactUntrusted { .. })
    ));

    let trusted = ComponentTrustDecision::new(ComponentTrustStatus::Trusted, "trusted fixture");
    assert!(matches!(
        validate_component_release_execution_trust(&trusted, false, true, false),
        Err(ReleaseSecurityError::UnsignedComponentDeniedInProduction)
    ));
    assert!(validate_component_release_execution_trust(&trusted, false, true, true).is_ok());
    assert!(validate_component_release_execution_trust(&trusted, false, false, false).is_ok());
    assert!(validate_component_release_execution_trust(&trusted, true, true, false).is_ok());
}

#[test]
fn component_release_authority_expansion_is_denied_for_os_and_handle_capabilities() {
    for capability in [
        "filesystem",
        "network-tool",
        "secret",
        "shell",
        "raw-provider-handle",
    ] {
        assert!(
            reject_component_release_authority_expansion(capability).is_err(),
            "expected capability '{capability}' to be denied"
        );
    }
    assert!(reject_component_release_authority_expansion("generation").is_ok());
}

#[test]
fn model_artifact_release_trust_ignores_recognized_format() {
    // `ModelTrustDecision::new` is deliberately `pub(crate)`-only inside
    // magnetar-runtime (see its struct-level doc comment) and not reachable
    // from this crate -- go through the same public `ModelTrustStore`
    // path any real caller uses instead.
    let untrusted_manifest = model_source_cache_roadmap::probe_manifest();
    let untrusted = ModelTrustStore::default().evaluate(&untrusted_manifest);
    assert!(matches!(
        validate_model_artifact_release_trust(&untrusted, true),
        Err(ReleaseSecurityError::ModelArtifactUntrusted { .. })
    ));

    let trusted_manifest = model_source_cache_roadmap::probe_manifest();
    let trusted = ModelTrustStore::default()
        .trust_digest(trusted_manifest.id.digest.value.clone())
        .evaluate(&trusted_manifest);
    assert!(validate_model_artifact_release_trust(&trusted, false).is_ok());
    assert!(validate_model_artifact_release_trust(&trusted, true).is_ok());
}

#[test]
fn fixture_model_trust_requires_explicit_test_policy() {
    let fixture_manifest = model_source_cache_roadmap::probe_manifest();
    let trusted = ModelTrustStore::default()
        .trust_digest(fixture_manifest.id.digest.value.clone())
        .evaluate(&fixture_manifest);
    let undocumented = FixtureModelTrustPolicy::default();
    assert!(matches!(
        validate_fixture_model_trust(&trusted, &undocumented),
        Err(ReleaseSecurityError::FixtureTrustPolicyUndocumented)
    ));

    let documented = FixtureModelTrustPolicy {
        explicit_test_policy_documented: true,
    };
    assert!(validate_fixture_model_trust(&trusted, &documented).is_ok());
}

#[test]
fn source_cache_trust_is_never_implied_by_a_non_trust_signal() {
    let mut entry = CacheEntryMetadata::new(
        ModelArtifactId {
            kind: ModelArtifactKind::ModelWeights,
            name: ModelName::new("qwen-test").unwrap(),
            revision: ModelRevision::new("v1").unwrap(),
            variant: None,
            digest: ModelDigest {
                algorithm: "sha256".into(),
                value: "cachedigest".into(),
            },
            source: None,
            shard: None,
        },
        ModelSourceKind::LocalDirectorySource,
    );
    assert_eq!(entry.trust_status, ModelTrustStatus::Unknown);
    assert!(matches!(
        validate_source_cache_release_trust(&entry),
        Err(ReleaseSecurityError::CacheEntryTrustNotEstablished { .. })
    ));

    for signal in [
        NonTrustCacheSignal::CacheHit,
        NonTrustCacheSignal::SourceKind,
        NonTrustCacheSignal::Alias,
        NonTrustCacheSignal::LocalFile,
        NonTrustCacheSignal::FixtureStatus,
    ] {
        assert!(matches!(
            reject_cache_signal_alone_as_trust(signal, &entry),
            Err(ReleaseSecurityError::CacheSignalDoesNotImplyTrust { .. })
        ));
    }

    entry.trust_status = ModelTrustStatus::Trusted;
    assert!(validate_source_cache_release_trust(&entry).is_ok());
}

#[test]
fn cli_authority_is_never_delegated_to_runtime() {
    for capability in ["filesystem", "git", "shell", "secret", "tool-call"] {
        assert!(matches!(
            validate_cli_authority_not_delegated_to_runtime(capability),
            Err(ReleaseSecurityError::CliAuthorityDelegatedToRuntime { .. })
        ));
    }
}

#[test]
fn runtime_inference_api_rejects_non_inference_authority() {
    for capability in [
        "filesystem",
        "network-tool",
        "secret",
        "shell",
        "git",
        "tool-call",
    ] {
        assert!(matches!(
            validate_runtime_inference_api_security(capability),
            Err(ReleaseSecurityError::RuntimeInferenceApiAuthorityExpansionDenied { .. })
        ));
    }
    assert!(validate_runtime_inference_api_security("generation").is_ok());
}

#[test]
fn unsafe_code_policy_denies_unreviewed_blocks_only_when_configured() {
    let unreviewed = UnsafeCodePolicy {
        reviews: vec![UnsafeCodeReview {
            location: "compute.rs:100".into(),
            justified: false,
            reviewed: false,
        }],
        deny_unreviewed: true,
    };
    assert!(matches!(
        unreviewed.validate(),
        Err(ReleaseSecurityError::UnsafeCodeUnreviewed { .. })
    ));

    let not_enforced = UnsafeCodePolicy {
        deny_unreviewed: false,
        ..unreviewed.clone()
    };
    assert!(not_enforced.validate().is_ok());

    let reviewed = UnsafeCodePolicy {
        reviews: vec![UnsafeCodeReview {
            location: "compute.rs:100".into(),
            justified: true,
            reviewed: true,
        }],
        deny_unreviewed: true,
    };
    assert!(reviewed.validate().is_ok());
}

#[test]
fn magnetar_runtime_unsafe_code_inventory_is_reviewed_and_justified() {
    let inventory = magnetar_runtime_unsafe_code_inventory();
    assert_eq!(inventory.reviews.len(), 3);
    assert!(inventory.deny_unreviewed);
    assert!(inventory.validate().is_ok());
    for review in &inventory.reviews {
        assert!(review.location.contains("provider.rs"));
        assert!(review.justified);
        assert!(review.reviewed);
    }
}

#[test]
fn unexpected_capability_expanding_dependency_feature_is_rejected() {
    let unexpected = DependencyFeatureReview {
        crate_name: "example-crate".into(),
        feature_name: "http-client".into(),
        capability: DependencyFeatureCapability::Networking,
        expected: false,
        accepted_exception: false,
    };
    assert!(matches!(
        reject_unexpected_capability_expanding_feature(&unexpected),
        Err(ReleaseSecurityError::UnexpectedCapabilityExpandingFeature { .. })
    ));

    let excepted = DependencyFeatureReview {
        accepted_exception: true,
        ..unexpected.clone()
    };
    assert!(reject_unexpected_capability_expanding_feature(&excepted).is_ok());

    let expected = DependencyFeatureReview {
        expected: true,
        ..unexpected
    };
    assert!(reject_unexpected_capability_expanding_feature(&expected).is_ok());
}

#[test]
fn vulnerability_handling_policy_requires_every_field_defined() {
    assert!(VulnerabilityHandlingPolicy::default().validate().is_err());
    let complete = VulnerabilityHandlingPolicy {
        advisory_severity_handling_defined: true,
        release_blocking_criteria_defined: true,
        mitigation_documentation_required: true,
        exception_approval_defined: true,
        follow_up_tracking_defined: true,
        patch_release_expectation_documented: true,
    };
    assert!(complete.validate().is_ok());
}

#[test]
fn security_release_notes_require_shall_strength_topics() {
    assert!(SecurityReleaseNotes::default().validate().is_err());
    let complete = SecurityReleaseNotes {
        v0_1_threat_model: Some("CPU-local baseline".into()),
        trusted_native_provider_model: Some("Providers are trusted native code".into()),
        no_raw_handle_policy: Some("no raw handles in public APIs".into()),
        default_redaction: Some("diagnostics redact by default".into()),
        reporting_process_placeholder: Some("reporting process TBD".into()),
        ..Default::default()
    };
    assert!(complete.validate().is_ok());
}

#[test]
fn release_security_blocking_reports_every_triggered_reason() {
    assert!(evaluate_release_security_blocking(&ReleaseSecurityGateInputs::default()).is_ok());

    let blocked = ReleaseSecurityGateInputs {
        secrets_detected: true,
        raw_handle_exposed: true,
        checksum_mismatch: true,
        ..Default::default()
    };
    match evaluate_release_security_blocking(&blocked) {
        Err(ReleaseSecurityError::ReleaseBlocked { reasons }) => assert_eq!(reasons.len(), 3),
        other => panic!("unexpected outcome: {other:?}"),
    }
}

#[test]
fn undocumented_security_exception_is_rejected_when_required() {
    assert!(reject_undocumented_security_exception(false, None).is_ok());
    assert!(matches!(
        reject_undocumented_security_exception(true, None),
        Err(ReleaseSecurityError::SecurityExceptionIncomplete)
    ));

    let incomplete = SecurityException {
        issue: "advisory".into(),
        affected_component: String::new(),
        severity: DependencyAdvisorySeverity::High,
        rationale: "reason".into(),
        mitigation: "patch".into(),
        owner: "team".into(),
        expiration_or_follow_up: "v0.2".into(),
        release_note_entry: true,
    };
    assert!(incomplete.validate().is_err());

    let complete = SecurityException {
        affected_component: "example-crate".into(),
        ..incomplete
    };
    assert!(reject_undocumented_security_exception(true, Some(&complete)).is_ok());
}

#[test]
fn release_security_observation_is_always_redacted() {
    let observation = record_release_security_observation(
        ReleaseSecurityObservationKind::SecretScanCompleted,
        "found credential abc123 in build.env",
    );
    assert!(!observation.detail.unwrap().contains("credential abc123"));

    let ordinary = record_release_security_observation(
        ReleaseSecurityObservationKind::ReleaseSecurityPassed,
        "all gates passed",
    );
    assert_eq!(ordinary.detail.as_deref(), Some("all gates passed"));
}

#[test]
fn release_security_conformance_report_is_conformant() {
    let report = run_release_security_conformance();
    assert!(!report.results.is_empty());
    for result in &report.results {
        assert!(
            result.passed,
            "{} failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
    assert!(report.is_conformant());
}

// ---------------------------------------------------------------------
// release_cutover
// ---------------------------------------------------------------------

#[test]
fn release_cutover_policy_version_constant_is_set() {
    assert_eq!(RELEASE_CUTOVER_POLICY_VERSION, "0.1.0");
}

#[test]
fn release_readiness_checklist_requires_every_field() {
    let missing_notes = ReleaseReadinessChecklist {
        release_branch_or_commit_selected: true,
        version_selected: true,
        openspec_baseline_selected: true,
        scope_selected: true,
        gates_selected: true,
        artifacts_selected: true,
        release_notes_draft_exists: false,
        compatibility_matrix_draft_exists: true,
        security_notes_draft_exists: true,
    };
    assert!(matches!(
        missing_notes.validate(),
        Err(ReleaseCutoverError::ReleaseReadinessIncomplete { .. })
    ));

    let complete = ReleaseReadinessChecklist {
        release_notes_draft_exists: true,
        ..missing_notes
    };
    assert!(complete.validate().is_ok());
}

#[test]
fn openspec_freeze_confirmation_requires_frozen_state() {
    let open = OpenSpecFreezeConfirmation {
        freeze_state: ReleaseFreezeState::Open,
        accepted_changes_list_final: true,
        pending_changes_excluded: true,
        wit_breaking_changes_have_version_bumps: true,
        checklist_references_correct_changes: true,
        roadmap_items_deferred_unless_included: true,
    };
    assert!(matches!(
        open.validate(),
        Err(ReleaseCutoverError::OpenSpecNotFrozen { .. })
    ));

    let frozen = OpenSpecFreezeConfirmation {
        freeze_state: ReleaseFreezeState::Frozen,
        ..open
    };
    assert!(frozen.validate().is_ok());
}

#[test]
fn semantic_change_after_freeze_is_blocked_or_freeze_is_restarted() {
    let outcome = reject_semantic_change_after_freeze(
        ReleaseFreezeState::Frozen,
        ReleaseFreezeChangeKind::SemanticContractChange,
    );
    assert!(matches!(
        outcome,
        Err(ReleaseCutoverError::SemanticChangeAfterFreeze { .. })
    ));
    assert!(
        reject_semantic_change_after_freeze(
            ReleaseFreezeState::Open,
            ReleaseFreezeChangeKind::SemanticContractChange,
        )
        .is_ok()
    );
}

#[test]
fn cuda_listed_as_included_is_blocked() {
    let outcome = validate_v0_1_scope_feature("cuda", true);
    assert!(matches!(
        outcome,
        Err(ReleaseCutoverError::RoadmapFeaturePresentedAsIncluded { .. })
    ));
    assert!(validate_v0_1_scope_feature("cuda", false).is_ok());
}

#[test]
fn missing_wit_version_blocks_release() {
    let missing = validate_wit_versions_confirmed(&[WitPackageVersionRecord {
        package: "magnetar:observability".into(),
        version: None,
    }]);
    assert!(matches!(
        missing,
        Err(ReleaseCutoverError::WitVersionMissing { .. })
    ));

    let confirmed: Vec<WitPackageVersionRecord> = release_wit_contract_versions()
        .iter()
        .map(WitPackageVersionRecord::from_interface)
        .collect();
    assert!(validate_wit_versions_confirmed(&confirmed).is_ok());
}

#[test]
fn missing_wit_package_blocks_the_cutover_compatibility_matrix() {
    let mut matrix = CutoverCompatibilityMatrix::default();
    for dimension in CUTOVER_COMPATIBILITY_DIMENSIONS {
        if *dimension != CutoverCompatibilityDimension::WitPackages {
            matrix.set(*dimension, CutoverCompatibilityStatus::StableForV01Baseline);
        }
    }
    let outcome = matrix.validate();
    assert!(matches!(
        outcome,
        Err(ReleaseCutoverError::CompatibilityDimensionMissing { dimension })
            if dimension == "wit-packages"
    ));
}

#[test]
fn cutover_version_confirmation_requires_every_field() {
    let wit_packages: Vec<WitPackageVersionRecord> = release_wit_contract_versions()
        .iter()
        .map(WitPackageVersionRecord::from_interface)
        .collect();
    let incomplete = CutoverVersionConfirmation {
        release_version: ReleaseVersion::new(0, 1, 0),
        crate_versions_confirmed: true,
        binary_version_confirmed: false,
        wit_packages: wit_packages.clone(),
        conformance_suite_versions_confirmed: true,
        openspec_baseline_version_confirmed: true,
        release_candidate_lineage_documented: None,
    };
    assert!(matches!(
        incomplete.validate(),
        Err(ReleaseCutoverError::VersionConfirmationIncomplete { .. })
    ));

    let complete = CutoverVersionConfirmation {
        binary_version_confirmed: true,
        ..incomplete
    };
    assert!(complete.validate().is_ok());
}

#[test]
fn runtime_version_mismatch_fails_cutover_verification() {
    let report = build_release_binary_version_report(
        ReleaseVersion::new(0, 1, 0),
        vec!["reference-cpu-provider".into()],
        "release",
        None,
    );
    let outcome = validate_runtime_version_matches_release_tag(&report, "0.1.0-rc.1");
    assert!(matches!(
        outcome,
        Err(ReleaseCutoverError::RuntimeVersionMismatch { .. })
    ));
    assert!(validate_runtime_version_matches_release_tag(&report, "0.1.0").is_ok());
}

#[test]
fn experimental_webgpu_enabled_by_default_is_blocked() {
    let flag = ReleaseFeatureFlag {
        name: "webgpu-provider".into(),
        class: ReleaseFeatureFlagClass::Experimental,
        enabled_by_default: true,
    };
    assert!(matches!(
        validate_cutover_feature_flag(&flag),
        Err(ReleaseCutoverError::ExperimentalFeatureEnabledByDefault { .. })
    ));
}

#[test]
fn test_only_flag_enabled_by_default_is_blocked_in_release_build() {
    let flag = ReleaseFeatureFlag {
        name: "test-harness".into(),
        class: ReleaseFeatureFlagClass::TestOnly,
        enabled_by_default: true,
    };
    assert!(matches!(
        validate_cutover_feature_flag(&flag),
        Err(ReleaseCutoverError::NonBaselineFeatureEnabledInRelease { .. })
    ));
}

#[test]
fn provider_abi_missing_blocks_compatibility_matrix_completion() {
    let mut matrix = CutoverCompatibilityMatrix::default();
    for dimension in CUTOVER_COMPATIBILITY_DIMENSIONS {
        if *dimension != CutoverCompatibilityDimension::ProviderAbi {
            matrix.set(*dimension, CutoverCompatibilityStatus::StableForV01Baseline);
        }
    }
    assert!(matches!(
        matrix.validate(),
        Err(ReleaseCutoverError::CompatibilityDimensionMissing { .. })
    ));
}

#[test]
fn cutover_compatibility_matrix_uses_approved_status_vocabulary() {
    assert_eq!(CUTOVER_COMPATIBILITY_DIMENSIONS.len(), 12);
    assert_eq!(
        cutover_compatibility_status_id(CutoverCompatibilityStatus::StableForV01Baseline),
        "stable-for-v0.1-baseline"
    );
    assert_eq!(
        cutover_compatibility_status_id(CutoverCompatibilityStatus::Unsupported),
        "unsupported"
    );
}

#[test]
fn experimental_api_presented_stable_is_blocked() {
    let outcome = reject_status_misrepresentation(
        CutoverCompatibilityStatus::Experimental,
        CutoverCompatibilityStatus::StableForV01Baseline,
        "cli-command-surface",
    );
    assert!(matches!(
        outcome,
        Err(ReleaseCutoverError::StatusMisrepresented { .. })
    ));
    assert!(
        reject_status_misrepresentation(
            CutoverCompatibilityStatus::StableForV01Baseline,
            CutoverCompatibilityStatus::StableForV01Baseline,
            "cli-command-surface",
        )
        .is_ok()
    );
}

#[test]
fn e2e_gate_not_run_blocks_required_gate_execution() {
    let without_e2e: Vec<ReleaseGateResult> = REQUIRED_RELEASE_GATES
        .iter()
        .filter(|gate| **gate != ReleaseGate::E2eLocalConformance)
        .map(|gate| ReleaseGateResult {
            gate: *gate,
            passed: true,
        })
        .collect();
    assert!(matches!(
        validate_required_gates_executed(&without_e2e),
        Err(ReleaseCutoverError::RequiredGateFailedOrMissing { .. })
    ));
}

#[test]
fn reference_cpu_gate_skip_is_disallowed() {
    let skip = GateSkip {
        gate: "reference-cpu-conformance".into(),
        outside_v0_1_scope: false,
        reason: Some("ran out of time".into()),
        hides_baseline_failure: false,
        included_in_release_report: true,
    };
    assert!(matches!(
        validate_gate_skips(&[skip]),
        Err(ReleaseCutoverError::DisallowedGateSkip { .. })
    ));
}

#[test]
fn allowed_skip_requires_every_condition() {
    let hides_failure = GateSkip {
        gate: "cuda-conformance".into(),
        outside_v0_1_scope: true,
        reason: Some("out of scope".into()),
        hides_baseline_failure: true,
        included_in_release_report: true,
    };
    assert!(hides_failure.validate().is_err());

    let allowed = GateSkip {
        hides_baseline_failure: false,
        ..hides_failure
    };
    assert!(allowed.validate().is_ok());
}

#[test]
fn undocumented_exception_blocks_release() {
    assert!(matches!(
        reject_undocumented_cutover_exception(true, None),
        Err(ReleaseCutoverError::UndocumentedException { .. })
    ));
    assert!(reject_undocumented_cutover_exception(false, None).is_ok());
}

#[test]
fn exception_missing_mitigation_or_owner_blocks_release() {
    let incomplete = CutoverException {
        gate: "dependency-audit".into(),
        exception: SecurityException {
            issue: "advisory RUSTSEC-0000-0000".into(),
            affected_component: "example-crate".into(),
            severity: DependencyAdvisorySeverity::High,
            rationale: "no fixed version available yet".into(),
            mitigation: String::new(),
            owner: String::new(),
            expiration_or_follow_up: "revisit in v0.2".into(),
            release_note_entry: true,
        },
    };
    assert!(matches!(
        incomplete.validate(),
        Err(ReleaseCutoverError::UndocumentedException { .. })
    ));

    let complete = CutoverException {
        exception: SecurityException {
            mitigation: "vendored patch applied".into(),
            owner: "release-team".into(),
            ..incomplete.exception.clone()
        },
        ..incomplete
    };
    assert!(complete.validate().is_ok());
    assert!(validate_cutover_exceptions(&[complete]).is_ok());
}

#[test]
fn secret_scan_missing_blocks_security_verification() {
    let missing = CutoverSecurityVerification {
        gate_inputs: ReleaseSecurityGateInputs::default(),
        security_notes: SecurityReleaseNotes {
            v0_1_threat_model: Some("CPU-local baseline".into()),
            trusted_native_provider_model: Some("Providers are trusted native code".into()),
            no_raw_handle_policy: Some("no raw handles in public APIs".into()),
            default_redaction: Some("diagnostics redact by default".into()),
            reporting_process_placeholder: Some("reporting process TBD".into()),
            ..Default::default()
        },
    };
    // A gate_inputs with every field false represents "not yet confirmed" at
    // this call site only through the surrounding checklist; the blocking
    // gate itself is exercised directly here.
    let blocked = CutoverSecurityVerification {
        gate_inputs: ReleaseSecurityGateInputs {
            secrets_detected: true,
            ..Default::default()
        },
        ..missing
    };
    assert!(blocked.validate().is_err());
}

#[test]
fn missing_conformance_report_blocks_stable_release_unless_marked_not_applicable() {
    let manifest = ReleaseArtifactManifest::default();
    assert!(matches!(
        validate_cutover_artifacts_generated(&manifest),
        Err(ReleaseCutoverError::ArtifactGenerationIncomplete { .. })
    ));

    let mut complete = ReleaseArtifactManifest::default();
    for kind in RELEASE_ARTIFACT_KINDS {
        complete.set(*kind, ReleaseArtifactStatus::NotApplicable);
    }
    assert!(validate_cutover_artifacts_generated(&complete).is_ok());
}

#[test]
fn checksum_mismatch_blocks_or_withdraws_release() {
    let checksum =
        ArtifactChecksum::new("magnetar-cli", ChecksumAlgorithm::Sha256, "deadbeef").unwrap();
    let outcome = verify_cutover_artifact_checksum(&checksum, "different-digest");
    assert!(matches!(
        outcome,
        Err(ReleaseCutoverError::ArtifactVerificationFailed { .. })
    ));
    assert!(verify_cutover_artifact_checksum(&checksum, "deadbeef").is_ok());
}

#[test]
fn cutover_artifact_verification_requires_integrity_and_cutover_specific_checks() {
    let incomplete = CutoverArtifactVerification {
        integrity: ArtifactIntegrityStatus {
            source_state_clean_or_ci_controlled: true,
            release_tag_matches_source: true,
            openspec_report_matches_baseline: true,
            conformance_reports_match_commit: true,
            checksums_match_final_artifacts: true,
        },
        release_notes_match_compatibility_matrix: false,
        artifact_names_include_version: true,
    };
    assert!(matches!(
        incomplete.validate(),
        Err(ReleaseCutoverError::ArtifactVerificationFailed { .. })
    ));

    let complete = CutoverArtifactVerification {
        release_notes_match_compatibility_matrix: true,
        ..incomplete
    };
    assert!(complete.validate().is_ok());
}

#[test]
fn known_limitation_missing_from_changelog_blocks_release() {
    let incomplete = CutoverChangelogChecklist {
        changelog: ReleaseChangelog {
            entries: vec![ChangelogEntry {
                kind: ChangelogEntryKind::AddedContract,
                description: "added the Runtime Inference API".into(),
            }],
        },
        includes_added_contracts: true,
        includes_changed_contracts: true,
        includes_removed_or_deprecated_contracts: true,
        includes_release_scope: true,
        includes_known_limitations: false,
        includes_compatibility_status: true,
        includes_security_notes: true,
        includes_conformance_status: true,
        includes_deferred_roadmap_items: true,
    };
    assert!(matches!(
        incomplete.validate(),
        Err(ReleaseCutoverError::ChangelogIncomplete { .. })
    ));

    let complete = CutoverChangelogChecklist {
        includes_known_limitations: true,
        ..incomplete
    };
    assert!(complete.validate().is_ok());
}

#[test]
fn release_notes_checklist_requires_every_topic() {
    let incomplete = CutoverReleaseNotesChecklist {
        explains_what_v0_1_is: true,
        ..Default::default()
    };
    assert!(matches!(
        incomplete.validate(),
        Err(ReleaseCutoverError::ReleaseNotesIncomplete { .. })
    ));

    let complete = CutoverReleaseNotesChecklist {
        explains_what_v0_1_is: true,
        explains_what_users_can_run: true,
        explains_stable_status: true,
        explains_preview_status: true,
        explains_experimental_status: true,
        explains_deferred_status: true,
        explains_unsupported_status: true,
        explains_artifact_verification: true,
        explains_security_limitations: true,
        explains_how_to_run_conformance: true,
        includes_compatibility_matrix: true,
        includes_security_notes: true,
        includes_known_limitations: true,
    };
    assert!(complete.validate().is_ok());
}

#[test]
fn tag_created_before_gates_pass_is_invalid() {
    let mut results: Vec<ReleaseGateResult> = REQUIRED_RELEASE_GATES
        .iter()
        .map(|gate| ReleaseGateResult {
            gate: *gate,
            passed: true,
        })
        .collect();
    results[0].passed = false;
    assert!(matches!(
        validate_tag_after_gates(&results, true),
        Err(ReleaseCutoverError::TagCreatedBeforeGatesPassed)
    ));
    // No tag created yet: an incomplete gate set does not itself block.
    assert!(validate_tag_after_gates(&results, false).is_ok());

    for result in &mut results {
        result.passed = true;
    }
    assert!(validate_tag_after_gates(&results, true).is_ok());
}

#[test]
fn wit_validation_must_complete_before_stable_tag() {
    let without_wit: Vec<ReleaseGateResult> = REQUIRED_RELEASE_GATES
        .iter()
        .filter(|gate| **gate != ReleaseGate::WitValidation)
        .map(|gate| ReleaseGateResult {
            gate: *gate,
            passed: true,
        })
        .collect();
    assert!(matches!(
        validate_tag_after_gates(&without_wit, true),
        Err(ReleaseCutoverError::TagCreatedBeforeGatesPassed)
    ));
}

#[test]
fn server_api_claimed_included_is_blocked() {
    let outcome = validate_publication_scope_preserved(
        "server-api-implementation",
        true,
        CutoverCompatibilityStatus::Deferred,
        CutoverCompatibilityStatus::StableForV01Baseline,
    );
    assert!(matches!(
        outcome,
        Err(ReleaseCutoverError::PublicationScopeViolation { .. })
    ));
}

#[test]
fn binary_version_mismatch_marks_post_publication_verification_invalid() {
    let mismatch = PostPublicationVerification {
        published_artifacts_match_checksums: true,
        release_notes_visible: true,
        reports_accessible: true,
        version_command_matches_tag: false,
        documentation_links_valid: true,
        compatibility_matrix_visible: true,
        security_notes_visible: true,
        deferred_roadmap_clearly_separated: true,
    };
    assert!(matches!(
        mismatch.validate(),
        Err(ReleaseCutoverError::PostPublicationVerificationFailed { .. })
    ));
}

#[test]
fn rollback_and_retraction_notes_cover_invalid_release_discovery() {
    let incomplete = RollbackRetractionNotes::default();
    assert!(incomplete.validate().is_err());

    let complete = RollbackRetractionNotes {
        withdrawal_procedure_documented: true,
        advisory_publication_procedure_documented: true,
        patch_release_procedure_documented: true,
        audit_trail_preservation_documented: true,
        release_notes_update_procedure_documented: true,
    };
    assert!(complete.validate().is_ok());
}

#[test]
fn post_v0_1_item_presented_as_release_claim_is_rejected() {
    let next_work = PostV01HandoffItem {
        name: "optimized-cpu-provider".into(),
        presented_as_v0_1_release_claim: false,
    };
    assert!(reject_post_v0_1_item_as_release_claim(&next_work).is_ok());

    let misclaimed = PostV01HandoffItem {
        name: "optimized-cpu-provider".into(),
        presented_as_v0_1_release_claim: true,
    };
    assert!(matches!(
        reject_post_v0_1_item_as_release_claim(&misclaimed),
        Err(ReleaseCutoverError::PostV01ItemPresentedAsReleaseClaim { .. })
    ));
}

#[test]
fn final_release_statement_describes_baseline_accurately() {
    assert!(validate_final_release_statement(V0_1_FINAL_RELEASE_STATEMENT).is_ok());
    assert!(validate_final_release_statement("Magnetar v0.1 ships full GPU support.").is_err());
}

#[test]
fn cli_boundary_gate_failure_blocks_cutover() {
    let outcome = validate_cutover_cli_boundary("filesystem");
    assert!(matches!(
        outcome,
        Err(ReleaseCutoverError::RuntimeScopeViolation { .. })
    ));
}

#[test]
fn runtime_tool_execution_blocks_cutover_scope_confirmation() {
    let outcome = validate_cutover_runtime_scope("shell");
    assert!(matches!(
        outcome,
        Err(ReleaseCutoverError::RuntimeScopeViolation { .. })
    ));
    assert!(validate_cutover_runtime_scope("generation").is_ok());
}

#[test]
fn release_cutover_observation_is_always_redacted() {
    let observation = record_release_cutover_observation(ReleaseCutoverObservationInput {
        correlation_id: CorrelationId::new("cutover-run-1"),
        kind: ReleaseSecurityObservationKind::SecretScanCompleted,
        gate: Some("secret-scan".into()),
        target: Some("reference-cpu-provider".into()),
        feature_set: vec!["reference-cpu-provider".into()],
        artifact: Some("magnetar-cli".into()),
        release_metadata: Some("v0.1.0-rc.1".into()),
        raw_detail: "found credential abc123 in build.env",
    });
    assert!(
        !observation
            .security_observation
            .detail
            .clone()
            .unwrap()
            .contains("credential abc123")
    );
    assert!(observation.gate.is_some());
    assert!(observation.target.is_some());
    assert!(!observation.feature_set.is_empty());
    assert!(observation.artifact.is_some());
}

#[test]
fn evaluate_release_cutover_reports_every_triggered_reason() {
    let clean = ReleaseCutoverGateInputs::default();
    assert!(evaluate_release_cutover(&clean).is_ok());

    let blocked = ReleaseCutoverGateInputs {
        openspec_not_frozen: true,
        tag_created_before_gates_passed: true,
        ..Default::default()
    };
    let outcome = evaluate_release_cutover(&blocked);
    assert!(matches!(
        &outcome,
        Err(ReleaseCutoverError::ReleaseCutoverBlocked { reasons })
            if reasons.len() == 2
    ));
}

#[test]
fn release_cutover_conformance_report_is_conformant() {
    let report = run_release_cutover_conformance();
    assert!(!report.results.is_empty());
    for result in &report.results {
        assert!(
            result.passed,
            "{} failed: {:?}",
            result.requirement, result.diagnostic
        );
    }
    assert!(report.is_conformant());
}
