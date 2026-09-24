//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::conformance::validate_first_native_component_engine_capabilities;

use crate::session::InferenceSessionId;
use ed25519_dalek::Signer;
use std::fs;
fn component_artifact_package(
    bytes: &[u8],
    source_kind: ComponentDistributionSourceKind,
) -> ComponentArtifactPackage {
    let digest = ComponentDigest::sha256(bytes);
    ComponentArtifactPackage::new(
        bytes.to_vec(),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION).into_bytes(),
        digest,
        ComponentDistributionSource::new(source_kind, "test-source"),
    )
}

fn manifest_yaml_with_authority(digest: &str, authorities: &[&str]) -> String {
    let requires = authorities
        .iter()
        .map(|authority| format!("    - {authority}"))
        .collect::<Vec<_>>()
        .join("\n");
    manifest_yaml(digest, MAGNETAR_RUNTIME_VERSION).replace(
        "authority:\n  requires: []",
        &format!("authority:\n  requires:\n{requires}"),
    )
}

fn manifest_yaml(digest: &str, runtime_version: &str) -> String {
    format!(
        r#"schema: magnetar-component-artifact
schema_version: 1
artifact:
  kind: component
  digest:
    algorithm: sha256
    value: "{digest}"
component:
  name: "magnetar.examples.hello"
  version: "0.1.0"
  description: "Minimal Magnetar Component fixture"
  role: "test-fixture"
runtime:
  magnetar:
    min_version: "{runtime_version}"
wit:
  imports:
    - package: "magnetar:test"
      interface: "echo"
      version: "1.0.0"
  exports:
    - package: "magnetar:test"
      interface: "run"
      version: "1.0.0"
capabilities:
  requires:
    - id: "magnetar:test/echo"
      version: "1.0.0"
authority:
  requires: []
publisher:
  id: "local-dev"
  name: "Local Development"
source:
  kind: "local"
  uri: "./fixtures/hello.component.wasm"
signatures: []
"#
    )
}

#[test]
fn first_native_component_engine_accepts_native_controlled_wasi_capabilities() {
    let capabilities = ComponentEngineCapabilities::native();

    assert!(validate_first_native_component_engine_capabilities(&capabilities).is_ok());
    assert!(capabilities.supports(ComponentEngineFeature::ComponentModel));
    assert!(capabilities.supports(ComponentEngineFeature::ControlledWasi));
}

#[test]
fn component_engine_profiles_declare_platform_capabilities() {
    let native = ComponentEngineCapabilities::native();
    assert_eq!(native.profile, ComponentEngineProfile::Native);
    assert!(native.supports(ComponentEngineFeature::ComponentModel));
    assert!(native.supports(ComponentEngineFeature::NativeProviderEndpoints));
    assert!(!native.supports(ComponentEngineFeature::BrowserCompatible));

    let web = ComponentEngineCapabilities::web();
    assert_eq!(web.profile, ComponentEngineProfile::Web);
    assert!(web.supports(ComponentEngineFeature::BrowserCompatible));
    assert!(web.supports(ComponentEngineFeature::JsMediatedHostCalls));
    assert!(web.supports(ComponentEngineFeature::BrowserMemory));
    assert!(!web.supports(ComponentEngineFeature::NativeProviderEndpoints));

    let test = ComponentEngineCapabilities::test();
    assert_eq!(test.profile, ComponentEngineProfile::Test);
    assert!(test.supports(ComponentEngineFeature::Interruption));
    assert!(!test.supports(ComponentEngineFeature::ControlledWasi));
}

#[test]
fn component_engine_requirements_fail_closed_on_profile_or_feature_mismatch() {
    let requirements = ComponentEngineRequirements::default()
        .require_profile(ComponentEngineProfile::Web)
        .require_feature(ComponentEngineFeature::BrowserCompatible);

    assert!(
        requirements
            .validate("browser-component", &ComponentEngineCapabilities::web())
            .is_ok()
    );

    assert!(matches!(
        requirements.validate("browser-component", &ComponentEngineCapabilities::native()),
        Err(ComponentError::EngineProfileMismatch {
            required: ComponentEngineProfile::Web,
            actual: ComponentEngineProfile::Native,
            ..
        })
    ));

    let requirements = ComponentEngineRequirements::default()
        .require_feature(ComponentEngineFeature::NativeProviderEndpoints);
    assert!(matches!(
        requirements.validate("native-component", &ComponentEngineCapabilities::web()),
        Err(ComponentError::EngineFeatureUnavailable {
            feature: ComponentEngineFeature::NativeProviderEndpoints,
            profile: ComponentEngineProfile::Web,
            ..
        })
    ));
}

#[test]
fn component_wasi_imports_fail_closed_without_authorization() {
    let filesystem = WitInterface::new("wasi:filesystem/types", "0.2.0");
    let environment = WitInterface::new("wasi:cli/environment", "0.2.0");
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("wasi-consumer", "1", "test component")
                .with_import(filesystem)
                .with_import(environment),
            "wasi-consumer.wasm",
        ))
        .unwrap();

    assert!(matches!(
        manager.instantiate_component("wasi-consumer"),
        Err(ComponentError::UnauthorizedImport { .. })
    ));
}

#[test]
fn prepared_component_contract_must_match_declared_imports() {
    let declared = WitInterface::new("example:declared/api", "1.0.0");
    let undeclared = WitInterface::new("example:undeclared/api", "1.0.0");
    let mut contract = ComponentContract::default();
    contract.imports.insert(ComponentImportRequirement::new(
        undeclared,
        ComponentInterfaceShape::Interface,
    ));
    let mut engine = MockComponentEngine::new();
    engine.prepared_contract = Some(contract);
    let mut manager = ComponentManager::with_engine(Box::new(engine));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("consumer", "1", "test component").with_import(declared),
            "consumer.wasm",
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("consumer"),
        Err(ComponentError::ContractValidationFailed { .. })
    ));
}

#[test]
fn prepared_component_contract_must_include_declared_exports() {
    let exported = WitInterface::new("example:export/api", "1.0.0");
    let mut engine = MockComponentEngine::new();
    engine.prepared_contract = Some(ComponentContract::default());
    let mut manager = ComponentManager::with_engine(Box::new(engine));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("producer", "1", "test component").with_export(exported),
            "producer.wasm",
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("producer"),
        Err(ComponentError::ContractValidationFailed { .. })
    ));
}

#[test]
fn component_exports_do_not_automatically_satisfy_imports() {
    let interface = WitInterface::new("example:service/api", "1.0.0");
    let producer =
        ComponentMetadata::new("producer", "1", "producer").with_export(interface.clone());
    let consumer = ComponentMetadata::new("consumer", "1", "consumer").with_import(interface);
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(producer, "producer.wasm"))
        .unwrap();
    manager
        .register_component(ComponentDescriptor::new(consumer, "consumer.wasm"))
        .unwrap();

    assert!(matches!(
        manager.instantiate_component("consumer"),
        Err(ComponentError::UnauthorizedImport { .. })
    ));
}

#[test]
fn pushed_component_package_is_validated_before_preparation() {
    let bytes = b"component-bytes";
    let digest = ComponentDigest::sha256(bytes);
    let package =
        component_artifact_package(bytes, ComponentDistributionSourceKind::ClientProvided);
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value.clone()));

    manager.prepare_pushed_package(package).unwrap();

    let definition = manager.definition("magnetar.examples.hello").unwrap();
    assert_eq!(definition.artifact_digest, Some(digest));
    assert!(matches!(
        definition.trust_decision,
        Some(ComponentTrustDecision {
            status: ComponentTrustStatus::Trusted,
            ..
        })
    ));
    assert!(manager.observations().iter().any(|observation| {
        observation.kind == ComponentObservationKind::Distribution
            && observation.message.contains("client-provided")
    }));
}

#[test]
fn pushed_component_package_rejects_source_digest_mismatch() {
    let mut package =
        component_artifact_package(b"component-bytes", ComponentDistributionSourceKind::Tachyon);
    package.declared_digest = ComponentDigest::sha256(b"different");
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_source("tachyon"));

    assert!(matches!(
        manager.prepare_pushed_package(package),
        Err(ComponentError::Distribution {
            category: ComponentDistributionErrorCategory::DigestMismatch,
            ..
        })
    ));
}

#[test]
fn trusted_distribution_source_still_rejects_forbidden_authority() {
    let bytes = b"component-bytes";
    let digest = ComponentDigest::sha256(bytes);
    let manifest = manifest_yaml_with_authority(&digest.value, &["filesystem"])
        .replace("  kind: \"local\"", "  kind: \"tachyon\"");
    let package = ComponentArtifactPackage::new(
        bytes.to_vec(),
        manifest.into_bytes(),
        digest,
        ComponentDistributionSource::new(ComponentDistributionSourceKind::Tachyon, "tachyon"),
    );
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_source("tachyon"));

    assert!(matches!(
        manager.prepare_pushed_package(package),
        Err(ComponentError::Manifest { message, .. })
            if message == "authority kind is outside Magnetar inference scope"
    ));
}

#[test]
fn local_distribution_does_not_require_tachyon_or_network() {
    let bytes = b"component-bytes";
    let digest = ComponentDigest::sha256(bytes);
    let package =
        component_artifact_package(bytes, ComponentDistributionSourceKind::LocalDirectory);
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));

    manager.prepare_pushed_package(package).unwrap();
    assert_eq!(
        manager.definition_state("magnetar.examples.hello"),
        Some(ComponentDefinitionState::Prepared)
    );
}

#[derive(Clone)]
struct TestComponentDistributionSource {
    package: ComponentArtifactPackage,
    candidates: Vec<ComponentDigest>,
}

impl ComponentDistributionSourceProvider for TestComponentDistributionSource {
    fn resolve(
        &self,
        component: &str,
        _version_requirement: Option<&str>,
    ) -> Result<Vec<ComponentDigest>, ComponentError> {
        if component == "magnetar.examples.hello" {
            Ok(self.candidates.clone())
        } else {
            Ok(Vec::new())
        }
    }

    fn fetch(&self, digest: &ComponentDigest) -> Result<ComponentArtifactPackage, ComponentError> {
        if self.package.declared_digest == *digest {
            Ok(self.package.clone())
        } else {
            Err(ComponentError::Distribution {
                category: ComponentDistributionErrorCategory::ArtifactNotFound,
                message: "digest not found".into(),
            })
        }
    }
}

#[test]
fn component_manager_observes_engine_selection_and_rejection() {
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component"),
            "component.wasm",
        ))
        .unwrap();

    manager.prepare_component("component").unwrap();
    assert!(
        manager.observations().iter().any(|observation| {
            observation.kind == ComponentObservationKind::EngineSelection
                && observation
                    .message
                    .contains(ComponentEngineProfile::Test.as_str())
        }),
        "selected engine profile should be observable"
    );

    let error = ComponentEngineRequirements::default()
        .require_feature(ComponentEngineFeature::ControlledWasi)
        .validate("component", &manager.engine_capabilities())
        .unwrap_err();
    assert!(matches!(
        error,
        ComponentError::EngineFeatureUnavailable {
            feature: ComponentEngineFeature::ControlledWasi,
            ..
        }
    ));
}

#[test]
fn component_imports_are_authorized_and_linked_explicitly() {
    let interface = WitInterface::new("magnetar:runtime/run", "1.0.0");
    let metadata =
        ComponentMetadata::new("consumer", "1", "test component").with_import(interface.clone());
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(metadata, "consumer.wasm"))
        .unwrap();

    assert!(matches!(
        manager.instantiate_component("consumer"),
        Err(ComponentError::UnauthorizedImport { .. })
    ));

    manager.authorize_interface(interface.clone());
    assert!(matches!(
        manager.instantiate_component("consumer"),
        Err(ComponentError::UnresolvedImport { .. })
    ));

    manager.provide_interface(interface);
    let instance = manager.instantiate_component("consumer").unwrap();
    assert_eq!(
        manager.instance_state(instance),
        Some(ComponentInstanceState::Ready)
    );
}

#[test]
fn component_ambient_network_process_and_secret_imports_fail_closed() {
    let interfaces = [
        WitInterface::new("wasi:sockets/tcp", "0.2.0"),
        WitInterface::new("wasi:cli/run", "0.2.0"),
        WitInterface::new("magnetar:secrets/read", "1.0.0"),
    ];
    for (index, interface) in interfaces.into_iter().enumerate() {
        let name = format!("authority-{index}");
        let mut manager = ComponentManager::new();
        manager
            .register_component(ComponentDescriptor::new(
                ComponentMetadata::new(&name, "1", "test component").with_import(interface),
                format!("{name}.wasm"),
            ))
            .unwrap();

        assert!(matches!(
            manager.instantiate_component(&name),
            Err(ComponentError::UnauthorizedImport { .. })
        ));
    }
}

#[test]
fn component_link_plan_is_runtime_owned_and_immutable_to_callers() {
    let interface = WitInterface::new("magnetar:runtime/run", "1.0.0");
    let metadata =
        ComponentMetadata::new("consumer", "1", "test component").with_import(interface.clone());
    let mut manager = ComponentManager::new();
    manager.provide_interface(interface.clone());
    manager
        .register_component(ComponentDescriptor::new(metadata, "consumer.wasm"))
        .unwrap();

    let plan = manager.link_plan("consumer").unwrap();
    let links = plan.links().collect::<Vec<_>>();
    assert_eq!(links.len(), 1);
    assert_eq!(links[0].0, &interface);
    assert!(matches!(
        links[0].1,
        ComponentEndpoint::Capability { interface: linked } if linked == &interface
    ));
    assert_eq!(plan.endpoint(&interface), Some(links[0].1));
}

#[test]
fn component_link_plan_rejects_forbidden_external_interfaces_even_if_provided() {
    for interface in [
        WitInterface::new("wasi:filesystem/types", "0.2.0"),
        WitInterface::new("wasi:sockets/tcp", "0.2.0"),
        WitInterface::new("magnetar:workspace/read", "1.0.0"),
        WitInterface::new("magnetar:git/status", "1.0.0"),
        WitInterface::new("magnetar:process/run", "1.0.0"),
        WitInterface::new("magnetar:secrets/read", "1.0.0"),
    ] {
        let mut manager = ComponentManager::new();
        manager.provide_interface(interface.clone());
        manager
            .register_component(ComponentDescriptor::new(
                ComponentMetadata::new("external", "1", "external component")
                    .with_import(interface),
                "external.wasm",
            ))
            .unwrap();

        assert!(matches!(
            manager.link_plan("external"),
            Err(ComponentError::UnauthorizedImport { .. })
        ));
    }
}

#[test]
fn component_authority_requirements_map_to_inference_runtime_endpoints() {
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "compute-capability".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::Capability { interface }
            if interface == WitInterface::new("magnetar:compute/run", "2.0.0")
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "model-artifact-read".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceArtifactRegistry {
            kind: InferenceArtifactKind::Model
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "tokenizer-artifact-read".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceArtifactRegistry {
            kind: InferenceArtifactKind::Tokenizer
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "prompt-template-read".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceArtifactRegistry {
            kind: InferenceArtifactKind::PromptTemplate
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "adapter-artifact-read".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceArtifactRegistry {
            kind: InferenceArtifactKind::Adapter
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "quantization-artifact-read".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceArtifactRegistry {
            kind: InferenceArtifactKind::Quantization
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "kv-cache-access".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceCacheService {
            kind: InferenceCacheKind::Kv
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "prefix-cache-access".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::InferenceCacheService {
            kind: InferenceCacheKind::Prefix
        }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "observability-emit".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::Observability
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "runtime-diagnostics".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::RuntimeDiagnostics
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "generation-capability".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::PendingRuntimeService { .. }
    ));
    assert!(matches!(
        (ComponentAuthorityRequirement {
            kind: "sampling-capability".into(),
        })
        .endpoint(),
        ComponentAuthorityEndpoint::PendingRuntimeService { .. }
    ));
}

#[test]
fn inference_artifact_registry_uses_identities_not_paths_and_scopes_sessions() {
    let mut manager = ComponentManager::new();
    let digest = ComponentDigest::sha256(b"model");
    let session = InferenceSessionId::new("session-a").unwrap();
    manager
        .register_inference_artifact(
            InferenceArtifactReference::new(InferenceArtifactKind::Model, "qwen-model", digest)
                .unwrap()
                .with_session(session.clone()),
        )
        .unwrap();

    let artifact = manager
        .resolve_inference_artifact(InferenceArtifactKind::Model, "qwen-model", Some(&session))
        .unwrap();
    assert_eq!(artifact.id, "qwen-model");
    assert!(matches!(
        manager.resolve_inference_artifact(InferenceArtifactKind::Model, "../qwen-model", None),
        Err(ComponentError::ArtifactRejected { .. })
    ));
    assert!(matches!(
        manager.resolve_inference_artifact(
            InferenceArtifactKind::Model,
            "qwen-model",
            Some(&InferenceSessionId::new("session-b").unwrap())
        ),
        Err(ComponentError::ArtifactRejected { .. })
    ));
}

#[test]
fn component_definition_can_create_multiple_isolated_instances() {
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component"),
            "component.wasm",
        ))
        .unwrap();

    let first = manager.instantiate_component("component").unwrap();
    let second = manager.instantiate_component("component").unwrap();
    assert_ne!(first, second);
    assert_eq!(
        manager.instance_state(first),
        Some(ComponentInstanceState::Ready)
    );
    assert_eq!(
        manager.instance_state(second),
        Some(ComponentInstanceState::Ready)
    );
}

#[test]
fn component_manager_enforces_instance_and_invocation_limits() {
    let mut manager = ComponentManager::new();
    manager.set_resource_limits(ComponentResourceLimits {
        max_instances: Some(1),
        ..ComponentResourceLimits::default()
    });
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component"),
            "component.wasm",
        ))
        .unwrap();

    manager.instantiate_component("component").unwrap();
    assert!(matches!(
        manager.instantiate_component("component"),
        Err(ComponentError::ResourceLimitExceeded {
            limit: "instances",
            ..
        })
    ));

    let interface = WitInterface::new("example:app/run", "1.0.0");
    let mut manager = ComponentManager::new();
    manager.set_resource_limits(ComponentResourceLimits {
        max_concurrent_invocations: Some(0),
        ..ComponentResourceLimits::default()
    });
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("callable", "1", "test component")
                .with_export(interface.clone()),
            "callable.wasm",
        ))
        .unwrap();
    let instance = manager.instantiate_component("callable").unwrap();
    assert!(matches!(
        manager.invoke(ComponentInvocation::new(instance, interface, "run")),
        Err(ComponentError::ResourceLimitExceeded {
            limit: "concurrent invocations",
            ..
        })
    ));
}

#[test]
fn component_engine_normalizes_traps_interruptions_and_limit_failures() {
    let interface = WitInterface::new("example:app/run", "1.0.0");
    let mut trapping_engine = MockComponentEngine::new();
    trapping_engine.trap_on_invoke = Some(ComponentTrapKind::Trap);
    let mut manager = ComponentManager::with_engine(Box::new(trapping_engine));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component")
                .with_export(interface.clone()),
            "component.wasm",
        ))
        .unwrap();
    let instance = manager.instantiate_component("component").unwrap();
    assert!(matches!(
        manager.invoke(ComponentInvocation::new(instance, interface, "run")),
        Err(ComponentError::Trap {
            kind: ComponentTrapKind::Trap,
            ..
        })
    ));

    let mut manager = ComponentManager::with_engine(Box::new(
        MockComponentEngine::new().without_resource_limits(),
    ));
    manager.set_resource_limits(ComponentResourceLimits {
        require_memory_limit: true,
        max_memory_bytes: Some(1024),
        ..ComponentResourceLimits::default()
    });
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("limited", "1", "test component"),
            "limited.wasm",
        ))
        .unwrap();
    assert!(matches!(
        manager.instantiate_component("limited"),
        Err(ComponentError::ResourceLimitUnsupported { .. })
    ));
}

#[test]
fn distribution_source_identity_does_not_imply_trust() {
    let package =
        component_artifact_package(b"component-bytes", ComponentDistributionSourceKind::Tachyon);
    let mut manager = ComponentManager::new();

    assert!(matches!(
        manager.prepare_pushed_package(package),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Unknown,
            ..
        })
    ));
}

#[test]
fn pulled_component_package_resolves_fetches_and_validates_locally() {
    let bytes = b"component-bytes";
    let digest = ComponentDigest::sha256(bytes);
    let package =
        component_artifact_package(bytes, ComponentDistributionSourceKind::LocalDirectory);
    let source = TestComponentDistributionSource {
        package,
        candidates: vec![digest.clone()],
    };
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value.clone()));

    manager
        .prepare_pulled_package(&source, "magnetar.examples.hello", Some(">=0.1.0,<1.0.0"))
        .unwrap();

    assert_eq!(
        manager
            .definition("magnetar.examples.hello")
            .and_then(|definition| definition.artifact_digest.clone()),
        Some(digest)
    );
    assert!(manager.observations().iter().any(|observation| {
        observation.kind == ComponentObservationKind::Distribution
            && observation.message.contains("candidate digest")
    }));
}

fn temp_component_artifact_dir(label: &str) -> std::path::PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "magnetar-component-artifact-{label}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir_all(&directory).unwrap();
    directory
}

fn tokenizer_manifest_yaml(digest: &str) -> String {
    format!(
        r#"schema: magnetar-component-artifact
schema_version: 1
artifact:
  kind: component
  digest:
    algorithm: sha256
    value: "{digest}"
component:
  name: "magnetar.examples.tokenizer"
  version: "0.1.0"
  description: "Tokenizer Component fixture"
  role: "tokenizer"
runtime:
  magnetar:
    min_version: "0.1.0"
wit:
  imports:
    - package: "magnetar:compute"
      interface: "run"
      version: "2.0.0"
  exports:
    - package: "magnetar:tokenizer"
      interface: "tokenize"
      version: "1.0.0"
capabilities:
  requires:
    - id: "magnetar:compute/run"
      version: "2.0.0"
authority:
  requires:
    - tokenizer-artifact-read
    - compute-capability
    - observability-emit
publisher:
  id: "local-dev"
  name: "Local Development"
source:
  kind: "local"
  uri: "./fixtures/tokenizer.component.wasm"
signatures: []
"#
    )
}

#[test]
fn component_runtime_instantiates_without_generic_start_or_stop() {
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component"),
            "component.wasm",
        ))
        .unwrap();

    let instance = manager.instantiate_component("component").unwrap();
    assert_eq!(
        manager.definition_state("component"),
        Some(ComponentDefinitionState::Prepared)
    );
    assert_eq!(
        manager.instance_state(instance),
        Some(ComponentInstanceState::Ready)
    );

    manager.shutdown();
    assert_eq!(
        manager.instance_state(instance),
        None,
        "shutdown removes Runtime-owned Component instances"
    );
}

#[test]
fn component_artifact_reference_prepares_future_artifact_model_without_trust_policy() {
    let descriptor = ComponentDescriptor::new(
        ComponentMetadata::new("component", "1", "test component"),
        "component.wasm",
    );

    assert!(matches!(
        descriptor.artifact_reference(),
        ComponentArtifactReference::LocalPath(path) if path == std::path::Path::new("component.wasm")
    ));
}

#[test]
fn component_import_version_must_match_authorized_interface() {
    let authorized = WitInterface::new("magnetar:runtime/run", "1.0.0");
    let requested = WitInterface::new("magnetar:runtime/run", "2.0.0");
    let metadata = ComponentMetadata::new("consumer", "1", "test component").with_import(requested);
    let mut manager = ComponentManager::new();
    manager.provide_interface(authorized);
    manager
        .register_component(ComponentDescriptor::new(metadata, "consumer.wasm"))
        .unwrap();

    assert!(matches!(
        manager.instantiate_component("consumer"),
        Err(ComponentError::UnauthorizedImport { .. })
    ));
}

#[test]
fn inference_cache_registry_scopes_access_to_session_and_model() {
    let mut registry = InferenceCacheRegistry::default();
    let session = InferenceSessionId::new("session-a").unwrap();
    let authorized =
        InferenceCacheScope::new(InferenceCacheKind::Kv, session.clone(), "qwen-model").unwrap();
    registry.authorize(authorized.clone());

    registry.authorize_access(&authorized).unwrap();
    assert!(matches!(
        registry.authorize_access(
            &InferenceCacheScope::new(InferenceCacheKind::Kv, session, "other-model").unwrap()
        ),
        Err(ComponentError::ArtifactRejected { .. })
    ));
    assert!(matches!(
        registry.authorize_access(
            &InferenceCacheScope::new(
                InferenceCacheKind::Prefix,
                InferenceSessionId::new("session-b").unwrap(),
                "qwen-model"
            )
            .unwrap()
        ),
        Err(ComponentError::ArtifactRejected { .. })
    ));
}

#[test]
fn component_invocation_after_destruction_fails() {
    let interface = WitInterface::new("example:app/run", "1.0.0");
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component")
                .with_export(interface.clone()),
            "component.wasm",
        ))
        .unwrap();
    let instance = manager.instantiate_component("component").unwrap();
    manager.destroy_instance(instance).unwrap();

    assert!(matches!(
        manager.invoke(ComponentInvocation::new(instance, interface, "run")),
        Err(ComponentError::InstanceNotFound(_))
    ));
}

#[test]
fn component_shutdown_prevents_new_lifecycle_operations() {
    let interface = WitInterface::new("example:app/run", "1.0.0");
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component")
                .with_export(interface.clone()),
            "component.wasm",
        ))
        .unwrap();
    let instance = manager.instantiate_component("component").unwrap();
    manager.shutdown();

    assert!(matches!(
        manager.invoke(ComponentInvocation::new(instance, interface, "run")),
        Err(ComponentError::RuntimeShutdown)
    ));
    assert!(matches!(
        manager.instantiate_component("component"),
        Err(ComponentError::RuntimeShutdown)
    ));
    assert!(matches!(
        manager.register_component(ComponentDescriptor::new(
            ComponentMetadata::new("other", "1", "test component"),
            "other.wasm",
        )),
        Err(ComponentError::RuntimeShutdown)
    ));
}

#[test]
fn component_observations_are_non_authoritative_and_redacted() {
    let interface = WitInterface::new("example:app/run", "1.0.0");
    let mut engine = MockComponentEngine::new();
    engine.trap_on_invoke = Some(ComponentTrapKind::Trap);
    let mut manager = ComponentManager::with_engine(Box::new(engine));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("component", "1", "test component")
                .with_export(interface.clone()),
            "component.wasm",
        ))
        .unwrap();
    let instance = manager.instantiate_component("component").unwrap();

    assert!(matches!(
        manager.invoke(ComponentInvocation::new(instance, interface, "run")),
        Err(ComponentError::Trap { .. })
    ));
    assert!(
        manager
            .observations()
            .iter()
            .any(
                |observation| observation.kind == ComponentObservationKind::Trap
                    && observation.instance == Some(instance)
                    && observation.message.contains("[redacted component trap]")
            )
    );
    assert!(
        !manager
            .observations()
            .iter()
            .any(|observation| observation.message.contains("wasmtime::"))
    );
    assert!(!manager.observations().iter().any(|observation| {
        observation.message.contains("Provider")
            || observation.message.contains("Device")
            || observation.message.contains("Store")
    }));
}

#[test]
fn pushed_component_package_temp_materialization_is_removed_with_manager() {
    let bytes = b"component-bytes-cleanup";
    let digest = ComponentDigest::sha256(bytes);
    let package =
        component_artifact_package(bytes, ComponentDistributionSourceKind::ClientProvided);
    // `prepare_pushed_package`'s temp directory name is
    // "magnetar-distributed-component-{digest}-{counter}": the digest
    // portion is this test's own (from its own literal `bytes`), so
    // matching on that full prefix -- rather than the bare
    // "magnetar-distributed-component-" prefix every such directory
    // shares -- can no longer pick up a directory some other,
    // concurrently-running test created under `cargo test`'s default
    // thread-parallel execution (a prior version of this test raced on
    // exactly that and flaked in CI).
    let expected_prefix = format!(
        "magnetar-distributed-component-{}-",
        digest.value.replace(':', "-")
    );
    let materialized = {
        let mut manager = ComponentManager::new();
        manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value.clone()));
        manager.prepare_pushed_package(package).unwrap();
        std::fs::read_dir(std::env::temp_dir())
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with(&expected_prefix))
            })
            .expect("distributed component package materialized")
    };

    assert!(!materialized.exists());
}

#[test]
fn pulled_component_package_rejects_empty_candidate_list() {
    let package = component_artifact_package(
        b"component-bytes",
        ComponentDistributionSourceKind::LocalCache,
    );
    let source = TestComponentDistributionSource {
        package,
        candidates: Vec::new(),
    };
    let mut manager = ComponentManager::new();

    assert!(matches!(
        manager.prepare_pulled_package(&source, "magnetar.examples.hello", None),
        Err(ComponentError::Distribution {
            category: ComponentDistributionErrorCategory::ArtifactNotFound,
            ..
        })
    ));
}

#[test]
fn component_artifact_accepts_target_tokenizer_manifest_authorities() {
    let directory = temp_component_artifact_dir("tokenizer-authority");
    let artifact = directory.join("tokenizer.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("tokenizer.component.wasm.magnetar-component.yaml"),
        tokenizer_manifest_yaml(&digest.value),
    )
    .unwrap();

    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.tokenizer", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:compute/run", "2.0.0"))
                .with_export(WitInterface::new("magnetar:tokenizer/tokenize", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    manager
        .prepare_component("magnetar.examples.tokenizer")
        .unwrap();
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_rejects_broad_authority_kinds() {
    for authority in [
        "filesystem",
        "network",
        "secrets",
        "git",
        "workspace",
        "process",
    ] {
        let directory = temp_component_artifact_dir(authority);
        let artifact = directory.join("hello.component.wasm");
        let bytes = b"component-bytes";
        fs::write(&artifact, bytes).unwrap();
        let digest = ComponentDigest::sha256(bytes);
        fs::write(
            directory.join("hello.component.wasm.magnetar-component.yaml"),
            manifest_yaml_with_authority(&digest.value, &[authority]),
        )
        .unwrap();
        let mut manager = ComponentManager::new();
        manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
        manager
            .register_component(ComponentDescriptor::new(
                ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                    .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                    .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
                &artifact,
            ))
            .unwrap();

        assert!(matches!(
            manager.prepare_component("magnetar.examples.hello"),
            Err(ComponentError::Manifest { message, .. })
                if message == "authority kind is outside Magnetar inference scope"
        ));
        fs::remove_dir_all(directory).unwrap();
    }
}

#[test]
fn component_artifact_rejects_unknown_authority_kinds() {
    let directory = temp_component_artifact_dir("unknown-authority");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml_with_authority(&digest.value, &["workspace-admin"]),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::Manifest { message, .. }) if message == "unsupported authority kind"
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_accepts_model_artifact_read_authority_when_trusted() {
    let directory = temp_component_artifact_dir("model-artifact-authority");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml_with_authority(&digest.value, &["model-artifact-read"]),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    manager
        .prepare_component("magnetar.examples.hello")
        .unwrap();
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_trust_overrides_do_not_allow_forbidden_authority() {
    for (label, trust_store, source_kind) in [
        ("trusted-digest", ComponentTrustStore::default(), "local"),
        (
            "development-mode",
            ComponentTrustStore::default().allow_unsigned_local_development(true),
            "local",
        ),
    ] {
        let directory = temp_component_artifact_dir(label);
        let artifact = directory.join("hello.component.wasm");
        let bytes = b"component-bytes";
        fs::write(&artifact, bytes).unwrap();
        let digest = ComponentDigest::sha256(bytes);
        let mut trust_store = trust_store;
        if label == "trusted-digest" {
            trust_store = trust_store.trust_digest(digest.value.clone());
        }
        let manifest = manifest_yaml_with_authority(&digest.value, &["filesystem"])
            .replace("  kind: \"local\"", &format!("  kind: \"{source_kind}\""));
        fs::write(
            directory.join("hello.component.wasm.magnetar-component.yaml"),
            manifest,
        )
        .unwrap();
        let mut manager = ComponentManager::new();
        manager.set_trust_store(trust_store);
        manager
            .register_component(ComponentDescriptor::new(
                ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                    .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                    .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
                &artifact,
            ))
            .unwrap();

        assert!(matches!(
            manager.prepare_component("magnetar.examples.hello"),
            Err(ComponentError::Manifest { message, .. })
                if message == "authority kind is outside Magnetar inference scope"
        ));
        fs::remove_dir_all(directory).unwrap();
    }
}

#[test]
fn component_authority_rejection_is_observed_with_reason_before_prepare() {
    let directory = temp_component_artifact_dir("authority-diagnostic");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml_with_authority(&digest.value, &["filesystem"]),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::Manifest { .. })
    ));
    assert_eq!(
        manager.definition_state("magnetar.examples.hello"),
        Some(ComponentDefinitionState::Failed)
    );
    assert!(manager.observations().iter().any(|observation| {
        observation.kind == ComponentObservationKind::Validation
            && observation.message.contains("component authority rejected")
            && observation
                .message
                .contains("authority kind is outside Magnetar inference scope")
    }));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_validation_observations_redact_paths_and_secrets() {
    let directory = temp_component_artifact_dir("redacted-diagnostic");
    let artifact = directory.join("hello.component.wasm");
    fs::write(&artifact, b"component-bytes").unwrap();
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture"),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ManifestMissing { .. })
    ));
    let messages = manager
        .observations()
        .iter()
        .map(|observation| observation.message.as_str())
        .collect::<Vec<_>>();
    assert!(
        messages
            .iter()
            .any(|message| message.contains("[redacted]"))
    );
    assert!(!messages.iter().any(|message| {
        message.contains(directory.to_string_lossy().as_ref())
            || message.to_ascii_lowercase().contains("secret")
    }));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_pipeline_requires_external_trust_policy_before_prepare() {
    let directory = temp_component_artifact_dir("trusted");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    let manifest = directory.join("hello.component.wasm.magnetar-component.yaml");
    fs::write(
        &manifest,
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();

    let import = WitInterface::new("magnetar:test/echo", "1.0.0");
    let export = WitInterface::new("magnetar:test/run", "1.0.0");
    let metadata = ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
        .with_import(import)
        .with_export(export);
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value.clone()));
    manager
        .register_component(ComponentDescriptor::new(metadata, &artifact))
        .unwrap();

    manager
        .prepare_component("magnetar.examples.hello")
        .unwrap();
    let definition = manager.definition("magnetar.examples.hello").unwrap();
    assert_eq!(definition.artifact_digest, Some(digest));
    assert!(matches!(
        definition.trust_decision,
        Some(ComponentTrustDecision {
            status: ComponentTrustStatus::Trusted,
            ..
        })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_rejects_local_wasm_without_manifest() {
    let directory = temp_component_artifact_dir("missing-manifest");
    let artifact = directory.join("unknown.component.wasm");
    fs::write(&artifact, b"component-bytes").unwrap();
    let mut manager = ComponentManager::new();
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("unknown", "0.1.0", "fixture"),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("unknown"),
        Err(ComponentError::ManifestMissing { .. })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_rejects_digest_mismatch_before_prepare() {
    let directory = temp_component_artifact_dir("digest-mismatch");
    let artifact = directory.join("hello.component.wasm");
    fs::write(&artifact, b"component-bytes").unwrap();
    let digest = ComponentDigest::sha256(b"different-bytes");
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Rejected,
            ..
        })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_rejects_manifest_wit_that_differs_from_actual_contract() {
    let directory = temp_component_artifact_dir("wit-mismatch");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();

    let mut engine = MockComponentEngine::new();
    let mut actual = ComponentContract::default();
    actual.imports.insert(ComponentImportRequirement::new(
        WitInterface::new("magnetar:test/other", "1.0.0"),
        ComponentInterfaceShape::Interface,
    ));
    engine.prepared_contract = Some(actual);
    let mut manager = ComponentManager::with_engine(Box::new(engine));
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture"),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ContractValidationFailed { .. })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_trust_store_revocation_overrides_digest_allowlist() {
    let directory = temp_component_artifact_dir("revoked");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(
        ComponentTrustStore::default()
            .trust_digest(digest.value.clone())
            .revoke_digest(digest.value),
    );
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Revoked,
            ..
        })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_manifest_may_declare_optional_wit_import_metadata() {
    let directory = temp_component_artifact_dir("optional-import");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    let manifest = manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION).replace(
        "  exports:",
        "    - package: \"magnetar:optional\"\n      interface: \"telemetry\"\n      version: \"1.0.0\"\n      optional: true\n  exports:",
    );
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest,
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    manager
        .prepare_component("magnetar.examples.hello")
        .unwrap();
    fs::remove_dir_all(directory).unwrap();
}

/// astorise/Magnetar#83 steps 6-7: the schema-level half of the
/// Component-vs-family compatibility gate, tested directly at
/// `ComponentManifest::from_yaml_bytes` (parsing/validation only, no
/// `ComponentManager`/trust/registration machinery needed) rather than
/// only through the higher-level integration proofs in
/// `first_native_runtime/tests.rs` and `inference-components`.
#[test]
fn component_manifest_parses_declared_architecture_families() {
    let digest = ComponentDigest::sha256(b"component-bytes");
    let yaml = manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION).replace(
        "wit:\n",
        "compatibility:\n  architecture_families:\n    - llama\n    - mistral\nwit:\n",
    );
    let manifest = ComponentManifest::from_yaml_bytes(yaml.as_bytes(), Path::new("<test>"))
        .expect("a manifest declaring architecture_families parses");
    assert_eq!(
        manifest.supported_architecture_families,
        BTreeSet::from(["llama".to_string(), "mistral".to_string()])
    );
}

/// The other half of astorise/Magnetar#83's backward-compatibility
/// requirement: every manifest written before this field existed has no
/// `compatibility` block at all, and must keep parsing to the same
/// permissive (empty) `supported_architecture_families` it always did.
#[test]
fn component_manifest_with_no_compatibility_block_is_permissive() {
    let digest = ComponentDigest::sha256(b"component-bytes");
    let yaml = manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION);
    let manifest = ComponentManifest::from_yaml_bytes(yaml.as_bytes(), Path::new("<test>"))
        .expect("a manifest with no compatibility block parses");
    assert!(
        manifest.supported_architecture_families.is_empty(),
        "no compatibility block declared must mean permissive, exactly like before this field \
         existed"
    );
}

/// An author writing `architecture_families: []` is declaring *something*
/// -- treating it the same as omitting the field entirely (permissive)
/// would let a typo'd or generated-empty list accidentally open a
/// Component to every architecture, silently.
#[test]
fn component_manifest_rejects_an_explicitly_empty_architecture_families_list() {
    let digest = ComponentDigest::sha256(b"component-bytes");
    let yaml = manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION).replace(
        "wit:\n",
        "compatibility:\n  architecture_families: []\nwit:\n",
    );
    let error = ComponentManifest::from_yaml_bytes(yaml.as_bytes(), Path::new("<test>"))
        .expect_err("an explicitly empty architecture_families list must be rejected");
    assert!(matches!(error, ComponentError::Manifest { .. }));
}

/// astorise/Magnetar#75's own schema-level half, mirroring
/// `component_manifest_parses_declared_architecture_families` one field
/// over: `compatibility.artifact_formats` populates
/// `ComponentManifest::supported_artifact_formats`.
#[test]
fn component_manifest_parses_declared_artifact_formats() {
    let digest = ComponentDigest::sha256(b"component-bytes");
    let yaml = manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION).replace(
        "wit:\n",
        "compatibility:\n  artifact_formats:\n    - huggingface\n    - gguf\nwit:\n",
    );
    let manifest = ComponentManifest::from_yaml_bytes(yaml.as_bytes(), Path::new("<test>"))
        .expect("a manifest declaring artifact_formats parses");
    assert_eq!(
        manifest.supported_artifact_formats,
        BTreeSet::from(["huggingface".to_string(), "gguf".to_string()])
    );
}

/// Mirrors `component_manifest_with_no_compatibility_block_is_permissive`:
/// no `compatibility` block declared must mean permissive on artifact
/// format too, exactly like it always has for architecture family.
#[test]
fn component_manifest_with_no_compatibility_block_is_permissive_on_artifact_format() {
    let digest = ComponentDigest::sha256(b"component-bytes");
    let yaml = manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION);
    let manifest = ComponentManifest::from_yaml_bytes(yaml.as_bytes(), Path::new("<test>"))
        .expect("a manifest with no compatibility block parses");
    assert!(
        manifest.supported_artifact_formats.is_empty(),
        "no compatibility block declared must mean permissive, any-format compatibility"
    );
}

/// Mirrors `component_manifest_rejects_an_explicitly_empty_architecture_families_list`:
/// an explicitly empty `artifact_formats: []` is a declaration of
/// *something*, not permissiveness, so it is rejected at parse time rather
/// than silently treated the same as omitting the field.
#[test]
fn component_manifest_rejects_an_explicitly_empty_artifact_formats_list() {
    let digest = ComponentDigest::sha256(b"component-bytes");
    let yaml = manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION)
        .replace("wit:\n", "compatibility:\n  artifact_formats: []\nwit:\n");
    let error = ComponentManifest::from_yaml_bytes(yaml.as_bytes(), Path::new("<test>"))
        .expect_err("an explicitly empty artifact_formats list must be rejected");
    assert!(matches!(error, ComponentError::Manifest { .. }));
}

/// An `artifact_formats` entry that is not a recognized
/// [`crate::model::ArtifactFormat`] string is rejected at parse time
/// (astorise/Magnetar#75) -- a typo here must never silently become a
/// permissive-by-accident restriction nobody's Artifact can ever satisfy,
/// nor pass through to become a value the runtime compatibility check
/// mysteriously never matches.
#[test]
fn component_manifest_rejects_an_unrecognized_artifact_format() {
    let digest = ComponentDigest::sha256(b"component-bytes");
    let yaml = manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION).replace(
        "wit:\n",
        "compatibility:\n  artifact_formats:\n    - onnx\nwit:\n",
    );
    let error = ComponentManifest::from_yaml_bytes(yaml.as_bytes(), Path::new("<test>"))
        .expect_err("an unrecognized artifact_formats entry must be rejected");
    assert!(matches!(error, ComponentError::Manifest { .. }));
}

#[test]
fn component_artifact_rejects_runtime_max_version_incompatibility() {
    let directory = temp_component_artifact_dir("runtime-max");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    let manifest = manifest_yaml(&digest.value, "0.0.1").replace(
        "    min_version: \"0.0.1\"",
        "    min_version: \"0.0.1\"\n    max_version: \"0.0.1\"",
    );
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest,
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Rejected,
            ..
        })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_rejects_incompatible_capability_versions() {
    for (label, capability_block) in [
        (
            "cap-major",
            "    - id: \"magnetar:test/echo\"\n      version: \"2.0.0\"",
        ),
        (
            "cap-range",
            "    - id: \"magnetar:test/echo\"\n      version: \"1.0.0\"\n      max_version: \"0.9.0\"",
        ),
    ] {
        let directory = temp_component_artifact_dir(label);
        let artifact = directory.join("hello.component.wasm");
        let bytes = b"component-bytes";
        fs::write(&artifact, bytes).unwrap();
        let digest = ComponentDigest::sha256(bytes);
        let manifest = manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION).replace(
            "    - id: \"magnetar:test/echo\"\n      version: \"1.0.0\"",
            capability_block,
        );
        fs::write(
            directory.join("hello.component.wasm.magnetar-component.yaml"),
            manifest,
        )
        .unwrap();
        let mut manager = ComponentManager::new();
        manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
        manager
            .register_component(ComponentDescriptor::new(
                ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                    .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                    .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
                &artifact,
            ))
            .unwrap();

        assert!(matches!(
            manager.prepare_component("magnetar.examples.hello"),
            Err(ComponentError::ArtifactRejected {
                status: ComponentTrustStatus::Rejected,
                ..
            })
        ));
        fs::remove_dir_all(directory).unwrap();
    }
}

#[test]
fn component_publisher_and_source_metadata_do_not_grant_trust() {
    let directory = temp_component_artifact_dir("publisher-policy");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    let descriptor = || {
        ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        )
    };
    let mut untrusted = ComponentManager::new();
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();
    untrusted.register_component(descriptor()).unwrap();
    assert!(matches!(
        untrusted.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Unknown,
            ..
        })
    ));

    let mut publisher_claim = ComponentManager::new();
    publisher_claim.set_trust_store(ComponentTrustStore::default().trust_publisher("local-dev"));
    publisher_claim.register_component(descriptor()).unwrap();
    assert!(matches!(
        publisher_claim.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Unknown,
            ..
        })
    ));

    let mut explicit_local_development = ComponentManager::new();
    explicit_local_development.set_trust_store(
        ComponentTrustStore::default()
            .trust_publisher("local-dev")
            .trust_source("local")
            .allow_unsigned_local_development(true),
    );
    explicit_local_development
        .register_component(descriptor())
        .unwrap();
    explicit_local_development
        .prepare_component("magnetar.examples.hello")
        .unwrap();

    let tachyon_manifest = manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION)
        .replace("  kind: \"local\"", "  kind: \"tachyon\"");
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        tachyon_manifest,
    )
    .unwrap();
    let mut source_claim = ComponentManager::new();
    source_claim.set_trust_store(ComponentTrustStore::default().trust_source("tachyon"));
    source_claim.register_component(descriptor()).unwrap();
    assert!(matches!(
        source_claim.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Unknown,
            ..
        })
    ));

    let mut digest_trusted = ComponentManager::new();
    digest_trusted.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    digest_trusted.register_component(descriptor()).unwrap();
    digest_trusted
        .prepare_component("magnetar.examples.hello")
        .unwrap();
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_signature_metadata_is_recorded_but_not_trusted_by_itself() {
    let directory = temp_component_artifact_dir("signature");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    let signed_manifest = manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION).replace(
        "signatures: []",
        &format!(
            "signatures:\n  - algorithm: \"test\"\n    digest: \"{}\"",
            digest.value
        ),
    );
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        signed_manifest,
    )
    .unwrap();
    let descriptor = || {
        ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        )
    };
    let mut no_trust = ComponentManager::new();
    no_trust.register_component(descriptor()).unwrap();
    assert!(matches!(
        no_trust.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Unknown,
            ..
        })
    ));

    let mut trusted = ComponentManager::new();
    trusted.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value.clone()));
    trusted.register_component(descriptor()).unwrap();
    trusted
        .prepare_component("magnetar.examples.hello")
        .unwrap();

    let bad_manifest = manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION).replace(
        "signatures: []",
        "signatures:\n  - algorithm: \"test\"\n    digest: \"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"",
    );
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        bad_manifest,
    )
    .unwrap();
    let mut mismatch = ComponentManager::new();
    mismatch.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    mismatch.register_component(descriptor()).unwrap();
    assert!(matches!(
        mismatch.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Rejected,
            ..
        })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_development_mode_is_explicit_and_still_validates_artifact() {
    let directory = temp_component_artifact_dir("development");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().allow_unsigned_local_development(true));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    manager
        .prepare_component("magnetar.examples.hello")
        .unwrap();
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_cache_is_digest_keyed_and_non_authoritative() {
    let bytes = b"component-bytes".to_vec();
    let mut cache = ComponentArtifactCache::default();
    let digest = cache.insert(bytes.clone());
    assert!(cache.contains_untrusted(&digest));
    assert_eq!(cache.get_verified(&digest).unwrap(), Some(bytes.as_slice()));

    let wrong_digest = ComponentDigest::sha256(b"wrong");
    cache.insert_unchecked_for_test(wrong_digest.clone(), bytes);
    assert!(matches!(
        cache.get_verified(&wrong_digest),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Rejected,
            ..
        })
    ));
}

#[test]
fn component_quarantine_prevents_preparation_and_preserves_diagnostic_status() {
    let directory = temp_component_artifact_dir("quarantine");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().quarantine_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    assert!(matches!(
        manager.prepare_component("magnetar.examples.hello"),
        Err(ComponentError::ArtifactRejected {
            status: ComponentTrustStatus::Quarantined,
            ..
        })
    ));
    assert!(
        manager
            .observations()
            .iter()
            .any(|observation| observation.message.contains("Quarantined"))
    );
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn component_artifact_validation_emits_structured_observations() {
    let directory = temp_component_artifact_dir("observations");
    let artifact = directory.join("hello.component.wasm");
    let bytes = b"component-bytes";
    fs::write(&artifact, bytes).unwrap();
    let digest = ComponentDigest::sha256(bytes);
    fs::write(
        directory.join("hello.component.wasm.magnetar-component.yaml"),
        manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION),
    )
    .unwrap();
    let mut manager = ComponentManager::new();
    manager.set_trust_store(ComponentTrustStore::default().trust_digest(digest.value));
    manager
        .register_component(ComponentDescriptor::new(
            ComponentMetadata::new("magnetar.examples.hello", "0.1.0", "fixture")
                .with_import(WitInterface::new("magnetar:test/echo", "1.0.0"))
                .with_export(WitInterface::new("magnetar:test/run", "1.0.0")),
            &artifact,
        ))
        .unwrap();

    manager
        .prepare_component("magnetar.examples.hello")
        .unwrap();
    let messages = manager
        .observations()
        .iter()
        .map(|observation| observation.message.as_str())
        .collect::<Vec<_>>();
    assert!(
        messages
            .iter()
            .any(|message| message.contains("discovered"))
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("digest computed"))
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("manifest loaded"))
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("WIT declarations match"))
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("compatibility validated"))
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("trust decision"))
    );
    fs::remove_dir_all(directory).unwrap();
}

// MAG-02: Ed25519 Component Artifact signature verification
// (docs/cryptographic-artifact-signatures.md). `ComponentTrustStore::evaluate`
// is a pure function of `(manifest, digest)`, so these tests parse a manifest
// directly via `from_yaml_bytes` and call `evaluate` directly, without the
// filesystem/engine machinery `ComponentManager::prepare_component` needs.

fn signature_test_keypair(seed: u8) -> (ed25519_dalek::SigningKey, VerifyingKeyBytes) {
    let signing_key = ed25519_dalek::SigningKey::from_bytes(&[seed; 32]);
    let public_key = signing_key.verifying_key().to_bytes();
    (signing_key, public_key)
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn manifest_yaml_with_signature(
    digest: &str,
    signed_digest: &str,
    key_id: &str,
    signature_hex: &str,
) -> String {
    manifest_yaml(digest, MAGNETAR_RUNTIME_VERSION).replace(
        "signatures: []",
        &format!(
            "signatures:\n  - kind: \"ed25519\"\n    key_id: \"{key_id}\"\n    digest: \"{signed_digest}\"\n    signature: \"{signature_hex}\"\n"
        ),
    )
}

fn parse_manifest(yaml: &str) -> ComponentManifest {
    ComponentManifest::from_yaml_bytes(yaml.as_bytes(), Path::new("<test>"))
        .expect("manifest parses")
}

#[test]
fn component_signature_from_trusted_key_grants_trust_without_digest_pinning() {
    let digest = ComponentDigest::sha256(b"component-bytes");
    let (signing_key, public_key) = signature_test_keypair(1);
    let key_id = key_id_for(&public_key);
    let signature = signing_key.sign(digest.value.as_bytes()).to_bytes();
    let manifest = parse_manifest(&manifest_yaml_with_signature(
        &digest.value,
        &digest.value,
        &key_id,
        &hex_encode(&signature),
    ));

    let trust_store = ComponentTrustStore::default()
        .trust_publisher_key(public_key, PublisherIdentity::new("Acme Publishing"));
    let decision = trust_store.evaluate(&manifest, &digest);

    assert_eq!(decision.status, ComponentTrustStatus::Trusted);
    assert_eq!(
        decision.authenticated_publisher,
        Some(PublisherIdentity::new("Acme Publishing"))
    );
}

#[test]
fn component_digest_pinning_continues_to_work_unchanged_without_a_signature() {
    let digest = ComponentDigest::sha256(b"component-bytes");
    let manifest = parse_manifest(&manifest_yaml(&digest.value, MAGNETAR_RUNTIME_VERSION));

    let trust_store = ComponentTrustStore::default().trust_digest(digest.value.clone());
    let decision = trust_store.evaluate(&manifest, &digest);

    assert_eq!(decision.status, ComponentTrustStatus::Trusted);
    assert_eq!(decision.reason, "digest trusted");
    assert_eq!(decision.authenticated_publisher, None);
}

#[test]
fn component_signature_from_an_untrusted_key_falls_through_to_unknown() {
    let digest = ComponentDigest::sha256(b"component-bytes");
    let (signing_key, public_key) = signature_test_keypair(2);
    let key_id = key_id_for(&public_key);
    let signature = signing_key.sign(digest.value.as_bytes()).to_bytes();
    let manifest = parse_manifest(&manifest_yaml_with_signature(
        &digest.value,
        &digest.value,
        &key_id,
        &hex_encode(&signature),
    ));

    // The trust store trusts no digest and no publisher key at all -- the
    // signature's key id resolves to nothing in `trusted_publisher_keys`.
    let trust_store = ComponentTrustStore::default();
    let decision = trust_store.evaluate(&manifest, &digest);

    assert_eq!(decision.status, ComponentTrustStatus::Unknown);
}

#[test]
fn component_signature_with_wrong_bytes_under_a_known_key_is_rejected() {
    let digest = ComponentDigest::sha256(b"component-bytes");
    let (signing_key, public_key) = signature_test_keypair(3);
    let key_id = key_id_for(&public_key);
    let mut signature = signing_key.sign(digest.value.as_bytes()).to_bytes();
    signature[0] ^= 0xFF;
    let manifest = parse_manifest(&manifest_yaml_with_signature(
        &digest.value,
        &digest.value,
        &key_id,
        &hex_encode(&signature),
    ));

    let trust_store = ComponentTrustStore::default()
        .trust_publisher_key(public_key, PublisherIdentity::new("Acme Publishing"));
    let decision = trust_store.evaluate(&manifest, &digest);

    assert_eq!(decision.status, ComponentTrustStatus::Rejected);
}

#[test]
fn component_signature_with_mismatched_signed_digest_under_a_known_key_is_rejected() {
    let digest = ComponentDigest::sha256(b"component-bytes");
    let (signing_key, public_key) = signature_test_keypair(4);
    let key_id = key_id_for(&public_key);
    let signature = signing_key.sign(digest.value.as_bytes()).to_bytes();
    // `signatures[].digest` claims a different digest than the artifact's
    // own actual computed digest -- rejected even though the signature
    // bytes would verify against the claimed (wrong) digest.
    let manifest = parse_manifest(&manifest_yaml_with_signature(
        &digest.value,
        "sha256:0000000000000000000000000000000000000000000000000000000000000000",
        &key_id,
        &hex_encode(&signature),
    ));

    let trust_store = ComponentTrustStore::default()
        .trust_publisher_key(public_key, PublisherIdentity::new("Acme Publishing"));
    let decision = trust_store.evaluate(&manifest, &digest);

    assert_eq!(decision.status, ComponentTrustStatus::Rejected);
}

#[test]
fn component_revoked_signature_key_is_revoked_even_though_otherwise_valid() {
    let digest = ComponentDigest::sha256(b"component-bytes");
    let (signing_key, public_key) = signature_test_keypair(5);
    let key_id = key_id_for(&public_key);
    let signature = signing_key.sign(digest.value.as_bytes()).to_bytes();
    let manifest = parse_manifest(&manifest_yaml_with_signature(
        &digest.value,
        &digest.value,
        &key_id,
        &hex_encode(&signature),
    ));

    let trust_store = ComponentTrustStore::default()
        .trust_publisher_key(public_key, PublisherIdentity::new("Acme Publishing"))
        .revoke_key(key_id);
    let decision = trust_store.evaluate(&manifest, &digest);

    assert_eq!(decision.status, ComponentTrustStatus::Revoked);
}

#[test]
fn component_revoking_a_key_does_not_distrust_a_digest_pinned_artifact() {
    let digest = ComponentDigest::sha256(b"component-bytes");
    let (signing_key, public_key) = signature_test_keypair(6);
    let key_id = key_id_for(&public_key);
    let signature = signing_key.sign(digest.value.as_bytes()).to_bytes();
    let manifest = parse_manifest(&manifest_yaml_with_signature(
        &digest.value,
        &digest.value,
        &key_id,
        &hex_encode(&signature),
    ));

    let trust_store = ComponentTrustStore::default()
        .trust_digest(digest.value.clone())
        .trust_publisher_key(public_key, PublisherIdentity::new("Acme Publishing"))
        .revoke_key(key_id);
    let decision = trust_store.evaluate(&manifest, &digest);

    assert_eq!(decision.status, ComponentTrustStatus::Trusted);
    assert_eq!(decision.reason, "digest trusted");
}
