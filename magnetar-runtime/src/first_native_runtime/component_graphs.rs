//! Graph construction from registered inference Components.

use super::*;

/// Holds the real Qwen Component's compiled artifact and its registered
/// [`GraphBuilderCapability`], built exactly once per process
/// (`qwen_real_component_runtime`) and reused across every call to
/// [`build_first_native_graphs_from_real_qwen_component`] -- compiling a
/// real (non-trivial) wasm Component is expensive (real, measured cost:
/// low seconds), while instantiating an already-compiled one and running
/// it is cheap. Without this cache, every dispatch call in
/// `execute_generation_step` -- once per generation *step*, not once per
/// generation -- would recompile the Component from scratch, which was
/// measured to take a single E2E test from milliseconds to ~28 seconds.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub(super) struct QwenRealComponentRuntime {
    pub(super) manager: Mutex<ComponentManager>,
    pub(super) capability: Arc<GraphBuilderCapability>,
    pub(super) model_config_capability: Arc<ModelConfigCapability>,
    pub(super) definition: ComponentDefinitionId,
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub(super) fn qwen_real_component_runtime()
-> Result<&'static QwenRealComponentRuntime, E2eConformanceError> {
    static RUNTIME: std::sync::OnceLock<QwenRealComponentRuntime> = std::sync::OnceLock::new();
    if let Some(runtime) = RUNTIME.get() {
        return Ok(runtime);
    }
    let capability = Arc::new(GraphBuilderCapability::new());
    let model_config_capability = Arc::new(ModelConfigCapability::new());
    let mut manager = ComponentManager::with_engine(Box::new(
        crate::component_wasmtime::WasmtimeComponentEngine::new().map_err(|error| {
            E2eConformanceError::ModelComponentFailed {
                reason: error.to_string(),
            }
        })?,
    ));
    manager.set_resource_limits(inference_component_runtime_limits());
    manager
        .set_trust_store(ComponentTrustStore::default().trust_digest(QWEN_REAL_COMPONENT_DIGEST));
    // `1.3.0`: the checked-in real Qwen Component now also exports
    // `build-prefill-graph-segment`/`build-decode-graph-segment`
    // (`add-real-multi-device-model-instance-placement`), a purely
    // additive evolution over `1.2.0`'s `attention-bias` field (task
    // 10.5's sibling QKV bias support), itself additive over `1.1.0`'s
    // `model-config` shape, itself additive over `1.0.0`'s
    // `graph-builder`-only world. `1.5.0` (`execute-and-cache-moe-experts`)
    // adds `model-config.mixture-of-experts`, and the Component also checks
    // `rope-scaling` (`1.4.0`).
    let graph_builder_interface =
        WitInterface::new("magnetar:model-component-graph/graph-builder", "1.5.0");
    manager.provide_capability(
        graph_builder_interface,
        capability.clone() as Arc<dyn HostCapability>,
    );
    let model_config_interface =
        WitInterface::new("magnetar:model-component-graph/model-config", "1.5.0");
    manager.provide_capability(
        model_config_interface,
        model_config_capability.clone() as Arc<dyn HostCapability>,
    );
    let definition = manager
        .prepare_pushed_package(qwen_real_component_package()?)
        .map_err(|error| E2eConformanceError::ModelComponentFailed {
            reason: error.to_string(),
        })?;
    // `OnceLock::set` losing a race is not an error here: the losing
    // thread's freshly-built `manager`/`capability` are simply dropped,
    // and every caller (winner and losers alike) reads back through
    // `RUNTIME.get()` afterwards, so they all observe the same instance.
    let _ = RUNTIME.set(QwenRealComponentRuntime {
        manager: Mutex::new(manager),
        capability,
        model_config_capability,
        definition,
    });
    Ok(RUNTIME.get().expect("just set or set by a racing caller"))
}

/// Test-oracle only (`reach-architecture-freeze-1` task 12.4): the checked-in
/// real Qwen Component binary, embedded for test fixtures. Production never
/// reads this -- see the `not(test)` branch of [`qwen_real_component_package`]
/// below.
#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine",
    test
))]
pub(super) const QWEN_REAL_COMPONENT_BYTES: &[u8] =
    include_bytes!("../../fixtures/components/qwen-real.component.wasm");

#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine",
    test
))]
pub(super) const QWEN_REAL_COMPONENT_MANIFEST_BYTES: &[u8] =
    include_bytes!("../../fixtures/components/qwen-real.component.wasm.magnetar-component.yaml");

#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine",
    test
))]
pub(super) fn qwen_real_component_package() -> Result<ComponentArtifactPackage, E2eConformanceError>
{
    Ok(ComponentArtifactPackage::new(
        QWEN_REAL_COMPONENT_BYTES.to_vec(),
        QWEN_REAL_COMPONENT_MANIFEST_BYTES.to_vec(),
        ComponentDigest::parse("sha256", QWEN_REAL_COMPONENT_DIGEST),
        ComponentDistributionSource::new(
            ComponentDistributionSourceKind::DevelopmentFixture,
            QWEN_REAL_COMPONENT_NAME,
        ),
    ))
}

// ---------------------------------------------------------------------------
// Generic, multi-Component runtime registry
// (`wire-generic-inference-component-runtime`)
// ---------------------------------------------------------------------------
//
// `qwen_real_component_runtime` above is a single, process-wide, hardcoded-
// digest slot: exactly one Component can ever be cached, and only a
// Component whose bytes hash to the one baked-in `QWEN_REAL_COMPONENT_DIGEST`
// constant is ever accepted, regardless of what a caller (e.g. an embedder
// like `magnetar-inference-component` acting on Tachyon's behalf) actually
// registers via `register_qwen_component_artifact`. That is deliberately
// left untouched here -- it is the CLI's own `"qwen-test"` self-test/demo
// alias, and a large existing test suite depends on its exact singleton
// timing and caching behavior.
//
// This section is the generic replacement an external embedder should use
// instead: an arbitrary number of distinct Components can be registered
// concurrently, each keyed by its own REAL digest (computed from its actual
// bytes via `ComponentDigest::sha256`, never a caller-supplied claim taken
// on faith), and trust is evaluated against a caller-supplied
// `ComponentTrustStore` -- never a hardcoded constant. Graph production
// itself (`build_first_native_graphs_with_runtime`) is unchanged and shared
// with the Qwen singleton path: nothing about building a graph from a
// Component was ever Qwen-specific, only *which* compiled Component and
// *whose* trust decision gated using it.

/// One caller-registered Component's compiled artifact and its registered
/// [`GraphBuilderCapability`]/[`ModelConfigCapability`], analogous to
/// [`QwenRealComponentRuntime`] but keyed by real digest in
/// [`register_inference_component_artifact`]'s registry rather than being a
/// single global slot.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub(super) struct RegisteredComponentRuntime {
    pub(super) manager: Mutex<ComponentManager>,
    pub(super) capability: Arc<GraphBuilderCapability>,
    pub(super) model_config_capability: Arc<ModelConfigCapability>,
    pub(super) definition: ComponentDefinitionId,
    /// This Component's own declared [`ComponentManifest::
    /// supported_architecture_families`] (astorise/Magnetar#83 steps 6-7),
    /// captured at registration time so a later caller holding only this
    /// Component's `ComponentDigest` (e.g. [`ProductionLoadedModel::
    /// load_with_component`]) can look up what it declares without
    /// threading the manifest itself through every intermediate call.
    pub(super) supported_architecture_families: BTreeSet<String>,
    /// This Component's own declared [`ComponentManifest::
    /// supported_artifact_formats`] (astorise/Magnetar#75), captured at
    /// registration time for the same reason
    /// `supported_architecture_families` is: [`ProductionLoadedModel::
    /// load_with_component`] needs it to reject an incompatible Artifact
    /// bundle format without re-parsing this Component's manifest.
    pub(super) supported_artifact_formats: BTreeSet<String>,
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub(super) static REGISTERED_COMPONENT_RUNTIMES: std::sync::OnceLock<
    Mutex<BTreeMap<String, Arc<RegisteredComponentRuntime>>>,
> = std::sync::OnceLock::new();

/// [`register_inference_component_artifact`]'s return value: the registered
/// Component's real digest, plus the already-validated [`ComponentManifest`]
/// parsed from `manifest_bytes` (astorise/Magnetar#83 steps 6-7). Carrying
/// the manifest out here means a caller that needs to know what this
/// specific Component declares -- most importantly
/// [`ComponentManifest::supported_architecture_families`] -- never needs a
/// second, independent YAML parse of bytes this function already parsed and
/// validated once.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[derive(Debug)]
pub struct RegisteredInferenceComponent {
    pub digest: ComponentDigest,
    pub manifest: ComponentManifest,
}

/// Registers an arbitrary Component artifact for generic first-native graph
/// production, trusted against `trust` (the caller's own
/// [`ComponentTrustStore`], never a hardcoded digest) rather than the single
/// `QWEN_REAL_COMPONENT_DIGEST` constant [`register_qwen_component_artifact`]
/// always uses. The returned [`RegisteredInferenceComponent::digest`] is
/// computed from `component_bytes` themselves (`ComponentDigest::sha256`) --
/// never a caller-supplied claim -- and doubles as the handle
/// [`build_first_native_graphs_from_named_component`] looks the registered
/// runtime back up by, so a caller never needs to invent or track its own
/// separate identifier for "the Component I just registered."
///
/// Idempotent per digest for *compilation*, like
/// [`register_qwen_component_artifact`]: a second registration of bytes that
/// hash to a digest already present in the registry never recompiles the
/// Component, so a caller can register unconditionally on every load rather
/// than tracking its own "have I already registered this one" state.
/// Distinct Components (distinct digests) coexist in the registry
/// simultaneously -- unlike the Qwen singleton, this is not a "last one
/// wins" slot.
///
/// Never idempotent for *authorization*, though (#71): reusing an already-
/// compiled runtime is a cache concern, not a trust one, so `trust` -- this
/// specific call's own [`ComponentTrustStore`] -- is evaluated against
/// `manifest_bytes` on every call, cache hit included. A digest some earlier
/// caller's trust store authorized is not thereby authorized for a later
/// caller whose own trust store does not.
///
/// Fails closed, exactly like every other Component loading path in this
/// crate: untrusted bytes (`trust` does not trust this digest) or a
/// manifest whose own embedded digest does not match the real bytes are
/// rejected before anything is returned, whether this call ends up
/// compiling the Component fresh (via
/// [`ComponentManager::prepare_pushed_package`]) or reusing an
/// already-registered one.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub fn register_inference_component_artifact(
    component_bytes: Vec<u8>,
    manifest_bytes: Vec<u8>,
    trust: &ComponentTrustStore,
) -> Result<RegisteredInferenceComponent, E2eConformanceError> {
    let digest = ComponentDigest::sha256(&component_bytes);
    // astorise/Magnetar#83 steps 6-7: parsed once here, shared by both
    // branches below, instead of only the cache-hit branch re-parsing it --
    // this is the one and only place `manifest_bytes` is interpreted, so a
    // caller (e.g. `inference-components`) never needs its own independent
    // YAML parse of the same bytes to learn what this Component declares
    // (`ComponentManifest::supported_architecture_families` included).
    let manifest = ComponentManifest::from_yaml_bytes(
        &manifest_bytes,
        std::path::Path::new("<registered-component-manifest>"),
    )
    .map_err(|error| E2eConformanceError::ModelComponentFailed {
        reason: error.to_string(),
    })?;
    if manifest.digest != digest {
        return Err(E2eConformanceError::ModelComponentFailed {
            reason: "manifest-declared digest does not match received bytes".into(),
        });
    }
    let registry = REGISTERED_COMPONENT_RUNTIMES.get_or_init(|| Mutex::new(BTreeMap::new()));
    let already_registered = {
        let existing = registry.lock().unwrap();
        existing.contains_key(&digest.value)
    };
    if already_registered {
        // #71: the compiled runtime is reused (that is the whole point of
        // the cache), but *this* caller's own `trust` must still authorize
        // `digest` on every call -- a digest another, earlier caller
        // authorized is not thereby authorized for everyone. Re-runs the
        // same trust decision `validate_component_artifact` applies on a
        // fresh registration, so a cache hit can never skip the
        // authorization a cache miss would have enforced.
        let trust_decision = trust.evaluate(&manifest, &digest);
        if trust_decision.status != ComponentTrustStatus::Trusted {
            return Err(E2eConformanceError::ModelComponentFailed {
                reason: trust_decision.reason,
            });
        }
        return Ok(RegisteredInferenceComponent { digest, manifest });
    }
    let capability = Arc::new(GraphBuilderCapability::new());
    let model_config_capability = Arc::new(ModelConfigCapability::new());
    let mut manager = ComponentManager::with_engine(Box::new(
        crate::component_wasmtime::WasmtimeComponentEngine::new().map_err(|error| {
            E2eConformanceError::ModelComponentFailed {
                reason: error.to_string(),
            }
        })?,
    ));
    manager.set_resource_limits(inference_component_runtime_limits());
    manager.set_trust_store(trust.clone());
    // This generic registry serves distinct, independently-compiled
    // Components concurrently (Llama and the synthetic-minimal test fixture
    // remain compiled against `1.2.0`, since the `1.3.0` bump
    // (`add-real-multi-device-model-instance-placement`) only added new
    // world exports neither of them needs -- see `model-component-graph.wit`
    // 1.3.0's doc comment). A compiled Component's real executable imports
    // are checked against its manifest's declared version, so the host must
    // keep authorizing `1.2.0` alongside `1.3.0` here, not merely replace
    // one with the other, or every still-`1.2.0`-compiled Component fails
    // `UnauthorizedImport`/`CapabilityVersionMismatch`.
    for version in ["1.2.0", "1.3.0", "1.4.0", "1.5.0"] {
        manager.provide_capability(
            WitInterface::new("magnetar:model-component-graph/graph-builder", version),
            capability.clone() as Arc<dyn HostCapability>,
        );
        manager.provide_capability(
            WitInterface::new("magnetar:model-component-graph/model-config", version),
            model_config_capability.clone() as Arc<dyn HostCapability>,
        );
    }
    let package = ComponentArtifactPackage::new(
        component_bytes,
        manifest_bytes,
        digest.clone(),
        ComponentDistributionSource::new(
            ComponentDistributionSourceKind::ClientProvided,
            format!("component:{}", digest.value),
        ),
    );
    let definition = manager.prepare_pushed_package(package).map_err(|error| {
        E2eConformanceError::ModelComponentFailed {
            reason: error.to_string(),
        }
    })?;
    let mut registry = registry.lock().unwrap();
    // A racing caller may have finished registering the same digest first
    // while this thread was compiling -- keep whichever won, exactly like
    // `qwen_real_component_runtime`'s own `OnceLock::set` race handling.
    registry.entry(digest.value.clone()).or_insert_with(|| {
        Arc::new(RegisteredComponentRuntime {
            manager: Mutex::new(manager),
            capability,
            model_config_capability,
            definition,
            supported_architecture_families: manifest.supported_architecture_families.clone(),
            supported_artifact_formats: manifest.supported_artifact_formats.clone(),
        })
    });
    Ok(RegisteredInferenceComponent { digest, manifest })
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub(super) fn named_component_runtime(
    digest: &ComponentDigest,
) -> Result<Arc<RegisteredComponentRuntime>, E2eConformanceError> {
    REGISTERED_COMPONENT_RUNTIMES
        .get()
        .and_then(|registry| registry.lock().unwrap().get(&digest.value).cloned())
        .ok_or_else(|| E2eConformanceError::ModelComponentFailed {
            reason: format!(
                "no Component is registered for digest '{}'; call \
                 register_inference_component_artifact first",
                digest.value
            ),
        })
}

/// [`build_first_native_graphs_from_real_qwen_component`]'s generic
/// counterpart: builds prefill/decode graphs from whichever Component
/// [`register_inference_component_artifact`] registered under `digest`,
/// instead of the single hardcoded Qwen singleton.
///
/// Takes the generic, WIT-facing [`ModelArchitectureConfig`] directly (#73 /
/// Tachyon integration audit MAG-02), not [`FirstNativeModelConfig`] -- a caller whose
/// own config is Qwen-shaped converts once via
/// `architecture_config_from_first_native_model_config` before calling, exactly like
/// `build_first_native_graphs_for_config` (the Qwen-singleton path) now
/// does internally; a caller for any other architecture never needs a
/// `FirstNativeModelConfig` to exist at all. See
/// `build_first_native_graphs_from_named_component_serves_a_real_second_
/// architecture_family` for the proof that an independently-compiled,
/// non-Qwen Component (real Llama) drives this exact entry point correctly.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub fn build_first_native_graphs_from_named_component(
    digest: &ComponentDigest,
    config: &ModelArchitectureConfig,
    identity: &ModelComponentIdentity,
    prompt_token_count: u64,
) -> Result<
    (
        FirstNativeComponentGraphs,
        ComponentDefinitionId,
        ComponentInstanceId,
    ),
    E2eConformanceError,
> {
    let runtime = named_component_runtime(digest)?;
    build_first_native_graphs_with_runtime(
        &runtime.manager,
        &runtime.capability,
        &runtime.model_config_capability,
        runtime.definition,
        config,
        identity,
        prompt_token_count,
    )
}

/// No Component Engine exists on this build target/feature combination
/// (`wasm32`, or `wasmtime-component-engine` disabled) -- fails closed,
/// structurally, exactly like [`first_native_component_graphs_for_prompt`]'s
/// own non-test fallback for the same reason. `E2eRuntimeModelExecutionEngine::
/// execute_generation_step` and `ProductionLoadedModel::prepare_generation`
/// both reach this whenever `component_digest` is `Some` on such a build,
/// so a `None`-digest instance (the pre-existing singleton path, unaffected
/// by this fallback) remains the only thing that can ever run here.
#[cfg(not(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine")))]
pub fn build_first_native_graphs_from_named_component(
    _digest: &ComponentDigest,
    _config: &ModelArchitectureConfig,
    _identity: &ModelComponentIdentity,
    _prompt_token_count: u64,
) -> Result<
    (
        FirstNativeComponentGraphs,
        ComponentDefinitionId,
        ComponentInstanceId,
    ),
    E2eConformanceError,
> {
    Err(E2eConformanceError::ModelComponentFailed {
        reason: "no Component engine is available on this build target; a caller-registered \
                 Component cannot be used here"
            .into(),
    })
}

/// Builds prefill and decode Execution Graphs by instantiating the real
/// Qwen Model Component (`QWEN_REAL_COMPONENT_BYTES`, compiled once and
/// cached -- see `qwen_real_component_runtime`) and calling its
/// `build-prefill-graph`/`build-decode-graph` exports, which produce the
/// graph through real `graph-builder` host calls into a
/// [`GraphBuilderCapability`] -- not `qwen_prefill_graph`/`qwen_decode_graph`
/// (task 11.5/12.6: this is production's *only* graph source; those Rust
/// functions are test-oracle only now, unreachable from any non-test
/// build). Returns the
/// component's own `ComponentDefinitionId`/`ComponentInstanceId` alongside
/// the graphs so the caller can still emit the same
/// `ComponentValidated`/`ComponentInstantiated` observations the prior
/// fixture-checksum path did. Destroys the fresh instance this call
/// creates before returning -- only the compiled artifact is cached, not
/// instance state, so nothing accumulates across repeated calls.
///
/// Public (`audit-complet-cuda-hot-path-2026-09-08` P1-1) so an external
/// caller with its own real Provider can obtain the exact same real,
/// Component-produced graph production itself dispatches, to drive
/// [`run_first_native_graph_with_provider`] end to end against it --
/// unlike `qwen_prefill_graph`, exposing this does not weaken "production
/// never Rust-synthesizes a graph": this function *is* the production
/// path, unconditionally the real Component either way.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub fn build_first_native_graphs_from_real_qwen_component(
    fixture: &E2eFixture,
    prompt_token_count: u64,
) -> Result<
    (
        FirstNativeComponentGraphs,
        ComponentDefinitionId,
        ComponentInstanceId,
    ),
    E2eConformanceError,
> {
    build_first_native_graphs_for_config(&fixture.config, &fixture.identity, prompt_token_count)
}

/// [`build_first_native_graphs_from_real_qwen_component`]'s actual body,
/// decoupled from [`E2eFixture`] (task group 10): the only two fixture
/// fields that function ever read. Additive -- the public, `E2eFixture`-
/// shaped entry point above is now a one-line wrapper with unchanged
/// behavior, so every existing caller is unaffected; this decoupled form
/// is what production loading (driven by a real ingested `FirstNativeModelConfig`/
/// `ModelComponentIdentity`, not a fixture) calls directly.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub(super) fn build_first_native_graphs_for_config(
    config: &FirstNativeModelConfig,
    identity: &ModelComponentIdentity,
    prompt_token_count: u64,
) -> Result<
    (
        FirstNativeComponentGraphs,
        ComponentDefinitionId,
        ComponentInstanceId,
    ),
    E2eConformanceError,
> {
    let runtime = qwen_real_component_runtime()?;
    let architecture_config = architecture_config_from_first_native_model_config(config);
    build_first_native_graphs_with_runtime(
        &runtime.manager,
        &runtime.capability,
        &runtime.model_config_capability,
        runtime.definition,
        &architecture_config,
        identity,
        prompt_token_count,
    )
}

/// [`build_first_native_graphs_for_config`]'s actual body, decoupled from
/// *which* compiled Component runtime it operates against
/// (`wire-generic-inference-component-runtime`): identical behavior to the
/// pre-existing single-Qwen-singleton version, just taking the runtime's
/// four pieces as parameters instead of always reading them off
/// [`qwen_real_component_runtime`]'s one process-wide slot. This is what
/// [`build_first_native_graphs_from_named_component`] calls for an
/// arbitrary caller-registered Component (`register_inference_component_
/// artifact`) instead of the single hardcoded-digest Qwen one -- nothing
/// about graph production itself is Qwen-specific: `config`/`identity`
/// already fully parameterize it, exactly as they did before this
/// extraction.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn build_first_native_graphs_with_runtime(
    manager: &Mutex<ComponentManager>,
    capability: &GraphBuilderCapability,
    model_config_capability: &ModelConfigCapability,
    definition: ComponentDefinitionId,
    config: &ModelArchitectureConfig,
    identity: &ModelComponentIdentity,
    prompt_token_count: u64,
) -> Result<
    (
        FirstNativeComponentGraphs,
        ComponentDefinitionId,
        ComponentInstanceId,
    ),
    E2eConformanceError,
> {
    let mut manager = manager.lock().unwrap();

    let instance = manager
        .instantiate_prepared_component(definition)
        .map_err(|error| E2eConformanceError::ModelComponentFailed {
            reason: error.to_string(),
        })?;
    let result = (|| {
        let engine_key = manager
            .engine_instance_key(instance)
            .ok_or_else(|| E2eConformanceError::ModelComponentFailed {
                reason: "component instance has no engine key".into(),
            })?
            .to_string();

        let export_interface = WitInterface::new(
            "magnetar:model-component-graph/model-component-graph-producer",
            "1.0.0",
        );
        let weight_shapes = weight_shapes_for_architecture_config(config);
        let compatibility_key = qwen_component_compatibility_key(identity);
        let session_context = |weight_shapes: BTreeMap<String, Vec<u64>>| SessionContext {
            component_id: identity.id.as_str().to_string(),
            compatibility_key: compatibility_key.clone(),
            kv_namespace: "qwen".to_string(),
            weight_shapes,
            output_edge_name: "logits".to_string(),
        };

        model_config_capability.bind_config(&engine_key, config.clone());

        capability.prepare_session(&engine_key, session_context(weight_shapes.clone()));
        let prefill_result = manager
            .invoke(
                ComponentInvocation::new(instance, export_interface.clone(), "build-prefill-graph")
                    .with_arguments(vec![ComponentValue::S64(prompt_token_count.max(1) as i64)]),
            )
            .map_err(|error| E2eConformanceError::ModelComponentFailed {
                reason: error.to_string(),
            })?;
        let prefill_handle =
            expect_single_string_invocation_result(&prefill_result, "build-prefill-graph")?;
        let prefill = capability
            .take_graph(&engine_key, &prefill_handle)
            .ok_or_else(|| E2eConformanceError::ModelComponentFailed {
                reason: "build-prefill-graph handle did not resolve to a finished graph".into(),
            })?;

        model_config_capability.bind_config(&engine_key, config.clone());
        capability.prepare_session(&engine_key, session_context(weight_shapes));
        let decode_result = manager
            .invoke(
                ComponentInvocation::new(instance, export_interface, "build-decode-graph")
                    .with_arguments(vec![ComponentValue::S64(prompt_token_count.max(1) as i64)]),
            )
            .map_err(|error| E2eConformanceError::ModelComponentFailed {
                reason: error.to_string(),
            })?;
        let decode_handle =
            expect_single_string_invocation_result(&decode_result, "build-decode-graph")?;
        let decode = capability
            .take_graph(&engine_key, &decode_handle)
            .ok_or_else(|| E2eConformanceError::ModelComponentFailed {
                reason: "build-decode-graph handle did not resolve to a finished graph".into(),
            })?;

        validate_first_scope_graph(&prefill)?;
        validate_first_scope_graph(&decode)?;
        validate_rope_frequency_scaling(&prefill, config)?;
        validate_rope_frequency_scaling(&decode, config)?;
        capability.clear_session(&engine_key);
        model_config_capability.clear_config(&engine_key);
        Ok(FirstNativeComponentGraphs {
            prefill_node_count: prefill.nodes.len(),
            decode_node_count: decode.nodes.len(),
            prefill,
            decode,
        })
    })();
    let _ = manager.destroy_instance(instance);
    result.map(|graphs| (graphs, definition, instance))
}

/// [`build_first_native_graphs_with_runtime`]'s per-Device pipeline-
/// placement counterpart (`add-real-multi-device-model-instance-
/// placement`): builds a real *prefill-segment* Execution Graph for decoder
/// layer range `[start_layer, end_layer)` only, via `build-prefill-graph-
/// segment` (`model-component-graph.wit` 1.3.0) instead of `build-prefill-
/// graph`'s always-whole-stack shape. Prefill-only for now -- decode-
/// segment wiring (`build-decode-graph-segment`) is deferred to whichever
/// phase first needs multi-step segment decode.
///
/// The produced graph's declared output edge is renamed to `"logits"`
/// exactly like a full graph's (via the same `SessionContext::
/// output_edge_name`), *whether or not* this segment actually reaches the
/// real lm-head -- so a caller reads a segment's real output the same way
/// regardless of range, a raw post-layer hidden state for an internal
/// segment or real logits for one ending at `num_hidden_layers`.
///
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn build_first_native_prefill_graph_segment_with_runtime(
    manager: &Mutex<ComponentManager>,
    capability: &GraphBuilderCapability,
    model_config_capability: &ModelConfigCapability,
    definition: ComponentDefinitionId,
    config: &FirstNativeModelConfig,
    identity: &ModelComponentIdentity,
    prompt_token_count: u64,
    start_layer: u32,
    end_layer: u32,
) -> Result<(ExecutionGraph, ComponentDefinitionId, ComponentInstanceId), E2eConformanceError> {
    let mut manager = manager.lock().unwrap();

    let instance = manager
        .instantiate_prepared_component(definition)
        .map_err(|error| E2eConformanceError::ModelComponentFailed {
            reason: error.to_string(),
        })?;
    let result = (|| {
        let engine_key = manager
            .engine_instance_key(instance)
            .ok_or_else(|| E2eConformanceError::ModelComponentFailed {
                reason: "component instance has no engine key".into(),
            })?
            .to_string();

        let export_interface = WitInterface::new(
            "magnetar:model-component-graph/model-component-graph-producer",
            "1.0.0",
        );
        let architecture_config = architecture_config_from_first_native_model_config(config);
        let weight_shapes = weight_shapes_for_architecture_config(&architecture_config);
        let compatibility_key = qwen_component_compatibility_key(identity);
        let session_context = SessionContext {
            component_id: identity.id.as_str().to_string(),
            compatibility_key,
            kv_namespace: "qwen".to_string(),
            weight_shapes,
            output_edge_name: "logits".to_string(),
        };

        let expected_scaling = architecture_config.clone();
        model_config_capability.bind_config(&engine_key, architecture_config);
        capability.prepare_session(&engine_key, session_context);
        let segment_result = manager
            .invoke(
                ComponentInvocation::new(instance, export_interface, "build-prefill-graph-segment")
                    .with_arguments(vec![
                        ComponentValue::S64(prompt_token_count.max(1) as i64),
                        ComponentValue::U32(start_layer),
                        ComponentValue::U32(end_layer),
                    ]),
            )
            .map_err(|error| E2eConformanceError::ModelComponentFailed {
                reason: error.to_string(),
            })?;
        let segment_handle =
            expect_single_string_invocation_result(&segment_result, "build-prefill-graph-segment")?;
        let segment = capability
            .take_graph(&engine_key, &segment_handle)
            .ok_or_else(|| E2eConformanceError::ModelComponentFailed {
                reason: "build-prefill-graph-segment handle did not resolve to a finished graph"
                    .into(),
            })?;

        validate_first_scope_graph(&segment)?;
        validate_rope_frequency_scaling(&segment, &expected_scaling)?;
        capability.clear_session(&engine_key);
        model_config_capability.clear_config(&engine_key);
        Ok(segment)
    })();
    let _ = manager.destroy_instance(instance);
    result.map(|graph| (graph, definition, instance))
}

/// `build_first_native_prefill_graph_segment_with_runtime`, bound to the
/// single real, checked-in Qwen Component singleton
/// (`qwen_real_component_runtime`) -- the segment-graph counterpart of
/// `build_first_native_graphs_for_config`.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub fn build_first_native_prefill_graph_segment_for_config(
    config: &FirstNativeModelConfig,
    identity: &ModelComponentIdentity,
    prompt_token_count: u64,
    start_layer: u32,
    end_layer: u32,
) -> Result<(ExecutionGraph, ComponentDefinitionId, ComponentInstanceId), E2eConformanceError> {
    let runtime = qwen_real_component_runtime()?;
    build_first_native_prefill_graph_segment_with_runtime(
        &runtime.manager,
        &runtime.capability,
        &runtime.model_config_capability,
        runtime.definition,
        config,
        identity,
        prompt_token_count,
        start_layer,
        end_layer,
    )
}

/// [`build_first_native_prefill_graph_segment_with_runtime`]'s decode
/// counterpart: builds a real *decode-segment* Execution Graph for decoder
/// layer range `[start_layer, end_layer)`, via `build-decode-graph-
/// segment` (`model-component-graph.wit` 1.3.0). `cached_token_count` is
/// this segment's own real count of already-committed historical tokens
/// (the same value a full graph's `build-decode-graph` receives), used by
/// the Component to size RoPE position/KV-append offsets correctly --
/// identical for every segment of one pipeline step, since every segment
/// advances through the same token positions together.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn build_first_native_decode_graph_segment_with_runtime(
    manager: &Mutex<ComponentManager>,
    capability: &GraphBuilderCapability,
    model_config_capability: &ModelConfigCapability,
    definition: ComponentDefinitionId,
    config: &FirstNativeModelConfig,
    identity: &ModelComponentIdentity,
    cached_token_count: u64,
    start_layer: u32,
    end_layer: u32,
) -> Result<(ExecutionGraph, ComponentDefinitionId, ComponentInstanceId), E2eConformanceError> {
    let mut manager = manager.lock().unwrap();

    let instance = manager
        .instantiate_prepared_component(definition)
        .map_err(|error| E2eConformanceError::ModelComponentFailed {
            reason: error.to_string(),
        })?;
    let result = (|| {
        let engine_key = manager
            .engine_instance_key(instance)
            .ok_or_else(|| E2eConformanceError::ModelComponentFailed {
                reason: "component instance has no engine key".into(),
            })?
            .to_string();

        let export_interface = WitInterface::new(
            "magnetar:model-component-graph/model-component-graph-producer",
            "1.0.0",
        );
        let architecture_config = architecture_config_from_first_native_model_config(config);
        let weight_shapes = weight_shapes_for_architecture_config(&architecture_config);
        let compatibility_key = qwen_component_compatibility_key(identity);
        let session_context = SessionContext {
            component_id: identity.id.as_str().to_string(),
            compatibility_key,
            kv_namespace: "qwen".to_string(),
            weight_shapes,
            output_edge_name: "logits".to_string(),
        };

        let expected_scaling = architecture_config.clone();
        model_config_capability.bind_config(&engine_key, architecture_config);
        capability.prepare_session(&engine_key, session_context);
        let segment_result = manager
            .invoke(
                ComponentInvocation::new(instance, export_interface, "build-decode-graph-segment")
                    .with_arguments(vec![
                        ComponentValue::S64(cached_token_count as i64),
                        ComponentValue::U32(start_layer),
                        ComponentValue::U32(end_layer),
                    ]),
            )
            .map_err(|error| E2eConformanceError::ModelComponentFailed {
                reason: error.to_string(),
            })?;
        let segment_handle =
            expect_single_string_invocation_result(&segment_result, "build-decode-graph-segment")?;
        let segment = capability
            .take_graph(&engine_key, &segment_handle)
            .ok_or_else(|| E2eConformanceError::ModelComponentFailed {
                reason: "build-decode-graph-segment handle did not resolve to a finished graph"
                    .into(),
            })?;

        validate_first_scope_graph(&segment)?;
        validate_rope_frequency_scaling(&segment, &expected_scaling)?;
        capability.clear_session(&engine_key);
        model_config_capability.clear_config(&engine_key);
        Ok(segment)
    })();
    let _ = manager.destroy_instance(instance);
    result.map(|graph| (graph, definition, instance))
}

/// `build_first_native_decode_graph_segment_with_runtime`, bound to the
/// single real, checked-in Qwen Component singleton
/// (`qwen_real_component_runtime`) -- the decode-segment counterpart of
/// [`build_first_native_prefill_graph_segment_for_config`].
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub fn build_first_native_decode_graph_segment_for_config(
    config: &FirstNativeModelConfig,
    identity: &ModelComponentIdentity,
    cached_token_count: u64,
    start_layer: u32,
    end_layer: u32,
) -> Result<(ExecutionGraph, ComponentDefinitionId, ComponentInstanceId), E2eConformanceError> {
    let runtime = qwen_real_component_runtime()?;
    build_first_native_decode_graph_segment_with_runtime(
        &runtime.manager,
        &runtime.capability,
        &runtime.model_config_capability,
        runtime.definition,
        config,
        identity,
        cached_token_count,
        start_layer,
        end_layer,
    )
}

/// True when `tensor_name` (a manifest tensor's logical name, matching the
/// real Qwen Component's own `weight-edge` naming convention -- see
/// `components/qwen/src/lib.rs`'s `weight_tensor_name`) is one
/// `build-prefill-graph-segment`/`build-decode-graph-segment` actually
/// references for decoder layer range `[start_layer, end_layer)`: either a
/// `"layers.{n}."`-prefixed per-layer weight with `start_layer <= n <
/// end_layer`, the token embedding (only when `start_layer == 0`, matching
/// the segment's `token_ids`-vs-`hidden_states_in` starting edge), or the
/// final norm/lm-head projection (only when `end_layer ==
/// num_hidden_layers`) -- see `model-component-graph.wit` 1.3.0's doc
/// comment for the same boundary semantics on the Component side.
///
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub(super) fn qwen_weight_name_in_layer_range(
    tensor_name: &str,
    start_layer: u32,
    end_layer: u32,
    num_hidden_layers: u32,
) -> bool {
    if let Some(rest) = tensor_name.strip_prefix("layers.") {
        let layer_index = rest
            .split('.')
            .next()
            .and_then(|text| text.parse::<u32>().ok());
        return matches!(layer_index, Some(index) if index >= start_layer && index < end_layer);
    }
    match tensor_name {
        "token_embedding" => start_layer == 0,
        "final_norm" | "lm_head" => end_layer == num_hidden_layers,
        _ => false,
    }
}

/// [`load_production_model_instance_for_provider`], restricted to decoder
/// layer range `[start_layer, end_layer)` (`add-real-multi-device-model-
/// instance-placement`, the "two `ModelInstance`s + explicit movement"
/// design): filters `manifest`'s own tensor inventory
/// (`qwen_weight_name_in_layer_range`) down to exactly the weights that
/// range's segment graph references, so this Model Instance's readiness
/// (`ModelInstanceDefinition::required_weight_names`, derived from
/// whatever manifest `load_model` receives) reflects only its own real
/// subset -- never the full architecture -- and materialization loads only
/// that subset's real bytes, never the other segment's.
///
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub(super) fn filter_manifest_for_layer_range(
    manifest: &ModelManifest,
    start_layer: u32,
    end_layer: u32,
    num_hidden_layers: u32,
) -> ModelManifest {
    let mut filtered = manifest.clone();
    filtered.tensors.retain(|tensor| {
        qwen_weight_name_in_layer_range(&tensor.name, start_layer, end_layer, num_hidden_layers)
    });
    filtered
}

/// A produced prefill/decode `ExecutionGraph` pair, from whichever source
/// this build's `first_native_component_graphs_for_prompt` actually used
/// (the real Qwen Component under the strict, default build; the Rust-
/// synthesized test-oracle recipe in a test build without a strict
/// engine). Public so an external caller with its own real Provider (see
/// `run_first_native_graph_with_provider`) can obtain the same real graph
/// production drives, via [`build_first_native_graphs_from_real_qwen_component`].
#[derive(Clone)]
pub struct FirstNativeComponentGraphs {
    pub prefill: ExecutionGraph,
    pub prefill_node_count: usize,
    pub decode: ExecutionGraph,
    pub decode_node_count: usize,
}

/// Resource limits every `ComponentManager` this file constructs runs
/// under -- the hardcoded-singleton `qwen_real_component_runtime` and the
/// generic, digest-keyed registry `register_inference_component_artifact`
/// populates alike (astorise/Magnetar audit round 3, MAG-01: this helper's
/// former name, `qwen_component_runtime_limits`, wrongly implied the
/// generic registry path depended on something Qwen-specific; nothing
/// here does -- these are plain resource-budget constants, not a
/// per-architecture policy). Not itself a Component-family-specific
/// concept: a real Llama Component runs under these exact same values,
/// with no branch anywhere selecting different limits per family. If a
/// Component ever legitimately needs a different budget, that has to be
/// expressed through a generic policy or the manifest, never a
/// family-specific branch here.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub(super) fn inference_component_runtime_limits() -> ComponentResourceLimits {
    ComponentResourceLimits {
        // 64 MiB: the checksum-only fixture Component fit in 1 MiB, and
        // the tiny 1-layer test fixture's real graph needed only ~19
        // graph-builder calls -- but every caller of this same shared
        // budget also builds graphs for real production checkpoints (task
        // 12.4/12.5), whose per-layer call count (and this session's
        // task-10.5/QKV-bias-support additions) scale with
        // `num_hidden_layers`: a real 24-layer Qwen2.5-0.5B-Instruct
        // config was observed to trip the previous
        // 8 MiB/1,000,000-fuel/1000ms budget (`ResourcePolicy`
        // interruption) building its ~500-call graph. Generous headroom
        // here (not per-config, since each caller is itself a shared,
        // one-time-initialized runtime across every config/Component it
        // ever builds a graph for) still fails closed against a genuinely
        // runaway Component -- it is bounded, not unlimited.
        max_memory_bytes: Some(1 << 26),
        execution_deadline_millis: Some(5_000),
        max_concurrent_invocations: Some(1),
        max_instances: Some(1),
        engine_execution_budget: Some(20_000_000),
        require_memory_limit: true,
    }
}

#[cfg(test)]
mod tests;
