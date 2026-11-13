//! Production generation: Component artifact resolution, loaded models, generation requests and streaming.

use super::*;

/// The first real, minimal Qwen Model Component (task 11.4/11.5,
/// `model-component-graph-contract`): a compiled `components/qwen`
/// Component binary, built via `cargo build --target wasm32-unknown-unknown
/// --release` then `wasm-tools component new`. Unlike
/// [`QWEN_GRAPH_COMPONENT_BYTES`] (a hand-written, imports-free checksum
/// fixture), this Component genuinely imports `graph-builder` and produces
/// its graphs through it -- this is what the strict, default production
/// path (`build_first_native_graphs_from_real_qwen_component`) actually
/// loads; `QWEN_GRAPH_COMPONENT_BYTES` remains in use only by tests
/// exercising Component-loading conformance generically (trust, digest,
/// resource limits), not by anything claiming to produce a real graph.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub(super) const QWEN_REAL_COMPONENT_NAME: &str = "magnetar.qwen.real";

/// The digest a Qwen Component artifact MUST have to be accepted, in
/// production or in tests -- not merely a description of whatever bytes
/// happen to be embedded. `prepare_distributed_package` computes the real
/// sha256 of whatever bytes it actually receives and rejects a mismatch
/// (`ComponentDistributionErrorCategory::DigestMismatch`), so a stale or
/// substituted artifact -- from either loading branch below -- is rejected
/// structurally, not just by convention.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub(super) const QWEN_REAL_COMPONENT_DIGEST: &str =
    "sha256:b09b2dcc852ef4f2010f6e98c8f156043679e2349155abb4ac54b16b56152d3a";

/// A Component artifact an embedder (e.g. `magnetar-cli`, the "deployment /
/// CLI / Component source adapter" layer in this task's own design) has
/// explicitly pushed for production first-native generation to use --
/// [`register_qwen_component_artifact`] is the only way to populate this.
/// Process-wide and set-once by design (`qwen_real_component_runtime`
/// itself already caches the *compiled* Component process-wide for the
/// same reason: compiling a real Component is expensive, and first-native
/// generation today has exactly one caller-facing model, `"qwen-test"`
/// (see [`run_first_native_generation`]), not a fleet of distinct
/// artifacts that would need per-call selection).
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub(super) static QWEN_COMPONENT_ARTIFACT: std::sync::OnceLock<ComponentArtifactPackage> =
    std::sync::OnceLock::new();

/// Lets an embedder push a real Qwen Component artifact for production
/// first-native generation to use (`reach-architecture-freeze-1` task
/// 12.4's "deployment / CLI / Component source adapter" boundary --
/// `magnetar-cli` calls this with a Component binary and manifest it owns
/// and embeds itself, e.g. for the bundled `"qwen-test"` self-test/demo
/// alias). Idempotent: a second call with the process already holding a
/// registered artifact is a harmless no-op, not an error -- a caller like
/// `magnetar-cli` can call this unconditionally before every generation
/// request rather than tracking its own "have I registered yet" state.
/// The pushed bytes still go through the same digest verification every
/// other loading path already required (`QWEN_REAL_COMPONENT_DIGEST`,
/// checked in `ComponentManager::prepare_distributed_package`): pushing the
/// wrong bytes fails closed exactly like a missing or corrupted external
/// source would, it does not bypass trust.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub fn register_qwen_component_artifact(component_bytes: Vec<u8>, manifest_bytes: Vec<u8>) {
    let _ = QWEN_COMPONENT_ARTIFACT.set(ComponentArtifactPackage::new(
        component_bytes,
        manifest_bytes,
        ComponentDigest::parse("sha256", QWEN_REAL_COMPONENT_DIGEST),
        ComponentDistributionSource::new(
            // Not `DevelopmentFixture`: this function is not test-gated --
            // it is `magnetar-cli`'s real, production embedder-facing call
            // path (see this function's own doc comment), which supplies
            // bytes it owns and embeds itself. `DevelopmentFixture` stays
            // reserved for the `#[cfg(test)]`-gated fixture packages
            // elsewhere in this file (`ClientProvided instead of
            // DevelopmentFixture` GitHub issue).
            ComponentDistributionSourceKind::ClientProvided,
            QWEN_REAL_COMPONENT_NAME,
        ),
    ));
}

/// Production Qwen Component loading (`reach-architecture-freeze-1` task
/// 12.4): no `include_bytes!` here, no silent fallback. First checks for an
/// artifact an embedder has explicitly pushed via
/// [`register_qwen_component_artifact`]; if none was pushed, falls back to
/// a caller-configured local path (`MAGNETAR_QWEN_COMPONENT_PATH` names the
/// Component `.wasm` file itself, with its manifest expected alongside it
/// as `<path>.magnetar-component.yaml`) -- the minimal external-source
/// resolution mechanisms this task calls for, deliberately not a full
/// Component registry/distribution service (a separate, larger piece of
/// infrastructure that does not exist yet and this task does not invent).
/// Neither configured, an unreadable file, or bytes that do not hash to
/// [`QWEN_REAL_COMPONENT_DIGEST`] all fail closed with a structured error --
/// never a fallback to an embedded fixture, because this crate has none.
#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine",
    not(test)
))]
pub(super) fn qwen_real_component_package() -> Result<ComponentArtifactPackage, E2eConformanceError>
{
    if let Some(package) = QWEN_COMPONENT_ARTIFACT.get() {
        return Ok(package.clone());
    }
    resolve_qwen_component_from_env_var("MAGNETAR_QWEN_COMPONENT_PATH")
}

/// The actual local-path resolution logic behind
/// [`qwen_real_component_package`]'s fallback branch, extracted as its own
/// testable function taking the env var *name* as a parameter (never
/// `#[cfg(not(test))]` itself, unlike its one production caller) so tests
/// can point it at a controlled variable instead of depending on -- or
/// polluting -- the real process environment. `env_var_name` names the
/// Component `.wasm` file itself; its manifest is expected alongside it as
/// `<path>.magnetar-component.yaml`.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub(super) fn resolve_qwen_component_from_env_var(
    env_var_name: &str,
) -> Result<ComponentArtifactPackage, E2eConformanceError> {
    resolve_qwen_component_from_lookup(env_var_name, std::env::var(env_var_name))
}

/// The pure part of [`resolve_qwen_component_from_env_var`]'s logic,
/// split out so tests can supply `lookup` directly (`Ok(path)` or
/// `Err(VarError::NotPresent)`) instead of mutating the real process
/// environment via `std::env::set_var`/`remove_var` (#68: those are
/// `unsafe` in Rust 2024 precisely because they race with any *concurrent*
/// environment access on another thread, and `cargo test` runs tests in
/// parallel threads within one process -- each test previously using its
/// own uniquely-named variable avoided colliding on the same key, but not
/// the actual hazard, which is any concurrent access at all).
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub(super) fn resolve_qwen_component_from_lookup(
    env_var_name: &str,
    lookup: Result<String, std::env::VarError>,
) -> Result<ComponentArtifactPackage, E2eConformanceError> {
    let component_path = lookup.map_err(|_| E2eConformanceError::ModelComponentFailed {
        reason: format!(
            "no Qwen Component artifact was registered (see \
             register_qwen_component_artifact) and {env_var_name} is not set; production \
             first-native generation requires an externally provided Qwen Model Component \
             artifact and has no embedded development fixture of its own to fall back to"
        ),
    })?;
    let component_bytes = std::fs::read(&component_path).map_err(|error| {
        E2eConformanceError::ModelComponentFailed {
            reason: format!(
                "failed to read Qwen Component bytes from '{component_path}' \
                 ({env_var_name}): {error}"
            ),
        }
    })?;
    let manifest_path = format!("{component_path}.magnetar-component.yaml");
    let manifest_bytes = std::fs::read(&manifest_path).map_err(|error| {
        E2eConformanceError::ModelComponentFailed {
            reason: format!(
                "failed to read Qwen Component manifest from '{manifest_path}': {error}"
            ),
        }
    })?;
    Ok(ComponentArtifactPackage::new(
        component_bytes,
        manifest_bytes,
        ComponentDigest::parse("sha256", QWEN_REAL_COMPONENT_DIGEST),
        ComponentDistributionSource::new(
            ComponentDistributionSourceKind::LocalDirectory,
            QWEN_REAL_COMPONENT_NAME,
        ),
    ))
}

/// Real per-weight dimensions for `config`'s architecture, keyed by the
/// canonical Model Artifact tensor name (`"layers.{N}.self_attn.q_proj"`,
/// `"token_embedding"`, ...) -- the same names `fixture.weights` is keyed
/// by, and what the real Qwen Component's `weight-edge` calls now supply
/// directly as their logical name (see `components/qwen/src/lib.rs`'s
/// `weight_tensor_name`), rather than a Component-internal shorthand the
/// Runtime would need its own mapping table to translate. Derived purely
/// from `config` (the same static architecture metadata `qwen_build_graph`
/// itself reads to size these edges), not from any bound `TensorResourceId`
/// -- weight *shape* is architecture metadata, resolving the real bytes
/// behind a weight edge happens later, at execution time
/// (`resolve_first_native_weight_edge`), unaffected by this function.
///
/// Takes the generic, WIT-facing [`ModelArchitectureConfig`] (#73 / Tachyon
/// integration audit MAG-02), not a Qwen-specific type: despite the name
/// this crate's [`FirstNativeModelConfig`] carried, the shape this function computes
/// (pre-norm decoder blocks, GQA-capable q/k/v/o projections, a gated MLP,
/// tied or untied embeddings) is the same generic decoder-only transformer
/// layout a real, independently-compiled Llama Component's weights need
/// too -- see `build_first_native_graphs_from_named_component_serves_a_
/// real_second_architecture_family`, which proves the Llama and Qwen
/// binaries produce identical graphs for the identical `architecture-
/// config`. Nothing here reads a Qwen-only field; every value comes
/// straight off [`ModelArchitectureConfig`].
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub(super) fn weight_shapes_for_architecture_config(
    config: &ModelArchitectureConfig,
) -> BTreeMap<String, Vec<u64>> {
    let q_dim = config.num_attention_heads as u64 * config.head_dim;
    let kv_dim = config.num_key_value_heads as u64 * config.head_dim;
    let mut shapes = BTreeMap::new();
    shapes.insert(
        "token_embedding".to_string(),
        vec![config.vocab_size, config.hidden_size],
    );
    shapes.insert("final_norm".to_string(), vec![config.hidden_size]);
    shapes.insert(
        "lm_head".to_string(),
        vec![config.hidden_size, config.vocab_size],
    );
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("layers.{layer}");
        shapes.insert(format!("{prefix}.input_norm"), vec![config.hidden_size]);
        shapes.insert(
            format!("{prefix}.self_attn.q_proj"),
            vec![config.hidden_size, q_dim],
        );
        shapes.insert(
            format!("{prefix}.self_attn.k_proj"),
            vec![config.hidden_size, kv_dim],
        );
        shapes.insert(
            format!("{prefix}.self_attn.v_proj"),
            vec![config.hidden_size, kv_dim],
        );
        if config.attention_bias {
            shapes.insert(format!("{prefix}.self_attn.q_bias"), vec![q_dim]);
            shapes.insert(format!("{prefix}.self_attn.k_bias"), vec![kv_dim]);
            shapes.insert(format!("{prefix}.self_attn.v_bias"), vec![kv_dim]);
        }
        shapes.insert(
            format!("{prefix}.self_attn.o_proj"),
            vec![q_dim, config.hidden_size],
        );
        shapes.insert(format!("{prefix}.post_attn_norm"), vec![config.hidden_size]);
        if let Some(moe) = &config.mixture_of_experts {
            // `execute-and-cache-moe-experts` F02: stacked expert weights,
            // each expert a column range (see the `moe-experts` Operator).
            let hidden = config.hidden_size;
            let experts = u64::from(moe.experts);
            let intermediate = moe.expert_intermediate_size;
            shapes.insert(format!("{prefix}.moe.router_proj"), vec![hidden, experts]);
            shapes.insert(
                format!("{prefix}.moe.experts_gate_proj"),
                vec![hidden, experts * intermediate],
            );
            shapes.insert(
                format!("{prefix}.moe.experts_up_proj"),
                vec![hidden, experts * intermediate],
            );
            shapes.insert(
                format!("{prefix}.moe.experts_down_proj"),
                vec![intermediate, experts * hidden],
            );
            if let Some(shared) = moe.shared_expert_intermediate_size {
                shapes.insert(
                    format!("{prefix}.moe.shared_gate_proj"),
                    vec![hidden, shared],
                );
                shapes.insert(format!("{prefix}.moe.shared_up_proj"), vec![hidden, shared]);
                shapes.insert(
                    format!("{prefix}.moe.shared_down_proj"),
                    vec![shared, hidden],
                );
                shapes.insert(format!("{prefix}.moe.shared_expert_gate"), vec![hidden, 1]);
            }
            continue;
        }
        shapes.insert(
            format!("{prefix}.mlp.gate_proj"),
            vec![config.hidden_size, config.intermediate_size],
        );
        shapes.insert(
            format!("{prefix}.mlp.up_proj"),
            vec![config.hidden_size, config.intermediate_size],
        );
        shapes.insert(
            format!("{prefix}.mlp.down_proj"),
            vec![config.intermediate_size, config.hidden_size],
        );
    }
    shapes
}

/// Bridges the Rust-side [`FirstNativeModelConfig`] (this crate's own test-oracle
/// architecture representation) into the generic, WIT-facing
/// [`ModelArchitectureConfig`] the `model-config` Capability hands to a
/// configurable Component (`implement-production-qwen-model-loading`
/// Decision 6). `FirstNativeModelConfig` carries no token id metadata today, so
/// `bos_token_id`/`eos_token_id` are `None` here -- real production
/// ingestion (task group 3) populates those from `config.json`.
///
/// Deliberately *not* `cfg`-gated to `wasmtime-component-engine`/non-`wasm32`
/// like most of this module's Component-engine machinery: both
/// `FirstNativeModelConfig` and `ModelArchitectureConfig` are plain, portable structs
/// with no Wasmtime dependency, and `E2eRuntimeModelExecutionEngine::
/// execute_generation_step` (itself never `cfg`-gated, since it must build
/// for every target/feature combination this crate supports) calls this
/// unconditionally before delegating to whichever `cfg`-selected variant of
/// `build_first_native_graphs_from_named_component` is actually compiled
/// (Tachyon integration audit MAG-03: gating this function specifically to
/// the non-`wasm32` Component-engine build broke `cargo check --target
/// wasm32-unknown-unknown` for exactly this call site).
pub(super) fn architecture_config_from_first_native_model_config(
    config: &FirstNativeModelConfig,
) -> ModelArchitectureConfig {
    let a = &config.architecture;
    ModelArchitectureConfig {
        hidden_size: a.hidden_size,
        intermediate_size: a.intermediate_size,
        num_hidden_layers: a.layer_count as u32,
        num_attention_heads: a.attention_head_count as u32,
        num_key_value_heads: a.kv_head_count as u32,
        head_dim: a.head_dimension,
        vocab_size: a.vocabulary_size,
        rms_norm_eps: config.rmsnorm_epsilon,
        rope_theta: config.rope.base,
        rope_scaling_factor: config.rope.scale.map(|value| value as f32),
        rope_frequency_scaling: config.rope.frequency_scaling,
        tie_word_embeddings: config.tied_embeddings,
        attention_bias: config.attention_bias,
        mixture_of_experts: config.mixture_of_experts,
        bos_token_id: None,
        eos_token_id: None,
    }
}

/// The inverse of [`architecture_config_from_first_native_model_config`]: builds a
/// [`FirstNativeModelConfig`] from a Runtime-authorized [`ModelArchitectureConfig`]
/// (production ingestion's normalized output) plus `context_length`
/// (`config.json`'s `max_position_embeddings`, not yet a field of
/// `ModelArchitectureConfig` itself -- threaded separately here rather
/// than widening that WIT-facing type for one production-only value), and
/// `family` -- the real ingested Model Artifact's own declared architecture
/// family (astorise/Magnetar#83: `first_native_architecture_metadata`
/// itself still defaults to `QWEN_ARCHITECTURE_FAMILY`, correct for its
/// other, genuinely Qwen-singleton callers, but wrong here; this
/// constructor overrides it with the real value instead of widening that
/// shared helper's own signature for its one production-only caller).
/// `require_bos`/`require_pad`/`chat_template_required` default to
/// permissive (`false`)/absent: production loading validates tokenizer
/// compatibility separately (task group 9), not through this constructor.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub(super) fn first_native_model_config_from_architecture_config(
    config: &ModelArchitectureConfig,
    context_length: u64,
    family: impl Into<String>,
) -> FirstNativeModelConfig {
    let mut architecture = first_native_architecture_metadata(
        config.hidden_size,
        config.num_hidden_layers as u64,
        config.num_attention_heads as u64,
        config.num_key_value_heads as u64,
        config.head_dim,
        config.intermediate_size,
        config.vocab_size,
        context_length,
    );
    architecture.family = family.into();
    FirstNativeModelConfig {
        architecture,
        rope: FirstNativeRopeConfig {
            base: config.rope_theta,
            scale: config.rope_scaling_factor.map(f64::from),
            dimension: config.head_dim,
            position_mode: FirstNativeRopePositionMode::Sequential,
            dynamic_scaling_supported: false,
            frequency_scaling: config.rope_frequency_scaling,
        },
        rmsnorm_epsilon: config.rms_norm_eps,
        tied_embeddings: config.tie_word_embeddings,
        attention_bias: config.attention_bias,
        mixture_of_experts: config.mixture_of_experts,
        require_bos: false,
        require_pad: false,
        expected_added_tokens: None,
        chat_template_required: false,
    }
}

/// The compiled Qwen Component identity production loading uses --
/// distinct from [`e2e_fixture_identity`] (a fixture-only id) but bound
/// to the same real, checkpoint-configurable Component artifact
/// [`qwen_real_component_runtime`] trusts by digest
/// (`QWEN_REAL_COMPONENT_DIGEST`), independent of `ModelComponentId`/
/// version: trust here is digest-based, not identity-based, so any
/// consistent identity works, and every production-loaded Model Instance
/// uses this same one.
///
/// Deliberately does *not* restrict `supported_architecture_families`
/// (astorise/Magnetar#83), unlike [`first_native_component_identity`]'s own
/// default: this identity backs the *generic* production path, which
/// threads the real ingested Model Artifact's own family through
/// (`production_model_fixture`) rather than assuming one. Restricting it to
/// a single hardcoded family here would reject every real architecture
/// whose family isn't literally that one string -- including real Qwen2
/// checkpoints, which declare `"qwen2"`, not `"qwen"`. An empty set means
/// no restriction at this layer (`ModelComponentIdentity::
/// supports_architecture`'s own documented semantics) -- the real
/// per-Component family compatibility check belongs where a specific
/// Component's own declared support is known, not here.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub(super) fn production_model_component_identity() -> ModelComponentIdentity {
    ModelComponentIdentity::new(
        ModelComponentId::new("production-model").expect("static id is valid"),
        ModelComponentVersion::new(1, 0, 0),
        ModelComponentImplementationKind::WebAssemblyComponent,
    )
    .trusted()
    .with_model_artifact_schema_version(MODEL_ARTIFACT_SCHEMA_VERSION)
}

/// Builds a real [`E2eFixture`] from production ingestion output (task
/// group 10): the same graph-execution/KV-cache/
/// `RuntimeModelExecutionEngine` machinery every existing first-native
/// caller already uses, driven by real Hugging Face bundle data instead
/// of the canonical tiny fixture. `weights` is deliberately empty --
/// production weight materialization goes through
/// [`stream_materialize_model_instance_weights`] against Provider-bound
/// Tensor Resources, never through this field (only the Rust-side test-
/// oracle forward-pass functions and the whole-map fixture loading path
/// read `E2eFixture::weights`; the real Component-graph execution path
/// resolves weights by Provider resource id per node -- see
/// `execute_first_native_graph`'s own doc comment).
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub fn production_model_fixture(
    manifest: ModelManifest,
    tokenizer_metadata: TokenizerMetadata,
    tokenizer: std::sync::Arc<dyn crate::tokenizer::Tokenizer + Send + Sync>,
) -> Result<E2eFixture, E2eConformanceError> {
    let architecture_config = manifest.architecture_config.clone().ok_or_else(|| {
        E2eConformanceError::ModelResolutionFailed {
            reason: "production Model Artifact has no Runtime-authorized architecture \
                     configuration bound to it"
                .into(),
        }
    })?;
    // See this function's doc comment: max_position_embeddings is not
    // yet threaded through ModelArchitectureConfig, so a generous
    // context length stands in until that gap closes. Documented, not
    // silent: a caller inspecting this fixture's config sees exactly
    // this value, and it never came from an unrecorded assumption.
    let context_length = 1_000_000u64;
    let config = first_native_model_config_from_architecture_config(
        &architecture_config,
        context_length,
        manifest.architecture.family.clone(),
    );
    let identity = production_model_component_identity();
    config
        .validate(&identity)
        .map_err(E2eConformanceError::from)?;
    let architecture_implementation = ModelArchitectureImplementation {
        architecture: manifest.architecture.clone(),
        kind: ModelArchitectureImplementationKind::ComponentBased,
        required_capabilities: Vec::new(),
    };
    let descriptor = first_native_component_descriptor(identity.clone(), &config)
        .map_err(E2eConformanceError::from)?;
    validate_first_native_model_artifact(&descriptor, &config, &manifest)
        .map_err(E2eConformanceError::from)?;
    Ok(E2eFixture {
        config,
        identity,
        architecture_implementation,
        manifest,
        tokenizer: FixtureTokenizer::wrapping(tokenizer_metadata, tokenizer),
        weights: BTreeMap::new(),
    })
}

/// Loads a production-ingested [`ModelManifest`] into a ready
/// [`ModelInstanceId`] through the same generic Runtime Inference API
/// primitives ([`load_model`]/[`create_model_instance`]) every loading
/// path already uses, then streams weight materialization from
/// `payload_source` without requiring the whole model to first exist as
/// a host `BTreeMap<String, HostTensor>` (task group 6's streaming
/// transactional materialization). No fixture manifest, no `qwen-test`
/// requirement -- this is the public production loading surface task
/// groups 10-11 describe: an authorized source's ingested output goes
/// straight to a ready Model Instance.
pub fn load_production_model_instance(
    runtime: &mut Runtime,
    manifest: &ModelManifest,
    payload_source: &dyn crate::production_model_ingestion::ProductionArtifactPayloadSource,
) -> Result<ModelInstanceId, InferenceApiError> {
    load_production_model_instance_for_provider(
        runtime,
        manifest,
        payload_source,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
}

/// [`load_production_model_instance`], generalized to bind the resulting
/// Model Instance's placement to an arbitrary registered Provider instead
/// of hardcoding Reference CPU. Matters even though
/// `WeightMaterializationTransaction::begin` itself resolves the
/// Provider it stages weights through from the instance's *own*
/// placement -- with no explicit `provider_binding` here that placement
/// carries no Provider, and `begin` falls back to Reference CPU
/// unconditionally when that happens (`generalize-first-native-provider-
/// dispatch`'s own "Runtime Treats Reference CPU As Normal Provider"
/// fix), which fails closed for a CUDA-only Runtime with a structured
/// "provider not registered" error rather than silently materializing
/// weights through the wrong Provider.
pub fn load_production_model_instance_for_provider(
    runtime: &mut Runtime,
    manifest: &ModelManifest,
    payload_source: &dyn crate::production_model_ingestion::ProductionArtifactPayloadSource,
    provider_binding: &ProviderBinding,
) -> Result<ModelInstanceId, InferenceApiError> {
    let mut coordinator = ModelLoadingCoordinator::new();
    let architecture_implementation = ModelArchitectureImplementation {
        architecture: manifest.architecture.clone(),
        kind: ModelArchitectureImplementationKind::ComponentBased,
        required_capabilities: Vec::new(),
    };
    coordinator.register_architecture(architecture_implementation.clone());
    let mut request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new(format!("production-{}", manifest.id.name.as_str())),
        manifest.id.clone(),
    );
    request.quantization_policy = ModelQuantizationPolicy::RejectUnsupported;
    let loaded = load_model(
        &mut coordinator,
        runtime,
        ModelLoadingApiRequest::new(request),
        manifest,
    )?;
    let instance = create_model_instance(
        runtime,
        &loaded,
        architecture_implementation,
        ResourceAffinity::new(FallbackClass::Transparent).with_provider(provider_binding.clone()),
    )?;
    stream_materialize_model_instance_weights(
        runtime,
        &instance,
        manifest.id.name.as_str(),
        &manifest.tensors,
        payload_source,
    )?;
    Ok(instance)
}

/// [`load_production_model_instance_for_provider`], restricted to decoder
/// layer range `[start_layer, end_layer)` (`add-real-multi-device-model-
/// instance-placement`): filters `manifest`'s own real tensor inventory
/// (`filter_manifest_for_layer_range`) down to exactly this segment's own
/// subset before streaming materialization, so only that subset's real
/// bytes are ever read from `payload_source` -- the other segment's
/// weights are never fetched, buffered, or admitted at all. Unlike the
/// fixture-based segment loader (`load_fixture_instance_segment_with_
/// weights_for_provider`), a real ingested production manifest already
/// carries a real, independently materializable `lm_head` tensor entry
/// when tied (the Hugging Face loader's own `DerivedLmHeadPayloadSource`
/// derives it once, ahead of this call), so no separate lm-head-
/// derivation step is needed here.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub fn load_production_model_instance_segment_for_provider(
    runtime: &mut Runtime,
    manifest: &ModelManifest,
    payload_source: &dyn crate::production_model_ingestion::ProductionArtifactPayloadSource,
    provider_binding: &ProviderBinding,
    start_layer: u32,
    end_layer: u32,
) -> Result<ModelInstanceId, InferenceApiError> {
    let num_hidden_layers = manifest
        .architecture_config
        .as_ref()
        .ok_or_else(|| InferenceApiError::ModelLoadingFailed {
            reason: "manifest has no Runtime-authorized architecture configuration bound to it \
                      -- segment loading needs its real num_hidden_layers"
                .into(),
        })?
        .num_hidden_layers;
    let segment_manifest =
        filter_manifest_for_layer_range(manifest, start_layer, end_layer, num_hidden_layers);
    let mut coordinator = ModelLoadingCoordinator::new();
    let architecture_implementation = ModelArchitectureImplementation {
        architecture: segment_manifest.architecture.clone(),
        kind: ModelArchitectureImplementationKind::ComponentBased,
        required_capabilities: Vec::new(),
    };
    coordinator.register_architecture(architecture_implementation.clone());
    let mut request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new(format!(
            "production-{}-segment-{start_layer}-{end_layer}",
            segment_manifest.id.name.as_str()
        )),
        segment_manifest.id.clone(),
    );
    request.quantization_policy = ModelQuantizationPolicy::RejectUnsupported;
    let loaded = load_model(
        &mut coordinator,
        runtime,
        ModelLoadingApiRequest::new(request),
        &segment_manifest,
    )?;
    let instance = create_model_instance(
        runtime,
        &loaded,
        architecture_implementation,
        ResourceAffinity::new(FallbackClass::Transparent).with_provider(provider_binding.clone()),
    )?;
    stream_materialize_model_instance_weights(
        runtime,
        &instance,
        segment_manifest.id.name.as_str(),
        &segment_manifest.tensors,
        payload_source,
    )?;
    Ok(instance)
}

/// One-shot production generation (task group 12): authorized source's
/// ingested manifest -> ready Model Instance -> real Component-produced
/// graphs -> real Provider dispatch -> decoded output, in one call. Builds
/// its own `Runtime` internally -- Runtime's execution-engine wiring is
/// deliberately not exposed to external callers at all (`Runtime::
/// builder().model_execution_engine` stays `pub(crate)`, a guarded
/// invariant `first_native_implementation_cut.rs` checks explicitly), so
/// this is the supported way an embedder actually drives generation for a
/// production-loaded instance, mirroring [`run_first_native_generation`]'s
/// own one-shot shape (fresh `Runtime`/session per call) but for real
/// ingested data instead of the `qwen-test` fixture.
///
/// `fixture` is [`production_model_fixture`]'s output; `payload_source` is
/// the same one the caller's ingestor produced; `trust_store` is the
/// caller's own trust policy (Decision 2: this function grants no trust
/// of its own -- an untrusted manifest fails inside `load_model` before
/// any materialization, exactly like every other loading path).
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub fn run_production_qwen_generation(
    fixture: E2eFixture,
    payload_source: &dyn crate::production_model_ingestion::ProductionArtifactPayloadSource,
    trust_store: ModelTrustStore,
    prompt: &str,
) -> Result<FirstNativeFixtureGeneration, InferenceApiError> {
    run_production_qwen_generation_for_provider(
        fixture,
        payload_source,
        trust_store,
        prompt,
        std::sync::Arc::new(ReferenceCpuProvider::new()),
    )
}

/// [`run_production_qwen_generation`], generalized to an arbitrary
/// registered Provider instead of hardcoding Reference CPU -- task
/// 10.6/12.3: the same production-loaded instance and Component-produced
/// graph can target CUDA (or any other registered Provider) this way,
/// with zero Provider-specific code in `magnetar-runtime` itself.
/// `provider` arrives as a `Provider` trait object the caller constructs
/// and owns; this function never imports or references a concrete non-
/// Reference-CPU Provider crate.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub fn run_production_qwen_generation_for_provider(
    fixture: E2eFixture,
    payload_source: &dyn crate::production_model_ingestion::ProductionArtifactPayloadSource,
    trust_store: ModelTrustStore,
    prompt: &str,
    provider: Arc<dyn Provider>,
) -> Result<FirstNativeFixtureGeneration, InferenceApiError> {
    run_production_qwen_generation_for_provider_with_prompt(
        fixture,
        payload_source,
        trust_store,
        PromptInput::PlainText(prompt.into()),
        None,
        provider,
    )
}

/// [`run_production_qwen_generation_for_provider`], generalized to accept
/// any [`PromptInput`] and an optional [`ChatTemplateFormatter`]
/// (`close-tachyon-scope-audit-gaps` task group 3): a caller whose
/// ingested manifest declares a real chat template renders
/// `PromptInput::ChatMessages` through it here, closing the gap where
/// every production generation call previously sent a bare, unformatted
/// prompt string even to an Instruct-tuned checkpoint expecting real chat
/// markup (a real Qwen2.5-Instruct checkpoint's own incoherent output on
/// a plain-text prompt is what this closes). `chat_formatter` is
/// consulted only for `PromptInput::ChatMessages`, matching
/// `tokenize_prompt_input`'s existing Prompt Input Boundary; passing
/// `PromptInput::PlainText` and `None` reproduces
/// `run_production_qwen_generation_for_provider`'s exact prior behavior,
/// which is exactly what that function's own thin wrapper above now does.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub fn run_production_qwen_generation_for_provider_with_prompt(
    fixture: E2eFixture,
    payload_source: &dyn crate::production_model_ingestion::ProductionArtifactPayloadSource,
    trust_store: ModelTrustStore,
    prompt: PromptInput,
    chat_formatter: Option<&dyn ChatTemplateFormatter>,
    provider: Arc<dyn Provider>,
) -> Result<FirstNativeFixtureGeneration, InferenceApiError> {
    run_production_qwen_generation_for_provider_with_request(
        fixture,
        payload_source,
        trust_store,
        ProductionGenerationRequest {
            prompt,
            parameters: GenerationParameters::greedy(),
            stop_conditions: StopConditions::default(),
            max_new_tokens: None,
            max_generation_millis: None,
        },
        chat_formatter,
        provider,
    )
}

/// Caller-supplied generation configuration for production Qwen generation
/// (`expose-production-generation-parameters`): the shape an embedder
/// (e.g. Tachyon translating an OpenAI-style request) fills in to reach
/// the real sampling/stop-matching contract every other Runtime
/// generation path already honors, without reimplementing any of it.
///
/// `max_new_tokens: None` preserves the checkpoint manifest's own
/// configured default (`fixture.manifest.generation.max_tokens`,
/// falling back to 64); `Some(n)` overrides it.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[derive(Clone, Debug)]
pub struct ProductionGenerationRequest {
    pub prompt: PromptInput,
    pub parameters: GenerationParameters,
    pub stop_conditions: StopConditions,
    pub max_new_tokens: Option<usize>,
    pub max_generation_millis: Option<u64>,
}

/// [`run_production_qwen_generation_for_provider_with_prompt`],
/// generalized to accept caller-supplied [`GenerationParameters`] and
/// [`StopConditions`] instead of the hardcoded greedy/default values
/// every other entry point in this family still passes
/// (`expose-production-generation-parameters`). `run_production_qwen_generation_for_provider_with_prompt`
/// is now a thin wrapper over this function reproducing its exact prior
/// hardcoded defaults, so no existing caller's behavior changes.
///
/// Text stop sequences (`request.stop_conditions.stop_text_sequences`)
/// are prepared against the real tokenizer here via
/// [`prepare_stop_sequences`] before generation starts -- a caller
/// supplies plain strings (e.g. an OpenAI `stop: ["END"]` list) and
/// never needs to construct the tokenizer-aware `prepared_stop_sequences`
/// form itself.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub fn run_production_qwen_generation_for_provider_with_request(
    fixture: E2eFixture,
    payload_source: &dyn crate::production_model_ingestion::ProductionArtifactPayloadSource,
    trust_store: ModelTrustStore,
    request: ProductionGenerationRequest,
    chat_formatter: Option<&dyn ChatTemplateFormatter>,
    provider: Arc<dyn Provider>,
) -> Result<FirstNativeFixtureGeneration, InferenceApiError> {
    let mut prepared = prepare_production_generation(
        fixture,
        payload_source,
        trust_store,
        request,
        chat_formatter,
        provider,
    )?;

    let mut execution_plans = RuntimeGenerationExecutionPlans {
        prefill: &mut prepared.prepared_plans.prefill,
        decode: &mut prepared.prepared_plans.decode,
    };
    let deadline_started_at = std::time::Instant::now();
    let max_generation_duration = prepared
        .max_generation_millis
        .map(std::time::Duration::from_millis);
    let generation_result = run_generation_loop_with_execution_plans(
        &mut prepared.runtime,
        &prepared.generation_request,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated_so_far| {
            max_generation_duration
                .is_some_and(|deadline| deadline_started_at.elapsed() >= deadline)
        },
        &mut prepared.observer,
        &mut execution_plans,
    )?;

    finish_production_generation(
        prepared.runtime,
        &prepared.fixture,
        &prepared.instance,
        &prepared.session,
        generation_result,
        prepared.observer,
    )
}

/// Every field [`run_production_qwen_generation_for_provider_with_request`]
/// and [`run_production_qwen_generation_for_provider_streaming`] both need
/// after setup (session creation, tokenization, graph/plan preparation)
/// completes -- extracted so the two entry points share exactly one setup
/// path rather than risking drift between a streaming and non-streaming
/// generation of the same request (`stream-production-generation-events`).
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub(super) struct PreparedProductionGeneration {
    pub(super) fixture: E2eFixture,
    pub(super) runtime: Runtime,
    pub(super) instance: ModelInstanceId,
    pub(super) session: InferenceSessionId,
    pub(super) generation_request: GenerationRequest,
    pub(super) prepared_plans: FirstNativePreparedPlans,
    pub(super) observer: InferenceApiObserver,
    pub(super) max_generation_millis: Option<u64>,
}

/// Shared setup for every production generation entry point from
/// `run_production_qwen_generation_for_provider_with_request` onward:
/// resolves the token budget and the multi-step-decode `Unsupported` gate,
/// builds the `Runtime`, loads the Model Instance, creates the session,
/// tokenizes the prompt, resolves stop sequences against the real
/// tokenizer, and prepares prefill/decode execution plans -- everything
/// up to but not including running the generation loop itself, since the
/// streaming and non-streaming callers run that loop differently.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub(super) fn prepare_production_generation(
    fixture: E2eFixture,
    payload_source: &dyn crate::production_model_ingestion::ProductionArtifactPayloadSource,
    trust_store: ModelTrustStore,
    request: ProductionGenerationRequest,
    chat_formatter: Option<&dyn ChatTemplateFormatter>,
    provider: Arc<dyn Provider>,
) -> Result<PreparedProductionGeneration, InferenceApiError> {
    let ProductionGenerationRequest {
        prompt,
        parameters,
        mut stop_conditions,
        max_new_tokens,
        max_generation_millis,
    } = request;

    let max_tokens = max_new_tokens.unwrap_or_else(|| {
        fixture
            .manifest
            .generation
            .as_ref()
            .and_then(|defaults| defaults.max_tokens)
            .unwrap_or(64) as usize
    });
    if max_tokens > 1 && !provider.supports_multi_step_decode() {
        return Err(InferenceApiError::Unsupported {
            reason: format!(
                "provider '{}' does not support multi-step decode (its KV history is not \
                 host-readable), but this request needs {max_tokens} decode steps",
                provider.metadata().name
            ),
        });
    }

    let provider_binding = ProviderBinding::new(provider.metadata().name.clone());
    let mut runtime = Runtime::builder()
        .register_provider(provider.clone())
        .model_execution_engine(std::sync::Arc::new(E2eRuntimeModelExecutionEngine {
            fixture: fixture.clone(),
            kv_states: Arc::new(Mutex::new(BTreeMap::new())),
            pending_kv_states: Arc::new(Mutex::new(BTreeMap::new())),
            component_digest: None,
            forced_token: None,
            graph_cache: None,
            copy_based_kv: false,
            step_replay: StepReplay::default(),
        }))
        .trust_store(trust_store)
        .build()
        .map_err(|error| InferenceApiError::InferenceApiUnavailable {
            reason: error.to_string(),
        })?;
    register_prepared_kernels_for_provider(&mut runtime, &*provider).map_err(|error| {
        InferenceApiError::KernelUnavailable {
            reason: error.to_string(),
        }
    })?;

    let instance = load_production_model_instance_for_provider(
        &mut runtime,
        &fixture.manifest,
        payload_source,
        &provider_binding,
    )?;
    require_ready_first_native_instance(&runtime, &instance)?;

    let session_request = SessionCreationRequest {
        model: GenerationModelReference::ModelInstance(instance.clone()),
        tokenizer: generation_tokenizer_reference(&fixture),
        generation_defaults: parameters.clone(),
        policy: SessionPolicy::default(),
        memory: SessionMemoryBudget::default(),
        allowed_capabilities: BTreeSet::new(),
        correlation_id: None,
        created_at_millis: 0,
    };
    let session = create_inference_session(&mut runtime, session_request)?;

    let tokenized = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(prompt),
        chat_formatter,
    )?;
    stop_conditions.prepared_stop_sequences = stop_conditions
        .stop_text_sequences
        .iter()
        .map(|text| fixture.tokenizer.resolve_stop_sequence(text))
        .collect::<Result<Vec<_>, _>>()
        .map_err(
            |error: TokenizerError| InferenceApiError::TokenizationFailed {
                reason: error.to_string(),
            },
        )?;
    let observer = InferenceApiObserver::new();

    let (component_graphs, _definition, _component_instance) =
        build_first_native_graphs_from_real_qwen_component(
            &fixture,
            tokenized.token_ids.len() as u64,
        )
        .map_err(|error| InferenceApiError::GraphPlanningFailed {
            reason: error.to_string(),
        })?;
    let prepared_plans = prepare_first_native_execution_plans_for_provider(
        &runtime,
        &instance,
        component_graphs,
        tokenized.token_ids.len() as u64,
        &provider_binding,
    )
    .map_err(|error| InferenceApiError::GraphPlanningFailed {
        reason: error.to_string(),
    })?;

    let generation_request = build_generation_request(
        GenerationRequestId::new("production-generation")?,
        Some(session.clone()),
        GenerationModelReference::ModelInstance(instance.clone()),
        generation_tokenizer_reference(&fixture),
        tokenized,
        max_tokens,
        parameters,
        stop_conditions,
        StreamingMode::TokenIds,
    );
    let generation_request = prepare_generation(&runtime, generation_request)?;

    Ok(PreparedProductionGeneration {
        fixture,
        runtime,
        instance,
        session,
        generation_request,
        prepared_plans,
        observer,
        max_generation_millis,
    })
}

/// Shared tail for every production generation entry point: decodes the
/// final text, closes the session, unloads the Model Instance, and builds
/// the returned [`FirstNativeFixtureGeneration`] -- run identically after
/// either the streaming or non-streaming generation loop, so both report
/// the same usage/finish-reason shape for an otherwise-identical request.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub(super) fn finish_production_generation(
    mut runtime: Runtime,
    fixture: &E2eFixture,
    instance: &ModelInstanceId,
    session: &InferenceSessionId,
    generation_result: GenerationResult,
    observer: InferenceApiObserver,
) -> Result<FirstNativeFixtureGeneration, InferenceApiError> {
    let decoded_text = decode_tokens_streaming(
        &fixture.tokenizer,
        StreamingDecodeRequest::new(generation_result.output.generated_token_ids.clone()),
    )
    .map(|decoded| decoded.text)
    .unwrap_or_else(|_| {
        let tokens = generation_result
            .output
            .generated_token_ids
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(" ");
        format!("[generated token ids: {tokens}]")
    });
    let generation_result = generation_result.with_decoded_text(decoded_text);

    close_inference_session(&mut runtime, session)?;
    unload_model_instance(
        &mut runtime,
        instance,
        ModelInstanceUnloadPolicy::DrainActiveUse,
    )?;

    let text = generation_result.decoded_text.clone().ok_or_else(|| {
        InferenceApiError::GenerationFailed {
            reason: "production generation produced no decoded text".into(),
        }
    })?;
    Ok(FirstNativeFixtureGeneration {
        text,
        result: generation_result,
        observer,
    })
}

/// Production model state loaded once into a Magnetar [`Runtime`].
/// Embedders with an external model registry keep this object behind their
/// own `LoadedModel` lifecycle and create per-request sessions on top of the
/// same ready [`ModelInstanceId`].
///
/// Despite [`E2eFixture::config`]'s own [`FirstNativeModelConfig`] type name, this state
/// machine carries no real Qwen-family-only gating: `component_digest`, when
/// set, makes an arbitrary caller-registered Component (any architecture the
/// registered Component's own graph producer implements, not just Qwen) the
/// real graph-production authority for both `prepare_generation` and
/// `E2eRuntimeModelExecutionEngine::execute_generation_step` (Tachyon
/// integration audit MAG-01/MAG-02, #72/#73) -- see
/// `loaded_inference_component_load_runs_a_real_second_architecture_end_to_end`
/// in `inference-components` for the proof that a real, independently-
/// compiled non-Qwen Component drives this exact type correctly end to end,
/// through `LoadedInferenceComponent::load` rather than only this crate's
/// own lower-level entry points.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub struct ProductionLoadedModel {
    pub(super) fixture: E2eFixture,
    pub(super) runtime: Runtime,
    pub(super) instance: ModelInstanceId,
    pub(super) provider_binding: ProviderBinding,
    pub(super) provider_supports_multi_step_decode: bool,
    pub(super) materialization_count: usize,
    /// `Some(digest)` when this instance was loaded against a caller-
    /// registered Component (`register_inference_component_artifact`,
    /// `wire-generic-inference-component-runtime`) instead of the CLI's
    /// single built-in `"qwen-test"` singleton -- threaded into both
    /// `prepare_generation` (plan production) and this instance's
    /// `E2eRuntimeModelExecutionEngine` (dispatch-time production), which
    /// must stay in lockstep: a plan's `graph_fingerprint` only matches the
    /// graph actually dispatched if both steps built it from the exact same
    /// Component. `None` preserves the pre-existing hardcoded-singleton
    /// behavior exactly, unchanged.
    pub(super) component_digest: Option<ComponentDigest>,
    /// Shared with this instance's execution engine.
    pub(super) graph_cache: ComponentGraphCache,
    /// Shared with this instance's execution engine.
    pub(super) step_replay: StepReplay,
}

/// `enforce-local-multigpu-and-expert-placement` tensor parallelism: with
/// `MAGNETAR_MULTI_GPU=tensor` and a Provider exposing several Devices,
/// the Runtime builds the Model Component's graphs, derives from the parallel
/// semantics it declared which weights are split
/// ([`crate::tensor_parallel::TensorParallelPlan`]) and hands the plan to the
/// Provider before any weight is written. `false` when no plan applies
/// (the layer placement then decides).
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub(super) fn apply_production_tensor_parallel(
    provider: &dyn Provider,
    fixture: &E2eFixture,
    component_digest: Option<&ComponentDigest>,
) -> Result<bool, InferenceApiError> {
    if std::env::var("MAGNETAR_MULTI_GPU").as_deref() != Ok("tensor") {
        return Ok(false);
    }
    let devices: Vec<_> = provider
        .devices()
        .iter()
        .map(|device| device.metadata().id.clone())
        .collect();
    let (Some(digest), Some(config), true) = (
        component_digest,
        fixture.manifest.architecture_config.as_ref(),
        devices.len() > 1,
    ) else {
        return Ok(false);
    };
    let failed = |reason: String| InferenceApiError::ModelLoadingFailed { reason };
    let (graphs, _definition, _instance) =
        build_first_native_graphs_from_named_component(digest, config, &fixture.identity, 2)
            .map_err(|error| failed(format!("tensor parallelism: {error}")))?;
    let plan = crate::tensor_parallel::TensorParallelPlan::from_graphs(
        [&graphs.prefill, &graphs.decode],
        devices,
    )
    .map_err(|error| failed(format!("tensor parallelism: {error}")))?;
    if !plan.splits() {
        return Ok(false);
    }
    provider
        .execution_api()
        .ok_or_else(|| failed("a multi-Device Provider exposes no execution API".into()))?
        .apply_tensor_parallel(&plan)
        .map_err(|error| failed(format!("tensor parallelism: {error}")))?;
    Ok(true)
}

/// The share of a Device's memory a layer placement may fill with weights:
/// the rest holds activations, KV history and Kernel workspace.
pub(super) const LAYER_PLACEMENT_WEIGHT_SHARE: (u64, u64) = (85, 100);

/// `enforce-local-multigpu-and-expert-placement` G05: when `provider`
/// exposes several Devices, the Runtime places the Model Instance's decoder
/// layers across them ([`crate::layer_placement::LayerPlacement::plan`],
/// policy from [`crate::layer_placement::LayerPlacementPolicy::from_env`])
/// and hands the plan to the Provider before any weight is written. A
/// single-Device Provider gets no plan.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub(super) fn apply_production_layer_placement(
    provider: &dyn Provider,
    manifest: &ModelManifest,
) -> Result<(), InferenceApiError> {
    use crate::layer_placement::{LayerPlacement, LayerPlacementPolicy, LayerWeightBytes};
    let devices: Vec<_> = provider
        .devices()
        .iter()
        .map(|device| {
            let metadata = device.metadata();
            let budget = metadata.memory_capacity / LAYER_PLACEMENT_WEIGHT_SHARE.1
                * LAYER_PLACEMENT_WEIGHT_SHARE.0;
            (metadata.id.clone(), budget)
        })
        .collect();
    if devices.len() < 2 {
        return Ok(());
    }
    let Some(layer_count) = manifest
        .architecture_config
        .as_ref()
        .map(|config| config.num_hidden_layers)
    else {
        return Ok(());
    };
    let bytes = LayerWeightBytes::from_weights(
        layer_count,
        manifest
            .tensors
            .iter()
            // What each weight occupies on its Device, not its checkpoint
            // bytes: an F16 or a dequantized quantized weight is larger.
            .map(|tensor| {
                (
                    tensor.name.as_str(),
                    crate::model_loading::resident_weight_bytes(tensor),
                )
            }),
    );
    let failed = |reason: String| InferenceApiError::ModelLoadingFailed { reason };
    let placement = LayerPlacement::plan(&bytes, &devices, LayerPlacementPolicy::from_env())
        .map_err(|error| failed(format!("layer placement: {error}")))?;
    let execution = provider.execution_api().ok_or_else(|| {
        failed("a multi-Device Provider exposes no execution API to place layers on".into())
    })?;
    execution
        .apply_layer_placement(&placement)
        .map_err(|error| failed(format!("layer placement: {error}")))
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
impl ProductionLoadedModel {
    pub fn load(
        fixture: E2eFixture,
        payload_source: &dyn crate::production_model_ingestion::ProductionArtifactPayloadSource,
        trust_store: ModelTrustStore,
        provider: Arc<dyn Provider>,
    ) -> Result<Self, InferenceApiError> {
        Self::load_with_component(fixture, payload_source, trust_store, provider, None)
    }

    /// [`Self::load`], additionally accepting which registered Component
    /// (`register_inference_component_artifact`) this instance's graph
    /// production should use instead of the CLI's single built-in
    /// `"qwen-test"` singleton. `None` reproduces [`Self::load`]'s exact
    /// existing behavior; `Some(digest)` is threaded into both this
    /// instance (`prepare_generation`) and its `E2eRuntimeModelExecutionEngine`
    /// (`execute_generation_step`), which must stay in lockstep for a
    /// published plan's `graph_fingerprint` to match what dispatch actually
    /// builds.
    pub fn load_with_component(
        fixture: E2eFixture,
        payload_source: &dyn crate::production_model_ingestion::ProductionArtifactPayloadSource,
        trust_store: ModelTrustStore,
        provider: Arc<dyn Provider>,
        component_digest: Option<ComponentDigest>,
    ) -> Result<Self, InferenceApiError> {
        // astorise/Magnetar#83 steps 6-7: the real Component-vs-family
        // compatibility gate. `None` (the pre-existing hardcoded-singleton
        // path) is unaffected -- there is no caller-registered manifest to
        // check against. `Some(digest)` looks up what that *specific*
        // registered Component itself declared at registration time
        // (`register_inference_component_artifact`, via `.magnetar-
        // component.yaml`'s `compatibility.architecture_families`) and
        // compares it against the real ingested Model Artifact's own
        // `fixture.manifest.architecture.family` -- mirroring
        // `ModelComponentIdentity::supports_architecture`'s exact
        // semantics (empty set means permissive) without constructing a
        // full identity for what is otherwise a one-shot boolean check.
        // Checked before building the Runtime below: rejecting a mismatch
        // here fails fast, instead of materializing weights and an
        // instance for a Component that was never going to be allowed to
        // execute them.
        if let Some(digest) = &component_digest {
            let registered = named_component_runtime(digest).map_err(|error| {
                InferenceApiError::ModelComponentUnavailable {
                    reason: error.to_string(),
                }
            })?;
            let declared = &registered.supported_architecture_families;
            if !declared.is_empty() && !declared.contains(&fixture.manifest.architecture.family) {
                return Err(InferenceApiError::ModelComponentUnavailable {
                    reason: ModelComponentError::ArchitectureUnsupported.to_string(),
                });
            }
            // astorise/Magnetar#75: the same gate one level down -- this
            // specific registered Component's own declared
            // `compatibility.artifact_formats` against the real ingested
            // Model Artifact's own declared `artifact_format`. Empty
            // declared set means permissive, exactly like
            // `supported_architecture_families` above.
            let declared_formats = &registered.supported_artifact_formats;
            if !declared_formats.is_empty()
                && !declared_formats.contains(fixture.manifest.artifact_format.as_str())
            {
                return Err(InferenceApiError::ModelComponentUnavailable {
                    reason: ModelComponentError::ArtifactFormatUnsupported.to_string(),
                });
            }
        }
        let provider_binding = ProviderBinding::new(provider.metadata().name.clone());
        let provider_supports_multi_step_decode = provider.supports_multi_step_decode();
        let graph_cache = ComponentGraphCache::default();
        let step_replay = StepReplay::default();
        let mut runtime = Runtime::builder()
            .register_provider(provider.clone())
            .model_execution_engine(std::sync::Arc::new(E2eRuntimeModelExecutionEngine {
                fixture: fixture.clone(),
                kv_states: Arc::new(Mutex::new(BTreeMap::new())),
                pending_kv_states: Arc::new(Mutex::new(BTreeMap::new())),
                component_digest: component_digest.clone(),
                forced_token: None,
                graph_cache: Some(graph_cache.clone()),
                copy_based_kv: false,
                step_replay: step_replay.clone(),
            }))
            .trust_store(trust_store)
            .build()
            .map_err(|error| InferenceApiError::InferenceApiUnavailable {
                reason: error.to_string(),
            })?;
        register_prepared_kernels_for_provider(&mut runtime, &*provider).map_err(|error| {
            InferenceApiError::KernelUnavailable {
                reason: error.to_string(),
            }
        })?;
        let tensor_parallel =
            apply_production_tensor_parallel(&*provider, &fixture, component_digest.as_ref())?;
        if !tensor_parallel {
            apply_production_layer_placement(&*provider, &fixture.manifest)?;
        }
        graph_cache.set_tensor_parallel(tensor_parallel);
        let instance = load_production_model_instance_for_provider(
            &mut runtime,
            &fixture.manifest,
            payload_source,
            &provider_binding,
        )?;
        require_ready_first_native_instance(&runtime, &instance)?;
        graph_cache.set_fusion_provider(Some(provider_binding.clone()));
        Ok(Self {
            fixture,
            runtime,
            instance,
            provider_binding,
            provider_supports_multi_step_decode,
            materialization_count: 1,
            component_digest,
            graph_cache,
            step_replay,
        })
    }

    pub fn model_instance_id(&self) -> &ModelInstanceId {
        &self.instance
    }

    pub fn materialization_count(&self) -> usize {
        self.materialization_count
    }

    /// Whether this instance runs split across Devices by tensor
    /// parallelism (`MAGNETAR_MULTI_GPU=tensor` with a plan that splits).
    pub fn tensor_parallel(&self) -> bool {
        self.graph_cache.tensor_parallel()
    }

    /// Turns graph fusion (`graph_fusion`) on or off for later requests. On
    /// by default; it only fuses what this instance's Provider advertises.
    pub fn set_graph_fusion(&mut self, enabled: bool) {
        self.graph_cache
            .set_fusion_provider(enabled.then(|| self.provider_binding.clone()));
    }

    /// Turns decode-step capture and replay (`capture-and-replay-cuda-graphs`)
    /// on or off for later steps. On by default; it only applies where this
    /// instance's Provider supports step capture.
    pub fn set_step_replay(&mut self, enabled: bool) {
        self.step_replay.set_enabled(enabled);
    }

    /// Starts attributing every later generation step's host time to its
    /// phases (A06). Off by default.
    pub fn enable_phase_attribution(&mut self) {
        self.runtime.phase_ledger_mut().enable();
    }

    /// Stops attribution and forgets anything not yet taken.
    pub fn disable_phase_attribution(&mut self) {
        self.runtime.phase_ledger_mut().disable();
    }

    /// Drains the steps attributed since the last call.
    pub fn take_phase_attribution(&mut self) -> PhaseAttributionReport {
        self.runtime.phase_ledger_mut().take_report()
    }

    pub fn generate(
        &mut self,
        request: ProductionGenerationRequest,
        chat_formatter: Option<&dyn ChatTemplateFormatter>,
    ) -> Result<FirstNativeFixtureGeneration, InferenceApiError> {
        let mut prepared = self.prepare_generation(request, chat_formatter)?;
        let mut execution_plans = RuntimeGenerationExecutionPlans {
            prefill: &mut prepared.prepared_plans.prefill,
            decode: &mut prepared.prepared_plans.decode,
        };
        let deadline_started_at = std::time::Instant::now();
        let max_generation_duration = prepared
            .max_generation_millis
            .map(std::time::Duration::from_millis);
        let generation_result = run_generation_loop_with_execution_plans(
            &mut self.runtime,
            &prepared.generation_request,
            SamplingPolicy::default(),
            CacheUsageSummary::default(),
            |_generated_so_far| {
                max_generation_duration
                    .is_some_and(|deadline| deadline_started_at.elapsed() >= deadline)
            },
            &mut prepared.observer,
            &mut execution_plans,
        )?;
        self.finish_generation(
            prepared.session,
            &prepared.generation_request,
            generation_result,
            prepared.observer,
        )
    }

    pub fn generate_streaming(
        &mut self,
        request: ProductionGenerationRequest,
        chat_formatter: Option<&dyn ChatTemplateFormatter>,
        on_event: &mut dyn FnMut(GenerationStreamEvent) -> std::ops::ControlFlow<()>,
    ) -> Result<FirstNativeFixtureGeneration, InferenceApiError> {
        self.generate_streaming_inner(request, chat_formatter, on_event, None)
    }

    /// [`Self::generate_streaming`], additionally handing each step's full
    /// next-token logits (host-visible, vocabulary-sized, pre-sampling) to
    /// `on_logits`, once per step and before that step's `Token` event
    /// (`qualify-real-checkpoint-performance` A02). Observation only: the
    /// logits, sampling and every event are exactly what
    /// [`Self::generate_streaming`] produces for the same request.
    pub fn generate_streaming_with_logits(
        &mut self,
        request: ProductionGenerationRequest,
        chat_formatter: Option<&dyn ChatTemplateFormatter>,
        on_event: &mut dyn FnMut(GenerationStreamEvent) -> std::ops::ControlFlow<()>,
        on_logits: &mut dyn FnMut(&[f32]),
    ) -> Result<FirstNativeFixtureGeneration, InferenceApiError> {
        self.generate_streaming_inner(request, chat_formatter, on_event, Some(on_logits))
    }

    pub(super) fn generate_streaming_inner(
        &mut self,
        request: ProductionGenerationRequest,
        chat_formatter: Option<&dyn ChatTemplateFormatter>,
        on_event: &mut dyn FnMut(GenerationStreamEvent) -> std::ops::ControlFlow<()>,
        on_logits: Option<LogitsObserver<'_>>,
    ) -> Result<FirstNativeFixtureGeneration, InferenceApiError> {
        let mut prepared = self.prepare_generation(request, chat_formatter)?;
        let prepared_tokenizer = self.fixture.tokenizer.clone();
        let mut generated_so_far: Vec<TokenId> = Vec::new();
        let mut previously_emitted_text = String::new();
        let mut on_token = |token_id: TokenId| -> std::ops::ControlFlow<()> {
            generated_so_far.push(token_id);
            let full_text = decode_tokens_streaming(
                &prepared_tokenizer,
                StreamingDecodeRequest::new(generated_so_far.clone()),
            )
            .map(|decoded| decoded.text)
            .ok();
            let text_delta = full_text.as_ref().and_then(|full_text| {
                full_text
                    .strip_prefix(previously_emitted_text.as_str())
                    .filter(|delta| !delta.is_empty())
                    .map(str::to_string)
            });
            if let Some(full_text) = full_text {
                previously_emitted_text = full_text;
            }
            on_event(GenerationStreamEvent::Token {
                token_id,
                text_delta,
            })
        };
        let mut execution_plans = RuntimeGenerationExecutionPlans {
            prefill: &mut prepared.prepared_plans.prefill,
            decode: &mut prepared.prepared_plans.decode,
        };
        let deadline_started_at = std::time::Instant::now();
        let max_generation_duration = prepared
            .max_generation_millis
            .map(std::time::Duration::from_millis);
        let should_cancel = |_generated_so_far: &[TokenId]| {
            max_generation_duration
                .is_some_and(|deadline| deadline_started_at.elapsed() >= deadline)
        };
        let generation_result = match on_logits {
            Some(on_logits) => {
                crate::inference_api::run_generation_loop_with_execution_plans_streaming_and_logits(
                    &mut self.runtime,
                    &prepared.generation_request,
                    SamplingPolicy::default(),
                    CacheUsageSummary::default(),
                    should_cancel,
                    &mut prepared.observer,
                    &mut execution_plans,
                    &mut on_token,
                    on_logits,
                )?
            }
            None => run_generation_loop_with_execution_plans_streaming(
                &mut self.runtime,
                &prepared.generation_request,
                SamplingPolicy::default(),
                CacheUsageSummary::default(),
                should_cancel,
                &mut prepared.observer,
                &mut execution_plans,
                &mut on_token,
            )?,
        };
        let _ = on_event(GenerationStreamEvent::Finished {
            finish_reason: generation_result.output.finish_reason,
            usage: generation_result.output.usage.clone(),
        });
        self.finish_generation(
            prepared.session,
            &prepared.generation_request,
            generation_result,
            prepared.observer,
        )
    }

    pub fn unload(mut self) -> Result<(), InferenceApiError> {
        unload_model_instance(
            &mut self.runtime,
            &self.instance,
            ModelInstanceUnloadPolicy::DrainActiveUse,
        )
        .map(|_| ())
    }

    pub(super) fn prepare_generation(
        &mut self,
        request: ProductionGenerationRequest,
        chat_formatter: Option<&dyn ChatTemplateFormatter>,
    ) -> Result<PreparedResidentProductionGeneration, InferenceApiError> {
        let ProductionGenerationRequest {
            prompt,
            parameters,
            mut stop_conditions,
            max_new_tokens,
            max_generation_millis,
        } = request;
        let max_tokens = max_new_tokens.unwrap_or_else(|| {
            self.fixture
                .manifest
                .generation
                .as_ref()
                .and_then(|defaults| defaults.max_tokens)
                .unwrap_or(64) as usize
        });
        if max_tokens > 1 && !self.provider_supports_multi_step_decode {
            return Err(InferenceApiError::Unsupported {
                reason: format!(
                    "provider '{}' does not support multi-step decode, but this request needs {max_tokens} decode steps",
                    self.provider_binding.as_str()
                ),
            });
        }
        let session_request = SessionCreationRequest {
            model: GenerationModelReference::ModelInstance(self.instance.clone()),
            tokenizer: generation_tokenizer_reference(&self.fixture),
            generation_defaults: parameters.clone(),
            policy: SessionPolicy::default(),
            memory: SessionMemoryBudget::default(),
            allowed_capabilities: BTreeSet::new(),
            correlation_id: None,
            created_at_millis: 0,
        };
        self.runtime
            .phase_ledger_mut()
            .begin_step(StepKind::RequestPreparation);
        let clock = self.runtime.phase_ledger().clock();
        let session = create_inference_session(&mut self.runtime, session_request)?;
        self.runtime
            .phase_ledger_mut()
            .record(clock, StepPhase::SessionCreate);
        let clock = self.runtime.phase_ledger().clock();
        let tokenized = tokenize_prompt_input(
            &self.fixture.tokenizer,
            TokenizationRequest::new(prompt),
            chat_formatter,
        )?;
        stop_conditions.prepared_stop_sequences = stop_conditions
            .stop_text_sequences
            .iter()
            .map(|text| self.fixture.tokenizer.resolve_stop_sequence(text))
            .collect::<Result<Vec<_>, _>>()
            .map_err(
                |error: TokenizerError| InferenceApiError::TokenizationFailed {
                    reason: error.to_string(),
                },
            )?;
        self.runtime
            .phase_ledger_mut()
            .record(clock, StepPhase::Tokenize);
        let observer = InferenceApiObserver::new();
        let prompt_token_count = tokenized.token_ids.len() as u64;
        let clock = self.runtime.phase_ledger().clock();
        let component_graphs = match self.graph_cache.get(prompt_token_count) {
            Some(graphs) => graphs,
            None => {
                let graphs = Arc::new(CachedGraphs::new(
                    match &self.component_digest {
                        Some(digest) => build_first_native_graphs_from_named_component(
                            digest,
                            &architecture_config_from_first_native_model_config(
                                &self.fixture.config,
                            ),
                            &self.fixture.identity,
                            prompt_token_count,
                        )
                        .map(|(graphs, _definition, _instance)| graphs),
                        None => build_first_native_graphs_from_real_qwen_component(
                            &self.fixture,
                            prompt_token_count,
                        )
                        .map(|(graphs, _definition, _instance)| graphs),
                    }
                    .map_err(|error| InferenceApiError::GraphPlanningFailed {
                        reason: error.to_string(),
                    })
                    .map(|graphs| self.graph_cache.optimize(&self.runtime, graphs))?,
                ));
                self.graph_cache
                    .insert(prompt_token_count, Arc::clone(&graphs));
                graphs
            }
        };
        self.runtime
            .phase_ledger_mut()
            .record(clock, StepPhase::GraphBuild);
        let clock = self.runtime.phase_ledger().clock();
        let prepared_plans = component_graphs
            .prepared_plans(
                &self.runtime,
                &self.instance,
                tokenized.token_ids.len() as u64,
                &self.provider_binding,
            )
            .map_err(|error| InferenceApiError::GraphPlanningFailed {
                reason: error.to_string(),
            })?;
        let generation_request = build_generation_request(
            GenerationRequestId::new("production-generation")?,
            Some(session.clone()),
            GenerationModelReference::ModelInstance(self.instance.clone()),
            generation_tokenizer_reference(&self.fixture),
            tokenized,
            max_tokens,
            parameters,
            stop_conditions,
            StreamingMode::TokenIds,
        );
        self.runtime
            .phase_ledger_mut()
            .record(clock, StepPhase::PlanPrepare);
        let clock = self.runtime.phase_ledger().clock();
        let generation_request = prepare_generation(&self.runtime, generation_request)?;
        self.runtime
            .phase_ledger_mut()
            .record(clock, StepPhase::RequestValidate);
        self.runtime.phase_ledger_mut().end_step();
        Ok(PreparedResidentProductionGeneration {
            session,
            generation_request,
            prepared_plans,
            observer,
            max_generation_millis,
        })
    }

    pub(super) fn finish_generation(
        &mut self,
        session: InferenceSessionId,
        request: &GenerationRequest,
        generation_result: GenerationResult,
        observer: InferenceApiObserver,
    ) -> Result<FirstNativeFixtureGeneration, InferenceApiError> {
        // A resident instance outlives every request it serves: release
        // the engine's per-request KV state before the session closes, or
        // each request's final KV tensors stay in Provider storage
        // (`bound-memory-manager-bookkeeping`).
        if let Some(engine) = self.runtime.model_execution_engine().cloned() {
            engine.release_request_state(&mut self.runtime, request)?;
        }
        let decoded_text = decode_tokens_streaming(
            &self.fixture.tokenizer,
            StreamingDecodeRequest::new(generation_result.output.generated_token_ids.clone()),
        )
        .map(|decoded| decoded.text)
        .unwrap_or_else(|_| {
            let tokens = generation_result
                .output
                .generated_token_ids
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(" ");
            format!("[generated token ids: {tokens}]")
        });
        let generation_result = generation_result.with_decoded_text(decoded_text);
        close_inference_session(&mut self.runtime, &session)?;
        self.runtime
            .forget_closed_inference_session(&session)
            .map_err(InferenceApiError::from)?;
        let text = generation_result.decoded_text.clone().ok_or_else(|| {
            InferenceApiError::GenerationFailed {
                reason: "production generation produced no decoded text".into(),
            }
        })?;
        Ok(FirstNativeFixtureGeneration {
            text,
            result: generation_result,
            observer,
        })
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub(super) struct PreparedResidentProductionGeneration {
    pub(super) session: InferenceSessionId,
    pub(super) generation_request: GenerationRequest,
    pub(super) prepared_plans: FirstNativePreparedPlans,
    pub(super) observer: InferenceApiObserver,
    pub(super) max_generation_millis: Option<u64>,
}

/// An incrementally-delivered production generation event
/// (`stream-production-generation-events`): `Token` is delivered once per
/// produced token, in production order, carrying an incremental,
/// tokenizer-decoded text delta (`Tokenizer::streaming_decode`'s own
/// UTF-8-boundary-safe accumulation -- never a duplicated or truncated
/// character); `Finished` is delivered exactly once, after the last
/// `Token`, whether generation ended by a stop condition, the token
/// budget, or caller-requested cancellation.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[derive(Clone, Debug, PartialEq)]
pub enum GenerationStreamEvent {
    Token {
        token_id: TokenId,
        text_delta: Option<String>,
    },
    Finished {
        finish_reason: FinishReason,
        usage: GenerationUsage,
    },
}

/// [`run_production_qwen_generation_for_provider_with_request`], delivering
/// each produced token to `on_event` as it happens instead of only
/// returning a final result (`stream-production-generation-events`,
/// MAG-P1-2). `on_event` returning `std::ops::ControlFlow::Break(())`
/// requests that generation stop after the token just delivered; it stops
/// cleanly (the exact same `FinishReason::Cancelled` path and resource
/// cleanup any other generation completion uses), not with a hard error.
/// Also returns the same `Result<FirstNativeFixtureGeneration, _>` shape
/// every other production entry point returns, once generation finishes,
/// for a caller that wants the aggregate result in addition to (or
/// instead of processing) the live stream.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub fn run_production_qwen_generation_for_provider_streaming(
    fixture: E2eFixture,
    payload_source: &dyn crate::production_model_ingestion::ProductionArtifactPayloadSource,
    trust_store: ModelTrustStore,
    request: ProductionGenerationRequest,
    chat_formatter: Option<&dyn ChatTemplateFormatter>,
    provider: Arc<dyn Provider>,
    on_event: &mut dyn FnMut(GenerationStreamEvent) -> std::ops::ControlFlow<()>,
) -> Result<FirstNativeFixtureGeneration, InferenceApiError> {
    let mut prepared = prepare_production_generation(
        fixture,
        payload_source,
        trust_store,
        request,
        chat_formatter,
        provider,
    )?;

    // Not built on `Tokenizer::streaming_decode`/`StreamingDecodeState`:
    // the real production tokenizer (`loaders/huggingface`'s
    // `HuggingFaceTokenizer`, reached through `FixtureTokenizer`'s
    // delegate) does not implement genuine incremental decode -- its
    // `decode()` ignores `streaming_state` entirely and simply decodes
    // whatever token slice it is given, discovered via this change's own
    // multi-token streaming test producing "!chi!" (per-token decode,
    // losing the underlying BPE decoder's inter-token joining) instead of
    // the correct "! c hi !" (whole-sequence decode). Full-redecoding the
    // cumulative token list every step and diffing against the
    // previously emitted text is slower (`O(n)` work per step) but is
    // correct for any `Tokenizer` implementation, including one that
    // (like the real production tokenizer today) carries no genuine
    // incremental state at all.
    let prepared_tokenizer = prepared.fixture.tokenizer.clone();
    let mut generated_so_far: Vec<TokenId> = Vec::new();
    let mut previously_emitted_text = String::new();
    let mut on_token = |token_id: TokenId| -> std::ops::ControlFlow<()> {
        generated_so_far.push(token_id);
        let full_text = decode_tokens_streaming(
            &prepared_tokenizer,
            StreamingDecodeRequest::new(generated_so_far.clone()),
        )
        .map(|decoded| decoded.text)
        .ok();
        let text_delta = full_text.as_ref().and_then(|full_text| {
            full_text
                .strip_prefix(previously_emitted_text.as_str())
                .filter(|delta| !delta.is_empty())
                .map(str::to_string)
        });
        if let Some(full_text) = full_text {
            previously_emitted_text = full_text;
        }
        on_event(GenerationStreamEvent::Token {
            token_id,
            text_delta,
        })
    };

    let mut execution_plans = RuntimeGenerationExecutionPlans {
        prefill: &mut prepared.prepared_plans.prefill,
        decode: &mut prepared.prepared_plans.decode,
    };
    let generation_result = run_generation_loop_with_execution_plans_streaming(
        &mut prepared.runtime,
        &prepared.generation_request,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated_so_far| false,
        &mut prepared.observer,
        &mut execution_plans,
        &mut on_token,
    )?;

    let _ = on_event(GenerationStreamEvent::Finished {
        finish_reason: generation_result.output.finish_reason,
        usage: generation_result.output.usage.clone(),
    });

    finish_production_generation(
        prepared.runtime,
        &prepared.fixture,
        &prepared.instance,
        &prepared.session,
        generation_result,
        prepared.observer,
    )
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub(super) fn expect_single_string_invocation_result(
    result: &ComponentInvocationResult,
    operation: &str,
) -> Result<String, E2eConformanceError> {
    match result.values.as_slice() {
        [ComponentValue::String(value)] => Ok(value.clone()),
        values => Err(E2eConformanceError::GraphValidationFailed {
            reason: format!(
                "Qwen Component export '{operation}' returned {values:?}, expected a single string"
            ),
        }),
    }
}
