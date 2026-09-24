use anyhow::{Context, Result, anyhow, bail};
use magnetar_loader_huggingface::{
    HuggingFaceChatTemplateFormatter, HuggingFaceIngestor, parse_tokenizer_config,
};
use magnetar_provider_cpu::ReferenceCpuProvider;
use magnetar_runtime::model::{ArtifactFormat, ModelTrustStatus, ModelTrustStore};
use magnetar_runtime::production_model_ingestion::{
    ProductionModelArtifactIngestor, ProductionModelSource, read_declared_artifact_format,
};
use magnetar_runtime::tokenizer::Tokenizer;
use magnetar_runtime::{
    ChatMessage, ComponentTrustStore, GenerationParameters, GenerationStreamEvent, GenerationUsage,
    ModelArtifactSource, ProductionGenerationRequest, PromptInput, Provider, StopConditions,
    register_inference_component_artifact,
};
use serde_json::Value;
use std::{
    fmt,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferenceComponentPlacement {
    ReferenceCpu,
    Cuda,
}

impl InferenceComponentPlacement {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "cpu" => Ok(Self::ReferenceCpu),
            "cuda" | "gpu" => Ok(Self::Cuda),
            other => bail!(
                "Magnetar inference Components support cpu and cuda placements, got `{other}`"
            ),
        }
    }
}

/// astorise/Magnetar audit round 5 (MAG-04): internal diagnostic-only
/// mirror of the resolved Provider/Device identity a loaded Component runs
/// under -- kept for this crate's own `Debug` output, never exported. An
/// embedder (Tachyon included) is expected to classify placement only via
/// the generic [`InferenceComponentPlacement`] it requested at load time,
/// never by reflecting Magnetar's own Provider/Device identity back out;
/// Tachyon's own boundary guard
/// (`scripts/validate_ai_component_boundary.sh`) explicitly forbids its
/// `core-host` from referencing this shape at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProviderDeviceClass {
    ReferenceCpu,
    // Only constructed by `capability_advertisement`'s `#[cfg(feature =
    // "cuda")]` branch; a `cuda`-less build (this crate's `cuda` feature is
    // default-off) never builds one, so `-D dead_code` flags it now that
    // MAG-04 made this enum crate-private and no longer publicly reachable.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    Cuda,
}

/// See [`ProviderDeviceClass`]'s own doc comment: internal-only, for
/// `Debug` formatting.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ComponentProviderAdvertisement {
    provider_name: String,
    provider_version: String,
    device_ids: Vec<String>,
    device_class: ProviderDeviceClass,
}

#[derive(Debug)]
pub struct InvalidComponentInvocation(String);

impl fmt::Display for InvalidComponentInvocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for InvalidComponentInvocation {}

pub fn is_invalid_component_invocation(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| cause.is::<InvalidComponentInvocation>())
}

fn invalid_request(message: impl Into<String>) -> anyhow::Error {
    anyhow!(InvalidComponentInvocation(message.into()))
}

#[derive(Clone, Debug)]
pub struct InferenceComponentSource {
    provenance: String,
    root: PathBuf,
}

impl InferenceComponentSource {
    pub fn authorized_local_bundle(
        provenance: impl Into<String>,
        root: impl Into<PathBuf>,
    ) -> Self {
        Self {
            provenance: provenance.into(),
            root: root.into(),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

#[derive(Clone, Debug)]
pub struct InferenceComponentArtifact {
    component_bytes: Vec<u8>,
    manifest_bytes: Vec<u8>,
}

impl InferenceComponentArtifact {
    pub fn from_bytes(component_bytes: Vec<u8>, manifest_bytes: Vec<u8>) -> Self {
        Self {
            component_bytes,
            manifest_bytes,
        }
    }
}

/// Two independent trust decisions (Tachyon integration audit MAG-03: a
/// Component's own executable trust must never be conflated with the
/// model/weights it happens to load): `model_trust_store` governs the
/// ingested Model Artifact (weights, tokenizer, config -- data, never
/// executed), while `component_trust_store` governs the WASM Component
/// binary itself (code, actually instantiated and invoked). Trusting one
/// says nothing about the other; a caller names both explicitly.
#[derive(Clone, Debug, Default)]
pub struct ArtifactTrustPolicy {
    model_trust_store: ModelTrustStore,
    component_trust_store: ComponentTrustStore,
}

impl ArtifactTrustPolicy {
    /// Trusts a Model Artifact digest (weights/tokenizer/config bundle) --
    /// never the Component binary. See [`Self::trust_component_digest`] for
    /// the separate, Component-specific trust decision.
    pub fn trust_digest(mut self, digest: &str) -> Self {
        self.model_trust_store = self.model_trust_store.trust_digest(digest);
        self
    }

    /// Trusts a Component (WASM) binary's own real content digest -- never
    /// the Model Artifact it happens to load. This is the digest
    /// `register_inference_component_artifact` computes from the actual
    /// bytes (`ComponentDigest::sha256`), not a value this policy invents;
    /// a caller names it explicitly here (e.g. one it has independently
    /// verified or pinned) before the artifact is ever registered.
    pub fn trust_component_digest(mut self, digest: &str) -> Self {
        self.component_trust_store = self.component_trust_store.trust_digest(digest);
        self
    }

    fn model_trust_store(&self) -> &ModelTrustStore {
        &self.model_trust_store
    }

    fn component_trust_store(&self) -> &ComponentTrustStore {
        &self.component_trust_store
    }

    fn into_model_trust_store(self) -> ModelTrustStore {
        self.model_trust_store
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InferenceComponentUsage {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InferenceComponentOutput {
    pub text: String,
    pub usage: InferenceComponentUsage,
}

/// [`InferenceComponentOutput`]'s opaque counterpart
/// (astorise/Magnetar#88): the same shape, with `text`/`usage` replaced by
/// raw bytes and string-keyed tags -- see
/// [`LoadedInferenceComponent::invoke_payload_opaque`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpaqueInferenceComponentOutput {
    pub bytes: Vec<u8>,
    pub tags: Vec<(String, String)>,
}

/// One resident, loaded inference Component instance.
///
/// # Concurrency model (Tachyon integration audit MAG-06)
///
/// **One active generation at a time per instance.** `invoke_payload`/
/// `invoke_payload_streaming` acquire `loaded_model`'s lock for the full
/// duration of a generation call (prefill through the last decode step);
/// a second call arriving while one is already in flight *blocks* until
/// the first finishes -- it is never rejected, dropped, or run
/// concurrently against the same `Runtime`/`ModelInstance`. This is a
/// deliberate choice, not an oversight: `ProductionLoadedModel` owns
/// one `Runtime` and one KV-cache-bearing `ModelInstance`, and
/// `magnetar-runtime`'s generation loop is not designed for two
/// generations to interleave their KV state within a single instance.
///
/// This is a per-*instance* constraint, not a process-wide one: an
/// embedder that wants concurrent generation across independent requests
/// loads multiple `LoadedInferenceComponent` instances (one `load()` call
/// each -- nothing here prevents that) and routes requests across them,
/// exactly the same way it would route across multiple model replicas on
/// separate hardware. This crate does not itself provide a scheduler,
/// batching, or an instance pool -- an embedder that needs one builds it
/// on top of this per-instance primitive; nothing here silently batches
/// requests together or reorders them.
pub struct LoadedInferenceComponent {
    name: String,
    root: PathBuf,
    chat_formatter: Option<Arc<HuggingFaceChatTemplateFormatter>>,
    provider: ComponentProviderAdvertisement,
    loaded_model: Mutex<magnetar_runtime::ProductionLoadedModel>,
}

pub fn local_bundle_manifest_digest(root: impl Into<PathBuf>) -> Result<String> {
    let root = root.into();
    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::Tachyon("tachyon:test-fixture".to_owned()),
        root.clone(),
    );
    // astorise/Magnetar#75: the declared-format sidecar, not filesystem
    // structure, picks the ingestor. A bundle missing/malformed sidecar
    // fails explicitly here instead of silently guessing a format.
    let declared_format = read_declared_artifact_format(&source).with_context(|| {
        format!(
            "Magnetar could not read the declared Artifact format for bundle at `{}`",
            root.display()
        )
    })?;
    let ingested = match declared_format {
        ArtifactFormat::Gguf => magnetar_loader_gguf::GgufIngestor::new().ingest(&source),
        ArtifactFormat::HuggingFace => HuggingFaceIngestor::new().ingest(&source),
    }
    .with_context(|| {
        format!(
            "Magnetar failed to inspect inference Component bundle at `{}`",
            root.display()
        )
    })?;
    Ok(ingested.manifest.id.digest.value)
}

/// Explicit, one-time legacy-migration path (astorise/Magnetar#75): derives
/// an [`ArtifactFormat`] for an old-style bundle that predates the
/// declared-format sidecar file, using the exact filesystem heuristic
/// `LoadedInferenceComponent::load`/`local_bundle_manifest_digest`
/// themselves used to rely on automatically, then writes the sidecar once
/// so every subsequent load goes through the explicit declaration path.
///
/// This is never called by `LoadedInferenceComponent::load` or
/// `local_bundle_manifest_digest` -- per the accepted decision on #75,
/// filesystem-structure detection may remain only as an explicit
/// import/migration mechanism for legacy Artifacts, never as a runtime
/// fallback. A caller migrating a legacy bundle invokes this explicitly,
/// once, as an upgrade step; it refuses to run again on a bundle that
/// already carries a sidecar, so it cannot silently overwrite an operator's
/// own explicit declaration.
pub fn migrate_legacy_bundle_artifact_format(root: impl Into<PathBuf>) -> Result<ArtifactFormat> {
    let root = root.into();
    let sidecar_path =
        root.join(magnetar_runtime::production_model_ingestion::ARTIFACT_FORMAT_SIDECAR_FILE_NAME);
    if sidecar_path.is_file() {
        bail!(
            "Magnetar will not run legacy artifact-format migration on `{}`: a declared-format \
             sidecar already exists at `{}`",
            root.display(),
            sidecar_path.display()
        );
    }
    let format = if root.join(magnetar_loader_gguf::GGUF_FILE_NAME).is_file() {
        ArtifactFormat::Gguf
    } else {
        ArtifactFormat::HuggingFace
    };
    std::fs::write(
        &sidecar_path,
        format!("artifact_format: {}\n", format.as_str()),
    )
    .with_context(|| {
        format!(
            "Magnetar failed to write declared-format sidecar at `{}`",
            sidecar_path.display()
        )
    })?;
    Ok(format)
}

pub fn cuda_provider_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        magnetar_provider_cuda::CudaProvider::new().is_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

impl fmt::Debug for LoadedInferenceComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoadedInferenceComponent")
            .field("name", &self.name)
            .field("root", &self.root)
            .field("provider", &self.provider)
            .finish_non_exhaustive()
    }
}

impl LoadedInferenceComponent {
    pub fn load(
        name: &str,
        artifact: InferenceComponentArtifact,
        source: InferenceComponentSource,
        trust_policy: ArtifactTrustPolicy,
        placement: InferenceComponentPlacement,
    ) -> Result<Self> {
        // The Component's own real digest (computed from its actual bytes,
        // never a claim) evaluated against this policy's `component_trust_
        // store` -- independent of the Model Artifact trust evaluated
        // below (Tachyon integration audit MAG-03). Fails closed here,
        // before any model ingestion happens, if this Component binary is
        // not one the caller has explicitly named as trusted.
        let component_digest = register_inference_component_artifact(
            artifact.component_bytes,
            artifact.manifest_bytes,
            trust_policy.component_trust_store(),
        )
        .map_err(|error| {
            anyhow!("Magnetar rejected inference Component artifact for `{name}`: {error}")
        })?
        .digest;

        let production_source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::Tachyon(source.provenance.clone()),
            source.root.clone(),
        );
        // Format selection: the bundle's own declared-format sidecar file
        // says which of this crate's two real ingestors applies -- Magnetar
        // never infers it from filesystem structure (astorise/Magnetar#75,
        // Tachyon integration audit MAG-01). A missing or malformed
        // declaration fails explicitly instead of guessing. Adding a third
        // real ingestor to this codebase would extend this same match, not
        // require touching every call site below -- all branches converge
        // on the same `ProductionIngestionResult`/`Arc<dyn Tokenizer>`/
        // `Option<String>` (raw chat template text) shapes.
        let declared_format =
            read_declared_artifact_format(&production_source).with_context(|| {
                format!(
                    "Magnetar could not read the declared Artifact format for Component `{name}` \
                 bundle at `{}`",
                    source.root.display()
                )
            })?;
        let ingested: magnetar_runtime::production_model_ingestion::ProductionIngestionResult =
            match declared_format {
                ArtifactFormat::Gguf => {
                    magnetar_loader_gguf::GgufIngestor::new().ingest(&production_source)
                }
                ArtifactFormat::HuggingFace => {
                    HuggingFaceIngestor::new().ingest(&production_source)
                }
            }
            .with_context(|| {
                format!(
                    "Magnetar failed to ingest inference Component bundle at `{}`",
                    source.root.display()
                )
            })?;
        let vocab_size = ingested
            .manifest
            .architecture_config
            .as_ref()
            .map(|config| config.vocab_size);

        let (real_tokenizer, chat_template_text): (
            Arc<dyn Tokenizer + Send + Sync>,
            Option<String>,
        ) = if declared_format == ArtifactFormat::Gguf {
            let tokenizer = magnetar_loader_gguf::load_gguf_tokenizer(
                &production_source,
                format!("{name}-tokenizer"),
                vocab_size,
            )
            .with_context(|| format!("Magnetar failed to load tokenizer for Component `{name}`"))?;
            let chat_template_text =
                magnetar_loader_gguf::load_gguf_chat_template(&production_source).with_context(
                    || format!("Magnetar failed to load chat template for Component `{name}`"),
                )?;
            (Arc::new(tokenizer), chat_template_text)
        } else {
            let tokenizer_path = source.root.join("tokenizer.json");
            let tokenizer_bytes = std::fs::read(&tokenizer_path)
                .with_context(|| format!("failed to read `{}`", tokenizer_path.display()))?;
            let tokenizer_config_path = source.root.join("tokenizer_config.json");
            let tokenizer_config_bytes = if tokenizer_config_path.is_file() {
                Some(std::fs::read(&tokenizer_config_path).with_context(|| {
                    format!("failed to read `{}`", tokenizer_config_path.display())
                })?)
            } else {
                None
            };
            let tokenizer_config_metadata = tokenizer_config_bytes
                .as_deref()
                .map(parse_tokenizer_config)
                .transpose()
                .with_context(|| {
                    format!(
                        "Magnetar failed to parse tokenizer configuration for Component `{name}`"
                    )
                })?;
            let chat_template_text = tokenizer_config_metadata
                .as_ref()
                .and_then(|metadata| metadata.chat_template_reference.clone());
            let tokenizer = magnetar_loader_huggingface::HuggingFaceTokenizer::from_bytes(
                &tokenizer_bytes,
                tokenizer_config_metadata.as_ref(),
                format!("{name}-tokenizer"),
                vocab_size,
            )
            .with_context(|| format!("Magnetar failed to load tokenizer for Component `{name}`"))?;
            (Arc::new(tokenizer), chat_template_text)
        };
        let tokenizer_metadata = real_tokenizer.metadata().clone();
        let chat_formatter = chat_template_text
            .map(HuggingFaceChatTemplateFormatter::new)
            .transpose()
            .with_context(|| {
                format!("Magnetar failed to load chat template for Component `{name}`")
            })?
            .map(Arc::new);

        let trust_decision = trust_policy
            .model_trust_store()
            .evaluate(&ingested.manifest);
        if trust_decision.status() != ModelTrustStatus::Trusted {
            bail!(
                "Magnetar Model Artifact trust rejected for `{name}`: {}",
                trust_decision.reason()
            );
        }
        let fixture = magnetar_runtime::production_model_fixture(
            ingested.manifest.clone(),
            tokenizer_metadata,
            real_tokenizer,
        )
        .with_context(|| {
            format!("Magnetar failed to build production fixture for Component `{name}`")
        })?;
        let provider = capability_advertisement(placement)?;
        let provider_for_generation = provider_for_placement(placement, name)?;
        // Threads the caller-registered Component's own digest through to
        // generation: `load_with_component` makes this specific Component
        // (not the CLI's hardcoded `"qwen-test"` singleton) the real graph-
        // production authority for both plan preparation and dispatch-time
        // execution (Tachyon integration audit MAG-02) -- verified to
        // produce identical generation to the singleton path for identical
        // Component bytes
        // (`production_loaded_model_load_with_component_matches_the_
        // singleton_path`).
        let loaded_model = magnetar_runtime::ProductionLoadedModel::load_with_component(
            fixture,
            ingested.payload_source.as_ref(),
            trust_policy.into_model_trust_store(),
            provider_for_generation,
            Some(component_digest),
        )
        .with_context(|| {
            format!("Magnetar failed to materialize resident inference Component `{name}`")
        })?;

        Ok(Self {
            name: name.to_owned(),
            root: source.root,
            chat_formatter,
            provider,
            loaded_model: Mutex::new(loaded_model),
        })
    }

    pub fn resident_debug(&self) -> Result<(String, usize)> {
        let loaded = self.loaded_model.lock().map_err(|_| {
            anyhow!(
                "resident Magnetar inference Component lock poisoned for `{}`",
                self.name
            )
        })?;
        Ok((
            loaded.model_instance_id().to_string(),
            loaded.materialization_count(),
        ))
    }

    /// Blocks until any generation already in flight on this instance
    /// finishes -- see [`LoadedInferenceComponent`]'s own "Concurrency
    /// model" doc comment.
    pub fn invoke_payload(&self, payload: &[u8]) -> Result<InferenceComponentOutput> {
        let request = InvocationPayload::parse(payload)?.into_generation_request()?;
        let outcome = self
            .loaded_model
            .lock()
            .map_err(|_| {
                anyhow!(
                    "resident Magnetar inference Component lock poisoned for `{}`",
                    self.name
                )
            })?
            .generate(request, self.chat_formatter())
            .map_err(|error| {
                anyhow!(
                    "Magnetar inference Component invocation for `{}` failed: {error}",
                    self.name
                )
            })?;
        let usage = outcome.result.output.usage;
        Ok(InferenceComponentOutput {
            text: outcome.text,
            usage: InferenceComponentUsage {
                prompt_tokens: usage.prompt_tokens,
                generated_tokens: usage.generated_tokens,
            },
        })
    }

    /// Blocks until any generation already in flight on this instance
    /// finishes -- see [`LoadedInferenceComponent`]'s own "Concurrency
    /// model" doc comment.
    pub fn invoke_payload_streaming(
        &self,
        payload: &[u8],
        on_event: &mut dyn FnMut(GenerationStreamEvent) -> std::ops::ControlFlow<()>,
    ) -> Result<InferenceComponentUsage> {
        let request = InvocationPayload::parse(payload)?.into_generation_request()?;
        let outcome = self
            .loaded_model
            .lock()
            .map_err(|_| {
                anyhow!(
                    "resident Magnetar inference Component lock poisoned for `{}`",
                    self.name
                )
            })?
            .generate_streaming(request, self.chat_formatter(), on_event)
            .map_err(|error| {
                anyhow!(
                    "Magnetar inference Component streaming invocation for `{}` failed: {error}",
                    self.name
                )
            })?;
        let usage = outcome.result.output.usage;
        Ok(InferenceComponentUsage {
            prompt_tokens: usage.prompt_tokens,
            generated_tokens: usage.generated_tokens,
        })
    }

    /// Opaque-bytes counterpart to [`Self::invoke_payload`]
    /// (astorise/Magnetar#88, Tachyon integration audit TACH-02): returns
    /// the generated text as raw UTF-8 bytes plus a string-keyed metadata
    /// tag list, instead of [`InferenceComponentOutput`]'s typed `text`/
    /// `usage` fields. An embedder whose own architecture must stay
    /// model-agnostic -- never touching token/usage vocabulary directly --
    /// can relay this straight through as opaque bytes and tags, instead of
    /// writing its own typed-to-opaque translation layer against
    /// `invoke_payload`'s output (exactly the boundary violation TACH-02
    /// flagged in Tachyon's own adapter).
    ///
    /// Tags always carry `prompt_tokens`/`generated_tokens`/`finish_reason`
    /// (the same fields [`InferenceComponentUsage`] and
    /// [`GenerationStreamEvent::Finished`] already carry, just as opaque
    /// string values instead of typed ones) -- a caller that wants
    /// structured accounting still has `invoke_payload` for that; this
    /// entry point is for one that does not.
    ///
    /// Blocks until any generation already in flight on this instance
    /// finishes -- see [`LoadedInferenceComponent`]'s own "Concurrency
    /// model" doc comment.
    pub fn invoke_payload_opaque(&self, payload: &[u8]) -> Result<OpaqueInferenceComponentOutput> {
        let request = InvocationPayload::parse(payload)?.into_generation_request()?;
        let outcome = self
            .loaded_model
            .lock()
            .map_err(|_| {
                anyhow!(
                    "resident Magnetar inference Component lock poisoned for `{}`",
                    self.name
                )
            })?
            .generate(request, self.chat_formatter())
            .map_err(|error| {
                anyhow!(
                    "Magnetar inference Component invocation for `{}` failed: {error}",
                    self.name
                )
            })?;
        let tags = generation_usage_tags(&outcome.result.output.usage);
        Ok(OpaqueInferenceComponentOutput {
            bytes: outcome.text.into_bytes(),
            tags,
        })
    }

    /// Opaque-bytes counterpart to [`Self::invoke_payload_streaming`]
    /// (astorise/Magnetar#88, Tachyon integration audit TACH-02): delivers
    /// each produced token's decoded text delta to `on_frame` as a raw
    /// UTF-8 byte frame, instead of [`GenerationStreamEvent`]'s typed
    /// `Token`/`Finished` variants -- a caller relaying frames opaquely has
    /// no typed event to fall back on, only bytes. A `Token` whose own
    /// `text_delta` is `None` (no new visible text yet, e.g. a byte-level
    /// tokenizer's not-yet-complete multi-byte sequence) delivers no frame
    /// at all, rather than an empty one -- an opaque byte-frame consumer has
    /// no way to distinguish "empty frame" from "no frame," so skipping
    /// keeps every delivered frame meaningful. Returns the same final
    /// metadata tags [`Self::invoke_payload_opaque`] does
    /// (`prompt_tokens`/`generated_tokens`/`finish_reason`), once
    /// generation finishes.
    ///
    /// Blocks until any generation already in flight on this instance
    /// finishes -- see [`LoadedInferenceComponent`]'s own "Concurrency
    /// model" doc comment.
    pub fn invoke_payload_streaming_opaque(
        &self,
        payload: &[u8],
        on_frame: &mut dyn FnMut(&[u8]) -> std::ops::ControlFlow<()>,
    ) -> Result<Vec<(String, String)>> {
        let request = InvocationPayload::parse(payload)?.into_generation_request()?;
        let mut tags = Vec::new();
        let mut on_event = |event: GenerationStreamEvent| -> std::ops::ControlFlow<()> {
            match event {
                GenerationStreamEvent::Token { text_delta, .. } => match text_delta {
                    Some(text_delta) if !text_delta.is_empty() => on_frame(text_delta.as_bytes()),
                    _ => std::ops::ControlFlow::Continue(()),
                },
                GenerationStreamEvent::Finished { usage, .. } => {
                    tags = generation_usage_tags(&usage);
                    std::ops::ControlFlow::Continue(())
                }
            }
        };
        self.loaded_model
            .lock()
            .map_err(|_| {
                anyhow!(
                    "resident Magnetar inference Component lock poisoned for `{}`",
                    self.name
                )
            })?
            .generate_streaming(request, self.chat_formatter(), &mut on_event)
            .map_err(|error| {
                anyhow!(
                    "Magnetar inference Component streaming invocation for `{}` failed: {error}",
                    self.name
                )
            })?;
        Ok(tags)
    }

    fn chat_formatter(&self) -> Option<&dyn magnetar_runtime::ChatTemplateFormatter> {
        self.chat_formatter
            .as_deref()
            .map(|formatter| formatter as &dyn magnetar_runtime::ChatTemplateFormatter)
    }
}

/// Shared by [`LoadedInferenceComponent::invoke_payload_opaque`] and
/// [`LoadedInferenceComponent::invoke_payload_streaming_opaque`]: the same
/// three fields, rendered as opaque string values rather than typed ones.
/// `finish_reason` uses `FinishReason`'s own `Debug` representation (its
/// exact variant name, e.g. `"EosToken"`) -- stable enough to be a useful
/// opaque tag without this crate maintaining a separate string mapping.
fn generation_usage_tags(usage: &GenerationUsage) -> Vec<(String, String)> {
    vec![
        ("prompt_tokens".to_string(), usage.prompt_tokens.to_string()),
        (
            "generated_tokens".to_string(),
            usage.generated_tokens.to_string(),
        ),
        (
            "finish_reason".to_string(),
            format!("{:?}", usage.finish_reason),
        ),
    ]
}

fn capability_advertisement(
    placement: InferenceComponentPlacement,
) -> Result<ComponentProviderAdvertisement> {
    match placement {
        InferenceComponentPlacement::ReferenceCpu => {
            let provider = ReferenceCpuProvider::new();
            Ok(provider_advertisement(
                &provider,
                ProviderDeviceClass::ReferenceCpu,
            ))
        }
        InferenceComponentPlacement::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let provider = magnetar_provider_cuda::CudaProvider::new();
                if !provider.is_available() {
                    bail!(
                        "Magnetar CUDA provider is unavailable; explicit CUDA placement cannot fall back to CPU"
                    );
                }
                Ok(provider_advertisement(&provider, ProviderDeviceClass::Cuda))
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!("Magnetar CUDA provider is not compiled into this Magnetar Component adapter")
            }
        }
    }
}

fn provider_for_placement(
    placement: InferenceComponentPlacement,
    name: &str,
) -> Result<Arc<dyn Provider>> {
    match placement {
        InferenceComponentPlacement::ReferenceCpu => {
            Ok(Arc::new(ReferenceCpuProvider::new()) as Arc<dyn Provider>)
        }
        InferenceComponentPlacement::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let provider = magnetar_provider_cuda::CudaProvider::new();
                if !provider.is_available() {
                    bail!(
                        "Magnetar CUDA provider is unavailable for `{name}`; explicit CUDA placement cannot fall back to CPU"
                    );
                }
                Ok(Arc::new(provider) as Arc<dyn Provider>)
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = name;
                bail!("Magnetar CUDA provider is not compiled into this Magnetar Component adapter")
            }
        }
    }
}

fn provider_advertisement(
    provider: &dyn Provider,
    device_class: ProviderDeviceClass,
) -> ComponentProviderAdvertisement {
    let metadata = provider.metadata();
    let device_ids = provider
        .devices()
        .iter()
        .map(|device| device.metadata().id.as_str().to_owned())
        .collect();
    ComponentProviderAdvertisement {
        provider_name: metadata.name,
        provider_version: metadata.version,
        device_ids,
        device_class,
    }
}

#[derive(Debug)]
struct InvocationPayload {
    prompt: PromptInput,
    parameters: GenerationParameters,
    stop_conditions: StopConditions,
    max_new_tokens: Option<usize>,
    max_generation_millis: Option<u64>,
}

impl InvocationPayload {
    fn parse(payload: &[u8]) -> Result<Self> {
        let text = std::str::from_utf8(payload).map_err(|error| {
            invalid_request(format!(
                "Component invocation payload must be UTF-8: {error}"
            ))
        })?;
        let Ok(value) = serde_json::from_str::<Value>(text) else {
            return Ok(Self::plain(text));
        };
        let Value::Object(object) = value else {
            return Ok(Self::plain(text));
        };
        reject_unsupported_fields(&object)?;
        reject_component_opaque_controls(&object)?;
        let prompt = if let Some(messages) = object.get("messages") {
            PromptInput::ChatMessages(parse_chat_messages(messages)?)
        } else {
            let prompt = object
                .get("prompt")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .unwrap_or_else(|| text.to_owned());
            PromptInput::PlainText(prompt)
        };
        let max_new_tokens = object
            .get("max_new_tokens")
            .or_else(|| object.get("max_tokens"))
            .and_then(Value::as_u64)
            .map(|value| value.min(usize::MAX as u64) as usize);
        let mut parameters = GenerationParameters::greedy();
        if let Some(temperature) = json_f32(&object, "temperature")? {
            if temperature == 0.0 {
                parameters = GenerationParameters::greedy();
            } else {
                parameters.temperature = temperature;
                parameters.greedy = false;
                parameters.sampling_enabled = true;
            }
        }
        if let Some(top_p) = json_f32(&object, "top_p")? {
            parameters.top_p = Some(top_p);
        }
        if let Some(top_k) = object
            .get("top_k")
            .and_then(Value::as_u64)
            .map(|value| value.min(u64::from(u32::MAX)) as u32)
        {
            parameters.top_k = Some(top_k);
        }
        if let Some(seed) = object.get("seed").and_then(Value::as_u64) {
            parameters.seed = Some(seed);
            parameters.deterministic = true;
        }
        parameters.frequency_penalty = json_f32(&object, "frequency_penalty")?;
        parameters.presence_penalty = json_f32(&object, "presence_penalty")?;
        parameters.repetition_penalty = json_f32(&object, "repetition_penalty")?;

        let mut stop_conditions = StopConditions::default();
        if let Some(stop) = object.get("stop") {
            stop_conditions.stop_text_sequences = parse_stop_sequences(stop)?;
        }
        Ok(Self {
            prompt,
            parameters,
            stop_conditions,
            max_new_tokens,
            max_generation_millis: parse_max_generation_millis(&object)?,
        })
    }

    fn plain(text: &str) -> Self {
        Self {
            prompt: PromptInput::PlainText(text.to_owned()),
            parameters: GenerationParameters::greedy(),
            stop_conditions: StopConditions::default(),
            max_new_tokens: None,
            max_generation_millis: None,
        }
    }

    fn into_generation_request(self) -> Result<ProductionGenerationRequest> {
        self.parameters
            .validate()
            .map_err(|error| invalid_request(format!("invalid generation parameters: {error}")))?;
        Ok(ProductionGenerationRequest {
            prompt: self.prompt,
            parameters: self.parameters,
            stop_conditions: self.stop_conditions,
            max_new_tokens: self.max_new_tokens,
            max_generation_millis: self.max_generation_millis,
        })
    }
}

fn reject_unsupported_fields(object: &serde_json::Map<String, Value>) -> Result<()> {
    const SUPPORTED: &[&str] = &[
        "prompt",
        "messages",
        "max_new_tokens",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "include_usage",
        "tools",
        "tool_choice",
        "tool_call_parser",
        "max_generation_ms",
        "json_schema",
    ];
    if let Some(key) = object.keys().find(|key| !SUPPORTED.contains(&key.as_str())) {
        return Err(invalid_request(format!(
            "inference Component invocation field `{key}` is not supported"
        )));
    }
    Ok(())
}

fn reject_component_opaque_controls(object: &serde_json::Map<String, Value>) -> Result<()> {
    if object
        .get("tools")
        .is_some_and(|tools| !is_empty_json(tools))
        || object
            .get("tool_choice")
            .is_some_and(|choice| choice.as_str() != Some("none"))
        || object.get("tool_call_parser").is_some()
    {
        return Err(invalid_request(
            "tool-call semantics must be handled by the guest or inference Component, not the Tachyon core boundary",
        ));
    }
    if object.get("json_schema").is_some() {
        return Err(invalid_request(
            "structured output is not advertised for this local inference Component",
        ));
    }
    Ok(())
}

fn parse_max_generation_millis(object: &serde_json::Map<String, Value>) -> Result<Option<u64>> {
    let Some(value) = object.get("max_generation_ms") else {
        return Ok(None);
    };
    let millis = value
        .as_u64()
        .ok_or_else(|| invalid_request("`max_generation_ms` must be an integer"))?;
    if millis == 0 {
        return Err(invalid_request(
            "`max_generation_ms` must be greater than zero",
        ));
    }
    Ok(Some(millis))
}

fn is_empty_json(value: &Value) -> bool {
    matches!(value, Value::Null)
        || value.as_array().is_some_and(Vec::is_empty)
        || value.as_object().is_some_and(serde_json::Map::is_empty)
}

fn parse_chat_messages(value: &Value) -> Result<Vec<ChatMessage>> {
    let messages = value
        .as_array()
        .ok_or_else(|| invalid_request("`messages` must be an array"))?;
    messages
        .iter()
        .map(|message| {
            let object = message
                .as_object()
                .ok_or_else(|| invalid_request("chat message entries must be objects"))?;
            let role = object
                .get("role")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid_request("chat message entries require string `role`"))?;
            let content = object
                .get("content")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid_request("chat message entries require string `content`"))?;
            Ok(ChatMessage::new(role, content))
        })
        .collect()
}

fn parse_stop_sequences(value: &Value) -> Result<Vec<String>> {
    if let Some(stop) = value.as_str() {
        return Ok(vec![stop.to_owned()]);
    }
    let stops = value
        .as_array()
        .ok_or_else(|| invalid_request("`stop` must be a string or string array"))?;
    stops
        .iter()
        .map(|stop| {
            stop.as_str()
                .map(str::to_owned)
                .ok_or_else(|| invalid_request("`stop` array entries must be strings"))
        })
        .collect()
}

fn json_f32(object: &serde_json::Map<String, Value>, key: &str) -> Result<Option<f32>> {
    object
        .get(key)
        .map(|value| {
            value
                .as_f64()
                .ok_or_else(|| invalid_request(format!("`{key}` must be a number")))
                .map(|value| value as f32)
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invocation_payload_maps_chat_controls_to_runtime_contracts() {
        let request = InvocationPayload::parse(
            br#"{
                "messages":[{"role":"system","content":"brief"},{"role":"user","content":"hi"}],
                "temperature":0.7,
                "top_p":0.9,
                "top_k":12,
                "seed":42,
                "frequency_penalty":0.1,
                "presence_penalty":0.2,
                "repetition_penalty":1.1,
                "max_new_tokens":16,
                "stop":["<eos>"],
                "include_usage":true
            }"#,
        )
        .expect("request should parse")
        .into_generation_request()
        .expect("supported controls should build a request");

        match request.prompt {
            PromptInput::ChatMessages(messages) => {
                assert_eq!(messages.len(), 2);
                assert_eq!(messages[0].role, "system");
                assert_eq!(messages[1].content, "hi");
            }
            other => panic!("expected chat messages, got {other:?}"),
        }
        assert_eq!(request.max_new_tokens, Some(16));
        assert_eq!(request.parameters.temperature, 0.7);
        assert_eq!(request.parameters.top_p, Some(0.9));
        assert_eq!(request.parameters.top_k, Some(12));
        assert_eq!(request.parameters.seed, Some(42));
        assert!(request.parameters.deterministic);
        assert_eq!(request.stop_conditions.stop_text_sequences, vec!["<eos>"]);
    }

    #[test]
    fn invocation_payload_rejects_boundary_owned_tool_and_json_controls() {
        for payload in [
            br#"{"prompt":"hi","tools":[{"type":"function","function":{"name":"x"}}]}"#.as_slice(),
            br#"{"prompt":"hi","tool_choice":"auto"}"#.as_slice(),
            br#"{"prompt":"hi","tool_call_parser":"qwen"}"#.as_slice(),
            br#"{"prompt":"hi","json_schema":"{\"type\":\"object\"}"}"#.as_slice(),
        ] {
            let error = InvocationPayload::parse(payload)
                .expect_err("opaque Component boundary controls must fail closed");
            assert!(is_invalid_component_invocation(&error));
        }
    }

    #[test]
    fn invocation_payload_carries_deadline() {
        let request = InvocationPayload::parse(br#"{"prompt":"hi","max_generation_ms":5000}"#)
            .expect("deadline should parse")
            .into_generation_request()
            .expect("deadline should build request");
        assert_eq!(request.max_generation_millis, Some(5000));

        let error = InvocationPayload::parse(br#"{"prompt":"hi","max_generation_ms":0}"#)
            .expect_err("zero deadline should fail closed");
        assert!(is_invalid_component_invocation(&error));
    }
}

/// This crate's own end-to-end integration test for `LoadedInferenceComponent::
/// load` (the noted gap left after the Tachyon integration audit closure: the
/// equivalent `magnetar-runtime` path is exercised by
/// `production_loaded_model_load_with_component_matches_the_singleton_path`,
/// but that test lives inside `magnetar-runtime` and never drives this
/// crate's own orchestration -- format detection, tokenizer construction,
/// Component registration, trust evaluation, `load_with_component` wiring --
/// through `LoadedInferenceComponent::load` itself). Builds a real,
/// completely-specified Hugging Face-shaped bundle on disk (every tensor the
/// real Qwen Component's graph resolves, at the exact shapes
/// `magnetar_runtime::first_native_expected_tensor_shape` expects) and runs a real
/// generation through `invoke_payload`, the same call shape a real embedder
/// (e.g. Tachyon) uses.
#[cfg(test)]
mod load_end_to_end_tests {
    use super::*;
    use std::io::Write as _;

    fn qwen_component_bytes() -> &'static [u8] {
        include_bytes!("../../magnetar-runtime/fixtures/components/qwen-real.component.wasm")
    }

    fn qwen_component_manifest_bytes() -> &'static [u8] {
        include_bytes!(
            "../../magnetar-runtime/fixtures/components/qwen-real.component.wasm.magnetar-component.yaml"
        )
    }

    fn tiny_config_json() -> Vec<u8> {
        serde_json::json!({
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "vocab_size": 4,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false,
            "torch_dtype": "float32",
            "bos_token_id": 0,
            "eos_token_id": 1
        })
        .to_string()
        .into_bytes()
    }

    /// A tiny, real `tokenizer.json` (WordLevel over a 4-token vocabulary,
    /// matching `tiny_config_json`'s `vocab_size`) -- the same recipe
    /// `loaders/huggingface`'s own tokenizer tests use, real `tokenizers`-
    /// crate input that round-trips "hello world" deterministically,
    /// unlike a byte-level BPE vocabulary too small to round-trip arbitrary
    /// text (a real failure hit and worked around earlier in this session's
    /// `loaders/gguf` tokenizer test).
    fn tiny_tokenizer_json() -> Vec<u8> {
        serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {
                    "id": 0, "content": "<bos>", "special": true,
                    "single_word": false, "lstrip": false, "rstrip": false, "normalized": false
                },
                {
                    "id": 1, "content": "<eos>", "special": true,
                    "single_word": false, "lstrip": false, "rstrip": false, "normalized": false
                }
            ],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"<bos>": 0, "<eos>": 1, "hello": 2, "world": 3},
                "unk_token": "hello"
            }
        })
        .to_string()
        .into_bytes()
    }

    fn write_safetensors(path: &Path, tensors: &[(&str, Vec<u64>, Vec<f32>)]) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, values) in tensors {
            let expected_elements: u64 = shape.iter().product();
            assert_eq!(
                expected_elements as usize,
                values.len(),
                "{name} declared shape does not match the number of values supplied"
            );
            let start = data.len() as u64;
            for value in values {
                data.extend_from_slice(&value.to_le_bytes());
            }
            let end = data.len() as u64;
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, end]}),
            );
        }
        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_bytes).unwrap();
        file.write_all(&data).unwrap();
    }

    /// Small, deterministic, bounded-magnitude values -- this test's only
    /// concern is that a real generation runs end to end through this
    /// crate's own orchestration, not numerical correctness (already
    /// covered by `magnetar-runtime`'s own equivalent test); values just
    /// need to be finite and not degenerate (e.g. all-zero, which would
    /// make RMSNorm divide a zero vector by its own near-zero norm).
    fn filled(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|index| (((index as f32) * 0.037 + seed) % 1.0 - 0.5) * 0.1)
            .collect()
    }

    /// Every tensor name/shape here matches `magnetar_runtime::
    /// first_native_expected_tensor_shape`'s HF-stored (pre-transpose) convention
    /// exactly for `tiny_config_json`'s dimensions: hidden_size=8,
    /// intermediate_size=16, num_hidden_layers=1, num_attention_heads=
    /// num_key_value_heads=2 (head_dim=4, so q/k/v_dim=8), vocab_size=4,
    /// untied embeddings (so `lm_head.weight` is declared explicitly,
    /// avoiding the separate tied-embedding derivation path).
    fn write_tiny_qwen_bundle(dir: &Path) {
        std::fs::write(dir.join("config.json"), tiny_config_json()).unwrap();
        std::fs::write(dir.join("tokenizer.json"), tiny_tokenizer_json()).unwrap();
        let tensors: Vec<(&str, Vec<u64>, Vec<f32>)> = vec![
            ("model.embed_tokens.weight", vec![4, 8], filled(32, 0.10)),
            ("model.norm.weight", vec![8], filled(8, 0.20)),
            (
                "model.layers.0.input_layernorm.weight",
                vec![8],
                filled(8, 0.30),
            ),
            (
                "model.layers.0.self_attn.q_proj.weight",
                vec![8, 8],
                filled(64, 0.40),
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                vec![8, 8],
                filled(64, 0.50),
            ),
            (
                "model.layers.0.self_attn.v_proj.weight",
                vec![8, 8],
                filled(64, 0.60),
            ),
            (
                "model.layers.0.self_attn.o_proj.weight",
                vec![8, 8],
                filled(64, 0.70),
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                vec![8],
                filled(8, 0.80),
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                vec![16, 8],
                filled(128, 0.90),
            ),
            (
                "model.layers.0.mlp.up_proj.weight",
                vec![16, 8],
                filled(128, 1.00),
            ),
            (
                "model.layers.0.mlp.down_proj.weight",
                vec![8, 16],
                filled(128, 1.10),
            ),
            ("lm_head.weight", vec![4, 8], filled(32, 1.20)),
        ];
        write_safetensors(&dir.join("model.safetensors"), &tensors);
        write_declared_artifact_format_sidecar(dir, ArtifactFormat::HuggingFace);
    }

    /// #81: a genuinely independent second Model Artifact -- distinct
    /// dimensions from `tiny_config_json`'s (not just a relabeled
    /// `model_type`), its own tensor set, and its own tokenizer vocabulary --
    /// for pairing with the real Llama Component instead of reusing
    /// `write_tiny_qwen_bundle`'s exact tensors/config/tokenizer. Grouped-
    /// query attention with a real head-count reduction this time
    /// (`num_attention_heads: 3`, `num_key_value_heads: 1`, so `q_dim: 12`
    /// but `kv_dim: 4` -- genuinely non-square projections, unlike the Qwen
    /// fixture's `num_attention_heads == num_key_value_heads` shape, which
    /// never exercises the GQA head-repetition path), and two decoder layers
    /// instead of one.
    fn llama_config_json() -> Vec<u8> {
        serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 12,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 3,
            "num_key_value_heads": 1,
            "vocab_size": 6,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false,
            "torch_dtype": "float32",
            "bos_token_id": 0,
            "eos_token_id": 1
        })
        .to_string()
        .into_bytes()
    }

    /// A real `tokenizer.json` (WordLevel over a 6-token vocabulary matching
    /// `llama_config_json`'s `vocab_size`), genuinely distinct from
    /// `tiny_tokenizer_json`'s vocabulary -- "the quick fox" round-trips to
    /// exactly 3 real tokens under this tokenizer, unlike the Qwen fixture's
    /// "hello world" (2 tokens).
    fn llama_tokenizer_json() -> Vec<u8> {
        serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {
                    "id": 0, "content": "<bos>", "special": true,
                    "single_word": false, "lstrip": false, "rstrip": false, "normalized": false
                },
                {
                    "id": 1, "content": "<eos>", "special": true,
                    "single_word": false, "lstrip": false, "rstrip": false, "normalized": false
                }
            ],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"<bos>": 0, "<eos>": 1, "the": 2, "quick": 3, "fox": 4, "jumps": 5},
                "unk_token": "the"
            }
        })
        .to_string()
        .into_bytes()
    }

    /// Every tensor name/shape here matches `magnetar_runtime::
    /// first_native_expected_tensor_shape`'s HF-stored (pre-transpose) convention
    /// exactly for `llama_config_json`'s dimensions: hidden_size=12,
    /// intermediate_size=24, num_hidden_layers=2, num_attention_heads=3,
    /// num_key_value_heads=1 (head_dim=4, so q_dim=12, kv_dim=4), vocab_size=6,
    /// untied embeddings.
    fn write_tiny_llama_bundle(dir: &Path) {
        std::fs::write(dir.join("config.json"), llama_config_json()).unwrap();
        std::fs::write(dir.join("tokenizer.json"), llama_tokenizer_json()).unwrap();
        let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = vec![
            (
                "model.embed_tokens.weight".to_string(),
                vec![6, 12],
                filled(72, 2.10),
            ),
            ("model.norm.weight".to_string(), vec![12], filled(12, 2.20)),
        ];
        for layer in 0..2u32 {
            let seed_offset = layer as f32 * 0.01;
            tensors.push((
                format!("model.layers.{layer}.input_layernorm.weight"),
                vec![12],
                filled(12, 2.30 + seed_offset),
            ));
            tensors.push((
                format!("model.layers.{layer}.self_attn.q_proj.weight"),
                vec![12, 12],
                filled(144, 2.40 + seed_offset),
            ));
            tensors.push((
                format!("model.layers.{layer}.self_attn.k_proj.weight"),
                vec![4, 12],
                filled(48, 2.50 + seed_offset),
            ));
            tensors.push((
                format!("model.layers.{layer}.self_attn.v_proj.weight"),
                vec![4, 12],
                filled(48, 2.60 + seed_offset),
            ));
            tensors.push((
                format!("model.layers.{layer}.self_attn.o_proj.weight"),
                vec![12, 12],
                filled(144, 2.70 + seed_offset),
            ));
            tensors.push((
                format!("model.layers.{layer}.post_attention_layernorm.weight"),
                vec![12],
                filled(12, 2.80 + seed_offset),
            ));
            tensors.push((
                format!("model.layers.{layer}.mlp.gate_proj.weight"),
                vec![24, 12],
                filled(288, 2.90 + seed_offset),
            ));
            tensors.push((
                format!("model.layers.{layer}.mlp.up_proj.weight"),
                vec![24, 12],
                filled(288, 3.00 + seed_offset),
            ));
            tensors.push((
                format!("model.layers.{layer}.mlp.down_proj.weight"),
                vec![12, 24],
                filled(288, 3.10 + seed_offset),
            ));
        }
        tensors.push(("lm_head.weight".to_string(), vec![6, 12], filled(72, 3.20)));
        let borrowed: Vec<(&str, Vec<u64>, Vec<f32>)> = tensors
            .iter()
            .map(|(name, shape, values)| (name.as_str(), shape.clone(), values.clone()))
            .collect();
        write_safetensors(&dir.join("model.safetensors"), &borrowed);
        write_declared_artifact_format_sidecar(dir, ArtifactFormat::HuggingFace);
    }

    /// astorise/Magnetar#75: writes the declared-format sidecar every real
    /// test bundle needs since `LoadedInferenceComponent::load`/
    /// `local_bundle_manifest_digest` no longer infer the format from
    /// filesystem structure -- a bundle missing this file fails to load
    /// explicitly (see `loaded_inference_component_load_rejects_a_bundle_
    /// missing_the_declared_format_sidecar`).
    fn write_declared_artifact_format_sidecar(dir: &Path, format: ArtifactFormat) {
        std::fs::write(
            dir.join(
                magnetar_runtime::production_model_ingestion::ARTIFACT_FORMAT_SIDECAR_FILE_NAME,
            ),
            format!("artifact_format: {}\n", format.as_str()),
        )
        .unwrap();
    }

    #[test]
    fn loaded_inference_component_load_runs_a_real_huggingface_bundle_end_to_end() {
        let dir = tempfile::tempdir().expect("temp dir creates");
        write_tiny_qwen_bundle(dir.path());

        // The real Model Artifact digest computed from the real bundle just
        // written -- never a hardcoded value -- via this crate's own
        // `local_bundle_manifest_digest`, the same helper a real embedder
        // uses to learn what to trust before calling `load`.
        let model_digest =
            local_bundle_manifest_digest(dir.path()).expect("the bundle inspects cleanly");
        // The real Component artifact digest, computed from the real
        // checked-in Qwen Component bytes -- never `QWEN_REAL_COMPONENT_
        // DIGEST` or any other hardcoded constant (Tachyon integration
        // audit MAG-03: Component trust is independent of Model Artifact
        // trust, and both are named explicitly here).
        let component_digest = magnetar_runtime::ComponentDigest::sha256(qwen_component_bytes());

        let trust_policy = ArtifactTrustPolicy::default()
            .trust_digest(&model_digest)
            .trust_component_digest(&component_digest.value);

        let component = LoadedInferenceComponent::load(
            "test-qwen",
            InferenceComponentArtifact::from_bytes(
                qwen_component_bytes().to_vec(),
                qwen_component_manifest_bytes().to_vec(),
            ),
            InferenceComponentSource::authorized_local_bundle("test-fixture", dir.path()),
            trust_policy,
            InferenceComponentPlacement::ReferenceCpu,
        )
        .expect(
            "a real, completely-specified Hugging Face-shaped bundle must load end to end \
             through this crate's own orchestration",
        );

        let output = component
            .invoke_payload(br#"{"prompt":"hello world","max_new_tokens":1}"#)
            .expect(
                "generation must run end to end through the real load()-produced instance, \
                 the same call shape a real embedder uses",
            );
        assert_eq!(
            output.usage.prompt_tokens, 2,
            "\"hello world\" tokenizes to exactly 2 real tokens under this bundle's own tokenizer"
        );
        assert_eq!(output.usage.generated_tokens, 1);
    }

    /// astorise/Magnetar#88 (Tachyon integration audit TACH-02): the opaque
    /// entry points must be a different *shape* for the same underlying
    /// call, not a parallel implementation that could silently diverge from
    /// the typed one. Loads the same real Component/Artifact pair as the
    /// test above and drives all three call shapes against it with the
    /// identical payload -- the typed batch path (`invoke_payload`), the
    /// opaque batch path (`invoke_payload_opaque`), and the opaque
    /// streaming path (`invoke_payload_streaming_opaque`) -- asserting all
    /// three agree on the generated text and on prompt/generated token
    /// counts.
    #[test]
    fn loaded_inference_component_opaque_entry_points_match_the_typed_path() {
        let dir = tempfile::tempdir().expect("temp dir creates");
        write_tiny_qwen_bundle(dir.path());

        let model_digest =
            local_bundle_manifest_digest(dir.path()).expect("the bundle inspects cleanly");
        let component_digest = magnetar_runtime::ComponentDigest::sha256(qwen_component_bytes());
        let trust_policy = ArtifactTrustPolicy::default()
            .trust_digest(&model_digest)
            .trust_component_digest(&component_digest.value);

        let component = LoadedInferenceComponent::load(
            "test-qwen-opaque",
            InferenceComponentArtifact::from_bytes(
                qwen_component_bytes().to_vec(),
                qwen_component_manifest_bytes().to_vec(),
            ),
            InferenceComponentSource::authorized_local_bundle("test-fixture", dir.path()),
            trust_policy,
            InferenceComponentPlacement::ReferenceCpu,
        )
        .expect("a real, completely-specified Hugging Face-shaped bundle must load end to end");

        let payload: &[u8] = br#"{"prompt":"hello world","max_new_tokens":1}"#;

        let typed = component
            .invoke_payload(payload)
            .expect("typed invocation succeeds");

        let opaque = component
            .invoke_payload_opaque(payload)
            .expect("opaque batch invocation succeeds");
        let opaque_text =
            String::from_utf8(opaque.bytes).expect("opaque batch output is valid UTF-8");
        assert_eq!(
            opaque_text, typed.text,
            "the opaque batch path must produce the identical generated text to the typed path"
        );
        let opaque_tags: std::collections::BTreeMap<_, _> = opaque.tags.into_iter().collect();
        assert_eq!(
            opaque_tags.get("prompt_tokens").map(String::as_str),
            Some(typed.usage.prompt_tokens.to_string()).as_deref()
        );
        assert_eq!(
            opaque_tags.get("generated_tokens").map(String::as_str),
            Some(typed.usage.generated_tokens.to_string()).as_deref()
        );

        let mut streamed_bytes = Vec::new();
        let mut on_frame = |frame: &[u8]| -> std::ops::ControlFlow<()> {
            streamed_bytes.extend_from_slice(frame);
            std::ops::ControlFlow::Continue(())
        };
        let streaming_tags = component
            .invoke_payload_streaming_opaque(payload, &mut on_frame)
            .expect("opaque streaming invocation succeeds");
        let streamed_text =
            String::from_utf8(streamed_bytes).expect("streamed frames are valid UTF-8");
        assert_eq!(
            streamed_text, typed.text,
            "the opaque streaming path's concatenated frames must produce the identical \
             generated text to the typed path"
        );
        let streaming_tags: std::collections::BTreeMap<_, _> = streaming_tags.into_iter().collect();
        assert_eq!(
            streaming_tags.get("prompt_tokens").map(String::as_str),
            Some(typed.usage.prompt_tokens.to_string()).as_deref()
        );
        assert_eq!(
            streaming_tags.get("generated_tokens").map(String::as_str),
            Some(typed.usage.generated_tokens.to_string()).as_deref()
        );
    }

    fn llama_component_bytes() -> &'static [u8] {
        include_bytes!("../../magnetar-runtime/fixtures/components/llama-real.component.wasm")
    }

    fn llama_component_manifest_bytes() -> &'static [u8] {
        include_bytes!(
            "../../magnetar-runtime/fixtures/components/llama-real.component.wasm.magnetar-component.yaml"
        )
    }

    /// Tachyon integration audit MAG-01/MAG-06 (#72) and MAG-02 (#81): the
    /// noted gap left even after `magnetar-runtime`'s own
    /// `build_first_native_graphs_from_named_component_serves_a_real_second_
    /// architecture_family` proved a real, independently-compiled non-Qwen
    /// Component (Llama) produces byte-identical graphs to Qwen for the same
    /// config -- that test never drove this crate's own `LoadedInferenceComponent::
    /// load`, so the audit's demand to "tester une seconde architecture
    /// complète via LoadedInferenceComponent::load" stayed unproven. A first
    /// version of this test (#72) closed the Component half of that gap but
    /// still loaded `write_tiny_qwen_bundle`'s exact Qwen-shaped Model
    /// Artifact -- proving the *Component* registry is multi-architecture,
    /// not that a real independent second *model* stack works. This version
    /// pairs the real checked-in Llama Component artifact with
    /// `write_tiny_llama_bundle`'s own genuinely independent config
    /// (different dimensions, a real GQA head-count reduction none of the
    /// Qwen fixtures exercise), tensors, and tokenizer -- registered and
    /// trusted under its own real digest exactly like the Qwen bundle is in
    /// the test above. `LoadedInferenceComponent` itself never names Qwen
    /// anywhere in this call: which Component drives graph production, and
    /// which Model Artifact is ingested, are both entirely a function of
    /// what the caller supplies.
    #[test]
    fn loaded_inference_component_load_runs_a_real_second_architecture_end_to_end() {
        let dir = tempfile::tempdir().expect("temp dir creates");
        write_tiny_llama_bundle(dir.path());

        let model_digest =
            local_bundle_manifest_digest(dir.path()).expect("the bundle inspects cleanly");
        // The real Llama Component artifact digest, computed from the real
        // checked-in Llama Component bytes -- an independently-compiled
        // binary, structurally distinct from the Qwen Component above
        // (Tachyon integration audit MAG-03: Component trust is independent
        // of Model Artifact trust, and both are named explicitly here).
        let component_digest = magnetar_runtime::ComponentDigest::sha256(llama_component_bytes());

        let trust_policy = ArtifactTrustPolicy::default()
            .trust_digest(&model_digest)
            .trust_component_digest(&component_digest.value);

        let component = LoadedInferenceComponent::load(
            "test-llama",
            InferenceComponentArtifact::from_bytes(
                llama_component_bytes().to_vec(),
                llama_component_manifest_bytes().to_vec(),
            ),
            InferenceComponentSource::authorized_local_bundle("test-fixture", dir.path()),
            trust_policy,
            InferenceComponentPlacement::ReferenceCpu,
        )
        .expect(
            "a real, independently-compiled non-Qwen Component (Llama) paired with its own \
             genuinely independent Model Artifact must load end to end through this crate's \
             own orchestration exactly like the Qwen Component/Artifact pair does",
        );

        let output = component
            .invoke_payload(br#"{"prompt":"the quick fox","max_new_tokens":1}"#)
            .expect(
                "generation must run end to end through the real Llama-Component-driven \
                 instance, the same call shape a real embedder uses",
            );
        assert_eq!(
            output.usage.prompt_tokens, 3,
            "\"the quick fox\" tokenizes to exactly 3 real tokens under this bundle's own tokenizer"
        );
        assert_eq!(output.usage.generated_tokens, 1);
    }

    /// astorise/Magnetar#83 steps 6-7's required negative proof, through
    /// the real `LoadedInferenceComponent::load` entry point end to end --
    /// no hand-built `ModelComponentIdentity` anywhere in this test. Pairs
    /// `write_tiny_llama_bundle`'s real Model Artifact (real ingestion
    /// normalizes its `model_type: "llama"` into `architecture.family:
    /// "llama"`, astorise/Magnetar#83 steps 1-5) with the real, checked-in
    /// Qwen Component -- whose own manifest now declares `compatibility.
    /// architecture_families: [qwen2]` -- both fully trusted (Model
    /// Artifact trust and Component trust are independent, MAG-03, and
    /// both are granted here so the *only* thing that can reject this load
    /// is the family mismatch itself, not an unrelated trust gap). Proves
    /// the Component-vs-family compatibility gate
    /// `ProductionLoadedModel::load_with_component` now enforces is real:
    /// before this fix, no such rejection existed anywhere in this call
    /// chain, and this exact pairing would have loaded successfully.
    #[test]
    fn loaded_inference_component_load_rejects_a_family_mismatched_component() {
        let dir = tempfile::tempdir().expect("temp dir creates");
        write_tiny_llama_bundle(dir.path());

        let model_digest =
            local_bundle_manifest_digest(dir.path()).expect("the bundle inspects cleanly");
        let component_digest = magnetar_runtime::ComponentDigest::sha256(qwen_component_bytes());

        let trust_policy = ArtifactTrustPolicy::default()
            .trust_digest(&model_digest)
            .trust_component_digest(&component_digest.value);

        let error = LoadedInferenceComponent::load(
            "test-mismatch",
            InferenceComponentArtifact::from_bytes(
                qwen_component_bytes().to_vec(),
                qwen_component_manifest_bytes().to_vec(),
            ),
            InferenceComponentSource::authorized_local_bundle("test-fixture", dir.path()),
            trust_policy,
            InferenceComponentPlacement::ReferenceCpu,
        )
        .expect_err(
            "a real Llama Model Artifact paired with a real Qwen Component that declares \
             `compatibility.architecture_families: [qwen2]` must be rejected, even though both \
             the Model Artifact and the Component are independently fully trusted",
        );
        assert!(
            error
                .chain()
                .any(|cause| cause.to_string().contains("architecture unsupported")),
            "expected the rejection to be the architecture-family compatibility gate \
             (`ModelComponentError::ArchitectureUnsupported`), got: {error:#}"
        );
    }

    #[test]
    fn loaded_inference_component_load_rejects_a_bundle_trusted_by_nothing() {
        let dir = tempfile::tempdir().expect("temp dir creates");
        write_tiny_qwen_bundle(dir.path());
        let component_digest = magnetar_runtime::ComponentDigest::sha256(qwen_component_bytes());
        // The Component itself is trusted, but the Model Artifact is not --
        // MAG-03's own independence guarantee, checked from this crate's
        // actual public `load` entry point rather than only at the
        // `magnetar-runtime` layer underneath it.
        let trust_policy =
            ArtifactTrustPolicy::default().trust_component_digest(&component_digest.value);

        let error = LoadedInferenceComponent::load(
            "test-qwen",
            InferenceComponentArtifact::from_bytes(
                qwen_component_bytes().to_vec(),
                qwen_component_manifest_bytes().to_vec(),
            ),
            InferenceComponentSource::authorized_local_bundle("test-fixture", dir.path()),
            trust_policy,
            InferenceComponentPlacement::ReferenceCpu,
        )
        .expect_err(
            "a Model Artifact trusted by nothing must not load, even with a trusted Component",
        );
        assert!(error.to_string().contains("Model Artifact trust rejected"));
    }

    /// astorise/Magnetar#75's required negative proof for the missing-
    /// declaration case: a bundle that has every real Hugging Face file
    /// (`config.json`, `tokenizer.json`, `model.safetensors`) but no
    /// `magnetar-artifact-format.yaml` sidecar must be rejected explicitly,
    /// never silently ingested by guessing from filesystem structure the
    /// way `LoadedInferenceComponent::load` used to. Proves there is no
    /// remaining fallback path inside `load` itself.
    #[test]
    fn loaded_inference_component_load_rejects_a_bundle_missing_the_declared_format_sidecar() {
        let dir = tempfile::tempdir().expect("temp dir creates");
        write_tiny_qwen_bundle(dir.path());
        std::fs::remove_file(
            dir.path().join(
                magnetar_runtime::production_model_ingestion::ARTIFACT_FORMAT_SIDECAR_FILE_NAME,
            ),
        )
        .expect("the sidecar written by write_tiny_qwen_bundle exists and can be removed");

        let error = local_bundle_manifest_digest(dir.path()).expect_err(
            "a bundle with no declared-format sidecar must fail explicitly, not fall back to \
             filesystem-structure guessing",
        );
        assert!(
            error.chain().any(|cause| cause.to_string().contains(
                magnetar_runtime::production_model_ingestion::ARTIFACT_FORMAT_SIDECAR_FILE_NAME
            )),
            "expected the rejection to name the missing sidecar file, got: {error:#}"
        );
    }

    /// astorise/Magnetar#75's legacy-migration proof: an old-style bundle
    /// with no declared-format sidecar is never auto-migrated by `load`
    /// itself (proven above), but `migrate_legacy_bundle_artifact_format`
    /// derives the same format the old filesystem heuristic always did and
    /// writes the sidecar once, after which the bundle loads through the
    /// normal explicit-declaration path -- and a second migration attempt
    /// is refused rather than silently overwriting it.
    #[test]
    fn migrate_legacy_bundle_artifact_format_derives_and_writes_the_sidecar_once() {
        let dir = tempfile::tempdir().expect("temp dir creates");
        write_tiny_qwen_bundle(dir.path());
        let sidecar_path = dir
            .path()
            .join(magnetar_runtime::production_model_ingestion::ARTIFACT_FORMAT_SIDECAR_FILE_NAME);
        std::fs::remove_file(&sidecar_path)
            .expect("the sidecar written by write_tiny_qwen_bundle exists and can be removed");

        let format = migrate_legacy_bundle_artifact_format(dir.path()).expect(
            "migration must derive a format for a legacy Hugging Face-shaped bundle (no \
             `model.gguf` file present)",
        );
        assert_eq!(format, ArtifactFormat::HuggingFace);
        let sidecar_contents =
            std::fs::read_to_string(&sidecar_path).expect("migration wrote the sidecar file");
        assert_eq!(sidecar_contents, "artifact_format: huggingface\n");

        // The now-migrated bundle loads cleanly through the normal
        // explicit-declaration path, exactly like a bundle that was always
        // written with the sidecar present.
        local_bundle_manifest_digest(dir.path())
            .expect("the migrated bundle now inspects cleanly through the declared format");

        let second_attempt = migrate_legacy_bundle_artifact_format(dir.path()).expect_err(
            "migration must refuse to run again once a declared-format sidecar already exists, \
             rather than silently overwriting an operator's own explicit declaration",
        );
        assert!(
            second_attempt
                .to_string()
                .contains("declared-format sidecar already exists"),
        );
    }
}
