use anyhow::{Context, Result, anyhow, bail};
use magnetar_loader_huggingface::{
    HuggingFaceChatTemplateFormatter, HuggingFaceIngestor, parse_tokenizer_config,
};
use magnetar_provider_cpu::ReferenceCpuProvider;
use magnetar_runtime::model::{ModelTrustStatus, ModelTrustStore};
use magnetar_runtime::production_model_ingestion::{
    ProductionModelArtifactIngestor, ProductionModelSource,
};
use magnetar_runtime::tokenizer::Tokenizer;
use magnetar_runtime::{
    ChatMessage, ComponentTrustStore, GenerationParameters, GenerationStreamEvent,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderDeviceClass {
    ReferenceCpu,
    Cuda,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComponentProviderAdvertisement {
    pub provider_name: String,
    pub provider_version: String,
    pub device_ids: Vec<String>,
    pub device_class: ProviderDeviceClass,
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

pub struct LoadedInferenceComponent {
    name: String,
    root: PathBuf,
    chat_formatter: Option<Arc<HuggingFaceChatTemplateFormatter>>,
    provider: ComponentProviderAdvertisement,
    loaded_model: Mutex<magnetar_runtime::ProductionQwenLoadedModel>,
}

pub fn local_bundle_manifest_digest(root: impl Into<PathBuf>) -> Result<String> {
    let root = root.into();
    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::Tachyon("tachyon:test-fixture".to_owned()),
        root.clone(),
    );
    let ingested = HuggingFaceIngestor::new()
        .ingest(&source)
        .with_context(|| {
            format!(
                "Magnetar failed to inspect inference Component bundle at `{}`",
                root.display()
            )
        })?;
    Ok(ingested.manifest.id.digest.value)
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
        })?;

        let production_source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::Tachyon(source.provenance.clone()),
            source.root.clone(),
        );
        let ingested = HuggingFaceIngestor::new()
            .ingest(&production_source)
            .with_context(|| {
                format!(
                    "Magnetar failed to ingest inference Component bundle at `{}`",
                    source.root.display()
                )
            })?;

        let tokenizer_path = source.root.join("tokenizer.json");
        let tokenizer_bytes = std::fs::read(&tokenizer_path)
            .with_context(|| format!("failed to read `{}`", tokenizer_path.display()))?;
        let tokenizer_config_path = source.root.join("tokenizer_config.json");
        let tokenizer_config_bytes =
            if tokenizer_config_path.is_file() {
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
                format!("Magnetar failed to parse tokenizer configuration for Component `{name}`")
            })?;
        let chat_formatter = tokenizer_config_metadata
            .as_ref()
            .and_then(|metadata| metadata.chat_template_reference.as_deref())
            .map(HuggingFaceChatTemplateFormatter::new)
            .transpose()
            .with_context(|| {
                format!("Magnetar failed to load chat template for Component `{name}`")
            })?
            .map(Arc::new);
        let real_tokenizer = magnetar_loader_huggingface::HuggingFaceTokenizer::from_bytes(
            &tokenizer_bytes,
            tokenizer_config_metadata.as_ref(),
            format!("{name}-tokenizer"),
            ingested
                .manifest
                .architecture_config
                .as_ref()
                .map(|config| config.vocab_size),
        )
        .with_context(|| format!("Magnetar failed to load tokenizer for Component `{name}`"))?;
        let tokenizer_metadata = real_tokenizer.metadata().clone();
        let real_tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(real_tokenizer);

        let trust_decision = trust_policy
            .model_trust_store()
            .evaluate(&ingested.manifest);
        if trust_decision.status() != ModelTrustStatus::Trusted {
            bail!(
                "Magnetar Model Artifact trust rejected for `{name}`: {}",
                trust_decision.reason()
            );
        }
        let fixture = magnetar_runtime::production_qwen_fixture(
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
        // (`production_qwen_loaded_model_load_with_component_matches_the_
        // singleton_path`).
        let loaded_model = magnetar_runtime::ProductionQwenLoadedModel::load_with_component(
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

    pub fn provider(&self) -> &ComponentProviderAdvertisement {
        &self.provider
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

    fn chat_formatter(&self) -> Option<&dyn magnetar_runtime::ChatTemplateFormatter> {
        self.chat_formatter
            .as_deref()
            .map(|formatter| formatter as &dyn magnetar_runtime::ChatTemplateFormatter)
    }
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
