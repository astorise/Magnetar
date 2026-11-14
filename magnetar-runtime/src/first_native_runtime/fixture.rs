//! The deterministic first-native fixture model, its weights, and fixture Runtimes.

use super::*;

pub const E2E_SUITE_VERSION: &str = "0.1.0";

pub const E2E_FIXTURE_VERSION: &str = "0.1.0";

pub const E2E_FIXTURE_WEIGHT_DIGEST_VERSION: &str = "e2e-qwen-fixture-weights-v1";

pub const E2E_FIXTURE_WEIGHT_DIGEST: &str =
    "sha256:8faec419da193fed6cc8f83e1cb8a3c1122acf0e577110a29b5c6422ed60aab1";

pub(super) const E2E_FIXTURE_EOS_TOKEN: TokenId = 0;

pub(super) const E2E_FIXTURE_BOS_TOKEN: TokenId = 257;

pub(super) const E2E_FIXTURE_VOCAB: u64 = 258;

pub(super) const E2E_FIXTURE_HIDDEN: u64 = 4;

pub(super) const E2E_FIXTURE_HEADS: u64 = 2;

pub(super) const E2E_FIXTURE_HEAD_DIM: u64 = 2;

pub(super) const E2E_FIXTURE_INTERMEDIATE: u64 = 8;

pub(super) const E2E_FIXTURE_CONTEXT: u64 = 32;

pub(super) const E2E_FIXTURE_LAYERS: u64 = 1;

// Task 11.5: the checksum-fixture Component (and everything below tied to
// it -- QwenComponentPreflight and friends) is no longer part of the
// production graph-production path, only of tests exercising Component-
// loading conformance generically. `#[cfg(test)]` reflects that; without it
// these items are genuinely dead code in a non-test build.

// ---------------------------------------------------------------------
// Error model
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// Fixture model
// ---------------------------------------------------------------------

/// A minimal, deterministic Qwen-like decoder-only fixture: config,
/// identity, architecture implementation, a Model Artifact manifest that
/// passes normal Model Artifact and Qwen baseline validation, a Tokenizer
/// Contract fixture, and deterministic weight tensors keyed by the same
/// canonical logical tensor names Model Loading expects.
#[derive(Clone)]
pub struct E2eFixture {
    pub config: FirstNativeModelConfig,
    pub identity: ModelComponentIdentity,
    pub architecture_implementation: ModelArchitectureImplementation,
    pub manifest: ModelManifest,
    pub tokenizer: FixtureTokenizer,
    pub weights: BTreeMap<String, HostTensor>,
}

pub fn e2e_fixture_config() -> FirstNativeModelConfig {
    let architecture = first_native_architecture_metadata(
        E2E_FIXTURE_HIDDEN,
        E2E_FIXTURE_LAYERS,
        E2E_FIXTURE_HEADS,
        E2E_FIXTURE_HEADS,
        E2E_FIXTURE_HEAD_DIM,
        E2E_FIXTURE_INTERMEDIATE,
        E2E_FIXTURE_VOCAB,
        E2E_FIXTURE_CONTEXT,
    );
    let mut config = FirstNativeModelConfig::new(
        architecture,
        FirstNativeRopeConfig::standard(E2E_FIXTURE_HEAD_DIM),
    );
    config.tied_embeddings = true;
    config
}

pub fn e2e_fixture_identity() -> ModelComponentIdentity {
    first_native_component_identity(
        ModelComponentId::new("e2e-qwen-fixture").expect("static id is valid"),
        ModelComponentVersion::new(1, 0, 0),
        ModelComponentImplementationKind::WebAssemblyComponent,
    )
}

pub fn e2e_fixture_tokenizer() -> Result<FixtureTokenizer, E2eConformanceError> {
    let metadata = TokenizerMetadata {
        id: TokenizerId::new("e2e-fixture-tokenizer")?,
        artifact: TokenizerArtifactId::new("e2e-fixture-tokenizer-artifact")?,
        digest: ModelDigest::parse(
            "sha256:0000000000000000000000000000000000000000000000000000000000000002",
        )
        .map_err(|error| E2eConformanceError::FixtureInvalid {
            reason: error.to_string(),
        })?,
        family: TokenizerFamily::new("byte-fixture")?,
        revision: TokenizerRevision::new("r1")?,
        vocabulary_size: E2E_FIXTURE_VOCAB as u32,
        added_token_count: 2,
        token_id_range: TokenIdRange::new(0, E2E_FIXTURE_BOS_TOKEN),
        model_max_length: Some(E2E_FIXTURE_CONTEXT as u32),
        special_tokens: vec![
            SpecialToken::new(SpecialTokenKind::Eos, "<eos>", E2E_FIXTURE_EOS_TOKEN),
            SpecialToken::new(SpecialTokenKind::Bos, "<bos>", E2E_FIXTURE_BOS_TOKEN),
        ],
        additional_special_tokens: Vec::new(),
        byte_fallback: true,
        normalization: None,
        pre_tokenizer: None,
        supports_offsets: true,
        supports_token_type_ids: false,
        supports_browser: true,
    };
    metadata.validate()?;
    Ok(FixtureTokenizer::new(metadata))
}

pub fn e2e_fixture_manifest(
    config: &FirstNativeModelConfig,
    architecture: &ModelArchitecture,
) -> Result<ModelManifest, E2eConformanceError> {
    const DIGEST: &str = "sha256:0000000000000000000000000000000000000000000000000000000000000003";
    let weights_digest = e2e_fixture_safetensors_digest();
    let weights_size_bytes = E2E_FIXTURE_SAFETENSORS_BYTES.len();
    // Real per-tensor content digests, computed from the checked-in
    // Safetensors file's actual bytes -- not the synthetic in-memory
    // `e2e_fixture_weights` -- so this manifest genuinely constrains what
    // counts as each tensor's content
    // (`bind-materialized-weight-content-to-model-artifact-digests`).
    let inventory = e2e_fixture_weight_inventory_with_digests(config)?;
    let mut tensor_yaml = String::new();
    for tensor in &inventory {
        let shape_text = tensor
            .shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let digest = tensor
            .digest
            .as_ref()
            .ok_or_else(|| E2eConformanceError::FixtureInvalid {
                reason: format!(
                    "fixture tensor '{}' unexpectedly has no computed content digest",
                    tensor.name
                ),
            })?;
        tensor_yaml.push_str(&format!(
            "  - name: {name}\n    shape: [{shape_text}]\n    storage_dtype: f32\n    digest: {digest}\n",
            name = tensor.name,
            digest = digest.value,
        ));
    }
    let yaml = format!(
        r#"
schema: magnetar-model-artifact
schema_version: 1
kind: model-bundle
digest: {DIGEST}
model:
  name: e2e-fixture-model
  revision: r1
architecture:
  family: {family}
  identifier: {identifier}
tokenizer: tokenizer
storage_dtype: f32
compute_dtype: f32
supported_compute_dtypes: [f32]
generation:
  temperature: 0.0
  max_tokens: 4
artifacts:
  weights:
    kind: model-weights
    digest: {weights_digest}
    size_bytes: {weights_size_bytes}
  config:
    kind: model-config
    digest: {DIGEST}
    size_bytes: 16
  tokenizer:
    kind: tokenizer
    digest: {DIGEST}
    size_bytes: 8
tensors:
{tensor_yaml}"#,
        family = architecture.family,
        identifier = architecture.identifier,
    );
    ModelManifest::from_yaml_str(&yaml).map_err(E2eConformanceError::from)
}

/// [`e2e_fixture_manifest`], generalized to compute every tensor's content
/// digest from a caller-supplied `weights` map instead of the checked-in
/// `E2E_FIXTURE_SAFETENSORS_BYTES` -- so a caller building a fixture for a
/// `FirstNativeModelConfig` other than the one canonical E2E fixture (for example a
/// genuinely grouped-query-shaped configuration) gets a manifest whose
/// declared digests actually match the weights it will go on to
/// materialize, instead of either failing the digest mismatch check in
/// `WeightMaterializationTransaction::stage_weight` or (worse) silently
/// validating the wrong content. Structurally identical to
/// `e2e_fixture_manifest` otherwise.
pub fn e2e_fixture_manifest_from_weights(
    config: &FirstNativeModelConfig,
    architecture: &ModelArchitecture,
    weights: &BTreeMap<String, HostTensor>,
) -> Result<ModelManifest, E2eConformanceError> {
    const DIGEST: &str = "sha256:0000000000000000000000000000000000000000000000000000000000000003";
    let weights_digest = e2e_fixture_weight_digest(weights);
    let weights_size_bytes: usize = weights
        .values()
        .map(|tensor| tensor.content_bytes().len())
        .sum();
    let inventory = e2e_fixture_weight_inventory(config)?;
    let mut tensor_yaml = String::new();
    for mut tensor in inventory {
        let content =
            weights
                .get(&tensor.name)
                .ok_or_else(|| E2eConformanceError::FixtureInvalid {
                    reason: format!("no supplied weight for fixture tensor '{}'", tensor.name),
                })?;
        tensor.digest = Some(ModelDigest::sha256(&content.content_bytes()));
        let shape_text = tensor
            .shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let digest = tensor.digest.as_ref().expect("just set above");
        tensor_yaml.push_str(&format!(
            "  - name: {name}\n    shape: [{shape_text}]\n    storage_dtype: f32\n    digest: {digest}\n",
            name = tensor.name,
            digest = digest.value,
        ));
    }
    let yaml = format!(
        r#"
schema: magnetar-model-artifact
schema_version: 1
kind: model-bundle
digest: {DIGEST}
model:
  name: e2e-fixture-model
  revision: r1
architecture:
  family: {family}
  identifier: {identifier}
tokenizer: tokenizer
storage_dtype: f32
compute_dtype: f32
supported_compute_dtypes: [f32]
generation:
  temperature: 0.0
  max_tokens: 4
artifacts:
  weights:
    kind: model-weights
    digest: {weights_digest}
    size_bytes: {weights_size_bytes}
  config:
    kind: model-config
    digest: {DIGEST}
    size_bytes: 16
  tokenizer:
    kind: tokenizer
    digest: {DIGEST}
    size_bytes: 8
tensors:
{tensor_yaml}"#,
        family = architecture.family,
        identifier = architecture.identifier,
    );
    ModelManifest::from_yaml_str(&yaml).map_err(E2eConformanceError::from)
}

/// Deterministic pseudo-random value in `[-0.5, 0.5]`, purely a function of
/// `seed` -- no RNG dependency, no shared mutable state, fully reproducible.
pub(super) fn fixture_value(seed: u64) -> f32 {
    let mut x = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x2545_F491_4F6C_DD1D);
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    ((x % 1000) as f32 / 1000.0) - 0.5
}

pub(super) fn fixture_seed(name: &str, offset: u64) -> u64 {
    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    for byte in name.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash.wrapping_add(offset)
}

pub(super) fn hex_digest(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>()
}

pub(super) fn fixture_tensor(name: &str, shape: &[u64]) -> Result<HostTensor, E2eConformanceError> {
    let len = shape.iter().product::<u64>();
    let is_norm_weight = shape.len() == 1;
    let data: Vec<f32> = (0..len)
        .map(|i| {
            let value = fixture_value(fixture_seed(name, i));
            if is_norm_weight {
                1.0 + value * 0.1
            } else {
                value
            }
        })
        .collect();
    HostTensor::new(shape.to_vec(), data).map_err(E2eConformanceError::from)
}

pub fn e2e_fixture_weight_digest(weights: &BTreeMap<String, HostTensor>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(E2E_FIXTURE_WEIGHT_DIGEST_VERSION.as_bytes());
    for (name, tensor) in weights {
        hasher.update(name.as_bytes());
        hasher.update([0]);
        for dimension in &tensor.shape {
            hasher.update(dimension.to_le_bytes());
        }
        hasher.update([0xff]);
        for value in &tensor.data {
            hasher.update(value.to_bits().to_le_bytes());
        }
        hasher.update([0xfe]);
    }
    let digest = hasher.finalize();
    format!("sha256:{}", hex_digest(&digest))
}

pub fn e2e_fixture_weights(
    config: &FirstNativeModelConfig,
) -> Result<BTreeMap<String, HostTensor>, E2eConformanceError> {
    let mut weights = BTreeMap::new();
    for name in
        first_native_expected_tensor_names(config.architecture.layer_count, config.tied_embeddings)
    {
        let shape = first_native_expected_tensor_shape(&name, config).ok_or_else(|| {
            E2eConformanceError::FixtureInvalid {
                reason: format!("no expected shape for fixture tensor '{name}'"),
            }
        })?;
        weights.insert(name.clone(), fixture_tensor(&name, &shape)?);
    }
    // `mlp.gate_up_proj` is defined as the horizontal concatenation of
    // `mlp.gate_proj`/`mlp.up_proj` -- not an independently-seeded fixture
    // tensor -- so that a fused `matmul` against it followed by `split`
    // (the Rust test-oracle graph's recipe, `qwen_build_graph`) is
    // mathematically identical to the two separate `matmul`s the hand-
    // written oracle forward passes below still compute
    // (`check_graph_executor_matches_full_sequence_oracle` and friends):
    // `x @ [gate_proj | up_proj] == [x @ gate_proj | x @ up_proj]`. An
    // independently-seeded `gate_up_proj` would make those two genuinely
    // different numeric results, which is exactly what an earlier attempt
    // at this fusion produced (`define-provider-prepared-kernel-execution-
    // contract` task group 3): the oracles diverged from the graph
    // executor's decode logits, not because either was wrong, but because
    // the "fused" weight was not actually the same math.
    for layer in 0..config.architecture.layer_count {
        let gate_name = format!("layers.{layer}.mlp.gate_proj");
        let up_name = format!("layers.{layer}.mlp.up_proj");
        let gate = weights
            .get(&gate_name)
            .ok_or_else(|| E2eConformanceError::FixtureInvalid {
                reason: format!("fixture is missing weight tensor '{gate_name}'"),
            })?;
        let up = weights
            .get(&up_name)
            .ok_or_else(|| E2eConformanceError::FixtureInvalid {
                reason: format!("fixture is missing weight tensor '{up_name}'"),
            })?;
        let [rows, gate_cols] = gate.shape.as_slice() else {
            return Err(E2eConformanceError::FixtureInvalid {
                reason: format!("'{gate_name}' is not rank-2: {:?}", gate.shape),
            });
        };
        let [up_rows, up_cols] = up.shape.as_slice() else {
            return Err(E2eConformanceError::FixtureInvalid {
                reason: format!("'{up_name}' is not rank-2: {:?}", up.shape),
            });
        };
        if rows != up_rows {
            return Err(E2eConformanceError::FixtureInvalid {
                reason: format!(
                    "'{gate_name}' and '{up_name}' row counts disagree: {rows} vs {up_rows}"
                ),
            });
        }
        let rows = *rows as usize;
        let gate_cols = *gate_cols as usize;
        let up_cols = *up_cols as usize;
        let mut data = Vec::with_capacity(rows * (gate_cols + up_cols));
        for row in 0..rows {
            data.extend_from_slice(&gate.data[row * gate_cols..(row + 1) * gate_cols]);
            data.extend_from_slice(&up.data[row * up_cols..(row + 1) * up_cols]);
        }
        let gate_up = HostTensor::new([rows as u64, (gate_cols + up_cols) as u64], data)?;
        weights.insert(format!("layers.{layer}.mlp.gate_up_proj"), gate_up);
    }
    Ok(weights)
}

/// Raw bytes of a real, checked-in Safetensors file encoding the exact same
/// deterministic weights [`e2e_fixture_weights`] builds in memory --
/// generated once via `formats/safetensors`'s own test suite (that crate,
/// not this one, is allowed to depend on both `magnetar-runtime` and a
/// Safetensors writer; see `materialize-weights-from-real-model-artifact`'s
/// design.md for why), committed here, and read by [`e2e_fixture_weight_inventory`]/
/// [`host_tensors_from_artifact_bytes`] as the real byte-level Model
/// Artifact the production first-native path actually materializes weights
/// from -- not merely an in-memory recreation.
pub const E2E_FIXTURE_SAFETENSORS_BYTES: &[u8] =
    include_bytes!("../../fixtures/e2e-fixture-weights.safetensors");

/// Size, in bytes, of the little-endian header-length prefix every
/// Safetensors file begins with -- a local copy of the same constant
/// `formats/safetensors::HEADER_LENGTH_PREFIX_BYTES` defines, since
/// `magnetar-runtime` cannot depend on that crate even to reuse a `usize`
/// (`externalize-runtime-extension-modules`).
pub(super) const HEADER_LENGTH_PREFIX_BYTES: usize = 8;

/// The real sha256 digest of [`E2E_FIXTURE_SAFETENSORS_BYTES`]'s actual
/// bytes -- computed at call time (not hardcoded) so it can never drift
/// from the checked-in file it describes; used by [`e2e_fixture_manifest`]'s
/// `artifacts.weights.digest` field, checked the same way
/// `bind_qwen_fixture_weights`'s existing digest gate already checks every
/// other declared digest.
pub fn e2e_fixture_safetensors_digest() -> String {
    let mut hasher = Sha256::new();
    hasher.update(E2E_FIXTURE_SAFETENSORS_BYTES);
    format!("sha256:{}", hex_digest(&hasher.finalize()))
}

/// The generic `ModelTensorMetadata` inventory [`E2E_FIXTURE_SAFETENSORS_BYTES`]'s
/// real, checked-in file actually contains, computed directly rather than by
/// calling a format parser: `magnetar-runtime` cannot depend on
/// `formats/safetensors` even at test time (`externalize-runtime-extension-modules`),
/// but this fixture's tensor names/shapes/generation order are already
/// fully known here (`first_native_expected_tensor_names`/`first_native_expected_tensor_shape`),
/// and the file was written by iterating that exact same sorted order (a
/// `BTreeSet`/`BTreeMap`'s iteration order, deterministic regardless of
/// insertion order) with each tensor's byte length equal to its element
/// count times 4 -- so the offsets below are provably the same offsets a
/// real parse of this file would produce, not merely assumed to match.
/// `formats/safetensors`'s own test suite is what actually proves the file
/// is parseable by the real, independent parser (see that crate's
/// `e2e_fixture_weights_round_trip_through_real_safetensors_bytes` test).
pub fn e2e_fixture_weight_inventory(
    config: &FirstNativeModelConfig,
) -> Result<Vec<ModelTensorMetadata>, E2eConformanceError> {
    let mut tensors = Vec::new();
    let mut offset = 0_u64;
    for name in
        first_native_expected_tensor_names(config.architecture.layer_count, config.tied_embeddings)
    {
        let shape = first_native_expected_tensor_shape(&name, config).ok_or_else(|| {
            E2eConformanceError::FixtureInvalid {
                reason: format!("no expected shape for fixture tensor '{name}'"),
            }
        })?;
        let element_count: u64 = shape.iter().product();
        let size_bytes = element_count * 4;
        tensors.push(ModelTensorMetadata {
            name: name.clone(),
            shape,
            storage_dtype: ModelDType::F32,
            layout: None,
            shard: None,
            offset_bytes: Some(offset),
            size_bytes: Some(size_bytes),
            quantization: None,
            expected_compute_dtype: None,
            digest: None,
        });
        offset += size_bytes;
    }
    Ok(tensors)
}

/// [`e2e_fixture_weight_inventory`], with each tensor's `digest` populated
/// from the real, checked-in Safetensors file's actual bytes -- computed
/// from [`e2e_fixture_weights_from_real_artifact`]'s already-real,
/// already-parsed [`HostTensor`]s (not from [`e2e_fixture_weights`]'s
/// synthetic in-memory values), so this is a real, non-circular content
/// binding: the fixture is not hashing its own assumptions about itself.
/// Implements `bind-materialized-weight-content-to-model-artifact-digests`'s
/// "Tensor Content Digest Binding" requirement for the one real artifact
/// source this crate has today.
pub fn e2e_fixture_weight_inventory_with_digests(
    config: &FirstNativeModelConfig,
) -> Result<Vec<ModelTensorMetadata>, E2eConformanceError> {
    let inventory = e2e_fixture_weight_inventory(config)?;
    let real_weights = e2e_fixture_weights_from_real_artifact(config)?;
    Ok(inventory
        .into_iter()
        .map(|mut metadata| {
            if let Some(tensor) = real_weights.get(&metadata.name) {
                metadata.digest = Some(ModelDigest::sha256(&tensor.content_bytes()));
            }
            metadata
        })
        .collect())
}

/// Materializes the E2E fixture's weights by reading
/// [`E2E_FIXTURE_SAFETENSORS_BYTES`]'s real bytes at
/// [`e2e_fixture_weight_inventory`]'s declared offsets -- the real-file
/// counterpart to [`e2e_fixture_weights`]'s in-memory construction, proven
/// equal to it by `tests::e2e_fixture_real_artifact_weights_match_in_memory_weights`.
pub fn e2e_fixture_weights_from_real_artifact(
    config: &FirstNativeModelConfig,
) -> Result<BTreeMap<String, HostTensor>, E2eConformanceError> {
    let inventory = e2e_fixture_weight_inventory(config)?;
    // The Safetensors envelope's own 8-byte little-endian header-length
    // prefix -- reading it is not "parsing" in the sense
    // `externalize-runtime-extension-modules` forbids (no JSON, no
    // validation, no `formats/safetensors` call): it is the one universally
    // fixed part of the container format, needed here only to locate where
    // the tensor-data section starts within the file
    // (`host_tensors_from_artifact_bytes`'s `data_section_start`).
    let header_length = E2E_FIXTURE_SAFETENSORS_BYTES
        .get(..HEADER_LENGTH_PREFIX_BYTES)
        .and_then(|prefix| prefix.try_into().ok())
        .map(u64::from_le_bytes)
        .ok_or_else(|| E2eConformanceError::FixtureInvalid {
            reason: "real artifact bytes are too short to hold a header length prefix".into(),
        })?;
    let data_section_start = HEADER_LENGTH_PREFIX_BYTES as u64 + header_length;
    host_tensors_from_artifact_bytes(
        &inventory,
        E2E_FIXTURE_SAFETENSORS_BYTES,
        data_section_start,
    )
    .map_err(|error| E2eConformanceError::FixtureInvalid {
        reason: format!("real artifact bytes failed to materialize: {error}"),
    })
}

pub fn e2e_fixture() -> Result<E2eFixture, E2eConformanceError> {
    let config = e2e_fixture_config();
    let identity = e2e_fixture_identity();
    config.validate(&identity)?;
    let architecture_implementation = qwen_model_component::qwen_architecture_implementation(
        &identity,
        ModelArchitectureImplementationKind::ComponentBased,
    );
    let manifest = e2e_fixture_manifest(&config, &architecture_implementation.architecture)?;
    let tokenizer = e2e_fixture_tokenizer()?;
    let weights = e2e_fixture_weights(&config)?;

    let descriptor = first_native_component_descriptor(identity.clone(), &config)?;
    validate_first_native_model_artifact(&descriptor, &config, &manifest)?;

    Ok(E2eFixture {
        config,
        identity,
        architecture_implementation,
        manifest,
        tokenizer,
        weights,
    })
}

// ---------------------------------------------------------------------
// Real Reference CPU forward pass (operator coverage + determinism)
// ---------------------------------------------------------------------

/// Every required-now Operator this forward pass exercises through the real
/// Reference CPU kernel functions (not a shortcut, not a placeholder).
pub const E2E_EXERCISED_OPERATORS: [&str; 10] = [
    "embedding",
    "rmsnorm",
    "matmul",
    "rope",
    "attention",
    "softmax",
    "silu",
    "add",
    "mul",
    "residual-add",
];

pub(super) fn fixture_tensor_by_name<'a>(
    weights: &'a BTreeMap<String, HostTensor>,
    name: &str,
) -> Result<&'a HostTensor, E2eConformanceError> {
    weights
        .get(name)
        .ok_or_else(|| E2eConformanceError::FixtureInvalid {
            reason: format!("fixture is missing weight tensor '{name}'"),
        })
}

// ---------------------------------------------------------------------
// Suite runner
// ---------------------------------------------------------------------

/// Outcome of a completed E2E success-path run, kept around so downstream
/// checks (determinism, streaming order, usage accounting) can inspect it.
pub(super) struct E2eRunOutcome {
    pub(super) generation_result: GenerationResult,
    pub(super) observer: InferenceApiObserver,
    pub(super) kv_observations: Vec<KvCacheObservation>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FirstNativeFixtureGeneration {
    pub text: String,
    pub result: GenerationResult,
    pub observer: InferenceApiObserver,
}

pub(super) fn build_runtime() -> Runtime {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .build()
        .expect("Reference CPU provider registers cleanly");
    register_reference_cpu_prepared_kernels(&mut runtime);
    runtime
}

/// Same as [`build_runtime`], but the Runtime's sealed trust configuration
/// (`seal-runtime-model-trust-and-provenance-authority`) trusts `fixture`'s
/// manifest digest -- for the many E2E tests that go on to actually load
/// `fixture` through it via [`load_fixture_instance`] or
/// [`load_fixture_instance_with_weights`]. Tests that only inspect Runtime
/// state without loading anything (kernel registry advertisement checks,
/// for instance) use [`build_runtime`] directly instead, since they have no
/// fixture to trust.
pub(super) fn build_runtime_trusting_fixture(fixture: &E2eFixture) -> Runtime {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .trust_store(
            ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
        )
        .build()
        .expect("Reference CPU provider registers cleanly");
    register_reference_cpu_prepared_kernels(&mut runtime);
    runtime
}

pub(super) fn build_runtime_with_model_execution_engine(fixture: &E2eFixture) -> Runtime {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
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
        .trust_store(
            ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
        )
        .build()
        .expect("Reference CPU provider registers cleanly");
    register_reference_cpu_prepared_kernels(&mut runtime);
    runtime
}

/// [`build_runtime_with_model_execution_engine`], generalized to register
/// an arbitrary Provider instead of hardcoding Reference CPU -- the actual
/// per-node dispatch this Runtime reaches
/// (`E2eRuntimeModelExecutionEngine::execute_generation_step` ->
/// `execute_first_native_graph`) was already Provider-generic, resolving through
/// `resolve_kernel_execution_provider`; this and
/// [`register_prepared_kernels_for_provider`] were the only two
/// Reference-CPU-specific pieces standing in the way of running the real
/// first-native pipeline against a different Provider end to end.
/// Never imports or references any concrete non-Reference-CPU Provider
/// type: `provider` arrives as a `Provider` trait object this crate
/// already depends on.
pub(super) fn build_runtime_with_model_execution_engine_and_provider(
    fixture: &E2eFixture,
    provider: Arc<dyn Provider>,
) -> Result<Runtime, E2eConformanceError> {
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
        .trust_store(
            ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
        )
        .build()
        .map_err(|error| E2eConformanceError::ModelComponentFailed {
            reason: format!("provider registration failed: {error}"),
        })?;
    register_prepared_kernels_for_provider(&mut runtime, &*provider)?;
    Ok(runtime)
}

/// Registers Reference CPU's `PreparedKernel` coverage for every Operator
/// the Qwen graph (Rust-synthesized or real-Component-produced) can emit,
/// so [`prepare_first_native_execution_plans`] finds a `PreparedKernel`
/// for every node instead of failing with `KernelCoverageMissing`. Public
/// alongside the other generation-preparation primitives (task group 12):
/// an embedder building its own `Runtime` for a production-loaded
/// instance needs this same registration step, exactly as
/// `build_runtime_with_model_execution_engine` (this crate's own
/// internal helper) already performs it.
pub fn register_reference_cpu_prepared_kernels(runtime: &mut Runtime) {
    let mut prepared_ids = PreparedKernelIdAllocator::default();
    for advertisement in reference_cpu_kernel_advertisements() {
        let id = prepared_ids.allocate();
        let mut prepared = PreparedKernel::new(
            id,
            advertisement.id.clone(),
            CompiledKernelArtifactId::from_digest(format!(
                "builtin:{}",
                advertisement.id.stable_key()
            )),
            ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
            DeviceBinding::new(DeviceId::new(REFERENCE_CPU_DEVICE_ID)),
            PreparedKernelGeneration::new(1),
        );
        prepared
            .mark_ready()
            .expect("Reference CPU prepared kernel fixture can become ready");
        runtime
            .kernel_registry_mut()
            .register_prepared_kernel(prepared);
        runtime
            .kernel_registry_mut()
            .promote_generation(&advertisement.id, id)
            .expect("Reference CPU prepared kernel fixture promotes cleanly");
    }
}

/// Loads the E2E fixture model and creates its Model Instance, then binds
/// its weight tensors to Runtime resources through that instance
/// (`bind_qwen_fixture_weights`). The returned `MemoryManager` is a
/// placeholder every caller discards (`_memory`): loading now allocates
/// through `runtime.memory_mut()` directly (task 6.2), the same Runtime-
/// owned ledger dispatch itself accounts through (see
/// `execute_first_native_graph`'s doc comment in section 5), rather than a
/// throwaway that is dropped along with any admission/accounting history it
/// recorded.
pub(super) fn load_fixture_instance(
    fixture: &E2eFixture,
    runtime: &mut Runtime,
) -> Result<(ModelInstanceId, MemoryManager), E2eConformanceError> {
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(fixture.architecture_implementation.clone());
    let mut request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new("e2e-fixture-load"),
        fixture.manifest.id.clone(),
    );
    request.quantization_policy = ModelQuantizationPolicy::RejectUnsupported;
    let loaded = load_model(
        &mut coordinator,
        runtime,
        ModelLoadingApiRequest::new(request),
        &fixture.manifest,
    )?;
    let instance = create_model_instance(
        runtime,
        &loaded,
        fixture.architecture_implementation.clone(),
        ResourceAffinity::new(FallbackClass::Transparent),
    )?;
    bind_qwen_fixture_weights(runtime, &instance, fixture)?;
    Ok((instance, MemoryManager::default()))
}

/// [`load_fixture_instance`], generalized to pin the Model Instance's own
/// placement `ResourceAffinity` to `provider` explicitly, instead of the
/// empty (no-provider) affinity `load_fixture_instance` always passes --
/// `audit-complet-cuda-hot-path-2026-09-08` P1-1: without an explicit
/// provider here, `WeightMaterializationTransaction::begin` (see its own
/// doc comment) falls back to Reference CPU whenever a Model Instance's
/// placement leaves no Provider bound, which is *every* Model Instance
/// `load_fixture_instance` itself creates -- so weight materialization for
/// an arbitrary Provider needs this explicit pin to actually reach that
/// Provider's own storage instead of silently reusing Reference CPU's.
/// Never imports or references any concrete non-Reference-CPU Provider
/// type: `provider_binding` is a plain name.
pub(super) fn load_fixture_instance_for_provider(
    fixture: &E2eFixture,
    runtime: &mut Runtime,
    provider_binding: &ProviderBinding,
) -> Result<(ModelInstanceId, MemoryManager), E2eConformanceError> {
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(fixture.architecture_implementation.clone());
    let mut request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new("e2e-fixture-load"),
        fixture.manifest.id.clone(),
    );
    request.quantization_policy = ModelQuantizationPolicy::RejectUnsupported;
    let loaded = load_model(
        &mut coordinator,
        runtime,
        ModelLoadingApiRequest::new(request),
        &fixture.manifest,
    )?;
    let instance = create_model_instance(
        runtime,
        &loaded,
        fixture.architecture_implementation.clone(),
        ResourceAffinity::new(FallbackClass::Transparent).with_provider(provider_binding.clone()),
    )?;
    bind_qwen_fixture_weights(runtime, &instance, fixture)?;
    Ok((instance, MemoryManager::default()))
}

/// [`load_fixture_instance_for_provider`], generalized further to
/// materialize caller-supplied `weights` directly through
/// `materialize_model_instance_weights` instead of the fixture's own
/// digest-checked `bind_qwen_fixture_weights` -- so a caller can run this
/// pipeline against a `fixture`/`FirstNativeModelConfig` of their own choosing (for
/// example a genuinely grouped-query-shaped configuration,
/// `attention_head_count != kv_head_count`, which the one canonical E2E
/// fixture this crate ships is not) instead of only the one canonical E2E
/// fixture's own weights. Mirrors `load_fixture_instance_with_weights`
/// (the equivalent `#[cfg(test)]`-only helper this crate's own weight-
/// sensitivity tests use), generalized the same way
/// `load_fixture_instance_for_provider` generalizes `load_fixture_instance`.
pub(super) fn load_fixture_instance_with_weights_for_provider(
    fixture: &E2eFixture,
    runtime: &mut Runtime,
    provider_binding: &ProviderBinding,
    weights: &BTreeMap<String, HostTensor>,
) -> Result<ModelInstanceId, E2eConformanceError> {
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(fixture.architecture_implementation.clone());
    let mut request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new("first-native-provider-load-with-weights"),
        fixture.manifest.id.clone(),
    );
    request.quantization_policy = ModelQuantizationPolicy::RejectUnsupported;
    let loaded = load_model(
        &mut coordinator,
        runtime,
        ModelLoadingApiRequest::new(request),
        &fixture.manifest,
    )?;
    let instance = create_model_instance(
        runtime,
        &loaded,
        fixture.architecture_implementation.clone(),
        ResourceAffinity::new(FallbackClass::Transparent).with_provider(provider_binding.clone()),
    )?;
    let mut weights = weights.clone();
    qwen_weights_with_derived_lm_head(fixture, &mut weights)?;
    materialize_model_instance_weights(
        runtime,
        &instance,
        fixture.manifest.id.name.as_str(),
        &weights,
    )
    .map_err(E2eConformanceError::from)?;
    Ok(instance)
}

#[cfg(test)]
mod tests;
