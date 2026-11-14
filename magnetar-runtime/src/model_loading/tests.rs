//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::adapter::*;
use crate::affinity::*;
use crate::e2e_conformance::*;
use crate::inference_api::*;
use crate::memory::*;
use crate::model::{ModelDType, ModelDigest, ModelTensorMetadata};
use crate::model_instance::*;
use crate::operator::*;
use crate::reference_cpu::*;
use crate::runtime::*;
use crate::tensor::*;
use crate::tests::*;
use crate::tokenizer::*;
use std::collections::BTreeMap;

use crate::memory::MemoryManager;
use crate::model::{
    ModelArchitecture, ModelManifest, ModelQuantizationFormat, ModelTrustDecision, ModelTrustStore,
};
fn f16_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn artifact_bytes_test_tensor(
    name: &str,
    shape: Vec<u64>,
    data: &[f32],
) -> (ModelTensorMetadata, Vec<u8>) {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let metadata = ModelTensorMetadata {
        name: name.to_string(),
        shape,
        storage_dtype: ModelDType::F32,
        layout: None,
        shard: None,
        offset_bytes: Some(0),
        size_bytes: Some(bytes.len() as u64),
        quantization: None,
        expected_compute_dtype: None,
        digest: None,
    };
    (metadata, bytes)
}

fn f16_tensor_metadata(name: &str, shape: Vec<u64>, byte_len: usize) -> ModelTensorMetadata {
    ModelTensorMetadata {
        name: name.to_string(),
        shape,
        storage_dtype: ModelDType::F16,
        layout: None,
        shard: None,
        offset_bytes: Some(0),
        size_bytes: Some(byte_len as u64),
        quantization: None,
        expected_compute_dtype: None,
        digest: None,
    }
}

#[test]
fn host_tensors_from_artifact_bytes_reads_a_well_formed_tensor() {
    let (mut metadata, tensor_bytes) =
        artifact_bytes_test_tensor("weight.a", vec![2, 2], &[1.0, 2.0, 3.0, 4.0]);
    // Simulate a second tensor's bytes preceding this one in a real file by
    // offsetting into a larger buffer, rather than only ever testing offset
    // zero.
    let mut file = vec![0u8; 16];
    file.extend_from_slice(&tensor_bytes);
    metadata.offset_bytes = Some(16);

    let weights = host_tensors_from_artifact_bytes(std::slice::from_ref(&metadata), &file, 0)
        .expect("well-formed tensor materializes");
    let tensor = weights.get("weight.a").expect("tensor present");
    assert_eq!(tensor.shape, vec![2, 2]);
    assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn f32_to_bf16_handles_named_numeric_classes() {
    assert_eq!(f32_to_bf16(0.0), 0x0000);
    assert_eq!(f32_to_bf16(-0.0), 0x8000);
    assert_eq!(f32_to_bf16(1.0), 0x3F80);
    assert_eq!(f32_to_bf16(-2.0), 0xC000);
    assert_eq!(f32_to_bf16(f32::INFINITY), 0x7F80);
    assert_eq!(f32_to_bf16(f32::NEG_INFINITY), 0xFF80);
    assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
}

/// Same exhaustive round-trip proof for `bfloat16`.
#[test]
fn f32_to_bf16_round_trips_every_possible_bf16_bit_pattern_exactly() {
    for bits in 0u32..=0xFFFF {
        let bits = bits as u16;
        let decoded = bf16_to_f32(bits);
        let reencoded = f32_to_bf16(decoded);
        if decoded.is_nan() {
            assert!(
                bf16_to_f32(reencoded).is_nan(),
                "bf16 bit pattern {bits:#06x} decoded to NaN {decoded:?} must re-encode to a NaN, got {reencoded:#06x}"
            );
        } else {
            assert_eq!(
                reencoded, bits,
                "bf16 bit pattern {bits:#06x} (decoded {decoded:?}) did not round-trip: got {reencoded:#06x}"
            );
        }
    }
}

#[test]
fn host_tensors_from_artifact_bytes_converts_f16_storage_to_f32() {
    // 1.0, -2.0, 0.5, 0.0 as IEEE754 binary16 bit patterns.
    let bytes = f16_bytes(&[0x3C00, 0xC000, 0x3800, 0x0000]);
    let metadata = f16_tensor_metadata("weight.a", vec![4], bytes.len());

    let weights = host_tensors_from_artifact_bytes(std::slice::from_ref(&metadata), &bytes, 0)
        .expect("F16 tensor materializes");
    let tensor = weights.get("weight.a").expect("tensor present");
    assert_eq!(tensor.data, vec![1.0, -2.0, 0.5, 0.0]);
}

#[test]
fn host_tensors_from_artifact_bytes_rejects_f16_digest_mismatch() {
    let bytes = f16_bytes(&[0x3C00]);
    let mut metadata = f16_tensor_metadata("weight.a", vec![1], bytes.len());
    metadata.digest = Some(ModelDigest::sha256(b"not the real storage bytes"));

    let error = host_tensors_from_artifact_bytes(std::slice::from_ref(&metadata), &bytes, 0)
        .expect_err("a digest mismatch against the original F16 bytes must be rejected");
    assert_eq!(error.code, ModelLoadingErrorCode::MaterializationFailed);
}

#[test]
fn host_tensors_from_artifact_bytes_rejects_out_of_bounds_range() {
    let (metadata, _bytes) = artifact_bytes_test_tensor("weight.a", vec![4], &[1.0, 2.0, 3.0, 4.0]);
    // Declare a range that does not actually fit in a much smaller buffer.
    let short_file = vec![0u8; 4];

    let error = host_tensors_from_artifact_bytes(std::slice::from_ref(&metadata), &short_file, 0)
        .expect_err("out-of-bounds range must be rejected");
    assert_eq!(error.code, ModelLoadingErrorCode::MaterializationFailed);
}

fn sealed_loading_valid_manifest() -> ModelManifest {
    ModelManifest::from_yaml_str(&format!(
        r#"
schema: magnetar-model-artifact
schema_version: 1
kind: model-bundle
digest: {}
model:
  name: qwen.example
  revision: r1
architecture:
  family: qwen
  identifier: qwen2
storage_dtype: int8
compute_dtype: bf16
supported_compute_dtypes: [bf16, fp16]
artifacts:
  weights:
    kind: model-weights
    digest: {}
    size_bytes: 128
  config:
    kind: model-config
    digest: {}
    size_bytes: 16
quantization:
  format: q4_k
  workspace_bytes: 64
shards:
  - id: shard0
    digest: {}
    size_bytes: 128
    order: 0
tensors:
  - name: transformer.wte.weight
    shape: [4, 8]
    storage_dtype: int8
    shard: shard0
"#,
        sealed_loading_digest(),
        sealed_loading_digest(),
        sealed_loading_digest(),
        sealed_loading_digest()
    ))
    .unwrap()
}

fn sealed_loading_trusted(manifest: &ModelManifest) -> ModelTrustDecision {
    ModelTrustStore::default()
        .trust_digest(manifest.id.digest.value.clone())
        .evaluate(manifest)
}

fn sealed_loading_coordinator() -> ModelLoadingCoordinator {
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(ModelArchitectureImplementation {
        architecture: ModelArchitecture::new("qwen", "qwen2"),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    });
    coordinator
}

fn sealed_loading_digest() -> String {
    "sha256:0000000000000000000000000000000000000000000000000000000000000001".into()
}

#[test]
fn sealed_loading_rejects_untrusted_artifact_before_memory_allocation() {
    let manifest = sealed_loading_valid_manifest();
    let mut coordinator = sealed_loading_coordinator();
    let mut memory = MemoryManager::default();
    let request =
        ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    let untrusted = ModelTrustStore::default()
        .reject_digest(manifest.id.digest.value.clone())
        .evaluate(&manifest);

    let error = coordinator
        .load(request, &manifest, &untrusted, &mut memory)
        .unwrap_err();

    assert_eq!(error.code, ModelLoadingErrorCode::ModelArtifactUntrusted);
    assert_eq!(memory.allocations().count(), 0);
}

#[test]
fn sealed_loading_creates_runtime_owned_ready_context_without_raw_handles() {
    let manifest = sealed_loading_valid_manifest();
    let mut coordinator = sealed_loading_coordinator();
    let mut memory = MemoryManager::default();
    let mut request =
        ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    request.quantization_policy = ModelQuantizationPolicy::DequantizeAtLoad;
    request.sharding_policy = ModelShardingPolicy::Sequential;

    let context = coordinator
        .load(
            request,
            &manifest,
            &sealed_loading_trusted(&manifest),
            &mut memory,
        )
        .unwrap();

    assert_eq!(context.state(), ModelLoadingState::Ready);
    assert!(context.can_start_inference());
    assert!(!context.plan().has_raw_native_handles());
    assert_eq!(
        context.plan().quantization_handling(),
        &ModelQuantizationHandling::DequantizeAtLoad(ModelQuantizationFormat::GgufQ4K)
    );
    assert_eq!(
        context.plan().memory_placements(),
        &[ModelResidencyLocation::Host]
    );
    assert_eq!(memory.allocations().count(), 1);
    assert!(
        coordinator
            .observations()
            .iter()
            .any(|observation| observation.kind == ModelLoadingObservationKind::ModelReady)
    );
}

#[test]
fn sealed_loading_memory_budget_failure_does_not_allocate() {
    let manifest = sealed_loading_valid_manifest();
    let mut coordinator = sealed_loading_coordinator();
    let mut memory = MemoryManager::default();
    let mut request =
        ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    request.quantization_policy = ModelQuantizationPolicy::DequantizeAtLoad;
    request.memory_budget_bytes = Some(1);

    let error = coordinator
        .load(
            request,
            &manifest,
            &sealed_loading_trusted(&manifest),
            &mut memory,
        )
        .unwrap_err();

    assert_eq!(error.code, ModelLoadingErrorCode::MemoryFeasibilityFailed);
    assert_eq!(memory.allocations().count(), 0);
}

#[test]
fn host_tensors_from_artifact_bytes_rejects_unsupported_dtype() {
    let (mut metadata, bytes) = artifact_bytes_test_tensor("weight.a", vec![1], &[1.0]);
    // I8 (not F32/F16/BF16/Q8_0/Q4_K/Q5_K) stays genuinely unsupported --
    // Q8_0/Q4_K/Q5_K moved to their own dedicated dequantization tests
    // once `support-gguf-quantized-tensor-dequantization` added real support for
    // them (they no longer belong in this "still rejected" test).
    metadata.storage_dtype = ModelDType::I8;

    let error = host_tensors_from_artifact_bytes(std::slice::from_ref(&metadata), &bytes, 0)
        .expect_err("an unsupported dtype must be rejected");
    assert_eq!(error.code, ModelLoadingErrorCode::StorageDTypeUnsupported);
}

#[test]
fn host_tensors_from_artifact_bytes_converts_bf16_storage_to_f32() {
    // 1.0 and -2.0 as bfloat16 bit patterns (f32's top 16 bits).
    let bytes: Vec<u8> = [0x3F80u16, 0xC000u16]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let metadata = ModelTensorMetadata {
        storage_dtype: ModelDType::Bf16,
        ..f16_tensor_metadata("weight.a", vec![2], bytes.len())
    };

    let weights = host_tensors_from_artifact_bytes(std::slice::from_ref(&metadata), &bytes, 0)
        .expect("BF16 tensor materializes");
    let tensor = weights.get("weight.a").expect("tensor present");
    assert_eq!(tensor.data, vec![1.0, -2.0]);
}

#[test]
fn host_tensors_from_artifact_bytes_rejects_shape_size_mismatch() {
    let (mut metadata, bytes) =
        artifact_bytes_test_tensor("weight.a", vec![4], &[1.0, 2.0, 3.0, 4.0]);
    // Shape says 4 elements (16 bytes), but size_bytes disagrees.
    metadata.size_bytes = Some(8);

    let error = host_tensors_from_artifact_bytes(std::slice::from_ref(&metadata), &bytes, 0)
        .expect_err("shape/size mismatch must be rejected");
    assert_eq!(error.code, ModelLoadingErrorCode::MaterializationFailed);
}

#[path = "k_quant_golden.rs"]
mod k_quant_golden;

/// The k-quant and i-quant dequantizations match llama.cpp's reference
/// implementation bit for bit on random super-blocks.
#[test]
fn k_quant_dequantization_matches_the_gguf_reference_bit_for_bit() {
    use k_quant_golden::*;
    let q6 = dequantize_q6_k(&Q6_K_GOLDEN_BYTES);
    assert_eq!(q6.len(), Q6_K_GOLDEN_BITS.len());
    for (index, (value, bits)) in q6.iter().zip(Q6_K_GOLDEN_BITS).enumerate() {
        assert_eq!(value.to_bits(), bits, "Q6_K value {index}: {value}");
    }
    let q2 = dequantize_q2_k(&Q2_K_GOLDEN_BYTES);
    assert_eq!(q2.len(), Q2_K_GOLDEN_BITS.len());
    for (index, (value, bits)) in q2.iter().zip(Q2_K_GOLDEN_BITS).enumerate() {
        assert_eq!(value.to_bits(), bits, "Q2_K value {index}: {value}");
    }
    let q3 = dequantize_q3_k(&Q3_K_GOLDEN_BYTES);
    assert_eq!(q3.len(), Q3_K_GOLDEN_BITS.len());
    for (index, (value, bits)) in q3.iter().zip(Q3_K_GOLDEN_BITS).enumerate() {
        assert_eq!(value.to_bits(), bits, "Q3_K value {index}: {value}");
    }
    let nl = dequantize_iq4_nl(&IQ4_NL_GOLDEN_BYTES);
    assert_eq!(nl.len(), IQ4_NL_GOLDEN_BITS.len());
    for (index, (value, bits)) in nl.iter().zip(IQ4_NL_GOLDEN_BITS).enumerate() {
        assert_eq!(value.to_bits(), bits, "IQ4_NL value {index}: {value}");
    }
    let xs = dequantize_iq4_xs(&IQ4_XS_GOLDEN_BYTES);
    assert_eq!(xs.len(), IQ4_XS_GOLDEN_BITS.len());
    for (index, (value, bits)) in xs.iter().zip(IQ4_XS_GOLDEN_BITS).enumerate() {
        assert_eq!(value.to_bits(), bits, "IQ4_XS value {index}: {value}");
    }
    let values = dequantize_iq2_xxs(&IQ2_XXS_GOLDEN_BYTES);
    assert_eq!(values.len(), IQ2_XXS_GOLDEN_BITS.len());
    for (index, (value, bits)) in values.iter().zip(IQ2_XXS_GOLDEN_BITS).enumerate() {
        assert_eq!(value.to_bits(), bits, "IQ2_XXS value {index}: {value}");
    }
    let values = dequantize_iq2_xs(&IQ2_XS_GOLDEN_BYTES);
    assert_eq!(values.len(), IQ2_XS_GOLDEN_BITS.len());
    for (index, (value, bits)) in values.iter().zip(IQ2_XS_GOLDEN_BITS).enumerate() {
        assert_eq!(value.to_bits(), bits, "IQ2_XS value {index}: {value}");
    }
    let values = dequantize_iq2_s(&IQ2_S_GOLDEN_BYTES);
    assert_eq!(values.len(), IQ2_S_GOLDEN_BITS.len());
    for (index, (value, bits)) in values.iter().zip(IQ2_S_GOLDEN_BITS).enumerate() {
        assert_eq!(value.to_bits(), bits, "IQ2_S value {index}: {value}");
    }
    let values = dequantize_iq3_xxs(&IQ3_XXS_GOLDEN_BYTES);
    assert_eq!(values.len(), IQ3_XXS_GOLDEN_BITS.len());
    for (index, (value, bits)) in values.iter().zip(IQ3_XXS_GOLDEN_BITS).enumerate() {
        assert_eq!(value.to_bits(), bits, "IQ3_XXS value {index}: {value}");
    }
    let values = dequantize_iq3_s(&IQ3_S_GOLDEN_BYTES);
    assert_eq!(values.len(), IQ3_S_GOLDEN_BITS.len());
    for (index, (value, bits)) in values.iter().zip(IQ3_S_GOLDEN_BITS).enumerate() {
        assert_eq!(value.to_bits(), bits, "IQ3_S value {index}: {value}");
    }
    let q4 = dequantize_q4_k(&Q4_K_GOLDEN_BYTES);
    assert_eq!(q4.len(), Q4_K_GOLDEN_BITS.len());
    for (index, (value, bits)) in q4.iter().zip(Q4_K_GOLDEN_BITS).enumerate() {
        assert_eq!(value.to_bits(), bits, "Q4_K value {index}: {value}");
    }
}

/// A transposed `Q6_K` projection keeps its bytes packed and dequantizes
/// exactly like the load-time path; an untransposed one (an embedding
/// table, read row by row) stays dequantized at load.
#[test]
fn transposed_q6_k_projections_stay_packed() {
    use k_quant_golden::*;
    let metadata = |layout: Option<&str>| ModelTensorMetadata {
        name: "w".to_string(),
        // Logical `[in, out]`: two output columns of 256 values each.
        shape: vec![256, 2],
        storage_dtype: ModelDType::Q6K,
        layout: layout.map(str::to_string),
        shard: None,
        offset_bytes: Some(0),
        size_bytes: Some(Q6_K_GOLDEN_BYTES.len() as u64),
        quantization: None,
        expected_compute_dtype: None,
        digest: None,
    };
    let packed = packed_weight_from_raw_bytes(
        &metadata(Some(crate::packed_weight::TRANSPOSED_STORAGE_LAYOUT)),
        &Q6_K_GOLDEN_BYTES,
    )
    .expect("valid bytes")
    .expect("a transposed Q6_K projection is packed");
    assert_eq!(
        packed.format(),
        crate::packed_weight::PackedWeightFormat::Q6K
    );
    assert_eq!(packed.logical_shape(), [256, 2]);
    let stored = packed.dequantize_stored();
    for (index, (value, bits)) in stored.iter().zip(Q6_K_GOLDEN_BITS).enumerate() {
        assert_eq!(value.to_bits(), bits, "Q6_K value {index}: {value}");
    }
    let untransposed = ModelTensorMetadata {
        shape: vec![2, 256],
        ..metadata(None)
    };
    assert!(
        packed_weight_from_raw_bytes(&untransposed, &Q6_K_GOLDEN_BYTES)
            .expect("valid bytes")
            .is_none()
    );
}

/// A transposed F32 tensor's declared digest describes its storage bytes:
/// they are verified before the transpose, and the materialized tensor is
/// the declared (transposed) matrix.
#[test]
fn transposed_f32_tensors_verify_their_storage_digest() {
    // Stored `[3, 2]`, declared `[2, 3]`.
    let stored = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (mut metadata, bytes) = artifact_bytes_test_tensor("weight.t", vec![2, 3], &stored);
    metadata.layout = Some(crate::packed_weight::TRANSPOSED_STORAGE_LAYOUT.to_string());
    metadata.digest = Some(crate::model::ModelDigest::sha256(&bytes));
    let tensor = tensor_from_raw_bytes(&metadata, &bytes).expect("storage digest matches");
    assert_eq!(tensor.shape, vec![2, 3]);
    assert_eq!(tensor.data, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    let mut wrong = bytes.clone();
    wrong[0] ^= 1;
    let error = tensor_from_raw_bytes(&metadata, &wrong).expect_err("tampered bytes rejected");
    assert_eq!(error.code, ModelLoadingErrorCode::MaterializationFailed);
}

/// A weight's resident bytes: its storage when it stays packed or is f32,
/// else its elements as f32.
#[test]
fn resident_weight_bytes_count_what_the_provider_holds() {
    let base = f16_tensor_metadata("weight.h", vec![4, 64], 4 * 64 * 2);
    assert_eq!(resident_weight_bytes(&base), 4 * 64 * 4);
    let (f32_tensor, _) = artifact_bytes_test_tensor("weight.f", vec![2], &[1.0, 2.0]);
    assert_eq!(resident_weight_bytes(&f32_tensor), 8);
    let q8 = ModelTensorMetadata {
        storage_dtype: ModelDType::Q8,
        size_bytes: Some(4 * 2 * 34),
        ..f16_tensor_metadata("weight.q", vec![4, 64], 0)
    };
    assert_eq!(resident_weight_bytes(&q8), 4 * 2 * 34);
    // An untransposed k-quant table is dequantized at load.
    let q4k_table = ModelTensorMetadata {
        storage_dtype: ModelDType::Q4K,
        size_bytes: Some(4 * 144),
        ..f16_tensor_metadata("weight.e", vec![4, 256], 0)
    };
    assert_eq!(resident_weight_bytes(&q4k_table), 4 * 256 * 4);
}

#[test]
fn inference_api_load_model_wires_coordinator_and_memory_manager() {
    let manifest = minimal_model_manifest();
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(ModelArchitectureImplementation {
        architecture: manifest.architecture.clone(),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    });
    let mut runtime = Runtime::builder()
        .trust_store(ModelTrustStore::default().trust_digest(manifest.id.digest.value.clone()))
        .build()
        .unwrap();

    let core = ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    let mut request = ModelLoadingApiRequest::new(core);
    request.tokenizer_reference = Some(TokenizerId::new("fixture").unwrap());
    request.adapter_references = vec![AdapterArtifactId {
        name: AdapterName::new("lora-a").unwrap(),
        revision: AdapterRevision::new("1").unwrap(),
        digest: AdapterDigest {
            algorithm: "sha256".into(),
            value: "deadbeef".into(),
        },
    }];
    request.layout_policy = Some(TensorLayoutKind::Contiguous);
    request.provider_preferences = vec![ProviderBinding::new("reference-cpu")];

    let loaded = load_model(&mut coordinator, &mut runtime, request, &manifest).unwrap();
    assert_eq!(loaded.artifact, manifest.id);
    assert_eq!(loaded.state, ModelLoadingState::Ready);
}

/// `seal-runtime-model-trust-and-provenance-authority` P0-A: `load_model`
/// no longer accepts a caller-supplied trust decision at all -- trust is
/// evaluated from the performing `Runtime`'s own sealed configuration. A
/// `Runtime` built without trusting this manifest's digest must reject the
/// load, proving there is no parameter through which a caller could still
/// supply a favorable decision for an untrusted artifact.
#[test]
fn inference_api_load_model_rejects_when_runtime_trust_store_does_not_trust_digest() {
    let manifest = minimal_model_manifest();
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(ModelArchitectureImplementation {
        architecture: manifest.architecture.clone(),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    });
    // Sealed with an empty trust store -- trusts nothing, matching
    // `RuntimeBuilder::trust_store`'s documented default.
    let mut runtime = Runtime::builder().build().unwrap();
    let core = ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());

    let error = load_model(
        &mut coordinator,
        &mut runtime,
        ModelLoadingApiRequest::new(core),
        &manifest,
    )
    .unwrap_err();
    assert!(error.to_string().contains("ModelArtifactUntrusted"));
}

/// P0-B: `create_model_instance` cross-checks the caller-supplied
/// architecture against `loaded.plan().architecture`, which the loading
/// phase already resolved from the same manifest -- a disagreement is
/// rejected rather than silently accepted.
#[test]
fn runtime_create_model_instance_rejects_architecture_disagreeing_with_resolved_plan() {
    let manifest = minimal_model_manifest();
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(ModelArchitectureImplementation {
        architecture: manifest.architecture.clone(),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    });
    let mut runtime = Runtime::builder()
        .trust_store(ModelTrustStore::default().trust_digest(manifest.id.digest.value.clone()))
        .build()
        .unwrap();
    let core = ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    let loaded = load_model(
        &mut coordinator,
        &mut runtime,
        ModelLoadingApiRequest::new(core),
        &manifest,
    )
    .unwrap();

    let disagreeing_architecture = ModelArchitectureImplementation {
        architecture: ModelArchitecture::new("a-different-family", "a-different-identifier"),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    };
    let error = runtime
        .create_model_instance(
            &loaded,
            disagreeing_architecture,
            ResourceAffinity::new(FallbackClass::Transparent),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        ModelInstanceError::ArchitectureMismatch { .. }
    ));

    // Agreeing architecture is unaffected.
    runtime
        .create_model_instance(
            &loaded,
            ModelArchitectureImplementation {
                architecture: manifest.architecture.clone(),
                kind: ModelArchitectureImplementationKind::TestFixture,
                required_capabilities: Vec::new(),
            },
            ResourceAffinity::new(FallbackClass::Transparent),
        )
        .unwrap();
}

/// P0-B: when the loading phase *did* resolve a provider binding for the
/// plan, a caller-supplied affinity naming a different provider is
/// rejected. Nothing in today's real loading pipeline resolves
/// `plan.provider_binding` yet (confirmed by inspection: it is
/// unconditionally `None` at construction), so this forces the scenario
/// directly on the loaded plan rather than only exercising the
/// permissive "unresolved" branch every other test already covers.
#[test]
fn runtime_create_model_instance_rejects_affinity_disagreeing_with_resolved_provider_binding() {
    let manifest = minimal_model_manifest();
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(ModelArchitectureImplementation {
        architecture: manifest.architecture.clone(),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    });
    let mut runtime = Runtime::builder()
        .trust_store(ModelTrustStore::default().trust_digest(manifest.id.digest.value.clone()))
        .build()
        .unwrap();
    let core = ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    let mut loaded = load_model(
        &mut coordinator,
        &mut runtime,
        ModelLoadingApiRequest::new(core),
        &manifest,
    )
    .unwrap();
    loaded.plan.provider_binding = Some(ProviderBinding::new("resolved-provider"));

    let disagreeing_affinity = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(ProviderBinding::new("a-different-provider"));
    let error = runtime
        .create_model_instance(
            &loaded,
            ModelArchitectureImplementation {
                architecture: manifest.architecture.clone(),
                kind: ModelArchitectureImplementationKind::TestFixture,
                required_capabilities: Vec::new(),
            },
            disagreeing_affinity,
        )
        .unwrap_err();
    assert!(matches!(error, ModelInstanceError::AffinityMismatch { .. }));

    let agreeing_affinity = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(ProviderBinding::new("resolved-provider"));
    runtime
        .create_model_instance(
            &loaded,
            ModelArchitectureImplementation {
                architecture: manifest.architecture.clone(),
                kind: ModelArchitectureImplementationKind::TestFixture,
                required_capabilities: Vec::new(),
            },
            agreeing_affinity,
        )
        .unwrap();
}

/// Task 6.7: peak transient host staging is bounded independently of total
/// model size. The exactly-once-per-tensor, in-order call log the other
/// two streaming tests already assert *is* the proof this function never
/// buffers ahead: this test additionally scales the tensor count well
/// past the tiny three-tensor fixture (50 tensors) to show that scaling up
/// the tensor *count* changes nothing about the one-tensor-at-a-time
/// shape of each `read_payload` call -- `stream_materialize_model_
/// instance_weights`'s loop body holds exactly one `HostTensor` local
/// variable at a time (never a growing collection), so this property does
/// not degrade as the model grows.
#[test]
fn stream_materialize_model_instance_weights_scales_tensor_count_without_buffering_ahead() {
    const TENSOR_COUNT: usize = 50;
    let names: Vec<String> = (0..TENSOR_COUNT).map(|i| format!("weight.{i}")).collect();
    let manifest = manifest_with_tensors(
        names
            .iter()
            .map(|name| (name.as_str(), vec![4, 4]))
            .collect(),
    );
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(ModelArchitectureImplementation {
        architecture: manifest.architecture.clone(),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    });
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .trust_store(ModelTrustStore::default().trust_digest(manifest.id.digest.value.clone()))
        .build()
        .unwrap();
    let core = ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    let loaded = load_model(
        &mut coordinator,
        &mut runtime,
        ModelLoadingApiRequest::new(core),
        &manifest,
    )
    .unwrap();
    let instance = runtime
        .create_model_instance(
            &loaded,
            ModelArchitectureImplementation {
                architecture: manifest.architecture.clone(),
                kind: ModelArchitectureImplementationKind::TestFixture,
                required_capabilities: Vec::new(),
            },
            ResourceAffinity::new(FallbackClass::Transparent),
        )
        .unwrap();

    let bytes_by_name: BTreeMap<String, Vec<u8>> = names
        .iter()
        .map(|name| (name.clone(), f32_tensor_bytes(&[1.0; 16])))
        .collect();
    let payload_source = RecordingPayloadSource::new(bytes_by_name);

    stream_materialize_model_instance_weights(
        &mut runtime,
        &instance,
        "test",
        &manifest.tensors,
        &payload_source,
    )
    .expect("streaming materialization of 50 tensors succeeds");

    assert_eq!(
        runtime.model_instance(&instance).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Ready
    );
    assert_eq!(
        payload_source.call_log(),
        names,
        "still exactly once, in order, at scale"
    );
}

/// P0-C regression guard: matching shape and F32 dtype still materializes
/// normally.
#[test]
fn materialize_model_instance_weights_accepts_matching_shape_and_dtype() {
    let manifest = manifest_with_one_tensor(vec![2, 2], ModelDType::F32);
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(ModelArchitectureImplementation {
        architecture: manifest.architecture.clone(),
        kind: ModelArchitectureImplementationKind::TestFixture,
        required_capabilities: Vec::new(),
    });
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .trust_store(ModelTrustStore::default().trust_digest(manifest.id.digest.value.clone()))
        .build()
        .unwrap();
    let core = ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    let loaded = load_model(
        &mut coordinator,
        &mut runtime,
        ModelLoadingApiRequest::new(core),
        &manifest,
    )
    .unwrap();
    let instance = runtime
        .create_model_instance(
            &loaded,
            ModelArchitectureImplementation {
                architecture: manifest.architecture.clone(),
                kind: ModelArchitectureImplementationKind::TestFixture,
                required_capabilities: Vec::new(),
            },
            ResourceAffinity::new(FallbackClass::Transparent),
        )
        .unwrap();

    let matching_weights = BTreeMap::from([(
        "the-only-tensor".to_string(),
        HostTensor::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
    )]);
    materialize_model_instance_weights(&mut runtime, &instance, "test", &matching_weights).unwrap();
    assert_eq!(
        runtime.model_instance(&instance).unwrap().lifecycle(),
        ModelInstanceLifecycleState::Ready
    );
}

/// `support-gguf-quantized-tensor-dequantization`: a manifest declaring a quantized
/// storage dtype (`Q8_0`/`Q4_K`/`Q5_K`) backed by already-dequantized F32
/// content -- exactly the shape `host_tensors_from_artifact_bytes`
/// produces for a real quantized GGUF tensor -- materializes successfully
/// through the same shape/dtype whitelist `materialize_model_instance_
/// weights_rejects_quantized_declared_dtype` proves still rejects a
/// genuinely unsupported dtype, mirroring F16/BF16's own already-working
/// "declared non-F32, content already F32" acceptance.
#[test]
fn materialize_model_instance_weights_accepts_dequantized_content_for_a_quantized_declared_dtype() {
    for quantized_dtype in [ModelDType::Q8, ModelDType::Q4K, ModelDType::Q5K] {
        let manifest = manifest_with_one_tensor(vec![2, 2], quantized_dtype);
        let mut coordinator = ModelLoadingCoordinator::new();
        coordinator.register_architecture(ModelArchitectureImplementation {
            architecture: manifest.architecture.clone(),
            kind: ModelArchitectureImplementationKind::TestFixture,
            required_capabilities: Vec::new(),
        });
        let mut runtime = Runtime::builder()
            .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
            .trust_store(ModelTrustStore::default().trust_digest(manifest.id.digest.value.clone()))
            .build()
            .unwrap();
        let core =
            ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
        let loaded = load_model(
            &mut coordinator,
            &mut runtime,
            ModelLoadingApiRequest::new(core),
            &manifest,
        )
        .unwrap();
        let instance = runtime
            .create_model_instance(
                &loaded,
                ModelArchitectureImplementation {
                    architecture: manifest.architecture.clone(),
                    kind: ModelArchitectureImplementationKind::TestFixture,
                    required_capabilities: Vec::new(),
                },
                ResourceAffinity::new(FallbackClass::Transparent),
            )
            .unwrap();

        let dequantized_weights = BTreeMap::from([(
            "the-only-tensor".to_string(),
            HostTensor::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        )]);
        materialize_model_instance_weights(&mut runtime, &instance, "test", &dequantized_weights)
            .unwrap_or_else(|error| {
                panic!("{quantized_dtype:?} declared dtype with dequantized F32 content must materialize, got: {error}")
            });
        assert_eq!(
            runtime.model_instance(&instance).unwrap().lifecycle(),
            ModelInstanceLifecycleState::Ready
        );
    }
}

#[test]
fn sealed_loading_unsupported_quantization_requires_explicit_policy() {
    let manifest = sealed_loading_valid_manifest();
    let mut coordinator = sealed_loading_coordinator();
    let mut memory = MemoryManager::default();
    let request =
        ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());

    let error = coordinator
        .load(
            request,
            &manifest,
            &sealed_loading_trusted(&manifest),
            &mut memory,
        )
        .unwrap_err();

    assert_eq!(error.code, ModelLoadingErrorCode::QuantizationUnsupported);
}

#[test]
fn sealed_loading_memory_manager_rejection_maps_to_loading_error() {
    let manifest = sealed_loading_valid_manifest();
    let mut coordinator = sealed_loading_coordinator();
    let mut memory = MemoryManager::new(MemoryManagerConfig {
        max_runtime_bytes: Some(1),
        ..MemoryManagerConfig::default()
    });
    let mut request =
        ModelLoadingRequest::new(ModelLoadingRequestId::new("load-1"), manifest.id.clone());
    request.quantization_policy = ModelQuantizationPolicy::DequantizeAtLoad;

    let error = coordinator
        .load(
            request,
            &manifest,
            &sealed_loading_trusted(&manifest),
            &mut memory,
        )
        .unwrap_err();

    assert_eq!(error.code, ModelLoadingErrorCode::MemoryAllocationFailed);
}
