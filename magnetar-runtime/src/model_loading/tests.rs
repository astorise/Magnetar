//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::model::{ModelDType, ModelDigest, ModelTensorMetadata};

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
