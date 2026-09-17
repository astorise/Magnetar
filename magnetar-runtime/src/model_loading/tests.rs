//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::model::{ModelDType, ModelDigest, ModelTensorMetadata};

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
