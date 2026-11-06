//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;

/// One `block_q8_0`: scale `d` (as f16 bits) then 32 quants.
fn q8_0_block(d_bits: u16, quants: [i8; 32]) -> Vec<u8> {
    let mut block = d_bits.to_le_bytes().to_vec();
    block.extend(quants.iter().map(|&q| q as u8));
    block
}

#[test]
fn rejects_rows_that_are_not_whole_blocks_and_wrong_byte_counts() {
    assert!(PackedWeight::new(PackedWeightFormat::Q8_0, 2, 48, false, vec![0; 68]).is_err());
    assert!(PackedWeight::new(PackedWeightFormat::Q8_0, 2, 32, false, vec![0; 67]).is_err());
    assert!(PackedWeight::new(PackedWeightFormat::Q8_0, 0, 32, false, vec![]).is_err());
    assert!(PackedWeight::new(PackedWeightFormat::Q8_0, 2, 32, false, vec![0; 68]).is_ok());
}

#[test]
fn dequantizes_q8_0_and_applies_the_transposed_storage() {
    // d = 0.5 (f16 0x3800) and d = 2.0 (f16 0x4000).
    let mut first = [0_i8; 32];
    let mut second = [0_i8; 32];
    for i in 0..32 {
        first[i] = i as i8 - 16;
        second[i] = 100 - i as i8;
    }
    let mut bytes = q8_0_block(0x3800, first);
    bytes.extend(q8_0_block(0x4000, second));

    // Stored as 2 rows of 32 values, the graph sees [32, 2].
    let packed = PackedWeight::new(PackedWeightFormat::Q8_0, 2, 32, true, bytes.clone()).unwrap();
    assert_eq!(packed.logical_shape(), [32, 2]);
    let logical = packed.dequantize();
    assert_eq!(logical.shape, vec![32, 2]);
    for k in 0..32 {
        assert_eq!(logical.data[k * 2], 0.5 * f32::from(first[k]));
        assert_eq!(logical.data[k * 2 + 1], 2.0 * f32::from(second[k]));
    }

    let untransposed = PackedWeight::new(PackedWeightFormat::Q8_0, 2, 32, false, bytes).unwrap();
    assert_eq!(untransposed.logical_shape(), [2, 32]);
    assert_eq!(
        untransposed.dequantize().data,
        untransposed.dequantize_stored()
    );
}
