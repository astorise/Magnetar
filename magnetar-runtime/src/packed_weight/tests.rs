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

/// A `Q4_K` packed weight dequantizes through `ggml-quants.c`'s
/// `dequantize_row_q4_K` arithmetic, and its transposed storage applies
/// like `Q8_0`'s.
#[test]
fn dequantizes_q4_k_and_applies_the_transposed_storage() {
    // Two stored rows of one 256-value super-block each.
    let mut bytes = Vec::new();
    for row in 0..2u8 {
        bytes.extend(0x3c00_u16.to_le_bytes()); // d = 1.0
        bytes.extend(0x3800_u16.to_le_bytes()); // dmin = 0.5
        // Sub-blocks 0..3: scale 2 + row, min 1; 4..7 from the high bits.
        bytes.extend([
            2 + row,
            2 + row,
            2 + row,
            2 + row,
            1,
            1,
            1,
            1,
            0x13,
            0x13,
            0x13,
            0x13,
        ]);
        bytes.extend((0..128u32).map(|i| (i * 7 + u32::from(row)) as u8));
    }
    let weight = PackedWeight::new(PackedWeightFormat::Q4K, 2, 256, true, bytes.clone())
        .expect("two rows of one super-block");
    assert_eq!(weight.logical_shape(), [256, 2]);
    let stored = weight.dequantize_stored();
    assert_eq!(stored, crate::model_loading::dequantize_q4_k(&bytes));
    // Element 0 of row 0: low nibble of qs[0] = 0, so -dmin * min = -0.5.
    assert_eq!(stored[0], -0.5);
    // Element 32 of row 0: high nibble of qs[0] = 0 -> -0.5 too; element 1:
    // low nibble of qs[1] = 7 -> 1.0 * 2 * 7 - 0.5.
    assert_eq!(stored[1], 13.5);
    let logical = weight.dequantize();
    assert_eq!(logical.shape, vec![256, 2]);
    for row in 0..2 {
        for col in 0..256 {
            assert_eq!(logical.data[col * 2 + row], stored[row * 256 + col]);
        }
    }
    assert!(PackedWeight::new(PackedWeightFormat::Q4K, 1, 128, true, vec![0; 72]).is_err());
}
