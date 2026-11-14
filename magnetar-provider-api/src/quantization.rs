//! Quantization tables and decoders.
//!
//! Block layouts of the GGUF quantized storage formats, the tables the
//! i-quants index into, the decoders that expand a block sequence to `f32`,
//! and the half-precision conversions they use. Pure functions of their
//! arguments: no Runtime state, no allocation beyond the returned vector.
//! Providers reuse these to decode packed weights on the host.

pub mod iq_grids;

/// GGUF `block_q8_0`: 32 elements, 34 bytes (`ggml_half d` then 32
/// signed `int8` quants), no padding. Verified against `ggml-org/llama.cpp`'s
/// real `ggml-common.h`/`ggml-quants.c` (`support-gguf-quantized-tensor-dequantization`
/// design.md), not recalled from memory alone.
#[doc(hidden)]
pub const Q8_0_BLOCK_ELEMENTS: u64 = 32;

#[doc(hidden)]
pub const Q8_0_BLOCK_BYTES: u64 = 34;

/// GGUF K-quant super-block element count (`QK_K` upstream): 8 sub-blocks
/// of 32 elements each, shared by `block_q4_K` and `block_q5_K`.
#[doc(hidden)]
pub const QK_BLOCK_ELEMENTS: u64 = 256;

/// `block_q4_K`: `2 (d) + 2 (dmin) + 12 (scales) + 128 (qs) = 144` bytes.
#[doc(hidden)]
pub const Q4_K_BLOCK_BYTES: u64 = 144;

/// `block_q5_K`: `2 (d) + 2 (dmin) + 12 (scales) + 32 (qh) + 128 (qs) = 176` bytes.
#[doc(hidden)]
pub const Q5_K_BLOCK_BYTES: u64 = 176;

/// `block_q6_K`: `128 (ql) + 64 (qh) + 16 (scales) + 2 (d) = 210` bytes.
#[doc(hidden)]
pub const Q6_K_BLOCK_BYTES: u64 = 210;

/// `block_q2_K`: `16 (scales) + 64 (qs) + 2 (d) + 2 (dmin) = 84` bytes.
#[doc(hidden)]
pub const Q2_K_BLOCK_BYTES: u64 = 84;

/// `block_q3_K`: `32 (hmask) + 64 (qs) + 12 (scales) + 2 (d) = 110` bytes.
#[doc(hidden)]
pub const Q3_K_BLOCK_BYTES: u64 = 110;

/// `block_iq4_nl`: `2 (d) + 16 (qs) = 18` bytes for 32 values.
#[doc(hidden)]
pub const IQ4_NL_BLOCK_BYTES: u64 = 18;

/// `block_iq4_xs`: `2 (d) + 2 (scales_h) + 4 (scales_l) + 128 (qs) = 136`
/// bytes for 256 values.
#[doc(hidden)]
pub const IQ4_XS_BLOCK_BYTES: u64 = 136;

/// `block_iq2_xxs`: 66 bytes for 256 values.
#[doc(hidden)]
pub const IQ2_XXS_BLOCK_BYTES: u64 = 66;

/// `block_iq2_xs`: 74 bytes for 256 values.
#[doc(hidden)]
pub const IQ2_XS_BLOCK_BYTES: u64 = 74;

/// `block_iq2_s`: 82 bytes for 256 values.
#[doc(hidden)]
pub const IQ2_S_BLOCK_BYTES: u64 = 82;

/// `block_iq3_xxs`: 98 bytes for 256 values.
#[doc(hidden)]
pub const IQ3_XXS_BLOCK_BYTES: u64 = 98;

/// `block_iq3_s`: 110 bytes for 256 values.
#[doc(hidden)]
pub const IQ3_S_BLOCK_BYTES: u64 = 110;

/// `kvalues_iq4nl`: the non-linear values the `IQ4_NL`/`IQ4_XS` 4-bit
/// indices select.
pub const IQ4_NL_VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Both K-quant formats' shared 12-byte packed 6-bit scale/min encoding.
#[doc(hidden)]
pub const K_SCALE_SIZE: usize = 12;

/// Unpacks one K-quant sub-block's 6-bit scale and 6-bit min from the
/// shared 12-byte packed `scales` field (`block_q4_K`/`block_q5_K` both use
/// this exact encoding) -- `j` is the sub-block index (0..7). Ports
/// `ggml-quants.c`'s `get_scale_min_k4` bit-for-bit: sub-blocks 0-3 read
/// their 6 bits directly from `scales[j]`/`scales[j+4]`; sub-blocks 4-7
/// split across a low nibble in `scales[j+4]` and the top 2 bits of
/// `scales[j-4]`/`scales[j]` respectively. This packing is the detail most
/// reimplementations get wrong -- verified against the real upstream
/// source directly, not recalled from memory (design.md).
#[doc(hidden)]
pub fn get_scale_min_k4(j: usize, scales: &[u8; K_SCALE_SIZE]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

/// Dequantizes one `block_q8_0` region (`raw.len()` a multiple of 34
/// bytes) into `f32` values: `value[i] = d * qs[i]`, `d` the block's `f16`
/// scale, `qs[i]` a signed `int8` quant taken as-is (`ggml-quants.c`'s
/// `dequantize_row_q8_0`).
#[doc(hidden)]
pub fn dequantize_q8_0(raw: &[u8]) -> Vec<f32> {
    raw.as_chunks::<{ Q8_0_BLOCK_BYTES as usize }>()
        .0
        .iter()
        .flat_map(|block| {
            let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            block[2..].iter().map(move |&q| d * f32::from(q as i8))
        })
        .collect()
}

/// Dequantizes one `block_q4_K` region (`raw.len()` a multiple of 144
/// bytes) into `f32` values, porting `ggml-quants.c`'s
/// `dequantize_row_q4_K` bit-for-bit. Each 256-element super-block has 8
/// sub-blocks of 32 elements; for sub-block `sb`, `d_sb = d * scale6[sb]`
/// and `m_sb = dmin * min6[sb]`, and `value[i] = d_sb * q4[i] - m_sb` where
/// `q4[i]` is the raw unsigned 4-bit nibble (0-15). The 128-byte `qs`
/// array is walked in 32-byte strides shared by *two* sub-blocks per
/// stride: that stride's low nibbles form the first sub-block, its high
/// nibbles form the second -- not two independent contiguous halves.
#[doc(hidden)]
pub fn dequantize_q4_k(raw: &[u8]) -> Vec<f32> {
    let mut out =
        Vec::with_capacity(raw.len() / Q4_K_BLOCK_BYTES as usize * QK_BLOCK_ELEMENTS as usize);
    for block in raw.as_chunks::<{ Q4_K_BLOCK_BYTES as usize }>().0 {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales: [u8; K_SCALE_SIZE] = block[4..4 + K_SCALE_SIZE].try_into().unwrap();
        let qs = &block[4 + K_SCALE_SIZE..4 + K_SCALE_SIZE + 128];

        let mut is = 0usize;
        for chunk_start in (0..QK_BLOCK_ELEMENTS as usize).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, &scales);
            let (d1, min1) = (d * f32::from(sc1), dmin * f32::from(m1));
            let (sc2, m2) = get_scale_min_k4(is + 1, &scales);
            let (d2, min2) = (d * f32::from(sc2), dmin * f32::from(m2));
            let q = &qs[chunk_start / 2..chunk_start / 2 + 32];
            out.extend(q.iter().map(|&byte| d1 * f32::from(byte & 0xF) - min1));
            out.extend(q.iter().map(|&byte| d2 * f32::from(byte >> 4) - min2));
            is += 2;
        }
    }
    out
}

/// Dequantizes one `block_q6_K` region (`raw.len()` a multiple of 210
/// bytes) into `f32` values, porting `ggml-quants.c`'s
/// `dequantize_row_q6_K` bit-for-bit. A super-block is `ql[128]` (low
/// four bits), `qh[64]` (high two bits), `scales[16]` (signed 8-bit, one
/// per 16 values) and `f16` `d`, last. Each 128-value half reads 64 `ql`
/// bytes, 32 `qh` bytes and 8 scales: for `l` in `0..32`, the four values
/// `l`, `l + 32`, `l + 64`, `l + 96` take `ql[l]`/`ql[l + 32]`'s low or
/// high nibble with `qh[l]`'s bit pairs `0`, `2`, `4`, `6` as their high
/// bits, minus 32, times `d * scales[l / 16 + 0, 2, 4, 6]`.
#[doc(hidden)]
pub fn dequantize_q6_k(raw: &[u8]) -> Vec<f32> {
    let mut out =
        Vec::with_capacity(raw.len() / Q6_K_BLOCK_BYTES as usize * QK_BLOCK_ELEMENTS as usize);
    for block in raw.as_chunks::<{ Q6_K_BLOCK_BYTES as usize }>().0 {
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let mut values = [0.0_f32; QK_BLOCK_ELEMENTS as usize];
        for half in 0..2 {
            let ql = &block[64 * half..64 * half + 64];
            let qh = &block[128 + 32 * half..128 + 32 * half + 32];
            let sc = &block[192 + 8 * half..192 + 8 * half + 8];
            let y = &mut values[128 * half..128 * half + 128];
            for l in 0..32 {
                let is = l / 16;
                let q1 = i32::from((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32;
                let q2 = i32::from((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                let q3 = i32::from((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                let q4 = i32::from((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                let scale = |index: usize| d * f32::from(sc[index] as i8);
                y[l] = scale(is) * q1 as f32;
                y[l + 32] = scale(is + 2) * q2 as f32;
                y[l + 64] = scale(is + 4) * q3 as f32;
                y[l + 96] = scale(is + 6) * q4 as f32;
            }
        }
        out.extend(values);
    }
    out
}

/// Dequantizes `block_q2_K` super-blocks (`raw.len()` a multiple of 84
/// bytes), porting `ggml-quants.c`'s `dequantize_row_q2_K`: value `128 h +
/// 32 j + 16 p + l` (h = 0..1, j = 0..3, p = 0..1, l = 0..15) is bits `2 j`
/// of `qs[32 h + 16 p + l]`, under sub-block `8 h + 2 j + p` whose scale
/// byte holds a 4-bit scale (low) and min (high): `d * scale * q - dmin *
/// min`.
#[doc(hidden)]
pub fn dequantize_q2_k(raw: &[u8]) -> Vec<f32> {
    let mut out =
        Vec::with_capacity(raw.len() / Q2_K_BLOCK_BYTES as usize * QK_BLOCK_ELEMENTS as usize);
    for block in raw.as_chunks::<{ Q2_K_BLOCK_BYTES as usize }>().0 {
        let scales = &block[..16];
        let qs = &block[16..80];
        let d = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));
        for h in 0..2 {
            for j in 0..4 {
                for p in 0..2 {
                    let sc = scales[8 * h + 2 * j + p];
                    let dl = d * f32::from(sc & 0xF);
                    let ml = dmin * f32::from(sc >> 4);
                    for l in 0..16 {
                        let q = (qs[32 * h + 16 * p + l] >> (2 * j)) & 3;
                        out.push(dl * f32::from(q) - ml);
                    }
                }
            }
        }
    }
    out
}

/// Dequantizes `block_q3_K` super-blocks (`raw.len()` a multiple of 110
/// bytes), porting `ggml-quants.c`'s `dequantize_row_q3_K`: value `128 h +
/// 32 j + 16 p + l` is bits `2 j` of `qs[32 h + 16 p + l]`, minus 4 unless
/// bit `4 h + j` of `hmask[16 p + l]` is set, times `d * (scale - 32)` for
/// sub-block `8 h + 2 j + p`'s 6-bit scale (low nibbles from the first
/// eight scale bytes, high bit pairs from the last four).
#[doc(hidden)]
pub fn dequantize_q3_k(raw: &[u8]) -> Vec<f32> {
    let mut out =
        Vec::with_capacity(raw.len() / Q3_K_BLOCK_BYTES as usize * QK_BLOCK_ELEMENTS as usize);
    for block in raw.as_chunks::<{ Q3_K_BLOCK_BYTES as usize }>().0 {
        let hmask = &block[..32];
        let qs = &block[32..96];
        let packed = &block[96..108];
        let d = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));
        let scales = q3_k_scales(packed);
        for h in 0..2 {
            for j in 0..4 {
                for p in 0..2 {
                    let dl = d * f32::from(scales[8 * h + 2 * j + p] - 32);
                    for l in 0..16 {
                        let q = i8::try_from((qs[32 * h + 16 * p + l] >> (2 * j)) & 3)
                            .expect("two bits");
                        let high = hmask[16 * p + l] & (1 << (4 * h + j)) != 0;
                        out.push(dl * f32::from(q - if high { 0 } else { 4 }));
                    }
                }
            }
        }
    }
    out
}

/// Dequantizes `block_iq4_nl` blocks (`raw.len()` a multiple of 18 bytes),
/// porting `ggml-quants.c`'s `dequantize_row_iq4_nl`: value `j` and `j +
/// 16` of a block are the low and high nibbles of `qs[j]`, each selecting
/// an [`IQ4_NL_VALUES`] entry scaled by `d`.
#[doc(hidden)]
pub fn dequantize_iq4_nl(raw: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(raw.len() / IQ4_NL_BLOCK_BYTES as usize * 32);
    for block in raw.as_chunks::<{ IQ4_NL_BLOCK_BYTES as usize }>().0 {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        iq4_values(d, &block[2..18], &mut out);
    }
    out
}

/// One 32-value `IQ4_NL`-encoded run: `qs[j]`'s low nibble is value `j`,
/// its high nibble value `j + 16`.
#[doc(hidden)]
pub fn iq4_values(scale: f32, qs: &[u8], out: &mut Vec<f32>) {
    for shift in [0, 4] {
        out.extend(
            qs.iter()
                .map(|&q| scale * f32::from(IQ4_NL_VALUES[usize::from((q >> shift) & 0xF)])),
        );
    }
}

/// Dequantizes `block_iq4_xs` super-blocks (`raw.len()` a multiple of 136
/// bytes), porting `ggml-quants.c`'s `dequantize_row_iq4_xs`: sub-block
/// `i` (32 values) uses `d * (scale - 32)`, its 6-bit scale's low nibble
/// from `scales_l[i / 2]`, high bit pair from bits `2 i` of `scales_h`.
#[doc(hidden)]
pub fn dequantize_iq4_xs(raw: &[u8]) -> Vec<f32> {
    let mut out =
        Vec::with_capacity(raw.len() / IQ4_XS_BLOCK_BYTES as usize * QK_BLOCK_ELEMENTS as usize);
    for block in raw.as_chunks::<{ IQ4_XS_BLOCK_BYTES as usize }>().0 {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for (i, scale) in iq4_xs_scales(&block[2..8]).into_iter().enumerate() {
            iq4_values(
                d * f32::from(scale),
                &block[8 + 16 * i..24 + 16 * i],
                &mut out,
            );
        }
    }
    out
}

/// `ksigns_iq2xs[index]`: the 7-bit sign pattern `index` plus an eighth
/// sign bit making the count of negative values even.
pub const fn iq_signs(index: u32) -> u8 {
    let index = index & 0x7F;
    (index | ((index.count_ones() & 1) << 7)) as u8
}

/// Pushes the eight values of a grid entry (`grid`, one per byte), times
/// `scale`, negated where `signs` has a bit set.
#[doc(hidden)]
pub fn push_signed_grid(scale: f32, grid: u64, signs: u8, out: &mut Vec<f32>) {
    for j in 0..8 {
        let value = scale * f32::from((grid >> (8 * j)) as u8);
        out.push(if signs >> j & 1 == 1 { -value } else { value });
    }
}

#[doc(hidden)]
pub fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("four bytes"))
}

/// Dequantizes `block_iq2_xxs` super-blocks (66 bytes), porting
/// `ggml-quants.c`'s `dequantize_row_iq2_xxs`: per 32 values, two words --
/// four grid indices, then four 7-bit sign indices and a 4-bit scale
/// (`d * (0.5 + scale) * 0.25`).
#[doc(hidden)]
pub fn dequantize_iq2_xxs(raw: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(raw.len() / IQ2_XXS_BLOCK_BYTES as usize * 256);
    for block in raw.as_chunks::<{ IQ2_XXS_BLOCK_BYTES as usize }>().0 {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for ib in 0..8 {
            let grids = u32_at(block, 2 + 8 * ib);
            let aux = u32_at(block, 6 + 8 * ib);
            let db = d * (0.5 + (aux >> 28) as f32) * 0.25;
            for l in 0..4 {
                let grid = iq_grids::IQ2XXS_GRID[((grids >> (8 * l)) & 0xFF) as usize];
                push_signed_grid(db, grid, iq_signs(aux >> (7 * l)), &mut out);
            }
        }
    }
    out
}

/// Dequantizes `block_iq2_xs` super-blocks (74 bytes), porting
/// `dequantize_row_iq2_xs`: 16-bit codes (9-bit grid index, 7-bit sign
/// index), then a 4-bit scale per 16 values.
#[doc(hidden)]
pub fn dequantize_iq2_xs(raw: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(raw.len() / IQ2_XS_BLOCK_BYTES as usize * 256);
    for block in raw.as_chunks::<{ IQ2_XS_BLOCK_BYTES as usize }>().0 {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for ib in 0..8 {
            for l in 0..4 {
                let at = 2 + 2 * (4 * ib + l);
                let code = u32::from(u16::from_le_bytes([block[at], block[at + 1]]));
                let scale = (block[66 + ib] >> (4 * (l / 2))) & 0xF;
                let db = d * (0.5 + f32::from(scale)) * 0.25;
                let grid = iq_grids::IQ2XS_GRID[(code & 511) as usize];
                push_signed_grid(db, grid, iq_signs(code >> 9), &mut out);
            }
        }
    }
    out
}

/// Dequantizes `block_iq2_s` super-blocks (82 bytes), porting
/// `dequantize_row_iq2_s`: 10-bit grid indices (low bytes, then two high
/// bits each in `qh`), one explicit sign byte per 8 values, a 4-bit scale
/// per 16 values.
#[doc(hidden)]
pub fn dequantize_iq2_s(raw: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(raw.len() / IQ2_S_BLOCK_BYTES as usize * 256);
    for block in raw.as_chunks::<{ IQ2_S_BLOCK_BYTES as usize }>().0 {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for ib in 0..8 {
            for l in 0..4 {
                let index = usize::from(block[2 + 4 * ib + l])
                    | ((usize::from(block[66 + ib]) >> (2 * l)) & 3) << 8;
                let scale = (block[74 + ib] >> (4 * (l / 2))) & 0xF;
                let db = d * (0.5 + f32::from(scale)) * 0.25;
                let grid = iq_grids::IQ2S_GRID[index];
                push_signed_grid(db, grid, block[34 + 4 * ib + l], &mut out);
            }
        }
    }
    out
}

/// Dequantizes `block_iq3_xxs` super-blocks (98 bytes), porting
/// `dequantize_row_iq3_xxs`: 4-value grid indices, then per 32 values a
/// word of four 7-bit sign indices and a 4-bit scale (`d * (0.5 + scale)
/// * 0.5`).
#[doc(hidden)]
pub fn dequantize_iq3_xxs(raw: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(raw.len() / IQ3_XXS_BLOCK_BYTES as usize * 256);
    for block in raw.as_chunks::<{ IQ3_XXS_BLOCK_BYTES as usize }>().0 {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for ib in 0..8 {
            let aux = u32_at(block, 66 + 4 * ib);
            let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
            for l in 0..4 {
                let low = iq_grids::IQ3XXS_GRID[usize::from(block[2 + 8 * ib + 2 * l])];
                let high = iq_grids::IQ3XXS_GRID[usize::from(block[3 + 8 * ib + 2 * l])];
                let grid = u64::from(low) | (u64::from(high) << 32);
                push_signed_grid(db, grid, iq_signs(aux >> (7 * l)), &mut out);
            }
        }
    }
    out
}

/// Dequantizes `block_iq3_s` super-blocks (110 bytes), porting
/// `dequantize_row_iq3_s`: 9-bit 4-value grid indices (low bytes, high
/// bits in `qh`), one explicit sign byte per 8 values, a 4-bit scale per
/// 32 values (`d * (1 + 2 scale)`).
#[doc(hidden)]
pub fn dequantize_iq3_s(raw: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(raw.len() / IQ3_S_BLOCK_BYTES as usize * 256);
    for block in raw.as_chunks::<{ IQ3_S_BLOCK_BYTES as usize }>().0 {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for ib in 0..8 {
            let scale = (block[106 + ib / 2] >> (4 * (ib % 2))) & 0xF;
            let db = d * f32::from(1 + 2 * scale);
            let index = |i: usize| {
                usize::from(block[2 + 8 * ib + i]) | ((usize::from(block[66 + ib]) >> i) & 1) << 8
            };
            for l in 0..4 {
                let low = iq_grids::IQ3S_GRID[index(2 * l)];
                let high = iq_grids::IQ3S_GRID[index(2 * l + 1)];
                let grid = u64::from(low) | (u64::from(high) << 32);
                push_signed_grid(db, grid, block[74 + 4 * ib + l], &mut out);
            }
        }
    }
    out
}

/// A `block_iq4_xs`'s eight sub-block scales with their -32 applied, from
/// its `scales_h` (2 bytes) and `scales_l` (4 bytes).
pub fn iq4_xs_scales(packed: &[u8]) -> [i8; 8] {
    let high = u16::from_le_bytes([packed[0], packed[1]]);
    let mut scales = [0_i8; 8];
    for (i, scale) in scales.iter_mut().enumerate() {
        let low = (packed[2 + i / 2] >> (4 * (i % 2))) & 0xF;
        let top = ((high >> (2 * i)) & 3) as u8;
        *scale = i8::try_from(low | (top << 4)).expect("six bits") - 32;
    }
    scales
}

/// A `block_q3_K`'s sixteen 6-bit sub-block scales (0..63, before the -32
/// offset): scale `i`'s low nibble is nibble `i / 8` of byte `i % 8`, its
/// high bit pair bits `2 (i / 4)` of byte `8 + i % 4`.
pub fn q3_k_scales(packed: &[u8]) -> [i8; 16] {
    let mut scales = [0_i8; 16];
    for (i, scale) in scales.iter_mut().enumerate() {
        let low = (packed[i % 8] >> (4 * (i / 8))) & 0xF;
        let high = (packed[8 + i % 4] >> (2 * (i / 4))) & 3;
        *scale = i8::try_from(low | (high << 4)).expect("six bits");
    }
    scales
}

/// Dequantizes one `block_q5_K` region (`raw.len()` a multiple of 176
/// bytes) into `f32` values, porting `ggml-quants.c`'s
/// `dequantize_row_q5_K` bit-for-bit. Identical sub-block scale/min
/// structure to [`dequantize_q4_k`], plus a 32-byte `qh` high-bit plane
/// (fixed across the whole 256-element super-block, never advanced per
/// 64-chunk) reconstructing a 5-bit value: `q5[i] = (ql[i] & 0xF) | (qh[i]
/// & bit ? 16 : 0)`, `value[i] = d_sb * q5[i] - m_sb`. The selected `qh`
/// bit-pair rotates `(1,2) -> (4,8) -> (16,32) -> (64,128)` across the four
/// 64-element chunks.
#[doc(hidden)]
pub fn dequantize_q5_k(raw: &[u8]) -> Vec<f32> {
    let mut out =
        Vec::with_capacity(raw.len() / Q5_K_BLOCK_BYTES as usize * QK_BLOCK_ELEMENTS as usize);
    for block in raw.as_chunks::<{ Q5_K_BLOCK_BYTES as usize }>().0 {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales: [u8; K_SCALE_SIZE] = block[4..4 + K_SCALE_SIZE].try_into().unwrap();
        let qh = &block[4 + K_SCALE_SIZE..4 + K_SCALE_SIZE + 32];
        let qs = &block[4 + K_SCALE_SIZE + 32..4 + K_SCALE_SIZE + 32 + 128];

        let mut is = 0usize;
        let (mut u1, mut u2) = (1u8, 2u8);
        for chunk_start in (0..QK_BLOCK_ELEMENTS as usize).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, &scales);
            let (d1, min1) = (d * f32::from(sc1), dmin * f32::from(m1));
            let (sc2, m2) = get_scale_min_k4(is + 1, &scales);
            let (d2, min2) = (d * f32::from(sc2), dmin * f32::from(m2));
            let ql = &qs[chunk_start / 2..chunk_start / 2 + 32];
            out.extend(ql.iter().zip(qh).map(|(&byte, &high)| {
                let value = (byte & 0xF) | if high & u1 != 0 { 16 } else { 0 };
                d1 * f32::from(value) - min1
            }));
            out.extend(ql.iter().zip(qh).map(|(&byte, &high)| {
                let value = (byte >> 4) | if high & u2 != 0 { 16 } else { 0 };
                d2 * f32::from(value) - min2
            }));
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    out
}

/// Converts one IEEE 754 binary16 ("half float") value to `f32`, exactly
/// (every `f16` value has an exact `f32` representation, since `f32`'s
/// exponent and mantissa ranges both strictly contain `f16`'s). Handles
/// every `f16` class explicitly rather than relying on a bit-shift
/// shortcut that only works for normalized values: `+0.0`/`-0.0` preserve
/// sign, subnormals are renormalized into `f32`'s wider exponent range
/// (never flushed to zero), `+Inf`/`-Inf` map to `f32` infinities, and
/// `NaN` maps to a `f32` `NaN` preserving the sign bit and (widened)
/// mantissa payload rather than canonicalizing to one fixed `NaN` pattern.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exponent = (bits >> 10) & 0x1F;
    let mantissa = u32::from(bits & 0x3FF);
    let magnitude_bits = if exponent == 0 {
        if mantissa == 0 {
            0
        } else {
            // Subnormal `f16`: renormalize by shifting the mantissa left
            // until its implicit leading bit would land at position 10,
            // counting how many shifts that took to compute the correct
            // (negative, then rebiased) `f32` exponent.
            let mut mantissa = mantissa;
            let mut shift = 0u32;
            while mantissa & 0x400 == 0 {
                mantissa <<= 1;
                shift += 1;
            }
            mantissa &= 0x3FF;
            let f32_exponent = 127 - 15 - shift + 1;
            (f32_exponent << 23) | (mantissa << 13)
        }
    } else if exponent == 0x1F {
        // Infinity (mantissa == 0) or NaN (mantissa != 0): `f32`'s
        // all-ones exponent field means the same thing.
        (0xFFu32 << 23) | (mantissa << 13)
    } else {
        // `exponent` (1..=30) minus f16's bias (15) can be negative before
        // rebiasing into f32's own (127) -- must go through signed
        // arithmetic, unlike the always-non-negative subnormal branch
        // above, or a small-but-normal f16 exponent (e.g. 14, for `0.5`)
        // underflows this as `u32` subtraction.
        let f32_exponent = (i32::from(exponent) - 15 + 127) as u32;
        (f32_exponent << 23) | (mantissa << 13)
    };
    f32::from_bits(sign | magnitude_bits)
}

/// Converts one `bfloat16` value to `f32`, exactly: `bfloat16` is defined
/// as `f32`'s sign/exponent/top-7-mantissa-bits truncated to 16 bits, so
/// widening back is a zero-extending left shift with no case analysis
/// needed -- `+0.0`/`-0.0`/subnormals/infinities/`NaN` all round-trip
/// correctly through this shift because they do not depend on any bits
/// `bfloat16` actually discards (the low 16 mantissa bits, which this
/// shift fills with zero).
#[doc(hidden)]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Converts one `f32` value to IEEE 754 binary16 ("half float"), rounding
/// to nearest with ties to even -- the inverse of [`f16_to_f32`], needed by
/// a native half-precision device buffer to convert host `f32` values into
/// on-device `f16` bytes. Handles every class explicitly: `+0.0`/`-0.0`
/// preserve sign, a magnitude below `f16`'s smallest subnormal flushes to
/// (signed) zero, a magnitude at or above `f16`'s subnormal range rounds
/// into a subnormal `f16` (rounding up through the subnormal/normal
/// boundary lands on the correct smallest-normal bit pattern automatically,
/// since IEEE 754's subnormal and normal encodings are bit-contiguous by
/// design), a magnitude beyond `f16`'s largest finite value rounds to
/// infinity, `f32` infinities map to `f16` infinities, and `f32` `NaN`
/// leaves a non-zero payload (never accidentally collapsing to infinity).
///
/// Unused outside tests today: this is Phase 1 groundwork for a future
/// native half-precision device buffer (`add-native-cuda-half-precision-
/// compute` design.md's Phase 2), which will convert host `f32` values to
/// on-device `f16` bytes at the upload boundary.
#[allow(dead_code)]
#[doc(hidden)]
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mantissa_f32 = bits & 0x007F_FFFF;
    let exp_f32 = ((bits >> 23) & 0xFF) as i32;

    if exp_f32 == 0xFF {
        if mantissa_f32 == 0 {
            return sign | 0x7C00;
        }
        let payload = (mantissa_f32 >> 13) as u16;
        let payload = if payload == 0 { 1 } else { payload };
        return sign | 0x7C00 | payload;
    }

    let unbiased = exp_f32 - 127;

    if unbiased > 15 {
        return sign | 0x7C00;
    }

    if unbiased < -14 {
        // Subnormal (or flush-to-zero, or rounds up into the smallest
        // normal) f16 output. `significand` includes the implicit leading
        // one (absent when `exp_f32 == 0`, i.e. the input itself was
        // already an `f32` subnormal).
        let significand = if exp_f32 == 0 {
            mantissa_f32
        } else {
            mantissa_f32 | 0x0080_0000
        };
        if significand == 0 {
            return sign;
        }
        let shift = (-(unbiased + 1)) as u32;
        if shift >= 32 {
            return sign;
        }
        let half = 1u32 << (shift - 1);
        let mask = (1u32 << shift) - 1;
        let truncated = significand >> shift;
        let remainder = significand & mask;
        let mut result = truncated;
        if remainder > half || (remainder == half && (result & 1) == 1) {
            result += 1;
        }
        return sign | (result as u16);
    }

    // Normal f16 output: round `mantissa_f32`'s top 10 bits (no implicit
    // bit -- f16, like f32, never stores it) to nearest-even.
    let half = 1u32 << 12;
    let mask = (1u32 << 13) - 1;
    let truncated = mantissa_f32 >> 13;
    let remainder = mantissa_f32 & mask;
    let mut mantissa10 = truncated;
    if remainder > half || (remainder == half && (mantissa10 & 1) == 1) {
        mantissa10 += 1;
    }
    let exp16 = (unbiased + 15) as u32;
    if mantissa10 == 0x400 {
        let new_exp = exp16 + 1;
        if new_exp >= 0x1F {
            return sign | 0x7C00;
        }
        return sign | ((new_exp as u16) << 10);
    }
    sign | ((exp16 as u16) << 10) | (mantissa10 as u16)
}

/// Converts one `f32` value to `bfloat16`, rounding to nearest with ties to
/// even -- the inverse of [`bf16_to_f32`]. `bfloat16` truncates `f32`'s
/// sign/exponent/top-7-mantissa-bits to 16 bits, so rounding is the
/// standard "add a rounding bias, then truncate" trick: adding `0x7FFF`
/// plus the bit that would become the rounded result's LSB, then
/// right-shifting away the low 16 bits, rounds to nearest and breaks exact
/// ties toward an even mantissa -- with no case analysis needed, exactly
/// like [`bf16_to_f32`]'s own unconditional shift, because this bias
/// addition cannot overflow into the sign bit for any finite input (the
/// worst case, `f32::MAX`, still leaves headroom below `f32`'s all-ones
/// exponent field). `NaN` is handled explicitly so a `NaN` whose rounded
/// mantissa would truncate to zero cannot collapse into infinity.
///
/// Unused outside tests today: Phase 1 groundwork, same as [`f32_to_f16`].
#[allow(dead_code)]
#[doc(hidden)]
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        let sign = ((bits >> 16) & 0x8000) as u16;
        let mantissa7 = ((bits >> 16) as u16) & 0x7F;
        let mantissa7 = if mantissa7 == 0 { 0x40 } else { mantissa7 };
        return sign | 0x7F80 | mantissa7;
    }
    let rounding_bias = 0x0000_7FFFu32 + ((bits >> 16) & 1);
    let rounded = bits.wrapping_add(rounding_bias);
    (rounded >> 16) as u16
}
