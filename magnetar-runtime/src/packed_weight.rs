//! Packed (block-quantized) Model Instance weights
//! (`execute-direct-quantized-weights`, B03).
//!
//! A packed weight keeps a checkpoint's validated quantized bytes as they
//! were stored instead of expanding them to `f32` at load. Model Loading
//! offers one to the Instance's Provider through
//! [`crate::provider::ProviderExecutionApi::write_packed_weight_tensor`]; a
//! Provider that computes directly from the format stores it, any other
//! declines, and Model Loading then dequantizes it to `f32` exactly as
//! before. Which of the two happened is never hidden: the Provider's answer
//! is the only thing that decides it.
//!
//! The packed bytes describe a 2D matrix *as stored*: `stored_rows` rows of
//! `stored_cols` values, each row a run of consecutive blocks. The matrix
//! the graph sees is either that matrix or, when `transposed`, its
//! transpose (a checkpoint stores a projection as `[out, in]` with its
//! blocks along `in`; the graph multiplies by `[in, out]`).

use crate::model_loading::f16_to_f32;
use crate::tensor::HostTensor;

/// [`crate::model::ModelTensorMetadata::layout`] value declaring that a
/// tensor's bytes store the transpose of its declared (logical) shape:
/// a 2D tensor declared `[rows, cols]` is stored as `[cols, rows]`.
pub const TRANSPOSED_STORAGE_LAYOUT: &str = "transposed";

/// A block quantization format a packed weight may use.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PackedWeightFormat {
    /// GGUF `block_q8_0`: 32 values per block, an `f16` scale `d` followed
    /// by 32 signed 8-bit quants; `value[i] = d * q[i]`.
    Q8_0,
    /// GGUF `block_q4_K`: 256 values per super-block, 144 bytes -- `f16`
    /// `d` and `dmin`, twelve bytes packing eight 6-bit sub-block scales and
    /// mins, then 128 bytes of 4-bit quants; `value = d * scale * q - dmin
    /// * min` per 32-value sub-block.
    Q4K,
    /// GGUF `block_q6_K`: 256 values per super-block, 210 bytes -- 128
    /// bytes of low 4-bit quants, 64 bytes of high 2-bit quants, sixteen
    /// signed 8-bit sub-block scales, then an `f16` `d`; `value = d *
    /// scale * (q - 32)` per 16-value sub-block.
    Q6K,
}

impl PackedWeightFormat {
    pub const fn block_elements(self) -> u64 {
        match self {
            Self::Q8_0 => 32,
            Self::Q4K | Self::Q6K => 256,
        }
    }

    pub const fn block_bytes(self) -> u64 {
        match self {
            Self::Q8_0 => 34,
            Self::Q4K => 144,
            Self::Q6K => 210,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Q8_0 => "q8_0",
            Self::Q4K => "q4_k",
            Self::Q6K => "q6_k",
        }
    }
}

/// A validated packed matrix; see the module documentation.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedWeight {
    format: PackedWeightFormat,
    stored_rows: u64,
    stored_cols: u64,
    transposed: bool,
    bytes: Vec<u8>,
}

impl PackedWeight {
    /// Validates that `bytes` holds exactly `stored_rows` rows of
    /// `stored_cols` values in `format`, each row a whole number of blocks.
    pub fn new(
        format: PackedWeightFormat,
        stored_rows: u64,
        stored_cols: u64,
        transposed: bool,
        bytes: Vec<u8>,
    ) -> Result<Self, String> {
        if stored_rows == 0 || stored_cols == 0 {
            return Err("a packed weight needs at least one row and one column".into());
        }
        if !stored_cols.is_multiple_of(format.block_elements()) {
            return Err(format!(
                "a {} row of {stored_cols} values is not a whole number of {}-value blocks",
                format.as_str(),
                format.block_elements()
            ));
        }
        let expected = (stored_cols / format.block_elements())
            .checked_mul(format.block_bytes())
            .and_then(|row_bytes| row_bytes.checked_mul(stored_rows))
            .ok_or_else(|| "packed weight byte size overflowed".to_string())?;
        if bytes.len() as u64 != expected {
            return Err(format!(
                "a {stored_rows}x{stored_cols} {} weight needs {expected} bytes, got {}",
                format.as_str(),
                bytes.len()
            ));
        }
        Ok(Self {
            format,
            stored_rows,
            stored_cols,
            transposed,
            bytes,
        })
    }

    pub fn format(&self) -> PackedWeightFormat {
        self.format
    }

    pub fn stored_rows(&self) -> u64 {
        self.stored_rows
    }

    pub fn stored_cols(&self) -> u64 {
        self.stored_cols
    }

    pub fn transposed(&self) -> bool {
        self.transposed
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The shape the graph sees: `[stored_cols, stored_rows]` when
    /// transposed, `[stored_rows, stored_cols]` otherwise.
    pub fn logical_shape(&self) -> [u64; 2] {
        if self.transposed {
            [self.stored_cols, self.stored_rows]
        } else {
            [self.stored_rows, self.stored_cols]
        }
    }

    /// The stored values in row-major stored order, dequantized exactly as
    /// `ggml-quants.c`'s `dequantize_row_q8_0` / `dequantize_row_q4_K` / `dequantize_row_q6_K`
    /// do.
    pub fn dequantize_stored(&self) -> Vec<f32> {
        match self.format {
            PackedWeightFormat::Q4K => crate::model_loading::dequantize_q4_k(&self.bytes),
            PackedWeightFormat::Q6K => crate::model_loading::dequantize_q6_k(&self.bytes),
            PackedWeightFormat::Q8_0 => self
                .bytes
                .as_chunks::<34>()
                .0
                .iter()
                .flat_map(|block| {
                    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                    block[2..].iter().map(move |&q| d * f32::from(q as i8))
                })
                .collect(),
        }
    }

    /// The logical matrix as `f32`: what a Provider that declines the packed
    /// weight receives instead.
    pub fn dequantize(&self) -> HostTensor {
        let stored = self.dequantize_stored();
        let shape = self.logical_shape();
        let data = if self.transposed {
            transpose(
                &stored,
                self.stored_rows as usize,
                self.stored_cols as usize,
            )
        } else {
            stored
        };
        HostTensor::new(shape.to_vec(), data)
            .expect("a validated packed weight dequantizes to its own element count")
    }
}

/// The row-major transpose of a `rows x cols` matrix.
pub(crate) fn transpose(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; values.len()];
    for row in 0..rows {
        for col in 0..cols {
            out[col * rows + row] = values[row * cols + col];
        }
    }
    out
}

/// A Provider's answer to [`crate::provider::ProviderExecutionApi::write_packed_weight_tensor`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackedWeightWrite {
    /// The Provider stored the packed bytes and computes from them.
    Stored,
    /// The Provider does not compute from this format; nothing was stored.
    Declined,
}

#[cfg(test)]
mod tests;
