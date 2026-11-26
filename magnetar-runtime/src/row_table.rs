//! A table read by row and never held whole (`support-qwen38-flash-next`,
//! design D8). Qwen4-Exp's n-gram embedding table has about 320 million
//! rows: it stays in its artifact, and a step reads only the rows it needs
//! through a [`MappedPayload`] the payload source owns.
//!
//! A step's rows are known before the step runs (the hash depends on token
//! ids alone), so [`RowTable::prefetch`] queues every read at once, sorted
//! and merged by page; a batched prefetch measured six times faster than the
//! best threaded `pread` on a cold cache (design D8). [`RowTable::read_rows`]
//! then copies each row out and converts it to `f32`.

use std::sync::Arc;

use crate::production_model_ingestion::{MappedPayload, ProductionIngestionError};
use crate::{ModelDType, quantization::bf16_to_f32, quantization::f16_to_f32};

/// The operating system's read granularity: prefetch spans are rounded to it
/// so that rows sharing a page become one read.
const PAGE: u64 = 4096;

/// The storage types a table row may have.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RowTableDType {
    F32,
    F16,
    Bf16,
}

impl RowTableDType {
    /// The row type for a tensor's storage type, `None` for a type with no
    /// plain per-element form (a quantized table is read dequantized
    /// instead).
    pub fn from_model(dtype: ModelDType) -> Option<Self> {
        match dtype {
            ModelDType::F32 => Some(Self::F32),
            ModelDType::F16 => Some(Self::F16),
            ModelDType::Bf16 => Some(Self::Bf16),
            _ => None,
        }
    }

    /// Bytes per element.
    pub const fn element_bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }
}

/// Why a row could not be read.
#[derive(Debug)]
pub enum RowTableError {
    /// The declared geometry does not match the mapping.
    Shape(String),
    /// A row index is past the table.
    OutOfRange { row: u64, rows: u64 },
    /// The mapping failed to read.
    Read(ProductionIngestionError),
}

impl std::fmt::Display for RowTableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape(reason) => write!(f, "row table shape: {reason}"),
            Self::OutOfRange { row, rows } => {
                write!(f, "row {row} is past the table's {rows} rows")
            }
            Self::Read(error) => write!(f, "row table read failed: {error:?}"),
        }
    }
}

impl std::error::Error for RowTableError {}

/// A `[rows, columns]` table whose bytes stay in their artifact.
#[derive(Debug)]
pub struct RowTable {
    payload: Arc<dyn MappedPayload>,
    rows: u64,
    columns: u64,
    dtype: RowTableDType,
}

impl RowTable {
    /// A table over `payload`, which must hold exactly `rows * columns`
    /// elements of `dtype`.
    pub fn new(
        payload: Arc<dyn MappedPayload>,
        rows: u64,
        columns: u64,
        dtype: RowTableDType,
    ) -> Result<Self, RowTableError> {
        let expected = rows
            .checked_mul(columns)
            .and_then(|elements| elements.checked_mul(dtype.element_bytes()))
            .ok_or_else(|| RowTableError::Shape("the table's byte size overflows".into()))?;
        if rows == 0 || columns == 0 || payload.len() != expected {
            return Err(RowTableError::Shape(format!(
                "{rows} x {columns} {dtype:?} needs {expected} bytes, the mapping has {}",
                payload.len()
            )));
        }
        Ok(Self {
            payload,
            rows,
            columns,
            dtype,
        })
    }

    pub const fn rows(&self) -> u64 {
        self.rows
    }

    pub const fn columns(&self) -> u64 {
        self.columns
    }

    fn row_bytes(&self) -> u64 {
        self.columns * self.dtype.element_bytes()
    }

    /// Hints that `rows` are about to be read: the pages they live in,
    /// sorted and merged, go to the mapping at once. Rows past the table are
    /// ignored (the read that follows reports them).
    pub fn prefetch(&self, rows: &[u64]) {
        let row_bytes = self.row_bytes();
        let mut spans: Vec<(u64, u64)> = rows
            .iter()
            .filter(|&&row| row < self.rows)
            .map(|&row| {
                let start = row * row_bytes;
                (
                    start / PAGE * PAGE,
                    (start + row_bytes).div_ceil(PAGE) * PAGE,
                )
            })
            .collect();
        spans.sort_unstable();
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(spans.len());
        for (start, end) in spans {
            match merged.last_mut() {
                Some(last) if start <= last.1 => last.1 = last.1.max(end),
                _ => merged.push((start, end)),
            }
        }
        let total = self.payload.len();
        let spans: Vec<(u64, u64)> = merged
            .into_iter()
            .map(|(start, end)| (start, end.min(total) - start))
            .collect();
        self.payload.prefetch(&spans);
    }

    /// The rows `rows`, in that order, as `f32`: `rows.len() * columns`
    /// values.
    pub fn read_rows(&self, rows: &[u64]) -> Result<Vec<f32>, RowTableError> {
        let columns = self.columns as usize;
        let mut out = Vec::with_capacity(rows.len() * columns);
        let mut bytes = vec![0u8; self.row_bytes() as usize];
        for &row in rows {
            if row >= self.rows {
                return Err(RowTableError::OutOfRange {
                    row,
                    rows: self.rows,
                });
            }
            self.payload
                .read_at(row * self.row_bytes(), &mut bytes)
                .map_err(RowTableError::Read)?;
            match self.dtype {
                RowTableDType::F32 => out.extend(
                    bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|chunk| f32::from_le_bytes(*chunk)),
                ),
                RowTableDType::F16 => out.extend(
                    bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|chunk| f16_to_f32(u16::from_le_bytes(*chunk))),
                ),
                RowTableDType::Bf16 => out.extend(
                    bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|chunk| bf16_to_f32(u16::from_le_bytes(*chunk))),
                ),
            }
        }
        Ok(out)
    }
}

/// A shared handle to a [`RowTable`] that two handles to the same table
/// compare equal through (a Model Instance's bindings compare as a whole).
#[derive(Clone, Debug)]
pub struct RowTableHandle(pub Arc<RowTable>);

impl PartialEq for RowTableHandle {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for RowTableHandle {}

#[cfg(test)]
mod tests;
