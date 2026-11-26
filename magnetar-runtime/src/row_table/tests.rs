//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use std::sync::Mutex;

/// A mapping over memory that records the prefetch hints it receives.
#[derive(Debug)]
struct MemoryPayload {
    bytes: Vec<u8>,
    hints: Mutex<Vec<Vec<(u64, u64)>>>,
}

impl MemoryPayload {
    fn new(bytes: Vec<u8>) -> Arc<Self> {
        Arc::new(Self {
            bytes,
            hints: Mutex::new(Vec::new()),
        })
    }
}

impl MappedPayload for MemoryPayload {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<(), ProductionIngestionError> {
        let start = offset as usize;
        let slice = self.bytes.get(start..start + out.len()).ok_or_else(|| {
            ProductionIngestionError::PayloadOutOfBounds {
                identity: "memory".into(),
            }
        })?;
        out.copy_from_slice(slice);
        Ok(())
    }

    fn prefetch(&self, spans: &[(u64, u64)]) {
        self.hints.lock().unwrap().push(spans.to_vec());
    }
}

fn f32_table(rows: u64, columns: u64) -> (Arc<MemoryPayload>, Vec<f32>) {
    let values: Vec<f32> = (0..rows * columns)
        .map(|index| index as f32 * 0.25 - 7.0)
        .collect();
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    (MemoryPayload::new(bytes), values)
}

#[test]
fn rows_are_read_in_the_order_asked_and_repeated_when_asked_twice() {
    let (payload, values) = f32_table(10, 4);
    let table = RowTable::new(payload, 10, 4, RowTableDType::F32).unwrap();
    let rows = table.read_rows(&[7, 0, 7, 9]).unwrap();
    let expected: Vec<f32> = [7usize, 0, 7, 9]
        .iter()
        .flat_map(|&row| values[row * 4..(row + 1) * 4].to_vec())
        .collect();
    assert_eq!(rows, expected);
    assert!(table.read_rows(&[]).unwrap().is_empty());
}

#[test]
fn half_precision_rows_convert_to_f32() {
    // bf16 1.5 is 0x3FC0, f16 1.5 is 0x3E00; each row holds [1.5, -2.0].
    for (dtype, one_and_a_half, minus_two) in [
        (RowTableDType::Bf16, 0x3FC0u16, 0xC000u16),
        (RowTableDType::F16, 0x3E00u16, 0xC000u16),
    ] {
        let bytes: Vec<u8> = (0..3)
            .flat_map(|_| [one_and_a_half, minus_two])
            .flat_map(u16::to_le_bytes)
            .collect();
        let table = RowTable::new(MemoryPayload::new(bytes), 3, 2, dtype).unwrap();
        assert_eq!(
            table.read_rows(&[2, 0]).unwrap(),
            [1.5, -2.0, 1.5, -2.0],
            "{dtype:?}"
        );
    }
}

#[test]
fn a_mapping_that_does_not_fit_the_declared_shape_is_refused() {
    let (payload, _) = f32_table(10, 4);
    for (rows, columns) in [(9, 4), (10, 5), (0, 4)] {
        assert!(
            RowTable::new(payload.clone(), rows, columns, RowTableDType::F32).is_err(),
            "{rows} x {columns}"
        );
    }
    assert!(RowTable::new(payload, 20, 4, RowTableDType::Bf16).is_ok());
}

#[test]
fn a_row_past_the_table_is_reported() {
    let (payload, _) = f32_table(10, 4);
    let table = RowTable::new(payload, 10, 4, RowTableDType::F32).unwrap();
    assert!(matches!(
        table.read_rows(&[3, 10]),
        Err(RowTableError::OutOfRange { row: 10, rows: 10 })
    ));
}

/// Prefetch sorts the rows' pages, merges the ones that touch and rounds to
/// the page size, so many rows become few reads.
#[test]
fn prefetch_hands_the_mapping_sorted_merged_page_spans() {
    // 1,000 rows of 256 bytes: 16 rows per page.
    let (payload, _) = f32_table(1_000, 64);
    let table = RowTable::new(payload.clone(), 1_000, 64, RowTableDType::F32).unwrap();
    // Rows 0 and 5 share page 0, row 16 is page 1 (touching page 0), row 300
    // is page 18, and 5 repeats.
    table.prefetch(&[300, 16, 5, 0, 5, 5_000]);
    let hints = payload.hints.lock().unwrap();
    assert_eq!(hints.len(), 1);
    assert_eq!(hints[0], vec![(0, 2 * PAGE), (18 * PAGE, PAGE)]);
}

#[test]
fn the_last_span_stops_at_the_end_of_the_mapping() {
    // 10 rows of 16 bytes: 160 bytes, less than a page.
    let (payload, _) = f32_table(10, 4);
    let table = RowTable::new(payload.clone(), 10, 4, RowTableDType::F32).unwrap();
    table.prefetch(&[9]);
    assert_eq!(payload.hints.lock().unwrap()[0], vec![(0, 160)]);
}

#[test]
fn two_handles_to_one_table_are_equal_and_two_tables_are_not() {
    let (payload, _) = f32_table(4, 2);
    let a = Arc::new(RowTable::new(payload.clone(), 4, 2, RowTableDType::F32).unwrap());
    let b = Arc::new(RowTable::new(payload, 4, 2, RowTableDType::F32).unwrap());
    assert_eq!(RowTableHandle(a.clone()), RowTableHandle(a.clone()));
    assert_ne!(RowTableHandle(a), RowTableHandle(b));
}
