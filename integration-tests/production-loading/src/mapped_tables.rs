//! A payload source that maps the n-gram tables of a checkpoint instead of
//! letting the Runtime read and stage them (`support-qwen38-flash-next`,
//! task 4.4, design D8): the Runtime then holds no copy of a table and reads
//! only the rows a step names. The mapping serves the same bytes the
//! loader's source reads, from memory, and counts what is asked of it.

use magnetar_runtime::production_model_ingestion::{
    MappedPayload, ProductionArtifactPayloadSource, ProductionIngestionError,
    ProductionPayloadRange,
};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

/// What the Runtime asked of the tables, shared with the test.
#[derive(Debug, Default)]
pub(crate) struct TableUse {
    /// The tensors the Runtime asked to map.
    pub(crate) mapped: Mutex<Vec<String>>,
    /// The tensors the Runtime read whole (a staged table would be one).
    pub(crate) read_whole: Mutex<Vec<String>>,
    /// Row reads served by the mappings.
    pub(crate) reads: AtomicU64,
    /// Bytes those reads copied.
    pub(crate) bytes_read: AtomicU64,
    /// Prefetch hints, and the spans they carried.
    pub(crate) prefetches: AtomicU64,
    pub(crate) prefetched_spans: AtomicU64,
}

/// `inner`, with every tensor whose name ends in `suffix` mappable.
pub(crate) struct MappedTables {
    inner: Arc<dyn ProductionArtifactPayloadSource>,
    suffix: &'static str,
    usage: Arc<TableUse>,
}

impl MappedTables {
    pub(crate) fn new(
        inner: Arc<dyn ProductionArtifactPayloadSource>,
        suffix: &'static str,
    ) -> (Self, Arc<TableUse>) {
        let usage = Arc::new(TableUse::default());
        (
            Self {
                inner,
                suffix,
                usage: Arc::clone(&usage),
            },
            usage,
        )
    }
}

#[derive(Debug)]
struct MemoryMapping {
    bytes: Vec<u8>,
    usage: Arc<TableUse>,
}

impl MappedPayload for MemoryMapping {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<(), ProductionIngestionError> {
        let start = offset as usize;
        let from = self.bytes.get(start..start + out.len()).ok_or_else(|| {
            ProductionIngestionError::PayloadOutOfBounds {
                identity: "mapped table".into(),
            }
        })?;
        out.copy_from_slice(from);
        self.usage.reads.fetch_add(1, Ordering::Relaxed);
        self.usage
            .bytes_read
            .fetch_add(out.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    fn prefetch(&self, spans: &[(u64, u64)]) {
        self.usage.prefetches.fetch_add(1, Ordering::Relaxed);
        self.usage
            .prefetched_spans
            .fetch_add(spans.len() as u64, Ordering::Relaxed);
    }
}

impl ProductionArtifactPayloadSource for MappedTables {
    fn read_payload(
        &self,
        range: &ProductionPayloadRange,
    ) -> Result<Vec<u8>, ProductionIngestionError> {
        if range.identity.ends_with(self.suffix) {
            self.usage
                .read_whole
                .lock()
                .unwrap()
                .push(range.identity.clone());
        }
        self.inner.read_payload(range)
    }

    fn map_payload(
        &self,
        range: &ProductionPayloadRange,
    ) -> Result<Option<Arc<dyn MappedPayload>>, ProductionIngestionError> {
        if !range.identity.ends_with(self.suffix) {
            return Ok(None);
        }
        self.usage
            .mapped
            .lock()
            .unwrap()
            .push(range.identity.clone());
        // The mapping's bytes come from the loader's own source, which
        // verifies the range's digest.
        let bytes = self.inner.read_payload(range)?;
        Ok(Some(Arc::new(MemoryMapping {
            bytes,
            usage: Arc::clone(&self.usage),
        })))
    }
}
