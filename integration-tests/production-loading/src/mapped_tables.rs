//! Payload sources that decide how the n-gram tables of a checkpoint reach
//! the Runtime (`support-qwen38-flash-next`, task 4.4, design D8), and count
//! what the Runtime asks of them:
//! - `mapping`: the source maps the tables itself, from memory, serving the
//!   bytes the loader's source reads, so the Runtime holds no copy of a table
//!   and reads only the rows a step names;
//! - `observing`: the loader's own source answers (its real file mapping),
//!   and this one only records what was mapped and what was read whole;
//! - `refusing`: the tables cannot be mapped, so the Runtime reads and stages
//!   them whole, as it does for every other weight: the dense baseline.

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Mapping,
    Observing,
    Refusing,
}

/// `inner`, with every tensor whose name ends in `suffix` treated as `mode`
/// says.
pub(crate) struct MappedTables {
    inner: Arc<dyn ProductionArtifactPayloadSource>,
    suffix: &'static str,
    mode: Mode,
    usage: Arc<TableUse>,
}

impl MappedTables {
    fn with_mode(
        inner: Arc<dyn ProductionArtifactPayloadSource>,
        suffix: &'static str,
        mode: Mode,
    ) -> (Self, Arc<TableUse>) {
        let usage = Arc::new(TableUse::default());
        (
            Self {
                inner,
                suffix,
                mode,
                usage: Arc::clone(&usage),
            },
            usage,
        )
    }

    /// The source maps the tables itself, from memory.
    pub(crate) fn mapping(
        inner: Arc<dyn ProductionArtifactPayloadSource>,
        suffix: &'static str,
    ) -> (Self, Arc<TableUse>) {
        Self::with_mode(inner, suffix, Mode::Mapping)
    }

    /// `inner` answers, and the tables' mappings and whole reads are counted.
    pub(crate) fn observing(
        inner: Arc<dyn ProductionArtifactPayloadSource>,
        suffix: &'static str,
    ) -> (Self, Arc<TableUse>) {
        Self::with_mode(inner, suffix, Mode::Observing)
    }

    /// The tables cannot be mapped: they are read whole.
    pub(crate) fn refusing(
        inner: Arc<dyn ProductionArtifactPayloadSource>,
        suffix: &'static str,
    ) -> (Self, Arc<TableUse>) {
        Self::with_mode(inner, suffix, Mode::Refusing)
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
            return self.inner.map_payload(range);
        }
        match self.mode {
            Mode::Refusing => Ok(None),
            Mode::Observing => {
                let mapping = self.inner.map_payload(range)?;
                if mapping.is_some() {
                    self.usage
                        .mapped
                        .lock()
                        .unwrap()
                        .push(range.identity.clone());
                }
                Ok(mapping)
            }
            Mode::Mapping => {
                self.usage
                    .mapped
                    .lock()
                    .unwrap()
                    .push(range.identity.clone());
                // The mapping's bytes come from the loader's own source,
                // which verifies the range's digest.
                let bytes = self.inner.read_payload(range)?;
                Ok(Some(Arc::new(MemoryMapping {
                    bytes,
                    usage: Arc::clone(&self.usage),
                })))
            }
        }
    }
}

/// Drops the page cache so that a table's pages are read from the disk
/// (best effort: it needs the right to write `/proc/sys/vm/drop_caches`).
/// Whether it worked.
pub(crate) fn drop_page_cache() -> bool {
    let _ = std::process::Command::new("sync").status();
    std::fs::write("/proc/sys/vm/drop_caches", "3").is_ok()
}
