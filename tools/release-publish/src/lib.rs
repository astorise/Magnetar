//! Tooling for `openspec/changes/implement-release-publication-automation/`
//! (MAG-06): credential-free release-proof generation (checksums, SBOM,
//! provenance, security-gate-input assembly) and OCI manifest/layout
//! construction for Component Artifacts and Kernel Exchange Bundles. Every
//! function in this crate runs with no network access and no registry
//! credentials -- the steps that need those (the real `cargo publish`, the
//! real GHCR push) are separate, explicitly credential-gated pipeline
//! steps this crate's output feeds into, not something this crate performs
//! itself.

pub mod checksum;
pub mod oci;
pub mod provenance;
pub mod release_gate;
pub mod sbom;
pub mod security_gate;

use std::{fmt, io};

use magnetar_roadmap_contracts::{ReleasePackagingError, ReleaseSecurityError};

/// This crate's own error type, composing the I/O, JSON, and
/// `magnetar-roadmap-contracts` policy errors every module here can
/// produce.
#[derive(Debug)]
pub enum ReleasePublishError {
    Io(io::Error),
    Json(serde_json::Error),
    Packaging(ReleasePackagingError),
    Security(ReleaseSecurityError),
    Metadata(String),
}

impl fmt::Display for ReleasePublishError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::Json(err) => write!(f, "JSON error: {err}"),
            Self::Packaging(err) => write!(f, "release packaging error: {err}"),
            Self::Security(err) => write!(f, "release security error: {err}"),
            Self::Metadata(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for ReleasePublishError {}

impl From<io::Error> for ReleasePublishError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<serde_json::Error> for ReleasePublishError {
    fn from(err: serde_json::Error) -> Self {
        Self::Json(err)
    }
}

impl From<ReleasePackagingError> for ReleasePublishError {
    fn from(err: ReleasePackagingError) -> Self {
        Self::Packaging(err)
    }
}

impl From<ReleaseSecurityError> for ReleasePublishError {
    fn from(err: ReleaseSecurityError) -> Self {
        Self::Security(err)
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::{
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// A fresh, empty temporary directory, removed when the returned guard
    /// is dropped -- so a panicking assertion still cleans up instead of
    /// leaking directories across test runs.
    pub(crate) struct TempDir(pub PathBuf);

    impl TempDir {
        pub(crate) fn new(prefix: &str) -> Self {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock before epoch")
                .as_nanos();
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("release-publish-{prefix}-{nanos}-{n}"));
            std::fs::create_dir_all(&path).expect("create temp dir");
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    impl std::ops::Deref for TempDir {
        type Target = std::path::Path;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
}
