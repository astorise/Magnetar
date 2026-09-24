//! SHA-256 checksums for release artifacts (tasks.md section 1, items
//! 1.2-1.3): recomputes real digests from files that were actually built,
//! rather than trusting a previously-recorded value, so
//! [`verify_bundle_against_dir`] composes
//! `magnetar_roadmap_contracts::verify_checksum_matches_final_artifact`
//! against real bytes instead of duplicating its comparison logic.

use std::{fs, path::Path};

use magnetar_roadmap_contracts::{
    ArtifactChecksum, ChecksumAlgorithm, verify_checksum_matches_final_artifact,
};
use sha2::{Digest, Sha256};

use crate::ReleasePublishError;

/// Hex-encoded SHA-256 digest of `bytes` (no `sha256:` prefix; callers that
/// need the OCI-style prefixed form use
/// [`crate::oci::manifest::digest_sha256`]).
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let mut out = String::with_capacity(Sha256::output_size() * 2);
    for byte in hasher.finalize() {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Computes a real [`ArtifactChecksum`] for the file at `path`, naming the
/// artifact `artifact_name` (typically its path relative to the release
/// bundle root, e.g. `"magnetar-cli-x86_64-unknown-linux-gnu"`).
pub fn checksum_for_file(
    path: &Path,
    artifact_name: &str,
) -> Result<ArtifactChecksum, ReleasePublishError> {
    let bytes = fs::read(path)?;
    Ok(ArtifactChecksum::new(
        artifact_name,
        ChecksumAlgorithm::Sha256,
        sha256_hex(&bytes),
    )?)
}

/// Walks `dir` recursively and returns one [`ArtifactChecksum`] per file,
/// named by its path relative to `dir` (forward-slash separated
/// regardless of host platform), sorted by that name so the result is
/// deterministic.
pub fn checksum_bundle_for_dir(dir: &Path) -> Result<Vec<ArtifactChecksum>, ReleasePublishError> {
    let mut checksums = Vec::new();
    collect_files(dir, dir, &mut checksums)?;
    checksums.sort_by(|a, b| a.artifact.cmp(&b.artifact));
    Ok(checksums)
}

fn collect_files(
    root: &Path,
    dir: &Path,
    out: &mut Vec<ArtifactChecksum>,
) -> Result<(), ReleasePublishError> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(root, &path, out)?;
        } else {
            let relative = path
                .strip_prefix(root)
                .expect("walked path is under its own root")
                .to_string_lossy()
                .replace('\\', "/");
            out.push(checksum_for_file(&path, &relative)?);
        }
    }
    Ok(())
}

/// Recomputes every checksum in `bundle` against the real file it names
/// under `dir`, via `verify_checksum_matches_final_artifact` -- the
/// "Release Proofs Generated From The Tagged Commit's Real Build"
/// requirement's own check, run here instead of only at upload time.
pub fn verify_bundle_against_dir(
    bundle: &[ArtifactChecksum],
    dir: &Path,
) -> Result<(), ReleasePublishError> {
    for checksum in bundle {
        let bytes = fs::read(dir.join(&checksum.artifact))?;
        let recomputed = sha256_hex(&bytes);
        verify_checksum_matches_final_artifact(checksum, &recomputed)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::TempDir;

    #[test]
    fn checksum_bundle_matches_written_files() {
        let dir = TempDir::new("checksum-bundle");
        fs::write(dir.join("a.txt"), b"hello").unwrap();
        fs::create_dir(dir.join("sub")).unwrap();
        fs::write(dir.join("sub/b.txt"), b"world").unwrap();

        let bundle = checksum_bundle_for_dir(&dir).unwrap();

        assert_eq!(bundle.len(), 2);
        assert_eq!(bundle[0].artifact, "a.txt");
        assert_eq!(bundle[1].artifact, "sub/b.txt");
        verify_bundle_against_dir(&bundle, &dir).unwrap();
    }

    #[test]
    fn tampered_file_is_detected_as_checksum_mismatch() {
        let dir = TempDir::new("checksum-tamper");
        fs::write(dir.join("a.txt"), b"hello").unwrap();
        let bundle = checksum_bundle_for_dir(&dir).unwrap();

        fs::write(dir.join("a.txt"), b"tampered").unwrap();

        let result = verify_bundle_against_dir(&bundle, &dir);
        assert!(matches!(
            result,
            Err(ReleasePublishError::Security(
                magnetar_roadmap_contracts::ReleaseSecurityError::ChecksumMismatch { .. }
            ))
        ));
    }

    #[test]
    fn sha256_hex_is_stable_and_content_sensitive() {
        let a = sha256_hex(b"hello");
        let b = sha256_hex(b"hello");
        let c = sha256_hex(b"hello!");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 64);
    }
}
