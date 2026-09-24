//! Writes a real local OCI Image Layout directory (OCI Image Layout Spec)
//! from a built manifest, with no network or registry involved (tasks.md
//! section 3, item 3.3): [`write_oci_layout`] and [`verify_oci_layout`]
//! round-trip through actual files on disk, so a digest mismatch between
//! what was written and what a real registry push/pull would see is
//! caught locally instead of only in a much slower, credentialed
//! integration test.

use std::{fs, path::Path};

use crate::{ReleasePublishError, checksum::sha256_hex, oci::manifest::BuiltArtifactManifest};

fn blob_path(root: &Path, digest: &str) -> Result<std::path::PathBuf, ReleasePublishError> {
    let hex = digest.strip_prefix("sha256:").ok_or_else(|| {
        ReleasePublishError::Metadata(format!("unsupported digest algorithm in '{digest}'"))
    })?;
    Ok(root.join("blobs").join("sha256").join(hex))
}

/// Writes `built.manifest`, an empty OCI config blob, and `layer_bytes`
/// (the artifact's own bytes) as content-addressed blobs under `root`,
/// plus `oci-layout` and `index.json`, per the OCI Image Layout Spec.
/// `tag` is recorded only as the `org.opencontainers.image.ref.name`
/// annotation on the index entry -- a convenience pointer, never trust
/// identity, per `component-artifact-distribution`'s "OCI Digest Is The
/// Only Trust Identity" requirement.
pub fn write_oci_layout(
    root: &Path,
    built: &BuiltArtifactManifest,
    layer_bytes: &[u8],
    tag: &str,
) -> Result<(), ReleasePublishError> {
    fs::create_dir_all(root.join("blobs").join("sha256"))?;
    fs::write(
        root.join("oci-layout"),
        br#"{"imageLayoutVersion":"1.0.0"}"#,
    )?;
    fs::write(
        blob_path(root, &built.manifest_digest)?,
        &built.manifest_bytes,
    )?;
    fs::write(blob_path(root, &built.layer_digest)?, layer_bytes)?;
    fs::write(blob_path(root, &built.manifest.config.digest)?, b"{}")?;

    let index = serde_json::json!({
        "schemaVersion": 2,
        "manifests": [{
            "mediaType": built.manifest.media_type,
            "digest": built.manifest_digest,
            "size": built.manifest_bytes.len(),
            "annotations": { "org.opencontainers.image.ref.name": tag },
        }],
    });
    fs::write(root.join("index.json"), serde_json::to_vec_pretty(&index)?)?;
    Ok(())
}

/// Reads every blob back from `root` and recomputes its digest, failing if
/// any recomputed digest disagrees with the filename it was stored under
/// -- the local equivalent of a registry push/pull round-trip, without a
/// registry.
pub fn verify_oci_layout(root: &Path) -> Result<(), ReleasePublishError> {
    let sha256_dir = root.join("blobs").join("sha256");
    for entry in fs::read_dir(&sha256_dir)? {
        let entry = entry?;
        let hex_name = entry.file_name().to_string_lossy().into_owned();
        let bytes = fs::read(entry.path())?;
        let recomputed = sha256_hex(&bytes);
        if recomputed != hex_name {
            return Err(ReleasePublishError::Metadata(format!(
                "OCI layout blob '{hex_name}' does not match its recomputed digest 'sha256:{recomputed}'"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        oci::manifest::build_artifact_manifest, oci::media_types::ArtifactKind,
        test_support::TempDir,
    };

    #[test]
    fn round_trips_a_component_artifact_through_a_local_layout() {
        let layer_bytes = b"fake wasm component bytes";
        let built =
            build_artifact_manifest(ArtifactKind::ComponentArtifact, layer_bytes, "demo").unwrap();
        let root = TempDir::new("oci-layout-roundtrip");

        write_oci_layout(&root, &built, layer_bytes, "latest").unwrap();

        assert!(root.join("oci-layout").is_file());
        assert!(root.join("index.json").is_file());
        verify_oci_layout(&root).unwrap();
    }

    #[test]
    fn tampered_blob_is_detected_by_verification() {
        let layer_bytes = b"fake wasm component bytes";
        let built =
            build_artifact_manifest(ArtifactKind::ComponentArtifact, layer_bytes, "demo").unwrap();
        let root = TempDir::new("oci-layout-tamper");
        write_oci_layout(&root, &built, layer_bytes, "latest").unwrap();

        let layer_blob = root
            .join("blobs")
            .join("sha256")
            .join(built.layer_digest.strip_prefix("sha256:").unwrap());
        fs::write(&layer_blob, b"tampered bytes").unwrap();

        let result = verify_oci_layout(&root);
        assert!(matches!(result, Err(ReleasePublishError::Metadata(_))));
    }
}
