//! Writes a real local OCI Image Layout directory (OCI Image Layout Spec)
//! from a built manifest, with no network or registry involved (tasks.md
//! section 3, item 3.3): [`write_oci_layout`] and [`verify_oci_layout`]
//! round-trip through actual files on disk, so a digest mismatch between
//! what was written and what a real registry push/pull would see is
//! caught locally instead of only in a much slower, credentialed
//! integration test.

use std::{fs, path::Path};

use crate::{
    ReleasePublishError,
    checksum::sha256_hex,
    oci::manifest::{BuiltArtifactManifest, digest_sha256},
};

fn blob_path(root: &Path, digest: &str) -> Result<std::path::PathBuf, ReleasePublishError> {
    let hex = digest.strip_prefix("sha256:").ok_or_else(|| {
        ReleasePublishError::Metadata(format!("unsupported digest algorithm in '{digest}'"))
    })?;
    Ok(root.join("blobs").join("sha256").join(hex))
}

/// Writes `built.manifest`, an empty OCI config blob, and `layer_blobs`
/// (the artifact's own layer bytes, one per entry of `built.manifest.
/// layers`, in the same order -- e.g. a Component's `[wasm bytes,
/// component manifest sidecar bytes]`, MAG-08) as content-addressed blobs
/// under `root`, plus `oci-layout` and `index.json`, per the OCI Image
/// Layout Spec. `tag` is recorded only as the
/// `org.opencontainers.image.ref.name` annotation on the index entry -- a
/// convenience pointer, never trust identity, per
/// `component-artifact-distribution`'s "OCI Digest Is The Only Trust
/// Identity" requirement.
pub fn write_oci_layout(
    root: &Path,
    built: &BuiltArtifactManifest,
    layer_blobs: &[&[u8]],
    tag: &str,
) -> Result<(), ReleasePublishError> {
    if layer_blobs.len() != built.manifest.layers.len() {
        return Err(ReleasePublishError::Metadata(format!(
            "expected {} layer blob(s) matching built.manifest.layers, got {}",
            built.manifest.layers.len(),
            layer_blobs.len()
        )));
    }
    fs::create_dir_all(root.join("blobs").join("sha256"))?;
    fs::write(
        root.join("oci-layout"),
        br#"{"imageLayoutVersion":"1.0.0"}"#,
    )?;
    fs::write(
        blob_path(root, &built.manifest_digest)?,
        &built.manifest_bytes,
    )?;
    for (layer, blob) in built.manifest.layers.iter().zip(layer_blobs) {
        let actual_digest = digest_sha256(blob);
        if actual_digest != layer.digest {
            return Err(ReleasePublishError::Metadata(format!(
                "layer blob for media type '{}' does not match its manifest digest (expected {}, got {actual_digest})",
                layer.media_type, layer.digest
            )));
        }
        fs::write(blob_path(root, &layer.digest)?, blob)?;
    }
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
        let layer_bytes: &[u8] = b"fake wasm component bytes";
        let manifest_bytes: &[u8] = b"component: {}\n";
        let built = build_artifact_manifest(
            ArtifactKind::ComponentArtifact,
            layer_bytes,
            "demo",
            Some(manifest_bytes),
        )
        .unwrap();
        let root = TempDir::new("oci-layout-roundtrip");

        write_oci_layout(&root, &built, &[layer_bytes, manifest_bytes], "latest").unwrap();

        assert!(root.join("oci-layout").is_file());
        assert!(root.join("index.json").is_file());
        verify_oci_layout(&root).unwrap();
    }

    #[test]
    fn round_trips_a_kernel_bundle_through_a_local_layout() {
        let layer_bytes: &[u8] = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03fake gzip payload";
        let built = build_artifact_manifest(
            ArtifactKind::KernelExchangeBundle,
            layer_bytes,
            "demo-kernel",
            None,
        )
        .unwrap();
        let root = TempDir::new("oci-layout-kernel-roundtrip");

        write_oci_layout(&root, &built, &[layer_bytes], "latest").unwrap();

        verify_oci_layout(&root).unwrap();
    }

    #[test]
    fn tampered_blob_is_detected_by_verification() {
        let layer_bytes: &[u8] = b"fake wasm component bytes";
        let manifest_bytes: &[u8] = b"component: {}\n";
        let built = build_artifact_manifest(
            ArtifactKind::ComponentArtifact,
            layer_bytes,
            "demo",
            Some(manifest_bytes),
        )
        .unwrap();
        let root = TempDir::new("oci-layout-tamper");
        write_oci_layout(&root, &built, &[layer_bytes, manifest_bytes], "latest").unwrap();

        let layer_blob = root
            .join("blobs")
            .join("sha256")
            .join(built.layer_digest.strip_prefix("sha256:").unwrap());
        fs::write(&layer_blob, b"tampered bytes").unwrap();

        let result = verify_oci_layout(&root);
        assert!(matches!(result, Err(ReleasePublishError::Metadata(_))));
    }

    #[test]
    fn mismatched_layer_blob_count_is_rejected() {
        let layer_bytes: &[u8] = b"fake wasm component bytes";
        let manifest_bytes: &[u8] = b"component: {}\n";
        let built = build_artifact_manifest(
            ArtifactKind::ComponentArtifact,
            layer_bytes,
            "demo",
            Some(manifest_bytes),
        )
        .unwrap();
        let root = TempDir::new("oci-layout-blob-count-mismatch");

        // Only one blob given for a two-layer manifest.
        let result = write_oci_layout(&root, &built, &[layer_bytes], "latest");
        assert!(matches!(result, Err(ReleasePublishError::Metadata(_))));
    }

    #[test]
    fn a_blob_in_the_wrong_position_is_rejected_before_writing() {
        let layer_bytes: &[u8] = b"fake wasm component bytes";
        let manifest_bytes: &[u8] = b"component: {}\n";
        let built = build_artifact_manifest(
            ArtifactKind::ComponentArtifact,
            layer_bytes,
            "demo",
            Some(manifest_bytes),
        )
        .unwrap();
        let root = TempDir::new("oci-layout-blob-order-mismatch");

        // Swapped order: this must be caught by the digest check, not
        // silently written under the wrong layer's declared digest.
        let result = write_oci_layout(&root, &built, &[manifest_bytes, layer_bytes], "latest");
        assert!(matches!(result, Err(ReleasePublishError::Metadata(_))));
    }
}
