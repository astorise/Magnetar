//! Builds an OCI Image Manifest (OCI Image Spec v1.1) for a Component
//! Artifact or Kernel Exchange Bundle without any network access (tasks.md
//! section 3, item 3.2): every digest is computed locally from the actual
//! bytes handed to [`build_artifact_manifest`], so the manifest this
//! function returns is exactly what a real registry push would record --
//! there is nothing left for a push step to fill in beyond the HTTP calls
//! themselves.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{checksum::sha256_hex, oci::media_types::ArtifactKind};

const EMPTY_CONFIG_MEDIA_TYPE: &str = "application/vnd.oci.empty.v1+json";
const EMPTY_CONFIG_BYTES: &[u8] = b"{}";
const OCI_IMAGE_MANIFEST_MEDIA_TYPE: &str = "application/vnd.oci.image.manifest.v1+json";

/// `sha256:<hex>` -- the OCI digest form used everywhere a digest is
/// recorded (manifest, layer, config), per
/// `component-artifact-distribution`'s "OCI Digest Is The Only Trust
/// Identity" requirement: this is the only identity this module ever
/// produces for an artifact; nothing here produces or accepts a mutable
/// tag as identity.
pub fn digest_sha256(bytes: &[u8]) -> String {
    format!("sha256:{}", sha256_hex(bytes))
}

/// One OCI content descriptor: a digest, size, media type, and optional
/// annotations.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct OciDescriptor {
    #[serde(rename = "mediaType")]
    pub media_type: String,
    pub digest: String,
    pub size: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<BTreeMap<String, String>>,
}

/// An OCI Image Manifest (schema version 2): a config descriptor plus the
/// artifact's layer descriptors.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct OciManifest {
    #[serde(rename = "schemaVersion")]
    pub schema_version: u32,
    #[serde(rename = "mediaType")]
    pub media_type: String,
    pub config: OciDescriptor,
    pub layers: Vec<OciDescriptor>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<BTreeMap<String, String>>,
}

/// The manifest, its serialized bytes, and its own digest, plus the
/// layer's digest, for one artifact (`layer_bytes`, e.g. a Component
/// Artifact's WASM binary or a Kernel Exchange Bundle's tar+zstd archive).
pub struct BuiltArtifactManifest {
    pub manifest: OciManifest,
    pub manifest_bytes: Vec<u8>,
    pub manifest_digest: String,
    pub layer_digest: String,
}

/// Builds a complete, self-contained OCI manifest for `layer_bytes` of
/// `kind`, titled `title`, with every digest computed from the actual
/// bytes given -- see this module's own doc comment.
pub fn build_artifact_manifest(
    kind: ArtifactKind,
    layer_bytes: &[u8],
    title: &str,
) -> Result<BuiltArtifactManifest, serde_json::Error> {
    let layer_digest = digest_sha256(layer_bytes);
    let config_digest = digest_sha256(EMPTY_CONFIG_BYTES);

    let mut annotations = BTreeMap::new();
    annotations.insert(
        "org.opencontainers.image.title".to_string(),
        title.to_string(),
    );
    annotations.insert(
        "org.magnetar.artifact-manifest-media-type".to_string(),
        kind.manifest_media_type().to_string(),
    );

    let manifest = OciManifest {
        schema_version: 2,
        media_type: OCI_IMAGE_MANIFEST_MEDIA_TYPE.to_string(),
        config: OciDescriptor {
            media_type: EMPTY_CONFIG_MEDIA_TYPE.to_string(),
            digest: config_digest,
            size: EMPTY_CONFIG_BYTES.len() as u64,
            annotations: None,
        },
        layers: vec![OciDescriptor {
            media_type: kind.layer_media_type().to_string(),
            digest: layer_digest.clone(),
            size: layer_bytes.len() as u64,
            annotations: None,
        }],
        annotations: Some(annotations),
    };

    let manifest_bytes = serde_json::to_vec(&manifest)?;
    let manifest_digest = digest_sha256(&manifest_bytes);

    Ok(BuiltArtifactManifest {
        manifest,
        manifest_bytes,
        manifest_digest,
        layer_digest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_layer_uses_the_component_media_type() {
        let built =
            build_artifact_manifest(ArtifactKind::ComponentArtifact, b"fake wasm bytes", "demo")
                .unwrap();
        assert_eq!(
            built.manifest.layers[0].media_type,
            "application/vnd.magnetar.component.v1+wasm"
        );
    }

    #[test]
    fn kernel_bundle_layer_uses_the_kernel_bundle_media_type() {
        let built = build_artifact_manifest(
            ArtifactKind::KernelExchangeBundle,
            b"fake tar+zstd bytes",
            "demo-kernel",
        )
        .unwrap();
        assert_eq!(
            built.manifest.layers[0].media_type,
            "application/vnd.magnetar.kernel-bundle.v1+tar+zstd"
        );
    }

    #[test]
    fn conformance_fixture_layer_uses_the_conformance_media_type() {
        let built = build_artifact_manifest(
            ArtifactKind::ConformanceFixture,
            br#"{"passed":true}"#,
            "demo-conformance",
        )
        .unwrap();
        assert_eq!(
            built.manifest.layers[0].media_type,
            "application/vnd.magnetar.conformance.v1+json"
        );
    }

    #[test]
    fn manifest_digest_is_stable_across_repeated_construction() {
        let first =
            build_artifact_manifest(ArtifactKind::ComponentArtifact, b"same bytes", "x").unwrap();
        let second =
            build_artifact_manifest(ArtifactKind::ComponentArtifact, b"same bytes", "x").unwrap();
        assert_eq!(first.manifest_digest, second.manifest_digest);
        assert_eq!(first.layer_digest, second.layer_digest);
    }

    #[test]
    fn different_bytes_produce_a_different_layer_digest() {
        let a = build_artifact_manifest(ArtifactKind::ComponentArtifact, b"bytes-a", "x").unwrap();
        let b = build_artifact_manifest(ArtifactKind::ComponentArtifact, b"bytes-b", "x").unwrap();
        assert_ne!(a.layer_digest, b.layer_digest);
        assert_ne!(a.manifest_digest, b.manifest_digest);
    }

    #[test]
    fn digest_sha256_has_the_sha256_prefix() {
        assert!(digest_sha256(b"anything").starts_with("sha256:"));
    }
}
