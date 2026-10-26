//! Builds an OCI Image Manifest (OCI Image Spec v1.1) for a Component
//! Artifact or Kernel Exchange Bundle without any network access (tasks.md
//! section 3, item 3.2): every digest is computed locally from the actual
//! bytes handed to [`build_artifact_manifest`], so the manifest this
//! function returns is exactly what a real registry push would record --
//! there is nothing left for a push step to fill in beyond the HTTP calls
//! themselves.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{ReleasePublishError, checksum::sha256_hex, oci::media_types::ArtifactKind};

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

/// Whether `bytes` starts with the gzip format's own two-byte magic
/// number (RFC 1952 section 2.3.1), the simplest real check that content
/// claiming to be gzip-compressed actually is (MAG-09).
fn looks_like_gzip(bytes: &[u8]) -> bool {
    bytes.first_chunk::<2>() == Some(&[0x1f, 0x8b])
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
/// primary layer's digest, for one artifact (`layer_bytes`, e.g. a
/// Component Artifact's WASM binary or a Kernel Exchange Bundle's
/// tar+zstd archive).
pub struct BuiltArtifactManifest {
    pub manifest: OciManifest,
    pub manifest_bytes: Vec<u8>,
    pub manifest_digest: String,
    pub layer_digest: String,
    /// The embedded Component manifest sidecar's own digest (MAG-08) --
    /// `Some` only for `ArtifactKind::ComponentArtifact`, where
    /// [`build_artifact_manifest`] requires one; `None` for every other
    /// kind, which have no such sidecar to embed.
    pub component_manifest_digest: Option<String>,
}

/// Builds a complete, self-contained OCI manifest for `layer_bytes` of
/// `kind`, titled `title`, with every digest computed from the actual
/// bytes given -- see this module's own doc comment.
///
/// `component_manifest_bytes` embeds the Component's own
/// `*.magnetar-component.yaml` sidecar as a second real layer (MAG-08: an
/// OCI Component package that only carries the WASM bytes cannot be
/// reconstructed from OCI alone -- a consumer like Tachyon has to stage
/// the manifest separately, out of band, with no digest tying it to the
/// published Component). Required (an error if absent) when `kind` is
/// `ComponentArtifact`, since every real Component always has one on
/// disk; rejected (an error if present) for every other kind, which has
/// no such sidecar.
pub fn build_artifact_manifest(
    kind: ArtifactKind,
    layer_bytes: &[u8],
    title: &str,
    component_manifest_bytes: Option<&[u8]>,
) -> Result<BuiltArtifactManifest, ReleasePublishError> {
    match (kind, component_manifest_bytes) {
        (ArtifactKind::ComponentArtifact, None) => {
            return Err(ReleasePublishError::Metadata(
                "ArtifactKind::ComponentArtifact requires its component manifest sidecar bytes (MAG-08): without it, the published OCI package cannot be reconstructed from OCI alone".into(),
            ));
        }
        (other, Some(_)) if other != ArtifactKind::ComponentArtifact => {
            return Err(ReleasePublishError::Metadata(format!(
                "a component manifest sidecar was given but {other:?} has no such sidecar to embed"
            )));
        }
        _ => {}
    }

    // MAG-09: KERNEL_BUNDLE_MEDIA_TYPE_V1 declares `+tar+gzip`; verify the
    // actual bytes before publishing a package whose declared format does
    // not match its real content (exactly how a `+tar+zstd` label ended
    // up on a gzip-compressed bundle in the first place -- nothing ever
    // checked the two agreed).
    if kind == ArtifactKind::KernelExchangeBundle && !looks_like_gzip(layer_bytes) {
        return Err(ReleasePublishError::Metadata(
            "Kernel Exchange Bundle bytes do not start with the gzip magic bytes (0x1f 0x8b) declared by KERNEL_BUNDLE_MEDIA_TYPE_V1 (+tar+gzip) -- MAG-09: refusing to publish a package whose declared format does not match its actual bytes".into(),
        ));
    }

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

    let mut layers = vec![OciDescriptor {
        media_type: kind.layer_media_type().to_string(),
        digest: layer_digest.clone(),
        size: layer_bytes.len() as u64,
        annotations: None,
    }];

    let component_manifest_digest = component_manifest_bytes.map(|manifest_bytes| {
        let digest = digest_sha256(manifest_bytes);
        let mut manifest_annotations = BTreeMap::new();
        manifest_annotations.insert(
            "org.opencontainers.image.title".to_string(),
            format!("{title}.magnetar-component.yaml"),
        );
        layers.push(OciDescriptor {
            media_type: kind.manifest_media_type().to_string(),
            digest: digest.clone(),
            size: manifest_bytes.len() as u64,
            annotations: Some(manifest_annotations),
        });
        digest
    });

    let manifest = OciManifest {
        schema_version: 2,
        media_type: OCI_IMAGE_MANIFEST_MEDIA_TYPE.to_string(),
        config: OciDescriptor {
            media_type: EMPTY_CONFIG_MEDIA_TYPE.to_string(),
            digest: config_digest,
            size: EMPTY_CONFIG_BYTES.len() as u64,
            annotations: None,
        },
        layers,
        annotations: Some(annotations),
    };

    let manifest_bytes = serde_json::to_vec(&manifest)?;
    let manifest_digest = digest_sha256(&manifest_bytes);

    Ok(BuiltArtifactManifest {
        manifest,
        manifest_bytes,
        manifest_digest,
        layer_digest,
        component_manifest_digest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plausible gzip header (magic bytes, deflate method, no flags)
    /// followed by arbitrary payload -- real enough to satisfy
    /// [`looks_like_gzip`] without needing a real compressor in this
    /// crate's own test dependencies.
    const GZIP_MAGIC_FIXTURE: &[u8] = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03fake gzip payload";

    #[test]
    fn component_layer_uses_the_component_media_type() {
        let built = build_artifact_manifest(
            ArtifactKind::ComponentArtifact,
            b"fake wasm bytes",
            "demo",
            Some(b"fake manifest yaml"),
        )
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
            GZIP_MAGIC_FIXTURE,
            "demo-kernel",
            None,
        )
        .unwrap();
        assert_eq!(
            built.manifest.layers[0].media_type,
            "application/vnd.magnetar.kernel-bundle.v1+tar+gzip"
        );
    }

    #[test]
    fn kernel_bundle_bytes_that_do_not_look_like_gzip_are_rejected() {
        // MAG-09: a declared +tar+gzip format must actually be gzip.
        let result = build_artifact_manifest(
            ArtifactKind::KernelExchangeBundle,
            b"not actually gzip-compressed bytes",
            "demo-kernel",
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn conformance_fixture_layer_uses_the_conformance_media_type() {
        let built = build_artifact_manifest(
            ArtifactKind::ConformanceFixture,
            br#"{"passed":true}"#,
            "demo-conformance",
            None,
        )
        .unwrap();
        assert_eq!(
            built.manifest.layers[0].media_type,
            "application/vnd.magnetar.conformance.v1+json"
        );
    }

    #[test]
    fn manifest_digest_is_stable_across_repeated_construction() {
        let first = build_artifact_manifest(
            ArtifactKind::ComponentArtifact,
            b"same bytes",
            "x",
            Some(b"same manifest"),
        )
        .unwrap();
        let second = build_artifact_manifest(
            ArtifactKind::ComponentArtifact,
            b"same bytes",
            "x",
            Some(b"same manifest"),
        )
        .unwrap();
        assert_eq!(first.manifest_digest, second.manifest_digest);
        assert_eq!(first.layer_digest, second.layer_digest);
    }

    #[test]
    fn different_bytes_produce_a_different_layer_digest() {
        let a = build_artifact_manifest(
            ArtifactKind::ComponentArtifact,
            b"bytes-a",
            "x",
            Some(b"manifest"),
        )
        .unwrap();
        let b = build_artifact_manifest(
            ArtifactKind::ComponentArtifact,
            b"bytes-b",
            "x",
            Some(b"manifest"),
        )
        .unwrap();
        assert_ne!(a.layer_digest, b.layer_digest);
        assert_ne!(a.manifest_digest, b.manifest_digest);
    }

    #[test]
    fn digest_sha256_has_the_sha256_prefix() {
        assert!(digest_sha256(b"anything").starts_with("sha256:"));
    }

    #[test]
    fn component_artifact_without_a_manifest_sidecar_is_rejected() {
        // MAG-08: a Component OCI package without its manifest sidecar
        // cannot be reconstructed from OCI alone -- this must fail
        // closed, not silently publish an incomplete package.
        let result = build_artifact_manifest(
            ArtifactKind::ComponentArtifact,
            b"fake wasm bytes",
            "demo",
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn non_component_kind_with_a_manifest_sidecar_is_rejected() {
        let result = build_artifact_manifest(
            ArtifactKind::KernelExchangeBundle,
            b"fake tar+zstd bytes",
            "demo-kernel",
            Some(b"unexpected manifest"),
        );
        assert!(result.is_err());
    }

    #[test]
    fn component_manifest_sidecar_is_embedded_as_its_own_layer_with_the_right_media_type() {
        let built = build_artifact_manifest(
            ArtifactKind::ComponentArtifact,
            b"fake wasm bytes",
            "demo",
            Some(b"component: {}\n"),
        )
        .unwrap();
        assert_eq!(built.manifest.layers.len(), 2);
        assert_eq!(
            built.manifest.layers[1].media_type,
            "application/vnd.magnetar.component-manifest.v1+yaml"
        );
        assert_eq!(
            built.manifest.layers[1].digest,
            digest_sha256(b"component: {}\n")
        );
        assert_eq!(
            built.component_manifest_digest,
            Some(digest_sha256(b"component: {}\n"))
        );
    }
}
