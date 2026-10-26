//! Versioned OCI media types for Magnetar-published artifacts (tasks.md
//! section 3, item 3.1), matching
//! `openspec/changes/implement-release-publication-automation/design.md`
//! Decision 6: the `v1` segment is part of the identifier, not a comment,
//! so a future breaking manifest-shape change ships as a distinct `v2`
//! media type instead of silently changing what `v1` consumers already
//! parse.

pub const COMPONENT_ARTIFACT_MEDIA_TYPE_V1: &str = "application/vnd.magnetar.component.v1+wasm";
/// `+yaml`, not `+json`: a Component's real manifest sidecar on disk is
/// always `<artifact>.magnetar-component.yaml` (see
/// `magnetar_runtime::component::manifest_path_for_definition`) and is
/// parsed as YAML (`ComponentManifest::from_yaml_bytes`). This constant
/// was never used as a real OCI layer `mediaType` before MAG-08 wired it
/// up as one -- only ever as an annotation value naming the manifest
/// shape -- so there is no real published artifact whose identity this
/// correction could break.
pub const COMPONENT_MANIFEST_MEDIA_TYPE_V1: &str =
    "application/vnd.magnetar.component-manifest.v1+yaml";
/// `+tar+gzip`, not `+tar+zstd`: every Kernel Exchange Bundle this
/// workspace can actually produce or ingest is gzip-compressed --
/// `extract_kernel_exchange_archive` (`magnetar-runtime`) implements
/// plain tar and tar.gz only, and zstd decompression is not wired into
/// the ingestion path at all (MAG-09). A real Kernel Exchange Bundle
/// (the `cpu_reference_matmul` fixture) was already published to GHCR
/// under the old, inaccurate `+tar+zstd` label before this correction --
/// see "Withdrawal / correction" in `docs/release-publication.md`: that
/// digest stays resolvable as published, and a corrected republish uses
/// this constant under its own new digest.
pub const KERNEL_BUNDLE_MEDIA_TYPE_V1: &str = "application/vnd.magnetar.kernel-bundle.v1+tar+gzip";
pub const KERNEL_MANIFEST_MEDIA_TYPE_V1: &str = "application/vnd.magnetar.kernel-manifest.v1+json";
pub const CONFORMANCE_FIXTURE_MEDIA_TYPE_V1: &str = "application/vnd.magnetar.conformance.v1+json";

/// The kind of artifact an OCI manifest is being built for -- each maps to
/// exactly one layer media type and one manifest-annotation media type, so
/// [`crate::oci::manifest`] cannot accidentally mismatch them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArtifactKind {
    ComponentArtifact,
    KernelExchangeBundle,
    ConformanceFixture,
}

impl ArtifactKind {
    /// The layer media type for this artifact kind's actual bytes.
    pub fn layer_media_type(self) -> &'static str {
        match self {
            Self::ComponentArtifact => COMPONENT_ARTIFACT_MEDIA_TYPE_V1,
            Self::KernelExchangeBundle => KERNEL_BUNDLE_MEDIA_TYPE_V1,
            Self::ConformanceFixture => CONFORMANCE_FIXTURE_MEDIA_TYPE_V1,
        }
    }

    /// The manifest media type naming what kind of artifact this OCI
    /// manifest describes, recorded as an annotation.
    /// `ConformanceFixture` has no separate manifest shape of its own, so
    /// it names its own layer type.
    pub fn manifest_media_type(self) -> &'static str {
        match self {
            Self::ComponentArtifact => COMPONENT_MANIFEST_MEDIA_TYPE_V1,
            Self::KernelExchangeBundle => KERNEL_MANIFEST_MEDIA_TYPE_V1,
            Self::ConformanceFixture => CONFORMANCE_FIXTURE_MEDIA_TYPE_V1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_artifact_kind_has_a_distinct_layer_media_type() {
        assert_eq!(
            ArtifactKind::ComponentArtifact.layer_media_type(),
            COMPONENT_ARTIFACT_MEDIA_TYPE_V1
        );
        assert_eq!(
            ArtifactKind::KernelExchangeBundle.layer_media_type(),
            KERNEL_BUNDLE_MEDIA_TYPE_V1
        );
        assert_eq!(
            ArtifactKind::ConformanceFixture.layer_media_type(),
            CONFORMANCE_FIXTURE_MEDIA_TYPE_V1
        );
        assert_ne!(
            ArtifactKind::ComponentArtifact.layer_media_type(),
            ArtifactKind::KernelExchangeBundle.layer_media_type()
        );
    }

    #[test]
    fn v1_media_types_do_not_collide_with_a_hypothetical_v2() {
        let hypothetical_v2 = COMPONENT_MANIFEST_MEDIA_TYPE_V1.replace("v1", "v2");
        assert_ne!(COMPONENT_MANIFEST_MEDIA_TYPE_V1, hypothetical_v2);
        assert!(COMPONENT_MANIFEST_MEDIA_TYPE_V1.contains("v1"));
    }
}
