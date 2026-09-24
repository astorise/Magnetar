//! Versioned OCI media types for Magnetar-published artifacts (tasks.md
//! section 3, item 3.1), matching
//! `openspec/changes/implement-release-publication-automation/design.md`
//! Decision 6: the `v1` segment is part of the identifier, not a comment,
//! so a future breaking manifest-shape change ships as a distinct `v2`
//! media type instead of silently changing what `v1` consumers already
//! parse.

pub const COMPONENT_ARTIFACT_MEDIA_TYPE_V1: &str = "application/vnd.magnetar.component.v1+wasm";
pub const COMPONENT_MANIFEST_MEDIA_TYPE_V1: &str =
    "application/vnd.magnetar.component-manifest.v1+json";
pub const KERNEL_BUNDLE_MEDIA_TYPE_V1: &str = "application/vnd.magnetar.kernel-bundle.v1+tar+zstd";
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
