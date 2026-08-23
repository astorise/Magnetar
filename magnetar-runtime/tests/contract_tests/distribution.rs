use magnetar_runtime::{ComponentDistributionSource, ComponentDistributionSourceKind};

#[test]
fn tachyon_distribution_source_metadata_is_only_source_identity() {
    let source = ComponentDistributionSource::new(
        ComponentDistributionSourceKind::Tachyon,
        "tachyon://component/example",
    );

    assert_eq!(source.kind.as_str(), "tachyon");
    assert_eq!(source.identity, "tachyon://component/example");
}
