use super::*;

fn device(name: &str) -> DeviceId {
    DeviceId::new(name)
}

#[test]
fn weight_layer_reads_the_canonical_layer_index() {
    assert_eq!(weight_layer("layers.12.self_attn.q_proj"), Some(12));
    assert_eq!(
        weight_layer("model.instance-1.weight.layers.3.moe.router_proj"),
        Some(3)
    );
    assert_eq!(weight_layer("token_embedding"), None);
    assert_eq!(weight_layer("model.i.weight.lm_head"), None);
}

#[test]
fn fill_stays_on_one_device_when_it_fits_and_balanced_splits() {
    let bytes = LayerWeightBytes {
        layers: vec![10; 4],
        first: 5,
        last: 5,
        ..Default::default()
    };
    let devices = [(device("gpu0"), 100), (device("gpu1"), 100)];
    let fill = LayerPlacement::plan(&bytes, &devices, LayerPlacementPolicy::Fill).unwrap();
    assert_eq!(fill.segments().len(), 1);
    let balanced = LayerPlacement::plan(&bytes, &devices, LayerPlacementPolicy::Balanced).unwrap();
    assert_eq!(
        balanced.segments(),
        &[
            LayerSegment {
                device: device("gpu0"),
                first_layer: 0,
                end_layer: 2,
            },
            LayerSegment {
                device: device("gpu1"),
                first_layer: 2,
                end_layer: 4,
            },
        ]
    );
    assert_eq!(
        balanced.device_for_weight("model.i.weight.token_embedding"),
        &device("gpu0")
    );
    assert_eq!(
        balanced.device_for_weight("model.i.weight.layers.3.mlp.up_proj"),
        &device("gpu1")
    );
    assert_eq!(
        balanced.device_for_weight("model.i.weight.lm_head"),
        &device("gpu1")
    );
}

#[test]
fn fill_spills_to_the_next_device_and_rejects_what_cannot_fit() {
    // 15 GB of layers on two 12 GB Devices.
    let gb = 1_u64 << 30;
    let bytes = LayerWeightBytes {
        layers: vec![gb / 2; 24],
        first: gb / 4,
        last: gb / 4,
        ..Default::default()
    };
    let devices = [(device("gpu0"), 10 * gb), (device("gpu1"), 10 * gb)];
    let plan = LayerPlacement::plan(&bytes, &devices, LayerPlacementPolicy::Fill).unwrap();
    assert_eq!(plan.segments().len(), 2);
    assert_eq!(plan.segments()[0].end_layer, 19);
    assert!(matches!(
        LayerPlacement::plan(&bytes, &devices[..1], LayerPlacementPolicy::Fill),
        Err(LayerPlacementError::DoesNotFit { .. })
    ));
    assert!(
        LayerPlacement::new(
            vec![LayerSegment {
                device: device("gpu0"),
                first_layer: 1,
                end_layer: 24,
            }],
            24
        )
        .is_err()
    );
}

/// Balanced shares are soft: when stopping at them leaves too much for the
/// later Devices, a placement that fits is still found.
#[test]
fn balanced_falls_back_to_a_placement_that_fits() {
    let bytes = LayerWeightBytes {
        layers: vec![10, 40, 20],
        first: 0,
        last: 0,
        ..Default::default()
    };
    let devices = [(device("a"), 100), (device("b"), 50)];
    let placement = LayerPlacement::plan(&bytes, &devices, LayerPlacementPolicy::Balanced).unwrap();
    let ends: Vec<_> = placement
        .segments()
        .iter()
        .map(|segment| (segment.device.as_str().to_string(), segment.end_layer))
        .collect();
    assert_eq!(ends, vec![("a".to_string(), 3)]);
}

fn ends(placement: &LayerPlacement) -> Vec<(String, u32)> {
    placement
        .segments()
        .iter()
        .map(|segment| (segment.device.as_str().to_string(), segment.end_layer))
        .collect()
}

/// A layer reusing an earlier layer's K/V stays in that layer's segment:
/// a balanced split that would fall between them moves past them, and a
/// Device that cannot hold them ends its segment before the source.
#[test]
fn layers_sharing_kv_stay_with_their_source() {
    let mut bytes = LayerWeightBytes {
        layers: vec![10; 6],
        ..Default::default()
    };
    // Layers 4 and 5 reuse layer 2's K/V: no segment starts at 3, 4 or 5.
    bytes.keep_kv_sources_together([(4, 2), (5, 2)]);
    assert_eq!(bytes.joined, vec![false, false, false, true, true, true]);
    let roomy = [(device("a"), 100), (device("b"), 100)];
    let balanced = LayerPlacement::plan(&bytes, &roomy, LayerPlacementPolicy::Balanced).unwrap();
    assert_eq!(ends(&balanced), vec![("a".to_string(), 6)]);
    // 40 bytes each: `a` cannot hold layers 0..=4, so it ends before the
    // source, layer 2, and `b` holds layers 2..=5 together.
    let tight = [(device("a"), 40), (device("b"), 40)];
    let fill = LayerPlacement::plan(&bytes, &tight, LayerPlacementPolicy::Fill).unwrap();
    assert_eq!(
        ends(&fill),
        vec![("a".to_string(), 2), ("b".to_string(), 6)]
    );
    // Without the constraint the same Devices split at layer 4.
    let free = LayerWeightBytes {
        layers: vec![10; 6],
        ..Default::default()
    };
    let unconstrained = LayerPlacement::plan(&free, &tight, LayerPlacementPolicy::Fill).unwrap();
    assert_eq!(
        ends(&unconstrained),
        vec![("a".to_string(), 4), ("b".to_string(), 6)]
    );
    // Joined layers no Device can hold together do not fit.
    let small = [(device("a"), 30), (device("b"), 30), (device("c"), 30)];
    assert!(matches!(
        LayerPlacement::plan(&bytes, &small, LayerPlacementPolicy::Fill),
        Err(LayerPlacementError::DoesNotFit { .. })
    ));
}
