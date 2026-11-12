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
