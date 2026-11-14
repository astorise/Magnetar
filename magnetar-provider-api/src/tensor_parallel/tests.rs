use super::*;

#[test]
fn shares_split_into_equal_granule_aligned_parts() {
    let plan = TensorParallelPlan {
        devices: vec![DeviceId::new("a"), DeviceId::new("b")],
        shards: BTreeMap::new(),
    };
    assert_eq!(plan.share(0, 4864, 1), (0, 2432));
    assert_eq!(plan.share(1, 4864, 1), (2432, 4864));
    assert_eq!(plan.share(0, 1408, 32), (0, 704));
    assert_eq!(plan.share(1, 96, 32), (32, 96));
}

#[test]
fn names_each_split_weight_by_its_canonical_name() {
    let plan = TensorParallelPlan {
        devices: vec![DeviceId::new("a"), DeviceId::new("b")],
        shards: BTreeMap::from([
            ("layers.0.mlp.gate_proj".to_string(), WeightShard::Columns),
            ("layers.0.mlp.down_proj".to_string(), WeightShard::Rows),
        ]),
    };
    assert!(plan.splits());
    assert_eq!(
        plan.shard_for_weight("model.i.weight.layers.0.mlp.down_proj"),
        Some(WeightShard::Rows)
    );
    assert_eq!(
        plan.shard_for_weight("layers.0.mlp.gate_proj"),
        Some(WeightShard::Columns)
    );
    assert_eq!(plan.shard_for_weight("model.i.weight.lm_head"), None);
}
