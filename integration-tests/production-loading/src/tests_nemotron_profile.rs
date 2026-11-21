//! Check the checkpoint profile against the portable operator interface.

use magnetar_component_nemotron::mamba2_profile::Mamba2Profile;
use magnetar_runtime::{
    ComputeDType, DTypeDescriptor, OperatorAttributeValue, OperatorFamily, OperatorId,
    ShapeDescriptor, TensorDescriptor, initial_operator_catalog, operator_semantics,
};
use std::collections::BTreeMap;

#[test]
fn pinned_nemotron_mixer_plan_matches_the_runtime_operator_interface() {
    let profile = Mamba2Profile::from_huggingface_config(include_bytes!(
        "../../../components/nemotron/tests/fixtures/nano-config.json"
    ))
    .unwrap();
    let catalog = initial_operator_catalog();
    let operator = catalog
        .get(&OperatorId::magnetar(
            "mamba2-mixer",
            1,
            OperatorFamily::LinearAlgebra,
        ))
        .unwrap();
    for tokens in [1, 7, profile.max_tokens] {
        let plan = profile.invocation(0, tokens).unwrap();
        let descriptor = |shape: &Vec<usize>| {
            TensorDescriptor::materialized(
                ShapeDescriptor::new(shape.iter().map(|&n| n as u64).collect::<Vec<_>>()),
                DTypeDescriptor::portable(ComputeDType::Float32),
            )
        };
        let inputs: Vec<_> = plan.input_shapes.iter().map(descriptor).collect();
        let outputs: Vec<_> = plan.output_shapes.iter().map(descriptor).collect();
        let mut attributes = BTreeMap::new();
        for (name, value) in [
            ("heads", plan.heads),
            ("head_dim", plan.head_dim),
            ("groups", plan.groups),
            ("state_dim", plan.state_dim),
            ("conv_kernel", plan.conv_kernel),
        ] {
            attributes.insert(name.into(), OperatorAttributeValue::Integer(value as i64));
        }
        for (name, value) in [("epsilon", plan.epsilon), ("dt_min", plan.dt_min)] {
            attributes.insert(name.into(), OperatorAttributeValue::Float(value));
        }
        if let Some(maximum) = plan.dt_max {
            attributes.insert("dt_max".into(), OperatorAttributeValue::Float(maximum));
        }
        attributes.insert(
            "conv_bias".into(),
            OperatorAttributeValue::Boolean(plan.conv_bias),
        );
        attributes.insert(
            "linear_bias".into(),
            OperatorAttributeValue::Boolean(plan.linear_bias),
        );
        operator
            .validate_invocation(&inputs, &outputs, &attributes)
            .unwrap();
        let semantic = operator_semantics::mamba2_mixer_from_attributes(&attributes).unwrap();
        assert_eq!(semantic.heads, plan.heads);
        assert_eq!(semantic.groups, plan.groups);
        assert_eq!(semantic.dt_min, 0.0);
        assert_eq!(semantic.dt_max, f32::INFINITY);
        let mut invalid_inputs = inputs.clone();
        invalid_inputs[1] = descriptor(&vec![1, 1]);
        assert!(
            operator
                .validate_invocation(&invalid_inputs, &outputs, &attributes)
                .is_err()
        );
        let mut invalid_outputs = outputs.clone();
        invalid_outputs[2] = descriptor(&vec![1, 1, 1]);
        assert!(
            operator
                .validate_invocation(&inputs, &invalid_outputs, &attributes)
                .is_err()
        );
        attributes.remove("groups");
        assert!(
            operator
                .validate_invocation(&inputs, &outputs, &attributes)
                .is_err()
        );
    }
}
