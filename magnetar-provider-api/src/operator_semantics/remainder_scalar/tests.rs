use super::*;

#[test]
fn finite_divisor_preserves_shape_and_uses_divisor_sign() {
    let input = HostTensor::new([2, 3], vec![-7.0, -1.0, 0.0, 1.0, 7.0, 12.5]).unwrap();
    let original = input.clone();
    let result = remainder_scalar(&input, 3.0).unwrap();
    assert_eq!(result.shape, [2, 3]);
    assert_eq!(result.data, [2.0, 2.0, 0.0, 1.0, 1.0, 0.5]);
    assert_eq!(input, original);
}

#[test]
fn invalid_divisors_inputs_and_shapes_are_rejected() {
    for value in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e-100] {
        let attrs = BTreeMap::from([("divisor".into(), OperatorAttributeValue::Float(value))]);
        assert!(remainder_scalar_divisor(&attrs).is_err());
    }
    assert!(remainder_scalar_divisor(&BTreeMap::new()).is_err());
    assert!(
        remainder_scalar_divisor(&BTreeMap::from([(
            "divisor".into(),
            OperatorAttributeValue::Integer(3),
        )]))
        .is_err()
    );
    let bad = HostTensor::new([1], vec![f32::NAN]).unwrap();
    assert!(remainder_scalar(&bad, 3.0).is_err());
    let malformed = HostTensor {
        shape: vec![2],
        data: vec![1.0],
    };
    assert!(remainder_scalar(&malformed, 3.0).is_err());
}
