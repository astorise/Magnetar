use super::*;

#[test]
fn positions_are_exact_and_invalid_ranges_are_rejected() {
    let input = HostTensor::new([3, 2], vec![f32::NAN; 6]).unwrap();
    assert_eq!(position_range(&input, 7).unwrap().data, [7.0, 8.0, 9.0]);
    assert_eq!(
        position_range(&input, MAX_POSITION - 2).unwrap().data,
        [16_777_213.0, 16_777_214.0, 16_777_215.0]
    );
    assert!(position_range(&input, MAX_POSITION - 1).is_err());
    assert!(position_range_validate(&[0, 2], 0).is_err());
    assert!(position_range_validate(&[1, 0], 0).is_err());
    assert!(position_range_validate(&[1], 0).is_err());
    assert!(position_range_validate(&[2, 1], u64::MAX).is_err());
    assert!(position_range_offset(&BTreeMap::new()).is_err());
    assert!(
        position_range_offset(&BTreeMap::from([(
            "position_offset".into(),
            OperatorAttributeValue::Integer(-1),
        )]))
        .is_err()
    );
}
