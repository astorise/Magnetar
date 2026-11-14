use magnetar_runtime::f16_to_f32;

/// `implement-production-qwen-model-loading` task 5.6: numeric edge cases
/// for F16 -> F32 conversion -- +0/-0, a subnormal, the largest finite
/// magnitude, both infinities, and NaN (payload/sign preserved, not
/// canonicalized).
#[test]
fn f16_to_f32_handles_every_numeric_class_exactly() {
    assert_eq!(f16_to_f32(0x0000).to_bits(), 0f32.to_bits());
    assert_eq!(f16_to_f32(0x8000).to_bits(), (-0f32).to_bits());
    // Smallest positive subnormal f16 (2^-24) must renormalize exactly.
    assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
    // Largest subnormal f16 mantissa.
    assert_eq!(f16_to_f32(0x03FF), 2f32.powi(-14) * (1023.0 / 1024.0));
    // Largest finite f16 magnitude (65504).
    assert_eq!(f16_to_f32(0x7BFF), 65504.0f32);
    assert_eq!(f16_to_f32(0xFBFF), -65504.0f32);
    assert!(f16_to_f32(0x7C00).is_infinite() && f16_to_f32(0x7C00) > 0.0);
    assert!(f16_to_f32(0xFC00).is_infinite() && f16_to_f32(0xFC00) < 0.0);
    let nan = f16_to_f32(0x7E00);
    assert!(nan.is_nan());
    assert_eq!(nan.to_bits() >> 31, 0, "sign bit must be preserved for NaN");
    let signed_nan = f16_to_f32(0xFE00);
    assert!(signed_nan.is_nan());
    assert_eq!(signed_nan.to_bits() >> 31, 1);
}
