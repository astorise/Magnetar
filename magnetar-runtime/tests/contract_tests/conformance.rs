use magnetar_runtime::{
    ComponentEngineCapabilities, ComponentEngineFeature, FirstNativeModelExecutionProfileError,
    validate_first_native_component_engine_capabilities,
};

#[test]
fn first_native_component_engine_rejects_non_native_or_ambient_test_profile() {
    let capabilities = ComponentEngineCapabilities::test();

    assert!(matches!(
        validate_first_native_component_engine_capabilities(&capabilities),
        Err(FirstNativeModelExecutionProfileError::ComponentEngineProfileMismatch)
    ));

    let mut capabilities = ComponentEngineCapabilities::native();
    capabilities.controlled_wasi = false;
    assert!(matches!(
        validate_first_native_component_engine_capabilities(&capabilities),
        Err(
            FirstNativeModelExecutionProfileError::ComponentEngineFeatureMissing(
                ComponentEngineFeature::ControlledWasi
            )
        )
    ));
}
