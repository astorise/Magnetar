# Tasks

## Data Movement Operations

- [x] Define Upload operation
- [x] Define Download operation
- [x] Define Copy operation
- [x] Define Materialize operation
- [x] Define Transfer operation
- [x] Define DType conversion operation
- [x] Define Placement conversion operation

## Host Data

- [x] Define HostBufferDescriptor
- [x] Define supported host data encodings
- [x] Define byte length validation
- [x] Define host-to-tensor upload rules
- [x] Define tensor-to-host download rules

## Tensor Resources

- [x] Allow upload to create TensorResource
- [x] Allow download from TensorResource
- [x] Allow copy between compatible TensorResources
- [x] Allow materialization of tensor views
- [x] Attach Resource Affinity to produced TensorResources

## Affinity Validation

- [x] Validate Provider affinity
- [x] Validate Device affinity
- [x] Validate Affinity Group compatibility
- [x] Reject incompatible resource movement
- [x] Require explicit movement between incompatible Providers or Devices

## Runtime Validation

- [x] Validate TensorDescriptor compatibility
- [x] Validate dtype support
- [x] Validate layout support
- [x] Validate byte size and element count
- [x] Validate transfer direction
- [x] Validate source and destination constraints

## Provider Advertisement

- [x] Allow Providers to advertise upload support
- [x] Allow Providers to advertise download support
- [x] Allow Providers to advertise copy support
- [x] Allow Providers to advertise materialization support
- [x] Allow Providers to advertise transfer support
- [x] Allow Providers to advertise supported dtypes and layouts

## Errors

- [x] Define unsupported-data-movement error
- [x] Define incompatible-affinity error
- [x] Define invalid-host-buffer error
- [x] Define invalid-transfer error
- [x] Define unsupported-conversion error
- [x] Define materialization-required error

## Documentation

- [x] Document explicit data movement model
- [x] Document no implicit CPU staging rule
- [x] Document view versus materialized copy movement
- [x] Document cross-Provider and cross-Device movement
- [x] Document relationship to `magnetar:compute/run`
