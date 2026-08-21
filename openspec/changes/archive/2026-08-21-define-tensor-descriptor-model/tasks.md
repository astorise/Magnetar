# Tasks

## Descriptor Types

- [x] Define TensorDescriptor
- [x] Define ShapeDescriptor
- [x] Define DTypeDescriptor
- [x] Define LayoutDescriptor
- [x] Define ViewDescriptor
- [x] Define TensorResourceDescriptor

## Fixed-Width Values

- [x] Use fixed-width integer types for dimensions
- [x] Use fixed-width integer types for strides
- [x] Use fixed-width integer types for offsets
- [x] Reject platform-sized integer assumptions

## Shape Validation

- [x] Validate rank limits
- [x] Validate dimension limits
- [x] Validate element count overflow
- [x] Validate byte-size overflow
- [x] Validate zero-sized tensor policy

## DType Validation

- [x] Define initial dtype set
- [x] Define unsupported dtype error
- [x] Define Provider dtype advertisement
- [x] Define dtype compatibility checks

## Layout and Views

- [x] Define contiguous layout descriptor
- [x] Define portable strided layout descriptor
- [x] Define opaque Provider-managed layout descriptor
- [x] Define view descriptor rules
- [x] Distinguish views from materialized copies

## Resource Model

- [x] Represent tensor storage as opaque resources
- [x] Attach Resource Affinity to tensor resources
- [x] Prevent raw storage exposure through WIT
- [x] Prevent native handle exposure through WIT

## Runtime Validation

- [x] Validate descriptors before compute submission
- [x] Validate Provider descriptor support
- [x] Return structured invalid-shape errors
- [x] Return structured unsupported-dtype errors
- [x] Return structured unsupported-layout errors

## Documentation

- [x] Document tensor descriptor semantics
- [x] Document view versus materialized copy semantics
- [x] Document relationship to `magnetar:compute/run`
- [x] Document exclusions from Candle Tensor semantics