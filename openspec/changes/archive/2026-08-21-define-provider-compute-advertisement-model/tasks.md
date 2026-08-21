# Tasks

## Advertisement Types

- [x] Define ProviderComputeAdvertisement
- [x] Define ComputeCapabilitySupport
- [x] Define OperationFamilySupport
- [x] Define OperationSchemaSupport
- [x] Define DTypeSupport
- [x] Define LayoutSupport
- [x] Define ShapeLimitSupport
- [x] Define PrecisionSupport
- [x] Define DataMovementSupport
- [x] Define DeviceComputeSupport

## Capability Support

- [x] Advertise supported `magnetar:compute/run` versions
- [x] Advertise supported operation catalog revision
- [x] Advertise supported operation schema revision
- [x] Advertise experimental extension support separately from portable support

## Operation Support

- [x] Advertise supported operation families
- [x] Advertise supported operation schemas
- [x] Advertise unsupported operation schemas explicitly when useful
- [x] Advertise Provider-specific operation extensions as non-portable

## DType and Layout Support

- [x] Advertise supported dtypes per operation schema
- [x] Advertise supported input layouts per operation schema
- [x] Advertise supported output layouts per operation schema
- [x] Advertise view support
- [x] Advertise materialization requirements

## Shape and Size Limits

- [x] Advertise maximum tensor rank
- [x] Advertise maximum dimension value
- [x] Advertise maximum element count
- [x] Advertise maximum byte size
- [x] Advertise supported broadcasting limits
- [x] Advertise supported batch dimensions

## Precision and Determinism

- [x] Advertise accumulation dtype support
- [x] Advertise approximate math support
- [x] Advertise deterministic execution support
- [x] Advertise random generation determinism support
- [x] Advertise precision policy constraints

## Data Movement

- [x] Advertise upload support
- [x] Advertise download support
- [x] Advertise copy support
- [x] Advertise transfer support
- [x] Advertise materialization support
- [x] Advertise dtype conversion support
- [x] Advertise layout conversion support
- [x] Advertise host-staged transfer requirements

## Device-Specific Support

- [x] Attach advertisements to Device identifiers when support differs by Device
- [x] Advertise memory constraints per Device
- [x] Advertise operation constraints per Device
- [x] Advertise data movement constraints per Device

## Runtime Integration

- [x] Use advertisements during Capability resolution
- [x] Use advertisements during Resolution Policy evaluation
- [x] Use advertisements during Compute Graph validation
- [x] Use advertisements during data movement validation
- [x] Return structured unsupported-advertisement errors

## Documentation

- [x] Document Provider advertisement format
- [x] Document portable versus Provider-specific support
- [x] Document Device-specific advertisements
- [x] Document examples for CPU, CUDA and OpenVINO Providers
- [x] Document how advertisements affect Provider selection