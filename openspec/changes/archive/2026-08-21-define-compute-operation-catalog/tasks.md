# Tasks

## Catalog Structure

- [x] Define ComputeOperationFamily
- [x] Define ComputeOperationDescriptor placeholder
- [x] Define operation family identifiers
- [x] Define operation family metadata

## Operation Families

- [x] Add descriptor and view operations
- [x] Add construction and allocation request operations
- [x] Add data movement and conversion operations
- [x] Add elementwise operations
- [x] Add comparison and selection operations
- [x] Add reduction operations
- [x] Add linear algebra operations
- [x] Add convolution and spatial transform operations
- [x] Add indexing and update operations
- [x] Add random generation operations
- [x] Add synchronization and completion operations

## Provider Advertisement

- [x] Allow Providers to advertise supported operation families
- [x] Allow Providers to advertise dtype support per operation family
- [x] Allow Providers to advertise layout constraints per operation family
- [x] Allow Providers to advertise precision constraints per operation family

## Runtime Validation

- [x] Validate that submitted compute work only uses known operation families
- [x] Validate Provider support before execution
- [x] Return structured unsupported-operation-family errors
- [x] Return structured unsupported-dtype errors
- [x] Return structured unsupported-layout errors

## Exclusions

- [x] Exclude autograd from the initial catalog
- [x] Exclude training graphs from the initial catalog
- [x] Exclude arbitrary Rust custom operations from WIT
- [x] Exclude backend-specific kernel names from portable contracts
- [x] Exclude direct device queue synchronization from Components

## Documentation

- [x] Document operation family scope
- [x] Document relationship with `magnetar:compute/run`
- [x] Document why operation families are not separate Capabilities
- [x] Document future operation schema work
