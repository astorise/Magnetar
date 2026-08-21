# Tasks

## Graph Model

- [x] Define ComputeGraph
- [x] Define ComputeNode
- [x] Define ComputeValue
- [x] Define ComputeInput
- [x] Define ComputeOutput
- [x] Define graph identifier rules
- [x] Define graph validation lifecycle

## Operation Model

- [x] Define ComputeOperation placeholder
- [x] Reference Compute Operation Catalog families
- [x] Validate operation family identifiers
- [x] Validate node input descriptors
- [x] Validate node output descriptors

## Submission Model

- [x] Define ComputeSubmission
- [x] Define submit semantics
- [x] Define await semantics
- [x] Define cancel semantics
- [x] Define terminal state semantics

## Resource Model

- [x] Allow graph inputs to reference TensorResource
- [x] Allow graph inputs to define TensorDescriptor
- [x] Allow graph outputs to return TensorResource
- [x] Attach Resource Affinity to produced TensorResource values
- [x] Validate Resource Affinity before Provider execution

## Runtime Validation

- [x] Validate graph acyclicity
- [x] Validate input/output references
- [x] Validate tensor descriptor compatibility
- [x] Validate Provider operation family support
- [x] Validate dtype and layout support
- [x] Validate shape and size limits
- [x] Return structured graph validation errors

## Provider Execution

- [x] Submit validated graph to selected Provider
- [x] Preserve Provider-owned execution details
- [x] Preserve Device and Provider affinity
- [x] Prevent native handle exposure through WIT

## Documentation

- [x] Document coarse graph submission
- [x] Document relationship to `magnetar:compute/run`
- [x] Document why eager per-operation WIT calls are excluded
- [x] Document graph execution lifecycle
