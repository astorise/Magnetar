# Define Compute Graph Submission Model

## Why

`magnetar:compute/run` is the coarse execution boundary for portable compute.

The Tensor Descriptor Model defines how tensors are described without exposing
native storage.

The Runtime now needs a portable way to submit compute work using those tensor
descriptors and opaque tensor resources.

Magnetar must avoid one WIT call per eager tensor operation.

Instead, Components submit a validated compute graph, batch or equivalent coarse
unit to the Runtime.

Providers execute the submitted unit using native kernels, memory management,
device queues and synchronization.

This change defines the compute graph submission model without finalizing every
operation schema or numerical rule.

## What Changes

This proposal introduces:

- ComputeGraph
- ComputeNode
- ComputeValue
- ComputeInput
- ComputeOutput
- ComputeSubmission
- ComputeOperation
- ComputeOperationState
- ComputeExecutionResult

A ComputeGraph describes a portable compute unit.

A ComputeSubmission submits that unit to `magnetar:compute/run`.

The Runtime validates graph structure, tensor descriptors, resource affinity,
operation family support and Provider compatibility before execution.

The Provider owns native execution details.

This proposal does not define the complete operation catalog schemas.

This proposal does not introduce autograd or training graphs.

This proposal does not expose backend kernels, raw queues, GPU pointers,
Candle tensors or Provider storage.

## Impact

Components gain a stable coarse-grained compute submission model.

Providers can implement `magnetar:compute/run` without exposing eager tensor
APIs.

The Runtime can validate compute work before Provider execution.

Future changes can add concrete operation schemas inside this submission model.