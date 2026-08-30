## ADDED Requirements

### Requirement: Device Does Not Own Synchronization API

Device abstraction SHALL remain descriptive/status-oriented and SHALL NOT expose
native synchronization creation or waiting methods.

#### Scenario: CUDA Device

Given Runtime sees CUDA-capable Device

When Device API is inspected

Then no `create_cuda_stream`, native-event, or queue-pointer operation exists.

### Requirement: Device Advertises Synchronization-Relevant Capabilities Through Provider Metadata

Provider/Device capability metadata SHALL describe properties relevant to
execution concurrency.

#### Scenario: Transfer overlap

Given hardware supports compute/transfer overlap

When capability discovery occurs

Then Runtime may learn this without receiving native transfer queue.

### Requirement: Device Loss Invalidates Bound Streams

Hard Device loss SHALL make bound ExecutionStreams unavailable for new work.

#### Scenario: GPU reset

Given logical streams target GPU

When Device becomes lost

Then Runtime prevents new submissions through those streams.
