## ADDED Requirements
### Requirement: Tensor Partition Descriptor

Runtime SHALL support explicit logical description of partitioned Tensor
Resources.

#### Scenario: Hidden dimension split

Given logical Tensor hidden size 8192 is split into two 4096 shards

When partition is inspected

Then shard ranges and parent relationship are explicit.

### Requirement: Tensor Shard Does Not Equal Full Tensor

A TensorShard SHALL not be consumed as complete Tensor unless Operator contract
explicitly permits that partitioned semantic.

#### Scenario: Non partition-aware RMSNorm

Given Kernel expects complete hidden dimension

When one shard is supplied

Then Runtime rejects binding or explicitly reconstructs required Tensor.

### Requirement: Partition Bounds Are Checked

Tensor partition ranges SHALL be validated for bounds and arithmetic overflow.

#### Scenario: Invalid shard extent

Given shard exceeds parent dimension

When descriptor is created

Then partition is rejected.

### Requirement: Non Replicated Partition Covers Declared Domain

A complete partition SHALL cover required logical domain without unexpected
gaps or overlap.

#### Scenario: Missing shard

Given dimension range 0..8192

But shards cover only 0..7168

When partition validates

Then missing coverage is reported.

### Requirement: Replica Is Distinct From Shard

Equivalent full copies SHALL be represented as replicas rather than partitions.

#### Scenario: Weight copied to GPU0 and GPU1

Given both represent complete same logical Tensor

When metadata is created

Then relationship is replication.

### Requirement: Shard Residency Is Explicit

Each TensorShard SHALL have explicit placement/residency.

#### Scenario: Two weight shards

Given shard 0 lives GPU0 and shard 1 GPU1

When partition is inspected

Then Device residency of both is known.

### Requirement: Partition Metadata Is Native Handle Free

Tensor partition descriptors SHALL not contain native Device addresses or peer
handles.

#### Scenario: CUDA shards

Given each has native pointer internally

When Core metadata is inspected

Then pointers are absent.