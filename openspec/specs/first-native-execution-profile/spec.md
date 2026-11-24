# first-native-execution-profile Specification

## Purpose
TBD - created by archiving change define-first-native-model-execution-profile. Update Purpose after archive.
## Requirements
### Requirement: First Native Execution Profile Is Bounded

Magnetar SHALL define a bounded mandatory profile for the first native model
vertical slice.

#### Scenario: Advanced feature missing

Given implementation supports Reference CPU Qwen execution

But does not support multi-Device Tensor Parallel

When first-profile conformance runs

Then missing Tensor Parallel does not fail the profile.

### Requirement: Native Execution Uses Magnetar Kernels

Mandatory profile SHALL execute model computation through Magnetar Kernel and
Provider contracts.

#### Scenario: Candle path available

Given repository still contains temporary Candle Provider

When first native profile runs

Then Candle model execution is not used.

### Requirement: Reference CPU Is Mandatory Provider

First profile SHALL require Reference CPU Provider.

#### Scenario: No accelerator installed

Given host has only CPU

When profile runs

Then Qwen fixture can still execute end to end.

### Requirement: Single Device Is Sufficient

First profile SHALL require only one logical local Device.

#### Scenario: Host has one CPU Device

Given no second Device exists

When conformance runs

Then topology is sufficient.

### Requirement: F32 Is Mandatory Baseline

First profile SHALL require f32 model execution.

#### Scenario: Provider lacks fp16

Given Reference CPU supports f32

When fixture executes

Then absence of fp16 does not fail profile.

### Requirement: Advanced Optimization Is Deferred

First profile conformance SHALL permit implementations without generated
Kernels, autotuning, adaptive performance optimization, or hot swap.

#### Scenario: Static Kernel selection

Given one Reference CPU Kernel exists per Operator

When Runtime selects it through Registry

Then profile remains conformant.

### Requirement: Multi Device Is Deferred

Multi-Device placement and collectives SHALL not be mandatory.

#### Scenario: Multi-Device code unimplemented

Given single Device path works

When profile validates

Then implementation is acceptable.

### Requirement: Simplification Is Allowed But Bypass Is Not

A generic contract MAY have a simple baseline realization, but SHALL not be
removed from the mandatory architecture path.

#### Scenario: Synchronous CPU stream

Given ExecutionStream executes synchronously

When Kernel completes

Then completed CompletionToken satisfies the stream contract.

#### Scenario: Direct CPU function call bypass

Given Qwen execution skips Registry/Provider

When profile validates

Then implementation is non-conformant.

### Requirement: Causal First-Native Datapath
The first-native execution profile SHALL prove that local inference is caused by the Model Component, Runtime-validated ExecutionGraph, PreparedExecutionPlan, Runtime ProviderLoader, Runtime MemoryManager, ModelInstance resources, Runtime-owned KV cache, and sampling path.

#### Scenario: Baseline path is complete
- **WHEN** first-native inference completes successfully
- **THEN** the emitted evidence identifies each required datapath layer as causally used for the produced token.

#### Scenario: Shortcut fails conformance
- **WHEN** any required datapath layer is bypassed in the first-native profile
- **THEN** the first-native conformance check fails.

### Requirement: Attention nodes carry their own geometry

The first-native engine SHALL forward an `attention` node's `head_count`, `kv_head_count`, `head_dimension`, `window_size` and `scale` to the Kernel, and SHALL use the architecture's geometry only for attributes the node omits. It SHALL send `scale` only to a Provider whose every active `attention` Kernel advertises `attention-scale`, and SHALL otherwise refuse the dispatch.

#### Scenario: Provider without attention scale support
- **WHEN** a graph's `attention` node sets `scale` and the bound Provider's attention Kernels do not advertise `attention-scale`
- **THEN** dispatch fails with a Kernel-unavailable error instead of computing with the default scale

### Requirement: Optional operator forms reach only Providers that declare them

The first-native engine SHALL forward each layered-model form to the Kernel:
- `embedding.scale`;
- `rmsnorm` with one input and its per-head attributes;
- `rope.partial_rotary_factor`;
- `gelu` with its attributes;
- `moe-experts` with its `activation` and a sixth input.

It SHALL first require each Kernel feature `required_kernel_features` names for the node from every active Kernel of the bound Provider for that operator, and SHALL otherwise refuse the dispatch.

#### Scenario: Provider ignoring partial rotation
- **WHEN** a `rope` node sets `partial_rotary_factor = 0.25` and the bound Provider's `rope` Kernels do not advertise `partial-rotary`
- **THEN** dispatch fails with a Kernel-unavailable error

#### Scenario: Reference CPU
- **WHEN** a graph uses every layered-model form on the Reference CPU
- **THEN** it runs, and its logits match an independent host computation

### Requirement: The sampled logits row is selected through row-wise nodes

When a prefill samples one row, the Runtime SHALL compute the LM head on that row only. This applies when the head is a plain or bias matmul producing `"logits"` directly, or through row-wise unary nodes that are each the only reader of their input.

#### Scenario: Logit softcap after the head
- **WHEN** a bucket graph's `"logits"` is a softcap over its LM head's output
- **THEN** the head reads the gathered row only, and the logits equal the uncapped graph's logits capped
