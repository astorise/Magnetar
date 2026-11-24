## ADDED Requirements

### Requirement: Attention nodes carry their own geometry

The first-native engine SHALL forward an `attention` node's `head_count`, `kv_head_count`, `head_dimension`, `window_size` and `scale` to the Kernel, and SHALL use the architecture's geometry only for attributes the node omits. It SHALL send `scale` only to a Provider whose every active `attention` Kernel advertises `attention-scale`, and SHALL otherwise refuse the dispatch.

#### Scenario: Provider without attention scale support
- **WHEN** a graph's `attention` node sets `scale` and the bound Provider's attention Kernels do not advertise `attention-scale`
- **THEN** dispatch fails with a Kernel-unavailable error instead of computing with the default scale
