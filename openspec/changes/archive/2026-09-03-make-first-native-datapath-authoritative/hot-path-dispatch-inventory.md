## First-Native Hot-Path Dispatch Inventory

Task: 2.1 Identify all first-native hot-path dispatch sites that still perform per-node kernel selection.

### Production Hot Path

- `magnetar-runtime/src/first_native_runtime.rs::dispatch_matmul`
  - Builds a `KernelSelectionRequest`.
  - Calls `runtime.kernel_registry().select(&selection_request)`.
  - Builds `KernelDispatchPlan::from_selection`.
  - Calls `KernelDispatcher::revalidate`.
  - Used by logits projection and the forced-token test proof path.

- `magnetar-runtime/src/first_native_runtime.rs::dispatch_reference_cpu_operator`
  - Builds a `KernelSelectionRequest`.
  - Calls `ctx.runtime.kernel_registry().select(&selection_request)`.
  - Builds `KernelDispatchPlan::from_selection`.
  - Calls `KernelDispatcher::revalidate`.
  - Used by Qwen embedding, matmul, RMSNorm, RoPE, attention, unary, and binary operations.

### First-Native Callers Affected

- `execute_qwen_prefill_hidden_states_through_dispatch`
  - Executes the Qwen prefill sequence by repeatedly calling the dispatch helpers above.

- `execute_qwen_decode_hidden_states_through_dispatch`
  - Executes the Qwen decode sequence by repeatedly calling the dispatch helpers above.

- `dispatch_qwen_logits_projection`
  - Executes logits projection through `dispatch_matmul`.

### Planning-Time Selection That Remains Allowed

- `prepare_first_native_plan_for_graph`
  - Builds the prepared execution plan from graph nodes.
  - Kernel selection here is planning-time work and remains allowed.

- test/static coverage helpers that directly exercise registry selection
  - These are not the production first-native decode/prefill hot path.

### Migration Implication

The next implementation step should introduce a prepared plan executor that provides an execution context per `GraphNodeId`/operator using the already-published `PlanNodeBinding`. The existing dispatch helpers can then be converted from "select and execute" to "execute prepared binding", with planning-time selection kept inside plan preparation.
