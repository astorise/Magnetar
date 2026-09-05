## Context

`reach-architecture-freeze-1` made the first-native graph executor's
node-to-node transport Resource-ID-*addressed*: `execute_qwen_graph_nodes`'s
`bindings` map is `BTreeMap<TensorEdgeId, TensorResourceId>`, never a private
`HostTensor` cache. That change's own investigation (tasks 3.3, 5.4-5.6) found
the remaining gap is narrower than "make the executor generic" -- it already
is. What is not generic is the *value* the trait exchanges at each Resource
ID: `ProviderExecutionApi::read_tensor`/`write_tensor`/`write_tensor_admitted`
are typed `HostTensor` -- Reference CPU's own host-visible tensor
representation, documented on `HostTensor` itself as "only Reference CPU
reads or writes this."

Two concrete things are blocked by that typing, independent of whether any
non-Reference-CPU Provider exists today:

1. `device-resident-resource`'s existing spec requires a Tensor Resource be
   able to "exist and execute entirely Device-side without an authoritative
   host byte buffer." A trait method typed `HostTensor` cannot honor that for
   a Provider that has no host-visible representation to offer.
2. Multi-output support (Correctif 5's 5.4/5.5) has no natural shape to grow
   into: Reference CPU's Kernels all call `store_output` at index 0 only, and
   there is no portable "this Kernel produced N outputs" description that
   does not first assume what an output value looks like.

Today, exactly one Provider (Reference CPU) does any real compute, and its
own Kernel implementations (`dispatch_qwen_matmul`, `rmsnorm`,
`rope_per_head`, attention, ...) are ordinary Rust functions operating on
`HostTensor.data: Vec<f32>` directly -- that is fine and expected to remain
true; a device-resident Provider's own Kernel implementations will operate on
its own device buffers, never touching `HostTensor`. The problem is narrower:
`HostTensor` leaks into the *generic* dispatch layer
(`execute_qwen_graph_nodes`, `dispatch_reference_cpu_operator`) that every
Provider's dispatch passes through, not just Reference CPU's own Kernel
bodies.

## Goals / Non-Goals

**Goals:**
- Define a Provider-agnostic tensor **value** contract on
  `ProviderExecutionApi` that a device-resident-only Provider can implement
  without ever exposing or accepting `HostTensor`.
- Define how a Kernel invocation reports **multiple** outputs under that
  contract, replacing the implicit "output index 0 only" assumption.
- Keep the change additive to the existing trait, not a breaking rewrite of
  Reference CPU or the dispatch pipeline's control flow.

**Non-Goals:**
- Migrating every `dispatch_qwen_*` call site in `first_native_runtime.rs`
  off `HostTensor` in this Change. Several of those genuinely need
  host-side computation today (RoPE's per-head rotation, attention,
  `concat_rows` for KV history) because Reference CPU's Kernel bodies *are*
  plain Rust over `Vec<f32>` -- that migration is real, node-type-by-node-type
  work this Change unblocks but does not itself perform. It becomes
  `reach-architecture-freeze-1`'s tasks 3.3/5.4/5.5/5.6 once this contract
  exists.
- Implementing a real GPU/CUDA/Metal Provider. This Change defines the
  contract a future one would implement against; it does not build one.
- Full device-resident zero-copy execution (`device-resident-resource`'s
  complete vision: replicas, peer access, async completion tracking). This
  Change only needs a Provider to be able to *decline* host materialization
  for a resource it holds -- not to implement every residency requirement
  that spec eventually describes.
- Changing `KernelSelectionRequest`/`KernelAdvertisement`-level Kernel
  *selection* semantics. This is about the value Kernels exchange once
  selected and dispatched, not how they are chosen.

## Status / Definition of Done

Per the post-freeze équipe review (`magnetar-pr36-decisions-equipe-2026-09-02.md`,
decisions 7-10): this Change's Goals section above states intent, not
completion, and multi-output specifically has landed only at the type level.
The actual Definition of Done for this Change is narrower than "every Goal
fully wired end to end":

- **Done:** a Provider-agnostic `TensorValue` contract exists on
  `ProviderExecutionApi` (`read_tensor_value`/`write_tensor_value`/
  `write_tensor_value_admitted`), a device-resident-only Provider can
  implement it without ever exposing `HostTensor` (proven by
  `DeviceResidentOnlyExecutor`, task 4.3), and `execute_qwen_graph_nodes`'s
  generic per-node transport reads/writes through it exclusively, with
  `TensorValue::into_host` as the explicit, structured-error boundary at
  every point that genuinely needs host bytes (weight binding, KV-history
  concatenation, final logits extraction, each node's own Kernel-input
  resolution) -- task group 5, closed.
- **Done, type-level only:** multi-output readiness. `KernelResult
  .updated_resources: Vec<TensorResourceDescriptor>` already carries an
  output-index -> `TensorResourceId` shape (task 3.1) -- the contract does
  not need to change again to support a real multi-output Kernel later.
- **Not done, deliberately deferred (task group 3, 3.2-3.5):** a real
  two-output Reference CPU Kernel, `execute_qwen_graph_nodes` resolving
  `node.outputs` plural, and an E2E multi-output test. No real Qwen operator
  produces more than one output today, so building this now would mean
  exercising invented plumbing against a synthetic Kernel rather than a real
  one -- acceptable follow-up work once a real multi-output operator exists,
  not a blocker for this Change or for `reach-architecture-freeze-1`.

So: **Provider-agnostic `TensorValue` contract + a Provider API able to
represent device-resident values + type-level readiness for future
multi-output** is this Change's actual Definition of Done. Full multi-output
*wiring* is out of scope here and tracked as `reach-architecture-freeze-1`
follow-up instead (see that change's tasks 5.4/5.5).

## Decisions

### Decision 1: Additive new methods, not a breaking signature change

**Choice:** Add new `ProviderExecutionApi` methods (`read_tensor_value`,
`write_tensor_value`, `write_tensor_value_admitted`) typed against the new
`TensorValue` (Decision 2), alongside the existing `read_tensor`/
`write_tensor`/`write_tensor_admitted`, which keep their current `HostTensor`
signature unchanged. The generic dispatch layer
(`execute_qwen_graph_nodes`'s per-node transport) migrates to the new
methods; the existing `HostTensor`-typed ones remain for hand-written test
oracles (`execute_qwen_prefill_hidden_states_through_dispatch` and similar,
which deliberately want raw `HostTensor` access to build fixtures) and any
caller that already knows it only ever talks to a host-visible Provider.

**Alternatives considered:**
- *Break the existing signatures in place* (what `proposal.md`'s first draft
  assumed): forces every current call site, including test oracles that have
  no reason to change, to migrate simultaneously. Rejected: the
  `write_tensor_admitted`/`release_admitted_tensor` precedent
  (`reach-architecture-freeze-1`, task 1.8) already established the additive
  pattern for exactly this kind of "new capability alongside an existing,
  narrower method" situation, at lower risk and with a smaller diff.
- *Associated type on `ProviderExecutionApi`* (`type Value: ...`): would make
  the trait not object-safe, breaking `Arc<dyn ProviderExecutionApi>`, which
  Correctif 3's generic Provider resolution depends on entirely. Rejected
  outright -- incompatible with the architecture `reach-architecture-freeze-1`
  just finished establishing.

### Decision 2: `TensorValue` as a small closed enum, not `Box<dyn Any>`

**Choice:**

```rust
pub enum TensorValue {
    /// Host-visible bytes. What every current Provider (Reference CPU) and
    /// test double produces and consumes today.
    Host(HostTensor),
    /// The Provider holds this value privately and declines to expose host
    /// bytes for it. Callers that only need to move data between two
    /// Kernels on the *same* Provider never need to unwrap this; a caller
    /// that genuinely needs host bytes (weight binding, KV-history concat,
    /// final logits extraction) gets a structured error instead of `None`,
    /// naming *why* (device-resident, not merely "not found").
    Opaque,
}

impl TensorValue {
    pub fn into_host(self) -> Result<HostTensor, TensorValueError> { ... }
}
```

**Alternatives considered:**
- `Box<dyn Any + Send + Sync>` with caller-side downcasting: reintroduces
  exactly the downcast pattern Correctif 3 (task group 3) removed from the
  generic dispatch path. Rejected on that precedent alone.
- An open, Provider-extensible value trait (`trait TensorValueRepr`): more
  "correct" in the abstract, but every consumer of a `TensorValue` (weight
  binding, KV concat, logits extraction, the three real host-materialization
  call sites identified in Context) would need to be generic over it or
  downcast anyway -- the closed two-variant enum gets the same practical
  benefit (a device-resident Provider can decline host bytes) with none of
  that complexity, and can grow a third variant later without breaking
  callers that already match on it exhaustively-with-a-catch-all.

### Decision 3: Multi-output via a `BTreeMap<usize, TensorResourceId>` on the dispatch result, not N return values

**Choice:** `KernelDispatchResult`/the Kernel-level completion path gains an
output-index -> `TensorResourceId` map (today's single implicit "index 0"
becomes a map with one entry). `execute_qwen_graph_nodes` stops assuming
`node.outputs.first()`; it resolves each declared output edge against the
Kernel's own output-index map (populated from `KernelResource` order in the
invocation, matching how inputs are already ordered).

**Alternatives considered:**
- A fixed-size array / tuple of outputs: rejected, Kernels do not have a
  compile-time-known output count in this generic path.
- Changing `KernelInvocation`'s `outputs: Vec<...>` (already a `Vec`, already
  supports multiple *declared* outputs) but leaving `execute_invocation`
  (Reference CPU) still writing only index 0: this is closer to the actual
  gap -- the *declaration* shape already supports N outputs; only the
  Reference CPU Kernel bodies and the graph executor's consumption of the
  result need to stop assuming one. This decision is really "finish wiring
  the `Vec` that already exists," not inventing a new shape.

## Risks / Trade-offs

- [Two tensor-value pathways coexist indefinitely (`HostTensor`-typed and
  `TensorValue`-typed methods on the same trait) if the follow-up migration
  in `reach-architecture-freeze-1` stalls] → Mitigate with a static guard
  test (matching this session's established pattern for similar
  "workaround must not spread" concerns) asserting the *count* of
  `HostTensor`-typed call sites in `execute_qwen_graph_nodes`'s per-node
  loop does not grow, so the additive path cannot silently become the
  permanent one by accretion.
- [`TensorValue::Opaque` callers that genuinely need host bytes get a new
  failure mode (structured error) where today they always got a value] →
  Correct and intended: Reference CPU never produces `Opaque` (it always
  returns `Host`), so no existing behavior changes until a real
  device-resident Provider exists. The failure mode exists so that Provider,
  when it does exist, fails closed and legibly rather than the caller
  silently getting stale or wrong host bytes.
- [Multi-output map adds a lookup indirection to every single-output node's
  hot path (the overwhelming majority today)] → Single-output remains the
  index-0 entry in a one-element map; the lookup is a `BTreeMap` get on a map
  with one key for every node that exists today, not a new branch in the
  common case.
