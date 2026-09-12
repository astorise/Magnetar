## Why

CUDA multi-step decode fails today not because the CUDA attention kernel can't handle growing KV history (it already computes a correct `query_position_offset` for `kv_seq_len > seq_len`), but because the Runtime's KV-history concatenation step is plain Rust over host `Vec<f32>` bytes: it calls `provider.read_tensor_value(..).into_host(..)`, which `CudaProvider` correctly declines (`TensorValue::Opaque`) since it never downloads a device-resident tensor without being asked to. `close-tachyon-scope-audit-gaps` made this limitation an explicit, checked-early `InferenceApiError::Unsupported` instead of a deep internal error, but the limitation itself -- and the two further host round-trips in the pending/commit path a real fix must also close -- remain. This is the one gap both external audits and the Tachyon scope charter named as the concrete blocker before Tachyon-Mesh can route real multi-token CUDA generation through Magnetar, and it is now well enough understood (root cause is one call, not an architectural rewrite) to close.

## What Changes

- Add a new portable `concat` Operator (`OperatorFamily::Tensor`, a new `ShapeRule::RowConcat`) to the Operator catalog, dispatched through the exact same Kernel Registry / `PreparedExecutionPlan` / dispatch machinery every other graph node already uses -- not a KV-specific trait method. Implemented on Reference CPU (`magnetar-runtime`'s in-crate double and the `providers/cpu` submodule, for parity/conformance) and CUDA (`providers/cuda`, a device-to-device buffer copy -- no new `.cu` kernel is required).
- `execute_qwen_graph_nodes`'s KV-history-append step (`first_native_runtime.rs`) dispatches historical-K/V concatenation through this new Operator instead of `provider.read_tensor_value(..).into_host(..)`, so the concatenated `[history + 1, kv_dim]` result is produced and stays device-resident for a Provider that holds KV device-resident. Reference CPU's own decode path is unaffected (same result, dispatched the same way every other node already is).
- Add a new, genuinely generic `ProviderExecutionApi::copy_tensor` method (a Tensor Resource duplicated to a fresh caller-chosen id, without host materialization) and use it to close the two remaining host round-trips in the decode pending-write and KV-commit paths (`promote_pending_kv_layer_role`), which succeed on CUDA today but pay a full download+upload per layer per role per step.
- `CudaProvider::supports_multi_step_decode()` returns `true` (the trait default), now truthfully: a multi-step decode request against CUDA proceeds and produces real, correct, multi-token output instead of failing fast with `Unsupported`.
- Fix a real, adjacent bookkeeping inconsistency found while implementing this: CUDA's own KV-commit path records `MemoryPlacement::ProviderOwnedOpaque`, while every other CUDA Device-resident write records `MemoryPlacement::Device` -- reconciled so Memory Manager accounting for CUDA KV state is consistent.
- Update the two tests `close-tachyon-scope-audit-gaps` added that currently assert CUDA multi-step decode *fails* (`multi_step_decode_on_real_cuda_hardware_fails_with_explicit_unsupported`, and the `max_tokens: 1` pins in `tests_production_loading_cuda_e2e.rs`/`tests_real_checkpoint_smoke.rs`) to instead prove it *succeeds*, matches Reference CPU output exactly, and does not grow Device memory unboundedly across steps -- verified on real CUDA hardware, both locally and via `gpu-runner-smoke.yml`.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `provider-prepared-kernel-execution`: adds a new requirement for a Provider-agnostic, host-materialization-free Tensor Resource copy, used to close the KV pending/commit round-trips. (The new `concat` Operator catalog entry itself is an implementation detail of the existing, already-generic "Operator Catalog"/"Shape Contract" requirements -- same precedent as this session's own `ShapeRule::RowBroadcastAdd` and the pre-existing "split" Kernel, neither of which needed a spec delta -- so `operator` is not listed as modified.)
- `cuda-provider`: adds a CUDA-specific conformance requirement that this Provider concretely satisfies `kv-cache`'s already-existing, generic "KV Cache Supports Device Residency" requirement (whose own scenario already states "KV does not require host round-trip between steps") -- CUDA is the Provider that did not yet satisfy it; `kv-cache` itself needs no text change, since it already anticipated exactly this.

(`generation`'s existing `Unsupported` fail-closed requirement, added by `close-tachyon-scope-audit-gaps`, is already Provider-agnostic in its wording and needs no text change; only the *test* exercising its CUDA-specific example is retargeted at a synthetic Provider, since CUDA no longer belongs in that scenario.)

## Impact

- `magnetar-runtime`: `operator.rs` (new Operator + `ShapeRule`), `first_native_runtime.rs` (KV-history-append dispatch, pending-write path), `reference_cpu.rs` (new Operator implementation), `provider.rs` (new `copy_tensor` method on `ProviderExecutionApi`, default-implementable or required -- decided in design).
- `providers/cpu` (submodule): new `concat` Operator implementation, for CPU/CUDA conformance parity.
- `providers/cuda` (submodule): new `concat` Operator implementation (device-to-device copy), new `copy_tensor` implementation, `supports_multi_step_decode` reverts to the trait default `true`, `MemoryPlacement` fix in the KV-commit path.
- `integration-tests/production-loading`: `tests_production_loading_cuda_e2e.rs`/`tests_real_checkpoint_smoke.rs` real-hardware tests generalized from prefill-only (`max_tokens: 1`) to genuine multi-token decode, with CPU/CUDA output parity and bounded-memory-growth assertions; `tests_production_loading_e2e.rs`'s synthetic-Provider `Unsupported` test is retargeted off CUDA (still proves the generic gate).
- No WIT contract changes (this is entirely below the Component/Capability boundary -- Components never see Tensor Resources or Providers). No changes to Resource Affinity, Memory Manager admission policy, Model Loading's lifecycle, or the Qwen Component's declared graph shapes (avoiding a Component fingerprint change).
- Out of scope, explicitly deferred: a ring-buffer/pre-allocated-capacity KV cache (this change matches Reference CPU's existing growing-fresh-buffer-per-step semantics on CUDA too, which is a real, bounded, but not maximally efficient design -- an optimization, not a correctness gap); native CUDA FP16/BF16 compute; multi-device or tensor-parallel decode; paged KV cache; quantization; GGUF; additional Providers or Model Components.
