## Why

MAG-06 from the Tachyon integration audit (`docs/audits/audit-magnetar-integration-tachyon-2026-09-13.md`) flagged that `LoadedInferenceComponent` holds `Mutex<ProductionQwenLoadedModel>` and serializes generation on one resident instance, with no explicit statement of whether that is the intended concurrency model or an oversight. The audit's own framing is explicit that this is not a Tachyon problem -- it is entirely Magnetar's own design decision to make and state.

## What Changes

- `LoadedInferenceComponent` gains a doc comment stating the concurrency model explicitly: one active generation at a time per instance (the pre-existing `Mutex` behavior, unchanged), a concurrent call blocks rather than being rejected or interleaved, and this is a deliberate consequence of `ProductionQwenLoadedModel` owning one `Runtime`/KV-cache-bearing `ModelInstance` per loaded Component -- not a limitation this crate works around by providing its own scheduler, batching, or instance pool. An embedder wanting concurrent generation across independent requests loads multiple instances and routes across them itself.
- `invoke_payload`/`invoke_payload_streaming` each gain a one-line pointer to that doc comment.
- **BREAKING**: none. Documentation only -- the actual locking behavior (`Mutex::lock`, blocking, not `try_lock`) is unchanged; this change states what it already does, it does not change what it does.

## Capabilities

### New Capabilities
(none)

### Modified Capabilities
- `component`: adds "A Resident Component Instance Serializes Its Own Generation" -- states the pre-existing, already-enforced (by `ProductionQwenLoadedModel::generate`/`generate_streaming` taking `&mut self`, not by this change's own new code) per-instance concurrency guarantee formally, closing MAG-06.

## Impact

- `inference-components/src/lib.rs`: doc comments only, no behavior change.
- README/`docs/audits/audit-magnetar-integration-tachyon-2026-09-13.md`: MAG-06 moves from open to closed.
