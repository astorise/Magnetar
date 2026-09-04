## Why

An eleventh audit round of PR #36 (commit `83a5d92`, closing round 10's
three P0s) found that both round-10 seals hold only on the facades they
touched, not on the primitives underneath: `ModelLoadingCoordinator::load`
remains `pub` and still accepts a caller-constructed `ModelTrustDecision`
directly, letting a caller build their own `ModelTrustStore`, evaluate it
faithfully (not forge it -- `ModelTrustStore::evaluate` is legitimately
public), and load an artifact `Runtime`'s own sealed trust policy would
reject. Likewise `ModelInstanceDefinition::from_loaded_context` and
`ModelInstanceManager::create` remain `pub`, and `Runtime::
model_instances_mut()` exposes `&mut ModelInstanceManager`, so a caller can
construct a definition with whatever architecture/affinity they choose and
register it directly, bypassing `Runtime::create_model_instance`'s
architecture/plan and affinity/plan cross-checks entirely. Both bypasses
are not hypothetical: `magnetar-runtime/tests/contract_tests/`, compiled
as an external consumer crate, already exercises both paths today for
otherwise-legitimate test coverage of the loading state machine and of
`ModelInstanceManager::create`'s own reset-on-create guarantee.

The same round also found a real contradiction this session introduced:
round 10's new `model-instance` requirement says an architecture
implementation kind, capability, or Resource Affinity field the loading
phase left unresolved is "a legitimate choice for the caller ... SHALL NOT
be constrained" -- but `inference-api`'s pre-existing "Provider Preferences
Are Non-Authoritative" requirement says "Runtime SHALL own Provider and
Device selection." In real code, `ModelInstancePlacement::new` copies
`affinity.provider()`/`.device()` directly into the instance's effective
placement whenever the loading plan resolved nothing -- there is no
Runtime-side resolution step distinguishing "caller preference" from
"caller-authoritative placement" at instance-creation time today (the
existing `ResolutionPolicy`/`BuiltInResolutionPolicy` mechanism resolves
Capability/Provider candidates at execution time, not at instance
creation). The user chose to correct the spec text to describe this
honestly as a documented limitation rather than build a new Runtime-side
resolution mechanism to make the stronger claim true.

## What Changes

- **BREAKING** (crate-internal API only -- no production or CLI call site
  is affected, confirmed by inspection before starting): `ModelLoadingCoordinator::load`
  becomes `pub(crate)`. The only remaining way to reach it is through
  `inference_api::load_model`/`load_model_observed`, both already
  Runtime-sealed since round 10.
- **BREAKING** (same scope note): `ModelInstanceDefinition::
  from_loaded_context` and `ModelInstanceManager::create` become
  `pub(crate)`. The only remaining way to create a Model Instance is
  `Runtime::create_model_instance`, already cross-checked since round 10.
- `magnetar-runtime/tests/contract_tests/model_loading.rs`'s tests of the
  loading state machine's own behavior (untrusted-artifact rejection,
  memory-budget/quantization/allocation failure mapping, Ready-context
  shape) move into `magnetar-runtime/src/tests.rs`, where `pub(crate)`
  access to the now-sealed `load` still applies -- these were always
  testing `ModelLoadingCoordinator`'s own contract, not "the public API
  surface an external consumer would use," so this is a relocation, not a
  loss of coverage.
- `contract_tests/model_instance.rs`'s two tests that construct or clone a
  `ModelInstanceDefinition` directly and call `.create()` on it (proving
  `ModelInstanceManager::create` unconditionally resets
  `resource_bindings` regardless of what the supplied definition carried)
  move into `magnetar-runtime/src/tests.rs` for the same reason. Every
  other `contract_tests/model_instance.rs` test -- the ones exercising
  Model Instance lifecycle, KV cache, sharing, warmup, and unload behavior
  through a definition it never needed direct field/constructor access to
  build -- migrates to `Runtime::create_model_instance` in place, staying
  in `contract_tests` since that is exactly the public entrypoint an
  external consumer now uses.
- New external-bypass regression tests (`magnetar-runtime/src/tests.rs`,
  which can observe both the sealed primitive and `Runtime`'s public
  surface in the same test): a `Runtime` that does not trust a digest
  cannot be loaded into by a hand-built `ModelTrustStore`/`.load()` call
  from crate-internal code standing in for what an external caller could
  no longer even compile; equivalent for the instance-creation bypass.
- `model-instance`'s "Model Instance References Architecture
  Implementation" requirement's "SHALL NOT be constrained" language for
  unresolved plan fields is corrected to describe today's actual,
  narrower guarantee: an unresolved field's caller-supplied value becomes
  the effective placement directly, which is a documented limitation (no
  Runtime-side resolution exists yet for instance-creation-time
  Provider/Device selection), not an intentional grant of caller
  authority. `inference-api`'s "Provider Preferences Are Non-Authoritative"
  requirement gains a note cross-referencing this limitation so the two
  specs no longer read as contradictory.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `model-loading`: new requirement that `ModelLoadingCoordinator::load` is
  reachable only through the Runtime-sealed loading API, not as an
  independently public entrypoint.
- `model-instance`: new requirement that `ModelInstanceDefinition::
  from_loaded_context` and `ModelInstanceManager::create` are reachable
  only through `Runtime::create_model_instance`; "Model Instance
  References Architecture Implementation" requirement corrected to
  accurately describe unresolved-field behavior as a documented
  limitation rather than an intentional authority grant.
- `inference-api`: "Provider Preferences Are Non-Authoritative" gains a
  cross-reference to the same documented limitation, so both specs
  describe one consistent, honest picture of today's behavior.

## Impact

- `magnetar-runtime/src/model_loading.rs`: `ModelLoadingCoordinator::load`
  visibility.
- `magnetar-runtime/src/model_instance.rs`: `ModelInstanceDefinition::
  from_loaded_context` and `ModelInstanceManager::create` visibility.
- `magnetar-runtime/src/tests.rs`: ~6 relocated loading-state-machine
  tests, 2 relocated definition-clone/create tests, new external-bypass
  regression tests.
- `magnetar-runtime/tests/contract_tests/model_loading.rs`: reduced to
  whatever, if anything, still exercises genuinely public surface after
  relocation (expected: little to nothing left needing its own file --
  confirmed during implementation, not assumed here).
- `magnetar-runtime/tests/contract_tests/model_instance.rs`: ~13 sites
  migrated from `model_instances_mut().create(definition())` to
  `Runtime::create_model_instance`.
- `openspec/specs/model-loading/spec.md`,
  `openspec/specs/model-instance/spec.md`,
  `openspec/specs/inference-api/spec.md`: requirement text as described
  above.
- `CHANGELOG.md`: round-11 findings and fixes recorded in the existing
  Architecture Freeze #1 history.
