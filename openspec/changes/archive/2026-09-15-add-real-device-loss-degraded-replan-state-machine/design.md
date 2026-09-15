## Context

This is the last of the four real gaps identified after `add-real-second-gpu-cuda-provider` closed: peer-to-peer movement and per-Device memory feasibility ranking were both closed with genuine hardware-verified behavior; Device loss is structurally different -- there is no safe way to make a real physical GPU disappear from a live, shared CI node mid-test. This change draws the same honest line the rest of this repository's development has followed throughout: implement and verify everything that genuinely can be, and explicitly, prominently document the one boundary that cannot be crossed here rather than silently mislabeling a software simulation as a hardware proof.

## Goals / Non-Goals

**Goals:**
- Exercise `magnetar_runtime::multi_device_placement`'s real, existing state machine with a real, Device-derived `MultiDevicePlacementPlan` for the first time.
- Prove the real, enforced invariant that an `Invalidated` Plan cannot silently revert to `Ready` -- a caller must build a new Plan generation instead.
- Be explicit, in both the test's own doc comment and this design, about exactly what is and is not hardware-verified here.

**Non-Goals:**
- Real hardware-failure injection (forcing an actual GPU to disappear) -- no safe mechanism exists.
- Wiring Device-loss detection into any real Provider/Runtime code path that would automatically drive this transition -- this change proves the state machine correctly enforces its own rules once transitioned into, not that anything currently detects real loss and triggers it.
- The `Stale`/`Retiring`/`Retired`/`Failed` states and their own transitions -- this change exercises `Ready` -> `Invalidated` specifically, matching the two requirements it targets; the module's own pre-existing synthetic tests already cover the rest of the state graph.

## Decisions

- **Use a real, Device-derived Plan with a caller-driven transition, rather than skip this item or simulate a fake Device-loss detector.** Two alternatives were considered: (a) skip this item as "not achievable," and (b) build a fake in-process "Device health monitor" that watches for a synthetic failure signal and calls `transition_to` automatically. (a) was rejected as leaving a real, closeable gap unclosed; (b) was rejected because building a fake detector would risk implying more automation exists than actually does -- nothing in this codebase currently detects real Device loss, and pretending otherwise would misrepresent this repository's real capabilities. The chosen approach -- real Plan data, explicit caller-driven transition, prominently documented as such -- is the most honest achievable proof.
- **Test the revert-rejection, not just the forward transition.** `Ready -> Invalidated` alone would only prove the state machine accepts an expected transition (already covered by its own synthetic tests). Also asserting that `Invalidated -> Ready` is rejected, and that the rejection leaves `plan.state` unchanged, is what actually demonstrates "Degraded Placement Requires Valid Plan" and "Placement Change Uses New Plan Generation" (a caller cannot cheaply resurrect a stale Plan -- it must build a new one).

## Risks / Trade-offs

- **This test's "Device loss" is not hardware-verified** -- a reader unfamiliar with this constraint could mistake it for the same class of proof as the peer-to-peer or memory-feasibility tests. Mitigated by the test's own prominent doc comment and this design.md stating the boundary explicitly, matching this repository's own established honesty convention.
