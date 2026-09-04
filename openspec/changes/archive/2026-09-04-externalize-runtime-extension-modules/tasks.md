## 1. CI guard generalization

- [x] 1.1 Replace `.github/workflows/quality.yml`'s format-crate-only dependency guard (`submodule-integration` job) with a generalized check matching `magnetar-(component|format|provider)-` against both `magnetar-runtime/Cargo.toml` and its resolved `cargo tree --all-features` output.
- [x] 1.2 Verify the generalized guard passes today (it must — every externalized module already conforms). (Verified locally: manifest grep and `cargo tree --all-features` both clean.)
- [x] 1.3 Verify the generalized guard actually catches a violation: temporarily add a dummy path dependency from `magnetar-runtime` to one externalized module's crate, confirm the guard fails, then revert. (Added a temporary `magnetar-provider-cpu` path dependency to `magnetar-runtime/Cargo.toml`, confirmed the manifest-grep half of the guard correctly flags it, then reverted via `git checkout --`. The `cargo tree` half was not separately live-tested — it relies on well-established, unmodified `cargo tree` behavior (any resolved dependency's package name appears in its output), not a novel mechanism this task wrote.)

## 2. Documentation cross-references

- [x] 2.1 Update `reach-architecture-freeze-1/design.md`'s open question about Change C to record that it has been decided (formalized as a normative requirement) and point at this Change.
- [x] 2.2 Update `SUBMODULES.md` to reference this Change's requirement as the normative source for "why every module lives externally," rather than only describing it as a versioning/pinning convention.

## 3. Verification and closure

- [x] 3.1 `openspec validate --strict externalize-runtime-extension-modules`. (Valid.)
- [x] 3.2 Confirm the generalized CI guard is syntactically valid YAML and runs in the same job/step position the format-only guard occupied. (`python -c "import yaml; yaml.safe_load(...)"` clean; step occupies the exact position the format-only step held, within `submodule-integration`.)
- [x] 3.3 Confirm no other CI job or code path assumed the old, narrower format-only guard's exact wording or step name. (Grepped the whole repository for the old step name/wording — no other reference existed; it was a leaf step with nothing depending on its exact text.)
