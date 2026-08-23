# Provider Conformance Suite

The Provider Conformance Suite defines the compatibility bar for Magnetar
Providers. A Provider is conformant because it passes Runtime-facing contract
checks, not because it can be loaded or registered.

The suite is scoped to Magnetar inference Runtime responsibilities:

- Provider metadata, identity, version, vendor metadata, and Runtime API
  compatibility.
- Capability advertisement and advertised Compute behavior.
- Device metadata, Provider ownership, duplicate Device detection, and public
  metadata redaction.
- Provider lifecycle, health, readiness, pressure, admission, stale-status, and
  status observation fields.
- Compute operation validation for advertised operation families.
- Data movement validation for advertised movement kinds.
- Resource Affinity preservation for Provider-owned and Device-bound resources.
- Cancellation support or explicit unsupported-cancellation reporting.
- Stable Provider error mapping and redacted diagnostics.
- Dynamic Provider ABI descriptor and loading-policy checks where dynamic
  loading is enabled.

The suite does not test filesystem, Git, shell, workspace, secret, network, or
general tool behavior. Those authorities are outside the Provider contract.

## Targets

A conformance target is a Provider implementation plus the configuration needed
to exercise it through normal Runtime contracts.

Supported target kinds are:

- `BuiltIn`: an in-process Provider compiled into the Runtime.
- `Mock`: a deterministic Provider fixture used by tests and default CI.
- `DynamicLibrary`: a native Provider library checked through explicit loading
  policy and ABI conformance.
- `Development`: a dynamic Provider target allowed only when development mode
  is explicit.

Default CI uses mock or built-in targets and SHALL NOT require GPU hardware,
vendor drivers, Tachyon, or network access.

## Profiles

Initial profiles are:

- `provider-core`: metadata, registration, Device metadata, status, admission,
  and redaction.
- `provider-compute`: Compute Capability advertisement and operation validation.
- `provider-data-movement`: upload, download, copy, materialize, transfer,
  dtype conversion, placement conversion, host-staging policy, and output
  affinity validation where advertised.
- `provider-cancellation`: cancellation support or stable unsupported result.
- `provider-observability`: status observation identity and redacted
  diagnostics.
- `provider-dynamic-abi`: dynamic factory symbol, ABI version, descriptor,
  function table, memory ownership, release/destroy behavior, and loading
  policy.

Optional hardware profiles are placeholders for CUDA, Metal, OpenVINO, and QNN.
They are opt-in and separately enabled. Passing a default CI profile does not
claim hardware-specific conformance.

## Report Format

The suite emits a structured report through `ProviderConformanceReport`.

The report includes:

- Provider identity and version.
- Runtime version.
- Conformance suite version.
- Target kind.
- Selected profiles.
- Passed tests.
- Failed tests.
- Skipped tests.
- Unsupported optional features.
- Redacted diagnostics.
- Unix timestamp.

Use `provider_conformance_report_json` to produce machine-readable JSON output
for CI artifacts or local review.

## Local Commands

Run the hardware-independent conformance tests:

```powershell
cargo test -p magnetar-runtime provider_conformance -- --nocapture
```

Run the full Runtime suite:

```powershell
cargo test --workspace --all-targets
```

Validate the OpenSpec change:

```powershell
openspec validate define-provider-conformance-suite --type change --strict
```

## Compatibility Versioning

The current suite version is `0.1.0`, exposed as
`PROVIDER_CONFORMANCE_SUITE_VERSION`.

Provider compatibility documentation must identify the suite version that was
passed. Passing one suite version does not imply passing future versions.

Providers that fail required profiles are non-conformant. Development mode may
load non-conformant Providers for local testing, but compatibility status must
make that visible.
