# Security Policy

## Scope

Magnetar executes untrusted input by design. The parts of the runtime where a
security report is most likely to apply are:

- **Component loading and execution.** Components are portable WebAssembly and
  are treated as untrusted. Reports about sandbox escape, resource-limit
  bypass, or a Component obtaining native handles are in scope.
- **Artifact and manifest parsing.** Component manifests, trust stores, and
  model manifests are parsed from files that may arrive alongside an untrusted
  artifact.
- **Trust decisions.** Anything that causes an artifact to be accepted that the
  configured trust policy should have rejected.

Providers are trusted native code by architectural definition. A malicious
Provider is outside the threat model; the operator chooses which Providers to
load.

## Implemented controls

The native Wasmtime Component engine is configured to run Components without
ambient WASI authority. Filesystem, environment, network, process, and runtime
resource imports are rejected unless Magnetar explicitly maps an allowed host
capability.

Wasmtime execution uses fuel metering and epoch-based interruption for
deadlines. Component resource limits, including memory limits, are part of the
Runtime Component policy surface and are enforced when the Wasmtime engine is
available.

Component artifact trust is still policy-driven: digest pinning or an explicit
local development policy may allow an artifact, while publisher/source metadata
alone is not treated as cryptographic proof.

## Known gaps

These are tracked publicly and do not need a private report:

- Component signatures carry no cryptographic material and are not verified.
  Publisher/source metadata alone is treated as `Unknown` and does not satisfy
  trust policy; acceptance still requires digest pinning or explicit local
  development policy. Design work for cryptographic artifact signatures and
  authenticated publisher identity is tracked in
  [#37](https://github.com/astorise/Magnetar/issues/37).
- Non-Wasmtime or future Component engines must provide equivalent fuel,
  deadline/interruption, resource-limit, and no-ambient-authority guarantees
  before they can satisfy the same native security profile.

## Reporting

Report suspected vulnerabilities through GitHub's private vulnerability
reporting on this repository ("Security" tab, "Report a vulnerability"). Please
include what you executed, what you expected the runtime to prevent, and what
happened instead.

Please do not open a public issue for a vulnerability that is not already
listed above.

## Status

Magnetar is pre-1.0 and its APIs are unstable. There are no security-supported
released versions yet; fixes land on `main`.
