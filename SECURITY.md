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

## Known gaps

These are tracked publicly and do not need a private report:

- Component signatures carry no cryptographic material and are not verified.
  Trust by publisher or source rests on manifest-declared identity. See
  [#9](https://github.com/astorise/Magnetar/issues/9).
- Component execution deadlines and fuel budgets are not enforced, so a
  Component can occupy a host thread indefinitely. See
  [#8](https://github.com/astorise/Magnetar/issues/8).

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
