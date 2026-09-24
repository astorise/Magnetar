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

Component artifact trust is policy-driven: digest pinning, a verified Ed25519
signature from an operator-trusted publisher key, or an explicit local
development policy may allow an artifact; publisher/source metadata alone is
never treated as cryptographic proof.

Component and Model Artifact signatures (`ComponentTrustStore`/
`ModelTrustStore`'s `trusted_publisher_keys`/`revoked_keys`) are verified
Ed25519 signatures (RFC 8032) over the artifact's existing SHA-256 content
digest, composing with digest pinning as a second, independent trust path
rather than replacing it -- every artifact trusted by digest pinning alone
continues to be trusted identically whether or not it also carries a
signature. A signature under an unrecognized key is treated the same as no
signature (falls through to digest pinning); a signature under a *known*
key that fails to verify is rejected outright, a stronger negative signal
than absence; a revoked key's signature is rejected even when the key was
previously trusted. The full design -- what gets signed, key
identification, trust/revocation, and the fail-closed verification
precedence -- is recorded in
[`docs/cryptographic-artifact-signatures.md`](docs/cryptographic-artifact-signatures.md).

## Known gaps

These are tracked publicly and do not need a private report:

- Trusted publisher keys and revocations are operator-configured local
  policy only (the same distribution model digest pinning already has): no
  key registry, discovery protocol, "well-known keys" list, or online
  revocation checking (OCSP/CRL-style) exists. An operator wanting timely
  revocation propagation is responsible for updating their local trust
  store, the same way they already are for `revoked_digests` today.
- Providers are trusted native code by architectural definition (see
  Scope above) and are explicitly out of scope for signature verification;
  a malicious Provider is outside this threat model regardless.
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
