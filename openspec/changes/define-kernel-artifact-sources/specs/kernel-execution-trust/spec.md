## ADDED Requirements

### Requirement: Kernel Trust Store

The Runtime SHALL evaluate kernel bundle trust against a deployment-owned kernel trust store holding trusted publisher Ed25519 keys, revoked keys and pinned manifest digests. It SHALL verify a bundle's signature envelope over the manifest digest with the same verifier Component and Model Artifacts use. The outcome SHALL be trusted (with the publisher bound to the key), untrusted, revoked or invalid, and only a verified signature under a trusted, unrevoked key, or a pinned manifest digest, SHALL yield trusted.

#### Scenario: A signed bundle from a trusted publisher

- **WHEN** a bundle's envelope signs its manifest digest with a trusted, unrevoked key
- **THEN** the bundle is trusted, and the decision records the publisher and key identifier

#### Scenario: A signature over another digest

- **WHEN** the envelope's signed digest is not the bundle's manifest digest
- **THEN** the bundle is invalid, whatever the signature verifies against

#### Scenario: A revoked key

- **WHEN** the signing key is listed as revoked
- **THEN** the bundle is revoked, and it stays revoked on re-import

#### Scenario: Envelope metadata alone

- **WHEN** a manifest carries a publisher claim and a signature envelope that does not verify
- **THEN** the bundle is not trusted, and the claim is recorded as a claim only

### Requirement: Execution Permission Is Separate From Trust

The Runtime SHALL require, before asking a Provider to prepare a kernel artifact, a deployment-granted execution permission for that Provider identity, artifact format and publisher, in addition to trust. Permissions SHALL be denied by default. Neither trust, a passed qualification nor a Component's own signature SHALL grant one.

#### Scenario: A trusted bundle without permission

- **WHEN** a trusted `nvidia:cubin` bundle has no execution permission for the CUDA Provider
- **THEN** it is not prepared, and the decision names the missing permission

#### Scenario: Permission is per format

- **WHEN** a publisher holds permission for `nvidia:cubin` on the CUDA Provider
- **THEN** a native CPU shared object from the same publisher still needs its own permission

### Requirement: Embedded Bundles Are Verified As Kernels

The Runtime SHALL verify a bundle served by a Component as a kernel bundle, with its own envelope and trust decision. The Component Artifact's trust SHALL NOT extend to the bundles it serves.

#### Scenario: A trusted Component serves an unsigned bundle

- **WHEN** a trusted, signed kernel-package Component serves a bundle with no kernel signature, under the production policy
- **THEN** the bundle is untrusted and is quarantined or rejected by policy

### Requirement: Preparation Eligibility

The Runtime SHALL consider a kernel artifact eligible for Provider preparation only when it is trusted, permitted for that Provider, qualified for the requested envelope, compatible with the target, and neither revoked nor quarantined.

#### Scenario: Revocation after qualification

- **WHEN** a qualified, prepared artifact's publisher key is revoked
- **THEN** it becomes ineligible, receives no new work, and follows the registry's revocation policy for work in flight
