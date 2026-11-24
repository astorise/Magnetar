## ADDED Requirements

### Requirement: Kernel Artifact Source

The Runtime SHALL obtain Kernel Exchange Bundle bytes only through a Kernel Artifact Source of a declared kind (Component-served, local path, authorized OCI locator, or cache), whose reads the Runtime's ingestion gateway drives and bounds. A source SHALL expose bundle descriptors and chunked reads only: no handle, path, URL or credential that the Runtime would follow on the source's own authority.

#### Scenario: A source serves a bundle

- **GIVEN** a source that describes one bundle by manifest digest and declared size
- **WHEN** the gateway reads it in chunks within the policy's chunk size and quota
- **THEN** the staged bytes enter an ingestion transaction whose validation is unchanged

#### Scenario: A source cannot redirect the Runtime

- **WHEN** a source descriptor or manifest names a location outside the configured source authority
- **THEN** the Runtime does not fetch it and records a structured rejection

### Requirement: Bounded Source Reads

The ingestion gateway SHALL cap every chunk at the policy chunk size, reject a chunk longer than requested, stop at the policy's cumulative byte quota and chunk-count limit whatever size the source declares, enforce a deadline, and hash bytes as it reads them.

#### Scenario: The declared size is a lie

- **WHEN** a source declares 4 KiB and keeps returning bytes past the policy quota
- **THEN** the transaction fails at the quota, nothing is staged, and the failure names the source and the quota

#### Scenario: An oversized chunk

- **WHEN** a source returns more bytes than the requested chunk size
- **THEN** the transaction fails without staging any byte of that chunk

#### Scenario: A source that never finishes

- **WHEN** a source keeps answering within the quota but past the deadline
- **THEN** the transaction is cancelled and its staged bytes are discarded

### Requirement: Distinct Kernel Artifact Identities

The Runtime SHALL distinguish the manifest digest (the canonical manifest bytes), blob digests (each blob), the bundle identity (the manifest digest with the verified set of blob digests it references) and the transport digest (an archive or OCI packaging). Trust, cache, registry and audit records SHALL key on the bundle identity, or on the manifest digest for signatures. A transport digest SHALL NOT grant trust or select a cached artifact.

#### Scenario: A repacked bundle

- **WHEN** the same manifest and blobs arrive in two archives with different transport digests
- **THEN** both resolve to the same bundle identity, and the cache holds one artifact

#### Scenario: A transport digest is not trust

- **WHEN** an OCI digest matches a pinned value but the manifest digest does not
- **THEN** the bundle is not trusted by that pin

### Requirement: Kernel Delivery Lifecycle

The Runtime SHALL record, per bundle identity and Provider, the delivery stage reached: discovered, staged, ingested (accepted, quarantined or rejected), qualified, prepared, benchmarked, ready, active, draining, retired or rolled back. It SHALL record which authority moved it there. A stage SHALL be entered only from its predecessor and only by its authority. Receiving bytes SHALL NOT imply preparation, and preparation SHALL NOT imply activation.

#### Scenario: Delivery does not execute

- **WHEN** a bundle is staged and accepted but not qualified
- **THEN** no Provider is asked to prepare it, and no registry candidate references it

#### Scenario: A skipped stage is refused

- **WHEN** a request asks to activate a bundle that is prepared but not promoted
- **THEN** the request fails with a lifecycle error that names the missing stage

### Requirement: Kernel Package Component Role

A Component in the `kernel-package` role SHALL act only as a Kernel Artifact Source. Its bundles SHALL go through the same gateway, ingestion, trust, permission and qualification as any other source. It SHALL NOT register kernels, select Providers or Devices, or receive native handles.

#### Scenario: A kernel-package Component asks to register

- **WHEN** a kernel-package Component's bundle is ingested
- **THEN** any registry entry for it is created under Provider authority after preparation, never under `KernelRegistrationAuthority::Component`
