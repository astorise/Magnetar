## ADDED Requirements

### Requirement: Packages Are Built, Qualified And Signed From Source

A published accelerator package variant SHALL be built from source in CI, qualified on its target hardware against the Reference CPU kernels, and signed with the publisher's key. A variant without a qualification run for a target SHALL NOT be published for that target.

#### Scenario: A variant never run on its target

- **WHEN** a CUBIN variant for `sm_90` has no qualification run on an `sm_90` Device
- **THEN** it is not published for `sm_90`

### Requirement: Evidence Bound To The Artifact

Qualification and benchmark evidence SHALL be published as documents bound to the bundle identity and blob digest they describe, with target, profile and oracle identity. The Runtime SHALL revalidate evidence against current policy and the local target, and evidence for another target SHALL NOT qualify an artifact locally.

#### Scenario: Evidence from another driver

- **WHEN** published evidence names a driver other than the local one
- **THEN** the artifact needs a local qualification before preparation

### Requirement: Third-Party Ports Record Provenance And License

A kernel ported from a third-party source SHALL record its upstream repository, revision, path and license. Its license SHALL be on the publication allowlist, and its attribution SHALL ship with the bundle. Provenance SHALL NOT grant trust or qualification.

#### Scenario: A copyleft upstream

- **WHEN** a port's upstream license is not on the allowlist
- **THEN** publication fails

#### Scenario: A permissive Hugging Face port

- **WHEN** a port of an Apache-2.0 `kernels-community` CUDA source passes qualification against the Reference CPU kernel
- **THEN** it is published with its attribution, and trusted only through the publisher's signature

### Requirement: Discovery Does Not Install Or Execute

Catalog discovery SHALL read only envelope and evidence metadata from configured sources and SHALL NOT fetch bundle bytes, prepare or promote anything. Installation SHALL be an explicit operator action that takes each bundle through source, trust, execution permission, local qualification and preparation, and SHALL NOT promote without an explicit request.

#### Scenario: Listing packages

- **WHEN** an operator lists packages matching the local CUDA Device
- **THEN** variants and their evidence status are shown, and no kernel bytes are downloaded

### Requirement: Signed Revocation Feed

Publishers SHALL be able to publish a signed revocation list of keys and manifest digests with a monotonic sequence number. The Runtime SHALL apply a list only on explicit refresh, SHALL refuse a list older than the last applied one, and SHALL apply entries through the kernel trust store and registry revocation.

#### Scenario: A replayed old list

- **WHEN** a list with a lower sequence number than the last applied one is offered
- **THEN** it is refused, and the current revocations stay in force
