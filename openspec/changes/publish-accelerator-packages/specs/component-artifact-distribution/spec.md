## ADDED Requirements

### Requirement: Accelerator Package And Evidence Media Types

Release publication SHALL publish accelerator package envelopes as `application/vnd.magnetar.accelerator-package.v1+json` and kernel evidence documents as `application/vnd.magnetar.kernel-evidence.v1+json`. It SHALL attach evidence as a referrer of the kernel bundle it describes, with the same manual-dispatch and new-digest-on-correction rules as Component and kernel bundle publication.

#### Scenario: A corrected evidence document

- **WHEN** a published evidence document is found wrong
- **THEN** a new document with a new digest is published, and the old one is never overwritten
