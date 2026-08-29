## ADDED Requirements

### Requirement: Kernel Distribution Is Transport Neutral

Kernel Exchange Bundle SHALL not depend on one distribution backend.

#### Scenario: Local CI artifact

Given bundle is delivered from filesystem

When Runtime ingests it

Then same logical validation applies as bundle delivered from object storage.

---

### Requirement: External Location Is Not Identity

Artifact locator SHALL not replace digest identity.

#### Scenario: Registry object moved

Given identical artifact is moved to another registry path

When digest remains unchanged

Then content identity remains unchanged.

---

### Requirement: Runtime Does Not Follow Arbitrary URLs

Portable manifest SHALL not grant ambient network fetch authority.

#### Scenario: Manifest names HTTPS URL

Given Runtime has no authorized source for that URL

When artifact is required

Then structured external-reference denial/unavailable error is returned.

---

### Requirement: Bundle Path Traversal Rejected

Bundle loader SHALL reject entries escaping logical bundle root.

#### Scenario: Malicious archive

Given entry path is `../../provider.so`

When bundle is loaded

Then bundle is rejected.

---

### Requirement: Symlink Escape Rejected

Bundle SHALL not rely on symlink-resolved artifact paths.

#### Scenario: Blob symlink

Given digest path is symlink to `/etc/...`

When loaded

Then bundle is rejected.

---

### Requirement: Repacking Does Not Change Logical Identity

Transport metadata SHALL not define Kernel logical identity.

#### Scenario: Archive timestamp changes

Given same logical bundle is repacked with new timestamps

When logical identity is computed

Then Kernel Artifact identity is unchanged.