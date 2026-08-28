## ADDED Requirements

### Requirement: Qualified Kernel Artifact Metadata

When present, the QualificationRecord SHALL be immutable for the identified artifact and qualification profile.

Compiled Kernel Artifact MAY be associated with immutable QualificationRecord.

#### Scenario: Artifact qualified

Given compiled artifact passes qualification

When qualification record is stored

Then record identifies artifact digest, profile, suite, oracle and compatibility
context.

---

### Requirement: Qualification Does Not Mutate Compiled Content

Qualification SHALL NOT modify compiled artifact bytes in place.

#### Scenario: New qualification profile

Given same binary is tested against stricter profile

When result is stored

Then compiled artifact digest remains unchanged and a distinct qualification
record is produced.

---

### Requirement: Revoked Qualification

Revocation SHALL NOT alter the underlying compiled artifact bytes.

Qualification record MAY be revoked independently of artifact bytes.

#### Scenario: Test suite bug found

Given qualification procedure was invalid

When qualification evidence is revoked

Then artifact is no longer treated as qualified under that evidence.