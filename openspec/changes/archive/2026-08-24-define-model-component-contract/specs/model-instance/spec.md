## ADDED Requirements
### Requirement: Model Instance References Architecture Implementation

A Model Instance SHALL be able to reference the Model Component or Runtime-native
architecture implementation used to create it.

#### Scenario: Instance compatibility

Given a Model Instance was created with Model Component C

When cache compatibility is evaluated

Then C's identity and version may be considered.

---

### Requirement: Model Instance Does Not Grant Component Authority

Referencing a Model Component from a Model Instance SHALL not grant additional
authority to the Component.

#### Scenario: Instance references component

Given Model Instance references Component C

When C requests network access

Then Runtime still denies forbidden authority.