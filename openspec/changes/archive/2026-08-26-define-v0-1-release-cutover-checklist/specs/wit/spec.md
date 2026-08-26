## ADDED Requirements

### Requirement: WIT Cutover Version Confirmation

Cutover SHALL confirm included WIT package versions before stable release.

#### Scenario: Missing WIT matrix

Given WIT package exists but is missing from compatibility matrix

When cutover runs

Then release is blocked.

---

### Requirement: WIT Cutover Gate Before Tag

WIT validation SHALL complete before stable release tag.

#### Scenario: Tag before WIT validation

Given stable tag is created before WIT validation

When cutover validates sequence

Then release is invalid.