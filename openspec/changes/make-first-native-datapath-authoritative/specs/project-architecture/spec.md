## ADDED Requirements

### Requirement: Architecture Freeze Status Matches Evidence
Architecture Freeze #1 status SHALL remain candidate until P0 first-native causal datapath requirements are implemented and proven by CI evidence.

#### Scenario: P0 datapath remains incomplete
- **WHEN** any P0 causal datapath requirement remains incomplete
- **THEN** README, OpenSpec notes, and release status do not claim the freeze is accepted.

#### Scenario: Freeze is accepted
- **WHEN** all P0 datapath requirements pass with linked commit and CI evidence
- **THEN** project status documents may mark Architecture Freeze #1 accepted.
