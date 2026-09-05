## ADDED Requirements

### Requirement: Loaded Artifact Resources Feed Execution
Model loading SHALL create the weight and constant resources used by first-native execution.

#### Scenario: Artifact bytes change
- **WHEN** the loaded model artifact bytes change and validation succeeds
- **THEN** first-native numerical outputs reflect the resources loaded from those bytes.

#### Scenario: Required weight missing
- **WHEN** a model artifact lacks a required weight resource
- **THEN** loading or binding fails before compute reads a substitute source.
