## ADDED Requirements
### Requirement: Model Loading Prepares Adapter Compatibility

Model Loading SHALL expose metadata needed for later adapter validation.

#### Scenario: Loaded model target modules

Given a model is loaded

When an adapter targets its modules

Then Runtime can validate target modules against loaded model metadata.

---

### Requirement: Model Loading Does Not Implicitly Activate Adapter

Loading a base model SHALL NOT implicitly activate adapters.

#### Scenario: Base model ready

Given a base model is loaded

When no adapter activation request exists

Then the loaded model runs without adapters.