## ADDED Requirements

### Requirement: Runtime Provider Executes First-Native Bindings
First-native compute SHALL execute through the Provider resolved from Runtime-owned provider registration for the prepared plan binding.

#### Scenario: Registered provider is replaced by a mock
- **WHEN** a mock provider is registered for the binding
- **THEN** first-native execution submits work to that mock provider.

#### Scenario: Bound provider is unavailable
- **WHEN** the provider bound by a ready plan is removed or invalidated
- **THEN** Runtime rejects new execution for that binding.
