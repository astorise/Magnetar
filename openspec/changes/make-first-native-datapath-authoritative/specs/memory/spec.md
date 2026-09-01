## ADDED Requirements

### Requirement: Runtime Memory Accounts Compute Resources
First-native outputs, workspaces, model weight resources, and KV resources SHALL be allocated, tracked, and released through the Runtime MemoryManager.

#### Scenario: Runtime memory limit is exceeded
- **WHEN** a first-native compute step requires memory beyond the Runtime MemoryManager limit
- **THEN** the compute step fails before unaccounted allocation occurs.

#### Scenario: Workspace lifecycle ends
- **WHEN** a provider workspace is no longer needed
- **THEN** Runtime memory accounting releases the workspace according to its lifecycle.
