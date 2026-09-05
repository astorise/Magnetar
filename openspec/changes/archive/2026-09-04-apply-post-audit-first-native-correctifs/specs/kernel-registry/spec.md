## ADDED Requirements

### Requirement: First-Native Qwen Operators Resolve Through Registry
Every executable operator in the first-native Qwen graph SHALL resolve through Kernel Registry before Provider dispatch.

#### Scenario: Operator dispatch has registry lineage
- **WHEN** Runtime executes a Qwen graph node
- **THEN** evidence links GraphNodeId to KernelRegistryResolutionId, KernelId, PreparedKernelId, ProviderSubmissionId, and CompletionId.

#### Scenario: Required kernel disabled
- **WHEN** a required Qwen kernel is unavailable or disabled
- **THEN** plan preparation or execution fails rather than calling a direct Reference CPU bypass.

### Requirement: Reference CPU Direct Calls Are Not Model E2E Execution
Direct Reference CPU kernel functions SHALL NOT execute the authoritative first-native Qwen model E2E path.

#### Scenario: Direct calls remain unit-test-only
- **WHEN** Reference CPU functions are used in unit tests, qualification oracles, or differential tests
- **THEN** they are allowed only outside the authoritative model E2E execution path.
