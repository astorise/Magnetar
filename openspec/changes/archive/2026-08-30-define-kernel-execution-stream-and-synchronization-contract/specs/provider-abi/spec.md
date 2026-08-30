## ADDED Requirements

### Requirement: Synchronization ABI Extension Is Optional

Native Provider ABI SHALL allow an optional versioned asynchronous synchronization
extension.

#### Scenario: Existing v1 Provider

Given Provider supports only synchronous Kernel execution

When loaded

Then it remains valid without asynchronous extension where conformance profile
permits synchronous execution.

### Requirement: Synchronization ABI Is C-Compatible

Synchronization ABI extension SHALL not expose Rust trait objects or Rust ABI
types.

#### Scenario: Native Provider implemented in C++

Given Provider implements stream operations

When ABI is inspected

Then operations use stable C-compatible descriptors and opaque identifiers.

### Requirement: Stream Identifier Is Opaque Across ABI

Provider ABI stream identifier SHALL not be interpreted by Runtime as native
pointer.

#### Scenario: uint64 stream ID

Given Provider returns value 91

When Runtime stores it

Then Runtime treats it as Provider-scoped opaque token.

### Requirement: Completion Identifier Is Opaque Across ABI

Provider ABI completion identifier SHALL be Provider-owned opaque state.

#### Scenario: Native event backing token

Given Provider internally maps token 53 to CUDA event

When Runtime polls token 53

Then Runtime never obtains CUDA event pointer.

### Requirement: ABI Failure Does Not Unwind Across Boundary

Provider asynchronous ABI operations SHALL normalize failures without unwinding
through ABI boundary.

#### Scenario: Provider throws internally

Given implementation encounters native exception

When ABI returns

Then failure is converted to structured Provider status/error.

### Requirement: ABI Ownership Is Explicit

Synchronization extension SHALL define ownership/release responsibility for
streams, completions, and returned buffers.

#### Scenario: Completion released

Given Runtime no longer requires token

When release operation occurs

Then Provider may safely destroy native event state according to ownership
contract.
