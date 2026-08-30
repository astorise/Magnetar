# provider-abi Specification

## Purpose
This specification defines the native Provider ABI's optional, versioned extension model, including buffer ownership, panic/exception normalization, and opaque identifier handling across the ABI boundary, starting with the Kernel Compilation ABI extension.
## Requirements
### Requirement: Kernel Compilation ABI Extension

Native Provider ABI SHALL expose Kernel Compilation as an optional versioned
extension rather than a Rust ABI surface.

#### Scenario: ABI v1 Provider without extension

Given Provider ABI v1 loads successfully

And Provider has no Kernel Compilation extension

When Runtime initializes it

Then Provider remains valid.

---

### Requirement: Compilation ABI Has Explicit Version

Kernel Compilation ABI extension SHALL have explicit version.

#### Scenario: Unsupported extension version

Given Provider exposes future incompatible compilation ABI

When Runtime loads it

Then extension is rejected without invalidating unrelated Provider capabilities
where policy permits.

---

### Requirement: Compilation ABI Buffer Ownership

Every buffer crossing compilation ABI SHALL have explicit allocation and
release ownership.

#### Scenario: Compiler returns binary

Given Provider allocates compiled output buffer

When Runtime consumes it

Then Provider-defined release operation is used according to ABI contract.

---

### Requirement: No Unwind Across Compilation ABI

Provider SHALL normalize panics/exceptions before crossing ABI.

#### Scenario: Compiler wrapper panics

Given internal Provider compiler wrapper fails

When control returns to Runtime

Then structured ABI error is returned rather than unwind.

---

### Requirement: Prepared Kernel Identifier Is Opaque ABI Value

When PreparedKernelId crosses the native ABI, it SHALL be represented as an opaque numeric identifier and SHALL NOT be treated as a native pointer.

#### Scenario: uint64 identifier

Given Provider returns `42`

When Runtime stores it

Then Runtime does not treat `42` as memory address.

---

### Requirement: ABI Compilation Jobs Are Opaque

Compilation job identifier SHALL be opaque.

#### Scenario: Provider compiler subprocess

Given Provider internally tracks OS process

When Runtime polls job

Then Runtime receives job ID, not process ID or process handle.

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
