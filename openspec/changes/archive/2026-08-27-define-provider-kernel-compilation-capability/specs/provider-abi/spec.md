## ADDED Requirements

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