## ADDED Requirements
### Requirement: Scheduler Uses Kernel Metadata

Scheduler SHALL use Kernel metadata for planning batched and asynchronous
execution.

#### Scenario: Batch size limit

Given a Kernel supports max batch size 8

When Scheduler forms a batch

Then it does not plan that Kernel for batch size 16.

---

### Requirement: Scheduler Respects Kernel Execution Mode

Scheduler SHALL respect Kernel execution mode, cancellation support, timeout,
and workspace lifetime.

#### Scenario: Asynchronous kernel

Given a Kernel is asynchronous

When Scheduler dispatches work

Then it tracks completion and resource lifetime accordingly.

---

### Requirement: Scheduler Does Not Select Raw Native Functions

Scheduler SHALL not select raw Kernel function pointers.

It SHALL operate on Runtime-validated Kernel metadata and invocations.

#### Scenario: Native function pointer

Given a Provider has internal native functions

When Scheduler plans execution

Then it uses Kernel metadata, not raw function addresses.
