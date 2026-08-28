## ADDED Requirements

### Requirement: Optimization Worker Provider State Is Not Portable

Provider-native state created in Optimization Plane SHALL not be transferred as
portable production artifact.

#### Scenario: Worker prepares CUDA Kernel

Given optimization worker has CUfunction handle

When candidate moves to production

Then compiled artifact is transferred, not CUfunction.

---

### Requirement: Optimization Does Not Mutate Production Provider Implicitly

Optimization Campaign SHALL NOT silently modify active production Provider
state.

#### Scenario: Benchmark candidate

Given production CUDA Provider serves inference

When campaign benchmarks candidate

Then it uses explicitly authorized environment/provider instance rather than
replacing active production state.

---

### Requirement: Provider Compilation Remains Existing Capability

Optimization orchestration SHALL compose Provider Kernel Compilation
Capability instead of adding separate compiler API.

#### Scenario: Candidate needs PTX compilation

Given campaign compiles source

When Provider compiler is used

Then existing compilation contract applies.