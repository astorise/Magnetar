## ADDED Requirements

### Requirement: Prefill And Decode May Use Different Kernels

Runtime SHALL ensure prefill and decode Kernel selections independently satisfy eligibility and feasibility constraints, and generation MAY select distinct eligible Kernels for each phase.

#### Scenario: Attention phases

Given throughput-oriented prefill and latency-oriented decode implementations

When generation executes

Then Runtime may use different Kernels by phase.

---

### Requirement: Generation Preference Is High-Level

Generation request MAY express optimization preference but SHALL NOT directly
provide PreparedKernelId or native handles.

#### Scenario: Low-latency request

Given request prefers low latency

When Runtime selects Kernel

Then preference maps into policy without bypassing eligibility.

---

### Requirement: Kernel Re-selection Preserves Generation Safety

Dynamic Kernel selection SHALL preserve valid in-flight Prepared Kernel
lifetime.

#### Scenario: Kernel changes between decode steps

Given policy allows re-selection between safe boundaries

When new Kernel is selected

Then no active invocation loses its Prepared Kernel state.