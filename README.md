# Magnetar

> **A Universal Compute Runtime for AI**

Magnetar is an open-source compute runtime written in Rust for executing AI workloads across heterogeneous hardware.

Unlike traditional inference engines, Magnetar is **hardware-agnostic** and **model-agnostic**. It provides a portable execution layer capable of orchestrating computation graphs on CPUs, GPUs, NPUs, TPUs and future accelerators through a modular provider architecture.

The goal of Magnetar is not to replace machine learning frameworks, but to become the execution runtime sitting between AI applications and compute devices.

---

## Vision

Modern AI evolves faster than execution engines.

Every few months new model architectures emerge:

- Large Language Models
- Diffusion Models
- Vision Models
- Audio Models
- Multimodal Models
- State Space Models

At the same time, compute hardware keeps diversifying:

- CPUs
- NVIDIA CUDA
- AMD ROCm
- Apple Metal
- Vulkan
- Intel NPUs
- Qualcomm AI Engine
- Google TPUs
- Future AI accelerators

Instead of coupling model implementations to hardware-specific code, Magnetar separates concerns through a unified execution runtime.

---

## Core Principles

- **Hardware Agnostic**
- **Model Agnostic**
- **Provider First**
- **Compiler Driven**
- **Zero Vendor Lock-in**
- **Portable Execution**
- **Zero-Cost Abstractions**
- **Rust Native**

---

## High-Level Architecture

```text
                +----------------------+
                |   AI Applications    |
                +----------+-----------+
                           |
                     Stable Runtime API
                           |
+-----------------------------------------------------------+
|                         Magnetar                          |
|-----------------------------------------------------------|
| Runtime                                                   |
| Scheduler                                                 |
| Graph Compiler                                            |
| Graph Optimizer                                           |
| Memory Planner                                            |
| Device Manager                                            |
| Provider Loader                                             |
| Capability Registry                                       |
+-------------------------+---------------------------------+
                          |
        +-----------------+-----------------------------+
        |                 |             |               |
     CPU Backend     CUDA Backend   Metal Backend   NPU Backend
        |                 |             |               |
     Optimized        Optimized     Optimized      Optimized
      Kernels          Kernels       Kernels        Kernels
```

---

## Architecture

Magnetar is organized as a collection of independent crates.

The source-derived boundaries for future Compute, model, generation, and
application contracts are documented in the
[capability contract taxonomy](docs/architecture/capability-taxonomy.md).

```text
magnetar
â”œâ”€â”€ magnetar-runtime
â”œâ”€â”€ magnetar-core
â”œâ”€â”€ magnetar-ir
â”œâ”€â”€ magnetar-compiler
â”œâ”€â”€ magnetar-memory
â”œâ”€â”€ magnetar-provider
â”œâ”€â”€ magnetar-device
â”œâ”€â”€ magnetar-kernel
â”‚
â”œâ”€â”€ magnetar-cpu
â”œâ”€â”€ magnetar-cuda
â”œâ”€â”€ magnetar-metal
â”œâ”€â”€ magnetar-vulkan
â”œâ”€â”€ magnetar-openvino
â”œâ”€â”€ magnetar-qnn
â”œâ”€â”€ magnetar-webgpu
â”‚
â”œâ”€â”€ magnetar-gguf
â”œâ”€â”€ magnetar-huggingface
â”œâ”€â”€ magnetar-onnx
â””â”€â”€ magnetar-safetensors
```

---

## Execution Pipeline

```text
          Model
            â”‚
            â–¼
     Model Loader
            â”‚
            â–¼
     Intermediate Graph
            â”‚
            â–¼
      Graph Compiler
            â”‚
            â–¼
    Optimization Passes
            â”‚
            â–¼
     Execution Planner
            â”‚
            â–¼
         Scheduler
            â”‚
            â–¼
     Selected Backend
            â”‚
            â–¼
         Hardware
```

---

## Features

### Runtime

- Stable execution API
- Hardware abstraction
- Runtime lifecycle
- Device discovery
- Backend selection

### Initial runtime architecture

The `magnetar-runtime` crate is the hardware-independent entry point. It owns
runtime lifecycle and backend registration, while concrete backends own device
discovery. The runtime can be initialized without any backend; callers select a
registered backend only when one is needed.

```text
Application
    |
    v
Runtime / RuntimeBuilder ----> registered Backend implementations
    |                                  |
    v                                  v
ExecutionContext                   Device implementations

Components declare WIT capability imports. The runtime resolves those imports
through the Capability Registry to one or more Providers; the first matching
Provider is used first and the remaining compatible Providers are fallbacks.
```

### Public API example

```rust
use magnetar_runtime::{Runtime, RuntimeConfig};

let mut runtime = Runtime::initialize(RuntimeConfig::default());
assert!(runtime.is_initialized());

// Backends are optional and may be registered and selected later.
runtime.shutdown();
```

### Compiler

- Intermediate Representation (IR)
- Graph lowering
- Operator fusion
- Memory planning
- Constant folding
- Dead code elimination

### Scheduler

- Multi-device execution
- Asynchronous scheduling
- Automatic backend selection
- Capability-aware execution

### Providers

- Dynamic loading
- Independent versioning
- Capability discovery
- Hot extensibility

### Backends

Planned support:

- CPU
- CUDA
- Metal
- Vulkan
- WebGPU
- Intel OpenVINO
- Qualcomm QNN
- AMD ROCm
- Future NPUs
- Future TPUs

---

## Roadmap

### Phase 1

- Runtime
- CPU backend
- Provider API

### Phase 2

- Intermediate Representation
- Graph execution
- Graph compiler

### Phase 3

- Graph optimization
- Memory planner
- Scheduler

### Phase 4

- CUDA backend
- Metal backend
- Vulkan backend

### Phase 5

- Multi-device execution
- Distributed execution
- Production optimizations

---

## Non Goals

Magnetar is **not**:

- a machine learning framework
- a training framework
- a tensor library
- a Python runtime
- a model zoo

Magnetar focuses exclusively on **portable AI execution**.

---

## Why Rust?

Rust enables Magnetar to deliver:

- predictable performance
- memory safety
- fearless concurrency
- zero-cost abstractions
- native portability

without requiring a garbage collector.

---

## ðŸ“„ License

Distributed under the MIT License. See `LICENSE` for more information.

---

## Status

âš ï¸ Magnetar is in active development.

The architecture is evolving rapidly and APIs should be considered unstable until the first stable release.
