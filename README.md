# Magnetar

> **A Universal Compute Runtime for AI**

Magnetar is an open-source compute runtime written in Rust for executing AI workloads across heterogeneous hardware.

Unlike traditional inference engines, Magnetar is **hardware-agnostic** and **model-agnostic**. It provides a portable execution layer capable of orchestrating computation graphs on CPUs, GPUs, NPUs, TPUs and future accelerators through a modular plugin architecture.

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
- **Plugin First**
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
| Plugin Loader                                             |
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

```text
magnetar
├── magnetar-runtime
├── magnetar-core
├── magnetar-ir
├── magnetar-compiler
├── magnetar-memory
├── magnetar-plugin
├── magnetar-device
├── magnetar-kernel
│
├── magnetar-cpu
├── magnetar-cuda
├── magnetar-metal
├── magnetar-vulkan
├── magnetar-openvino
├── magnetar-qnn
├── magnetar-webgpu
│
├── magnetar-gguf
├── magnetar-huggingface
├── magnetar-onnx
└── magnetar-safetensors
```

---

## Execution Pipeline

```text
          Model
            │
            ▼
     Model Loader
            │
            ▼
     Intermediate Graph
            │
            ▼
      Graph Compiler
            │
            ▼
    Optimization Passes
            │
            ▼
     Execution Planner
            │
            ▼
         Scheduler
            │
            ▼
     Selected Backend
            │
            ▼
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

### Plugins

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
- Plugin API

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

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## Status

⚠️ Magnetar is in active development.

The architecture is evolving rapidly and APIs should be considered unstable until the first stable release.
