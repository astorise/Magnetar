# Modularize Runtime Core

## Why

`magnetar-runtime` had grown into a monolithic `src/lib.rs` containing Runtime orchestration, Components, Capabilities, Providers, Devices, Resource Affinity, Resolution, Compute, Planning, Scheduler, Provider execution, Provider health, and Observability.

That layout was useful while the architecture was being discovered, but it is now the wrong foundation for the next stage. The important point is not only readability: this modularization creates the stable base on which later Runtime and AI work can land without turning the crate root into a new roadmap-shaped monolith.

The intended layering is:

```text
                   magnetar-runtime
                          |
            +-------------+-------------+
            |                           |
       Runtime Core                future AI
            |                           |
+-----------+-----------+       +-------+--------+
|           |           |       |       |        |
Compute   Provider    Component  Model Generation Agent
|           |           |
Planning   Device     WASM engine
|
Scheduler
```

This change deliberately introduces only the Runtime Core source boundaries. It does not create `model`, `generation`, or `agent` modules. Those domains should appear only when their contracts are defined.

The follow-up change is `stabilize-component-runtime-boundary`. That change will define the internal boundary of the real WASM engine before selecting or wiring Wasmtime: `ComponentEngine`, compilation and validation, `ComponentInstance`, `Store`, Capability linking, resources, traps, cancellation, sandboxing, and the separation between the abstract Runtime Component contract and the Wasmtime implementation.

## What Changes

The `magnetar-runtime` crate is reorganized into explicit architectural modules:

- `runtime`
- `component`
- `capability`
- `provider`
- `device`
- `affinity`
- `resolution`
- `compute`
- `planning`
- `scheduler`
- `observability`

`src/lib.rs` becomes the crate facade. It contains crate documentation, module declarations, and explicit public re-exports. Production implementation is moved into owned modules.

The existing `observability_exporter.rs` implementation is integrated under `observability/exporter.rs`.

## Non-Goals

This change does not:

- implement the WebAssembly Component engine
- choose or wire Wasmtime
- introduce `model`, `generation`, `inference`, `agent`, or `tool` production modules
- split `magnetar-runtime` into many crates
- redesign Runtime semantics
- redesign Component, Capability, Provider, Device, Compute, Planning, Scheduler, or Observability behavior
- reintroduce Backend or Plugin architecture
- introduce Tachyon coupling

## Impact

`magnetar-runtime` now has source ownership boundaries that match the architecture. Future changes can add contracts to the correct layer instead of expanding `lib.rs`.

The next Component Runtime work can define the WASM engine boundary in a focused change rather than mixing engine design with mechanical source movement.
