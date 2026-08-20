# Add WebAssembly Component Model

## Why

Magnetar aims to separate hardware-specific execution from reusable runtime functionality.

Traditional plugin systems require native implementations for every extension,
making portability difficult and increasing maintenance costs.

This proposal introduces a WebAssembly Component Model.

Components become portable runtime modules implementing well-defined WIT
contracts.

Native Hosts provide access to hardware resources while Components implement
higher-level functionality.

This separation allows the same Component to execute on different hardware
architectures without recompilation.

## What Changes

This proposal introduces:

- Component abstraction
- Component lifecycle
- WIT interface contracts
- Component discovery
- Component instantiation
- Component dependency resolution

This proposal intentionally excludes:

- WIT definitions
- Component implementations
- WASM runtime integration
- Hardware Hosts

Those capabilities will be introduced in later changes.

## Impact

Magnetar gains a portable extension mechanism based on WebAssembly Components.

Future functionality can be implemented independently from hardware
implementations.