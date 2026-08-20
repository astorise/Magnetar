# Define the Capability Model

## Why

Capabilities are the fundamental abstraction of Magnetar.

Components should never communicate directly with Providers.

Instead, Components express requirements through WIT capability contracts.

Providers advertise the capabilities they implement.

The Runtime resolves Components to compatible Providers.

This separation enables hardware independence, automatic provider selection,
runtime fallback and future distributed execution.

## What Changes

This proposal introduces the Capability Model.

Capabilities become first-class runtime objects.

A Capability defines:

- a unique identifier
- a semantic version
- one or more WIT contracts
- compatibility rules
- dependency information

Capabilities are registered independently from Providers.

Providers may expose multiple Capabilities.

Components may consume multiple Capabilities.

This proposal intentionally excludes concrete capability definitions.

No compute, memory or device interfaces are introduced.

Those will be defined by future proposals.

## Impact

The runtime becomes capability-centric.

Future Providers, Components and Hosts will rely on the same capability
resolution mechanism.