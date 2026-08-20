# Device Model Design

## Overview

Devices are immutable descriptions of execution targets discovered by a
Provider.  The runtime owns the device registry and schedules against its
contents; it never needs a Provider implementation to enumerate a target.

## Device contract

`DeviceMetadata` contains a globally unique `DeviceId`, a human-readable name,
type, vendor, architecture, memory capacity, compute-unit count, execution
capabilities, and the name of its owning Provider.  `Device` exposes that
metadata.  A `DeviceDescriptor` provides a reusable concrete implementation
for providers that do not need a specialised device type.

Device IDs are opaque strings.  They are unique across every registered
Provider, rather than only within a provider.

## Discovery and ownership

`Provider::devices` returns all devices discovered by that provider.  During
provider registration the loader registers each returned device in the
`ProviderRegistry`.  Registration fails atomically when an ID is duplicated or
when a device's declared owner differs from the registering provider.  The
registry stores both the `Arc<dyn Device>` and the owner's name.

The device's owner is represented by the stable provider name rather than an
`Arc<dyn Provider>`: this prevents a reference cycle and lets a device outlive
an individual provider object only as long as the registry does.

## Runtime API and lifecycle

`Runtime::devices` and `Runtime::device` delegate to the registry, so callers
can enumerate/select devices without depending on provider implementations.
Device registration happens after a Provider successfully runs `register` and
`initialize`, and is rolled back with all other contributions if any
registration step fails.  Provider shutdown clears the device registry along
with its other contributions.

Existing `Backend` registration remains available for compatibility; the new
device model is provider-based and does not depend on a Backend.
