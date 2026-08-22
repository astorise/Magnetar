# plugin Specification

## Purpose

This specification is retained for compatibility with historical Plugin work.
`Plugin` is not a canonical primary Magnetar architectural concept for new
designs. New trusted native extensions SHALL use Provider terminology. New
portable WebAssembly extensions SHALL use Component terminology.

## Requirements

### Requirement: Plugin Terminology Is Compatibility-Only

The runtime SHALL NOT require new architecture to use Plugin as a primary
extension concept.

Compatibility references to Plugin MAY remain when describing historical
artifacts or migration from the earlier plugin model.

#### Scenario: Classify new extension

Given a new Magnetar extension is proposed

When it is trusted native code

Then it is modeled as a Provider

And when it is portable WebAssembly code

Then it is modeled as a Component.

---

### Requirement: Legacy Plugin Migration

Historical Plugin requirements SHALL be migrated to Provider or Component
requirements before they are implemented as current architecture.

#### Scenario: Migrate backend-contributing plugin

Given a historical Plugin contributed a hardware backend

When the concept is updated to current Magnetar architecture

Then the native implementation is represented as a Provider that exposes
Devices and implements Capabilities.
