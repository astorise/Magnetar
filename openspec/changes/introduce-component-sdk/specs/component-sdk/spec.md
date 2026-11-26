## ADDED Requirements

### Requirement: Components share their graph scaffolding from one crate

The `magnetar-component-sdk` crate SHALL define the graph description a Model Component builds (phase, KV role, attribute value, attribute, KV involvement and node) and the `GraphSink` trait that the Component's host-independent graph code is written against. A Model Component that builds its graphs through `GraphSink` SHALL use these definitions and SHALL NOT define its own copies of them.

#### Scenario: One graph code, two sinks

- **WHEN** a Component's graph code is written against `GraphSink`
- **THEN** the same code builds against the Runtime's `graph-builder` Capability, through the adapter, and against `RecordingSink` in the Component's tests

#### Scenario: A second copy

- **WHEN** a Component's source defines its own `GraphSink`, `Node` or `Attr`
- **THEN** the change is refused in review as a copy of the crate's

### Requirement: The adapter to the Runtime's graph-builder is written once

The crate SHALL provide `wit_graph_sink!`, which implements `GraphSink` for a type on the `graph-builder` interface of the bindings a Component generated. A Component SHALL NOT hand-write that adapter. The macro SHALL rely only on the items the contract's `graph-builder` has had since version 1.8.0, so that a Component built against any later contract version can use it.

#### Scenario: The contract changes an item the adapter uses

- **WHEN** a contract version renames or reshapes a `graph-builder` type the adapter uses
- **THEN** every Component that calls the macro fails to compile at the call, and the fix is made once, in the crate

#### Scenario: A Component built against an older contract

- **WHEN** a Component generated its bindings from the 1.8.0 contract and calls the macro
- **THEN** it compiles and links against a Runtime at a later version

### Requirement: The crate depends on nothing

The crate SHALL have no dependency: not `magnetar-runtime`, not a model family's crate, not generated bindings. It SHALL name no model family.

#### Scenario: A Component for the WebAssembly Component target

- **WHEN** a Component builds for `wasm32-wasip2` with the crate
- **THEN** the crate adds no other crate to its dependency graph

### Requirement: A migration keeps the graph

A Component that moves from its own copies to the crate SHALL emit, for every configuration its tests build, the same graph as before: the same nodes with the same ids, operators, attributes, inputs, shapes and KV involvements, in the same order. The migration SHALL be proved by a golden graph recorded from the Component's code before it changes.

#### Scenario: A Component moves to the crate

- **WHEN** a Component's types, trait and adapter are replaced by the crate's
- **THEN** its golden graphs, recorded before the change, are equal to the graphs it builds after it

### Requirement: The crate is mounted beside the Components that use it

Magnetar SHALL pin the crate as a submodule at `components/sdk`. Components SHALL depend on it by the relative path `../sdk`, and the Magnetar integration jobs SHALL build every Component against the pinned revision.

#### Scenario: A Component in its own repository

- **WHEN** a Component's repository runs its checks inside a Magnetar checkout
- **THEN** `../sdk` resolves to the crate at the revision Magnetar pins
