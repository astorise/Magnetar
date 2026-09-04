## ADDED Requirements

### Requirement: Model Components, Providers, and Formats Are Externalized

Model-architecture Components, Providers beyond one minimal built-in reference baseline, and Model Artifact Format parsers SHALL live in repositories separate from the Magnetar Core repository, pinned into it as git submodules.

The Core repository SHALL have zero compile-time dependency on the concrete implementation crate of any such externalized module.

One minimal, generic Reference CPU Provider implementation MAY remain inside the Core repository, solely to serve as the Core's own test double. This in-crate implementation SHALL NOT be treated as, or substitute for, the real externally-distributable Reference CPU Provider implementation, which SHALL itself live externally like any other Provider.

#### Scenario: Add a new model architecture

Given a new model family (for example Llama) gains a real Component implementation

When that implementation is added to the project

Then it lives in its own repository, pinned into the Core repository as a git submodule

And the Core repository's own crate does not depend on it.

#### Scenario: Add a new hardware Provider

Given a new hardware Provider (for example CUDA) gains a real implementation

When that implementation is added to the project

Then it lives in its own repository, pinned into the Core repository as a git submodule

And the Core repository's own crate does not depend on it.

#### Scenario: Reference CPU's in-crate double coexists with its real implementation

Given the Core repository's own test suite instantiates a generic Reference CPU Provider double directly

When the real, externally-distributable Reference CPU Provider is built

Then the two are independent implementations sharing no types beyond what the Core's own generic contracts already require

And the Core repository does not depend on the externally-distributable implementation's crate.

#### Scenario: Add a new Model Artifact format parser

Given a new Model Artifact format (for example a new quantization container) gains a real parser

When that parser is added to the project

Then it lives in its own repository, pinned into the Core repository as a git submodule

And the Core repository's own crate does not depend on it.

#### Scenario: CI verifies the boundary automatically

Given the Core repository's manifest and resolved dependency graph

When continuous integration runs

Then it verifies neither names any externalized Component, Provider, or Format crate

And the check applies to every module following the established naming convention, not only to modules enumerated by name.
