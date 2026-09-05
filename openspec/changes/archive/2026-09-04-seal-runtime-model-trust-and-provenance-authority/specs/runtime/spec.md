## ADDED Requirements

### Requirement: Runtime-Sealed Trust Configuration

Runtime SHALL accept its Model Artifact trust policy only through its builder, configured once before the Runtime instance exists, and SHALL NOT expose any way to replace that policy for the instance's remaining lifetime.

The builder-configured trust policy SHALL default to a policy that trusts nothing, so that Runtime instances built without explicit trust configuration reject every Model Artifact trust evaluation rather than defaulting open.

#### Scenario: Trust policy set at build time

Given a Runtime is built with an explicit trust policy

When the Runtime later evaluates trust for a Model Artifact during loading

Then it uses the policy supplied at build time

#### Scenario: No post-build trust reconfiguration

Given a Runtime instance already exists

When code holding a reference to that Runtime instance attempts to change its trust policy

Then no public operation exists to do so

#### Scenario: Default Runtime trusts nothing

Given a Runtime is built without explicit trust configuration

When it evaluates trust for any Model Artifact

Then the artifact is not trusted
