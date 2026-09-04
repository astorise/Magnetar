## ADDED Requirements

### Requirement: Materialized Weight Content Matches Its Declared Digest When One Exists

Materialization evidence for a weight tensor whose inventory entry declares a content digest SHALL only be produced for content that matches that digest.

This extends -- it does not replace -- the existing requirement that materialization evidence is Runtime-issued and instance-bound: a caller with access to the one authorized weight-materialization transaction still cannot make an arbitrary tensor count as a declared tensor's content when a content digest for that tensor exists, even though the transaction itself is the legitimate, non-forgeable path. A tensor whose inventory entry declares no content digest is unaffected by this requirement; its materialization evidence continues to be governed only by the existing instance-bound, artifact-bound, and inventory-completeness requirements.

#### Scenario: Matching content is evidenced normally

Given a Model Instance's loaded artifact declares a content digest for a mandatory tensor

When the authorized materialization transaction stages content that matches that digest

Then materialization evidence is minted for that tensor as usual

#### Scenario: Mismatched content does not become evidenced

Given a Model Instance's loaded artifact declares a content digest for a mandatory tensor

When a caller attempts to materialize different content under that tensor's name

Then no materialization evidence is minted for that tensor, and the instance does not become Ready on the strength of it
