## ADDED Requirements

### Requirement: Component Trust Is Independent Of Model Artifact Trust

An embedder that both loads a Component (executable WASM) and ingests a Model Artifact (weights/tokenizer/config) it will use SHALL evaluate trust for each independently, using a distinct trust policy per concern. Trusting a Component's own digest SHALL NOT imply trust of any Model Artifact digest, and trusting a Model Artifact digest SHALL NOT imply trust of any Component digest -- each is evaluated against its own trust store.

#### Scenario: A trusted Model Artifact does not grant Component trust

Given a Model Artifact digest is trusted

And no Component digest is trusted

When an embedder attempts to register a Component artifact

Then registration is rejected on Component trust grounds, regardless of the Model Artifact's trust status

#### Scenario: A trusted Component does not grant Model Artifact trust

Given a Component digest is trusted and its artifact registers successfully

And no Model Artifact digest is trusted

When the embedder attempts to ingest the Model Artifact

Then ingestion's trust evaluation is rejected on Model Artifact trust grounds, regardless of the Component's trust status
