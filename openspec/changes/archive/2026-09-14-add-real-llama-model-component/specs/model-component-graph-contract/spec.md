## MODIFIED Requirements

### Requirement: Multiple Registered Components May Coexist Under Caller-Supplied Trust

The Runtime SHALL provide a registration mechanism through which an embedder can register an arbitrary Component artifact for graph production, keyed by that artifact's own real content digest (computed from the artifact's actual bytes, never a value the caller merely asserts) and evaluated against a trust decision the embedder itself supplies -- never a single hardcoded digest baked into the Runtime. Registering a second, distinct artifact SHALL NOT replace, invalidate, or require re-registering any artifact already registered: any number of distinct, real-digest-keyed Components SHALL remain simultaneously usable for graph production within one process.

Graph production for a registered artifact SHALL follow the same `model-component-graph-contract` Capability semantics (`graph-builder`/`model-config` host imports, `build-prefill-graph`/`build-decode-graph` exports) as any other Component-produced graph under this contract -- registration identity is orthogonal to graph-production semantics, which remain identical regardless of which registered artifact produced the graph.

#### Scenario: An artifact trusted by nothing is rejected

- **GIVEN** a well-formed Component artifact and an embedder-supplied trust decision that trusts nothing
- **WHEN** the embedder attempts to register the artifact
- **THEN** registration fails and no Component runtime is cached for it

#### Scenario: An artifact trusted by its own real digest registers and produces graphs

- **GIVEN** a well-formed Component artifact and an embedder-supplied trust decision that trusts that artifact's own real content digest
- **WHEN** the embedder registers the artifact and later requests prefill/decode graph production against the returned digest
- **THEN** registration succeeds, and the produced graphs are identical (in content, not merely well-formed) to graphs produced from the same underlying Component bytes through any other registration path

#### Scenario: Graph production against an unregistered digest fails closed

- **GIVEN** a content digest that was never registered
- **WHEN** graph production is requested against it
- **THEN** the request fails with a structured error, and no fallback graph source is substituted

#### Scenario: Two structurally distinct, simultaneously registered Components remain independently usable

- **GIVEN** two Component artifacts implementing the same contract but producing structurally different graphs, both registered under their own real digests and each trusted by the embedder
- **WHEN** graph production is requested against each registered digest, in either order
- **THEN** each produces its own distinct graphs (different node counts and operator-sequence hashes from the other), and requesting graphs from one registered Component does not alter, evict, or otherwise disturb the other's continued correct behavior

#### Scenario: A genuine second real architecture family registers and coexists with Qwen

- **GIVEN** the real, independently-compiled Llama Model Component registered under its own real digest, alongside the real Qwen Model Component already registered
- **WHEN** graph production is requested against the Llama digest
- **THEN** it succeeds, produces graphs consistent with the same generic contract, and does not disturb the Qwen Component's own continued correct behavior -- the registry's multiplicity holds for a real second production architecture family, not only a caller-supplied digest that always resolves back to Qwen or a synthetic fixture
