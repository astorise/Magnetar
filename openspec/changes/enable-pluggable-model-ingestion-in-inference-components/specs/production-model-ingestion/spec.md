## ADDED Requirements

### Requirement: An Embedder Selects Among Multiple Format-Specific Ingestors

An embedder that has more than one real, compatible `ProductionModelArtifactIngestor` implementation available SHALL select among them by inspecting the bundle it was given, never by hardcoding a single format's ingestor as the only reachable path. Tokenizer construction and chat-template loading performed alongside ingestion SHALL be selected the same way, consistently with which ingestor was chosen for that bundle.

#### Scenario: A bundle in one supported format ingests through its own real ingestor

- **GIVEN** an embedder with two real, compatible ingestors registered for two different bundle shapes
- **WHEN** it is given a bundle matching one of those shapes
- **THEN** it selects and uses that shape's real ingestor, and its tokenizer/chat-template construction for the same bundle

#### Scenario: Adding a bundle in a different supported format does not require touching the other format's path

- **GIVEN** an embedder already ingesting bundles of one format successfully
- **WHEN** it is given a bundle in a second, different but also-supported format
- **THEN** the first format's ingestion, tokenizer, and chat-template construction remain exactly as they were
