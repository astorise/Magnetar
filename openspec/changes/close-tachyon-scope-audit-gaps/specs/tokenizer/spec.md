## MODIFIED Requirements

### Requirement: Chat Template Is Artifact-Bound

A production chat template used for Qwen inference SHALL come from Runtime-authorized artifact/config data and SHALL NOT trigger arbitrary filesystem or network access during inference. When ingestion discovers a real chat template declared by the artifact's own configuration, normalization SHALL preserve it on the produced Model Artifact rather than discarding it, and production message rendering SHALL use that real template rather than a generic placeholder that ignores it. A chat template using a construct outside the supported rendering subset SHALL be rejected at ingestion time with a structured error naming the unsupported construct, not silently rendered incorrectly or silently ignored.

#### Scenario: Template path points outside artifact
- **WHEN** source metadata attempts to reference an unauthorized external template path
- **THEN** normalization/loading rejects it or treats it as non-authoritative metadata rather than fetching it during inference.

#### Scenario: Bundle declares a real chat template
- **WHEN** an authorized bundle's tokenizer configuration declares a real chat template string
- **THEN** the normalized Model Artifact carries that template, and rendering chat messages for that artifact uses it.

#### Scenario: Bundle declares no chat template
- **WHEN** an authorized bundle's tokenizer configuration declares no chat template
- **THEN** the normalized Model Artifact carries none, and a caller-supplied or Runtime-default plain-text rendering applies instead, unchanged from prior behavior.

#### Scenario: Chat template uses an unsupported construct
- **WHEN** a declared chat template uses a Jinja2 construct outside the supported rendering subset
- **THEN** ingestion rejects it with a structured error naming the unsupported construct, rather than producing a Model Artifact whose template would render incorrectly or be silently ignored later.
