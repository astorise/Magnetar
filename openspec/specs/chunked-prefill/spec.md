# chunked-prefill Specification

## Purpose
This specification defines chunked prefill: long prompts run as consecutive passes of one prefill graph that continue the KV history, with eligibility rules and controls.
## Requirements
### Requirement: Chunked Prefill

The Runtime SHALL prefill a prompt longer than the configured chunk length as consecutive passes of one prefill graph of the chunk length, each continuing the KV history the earlier passes wrote in place from its first row's position, so that a pass holds activations for at most the chunk length's rows.

#### Scenario: Same generation as one pass
- **WHEN** a prompt longer than the chunk length is prefilled in passes
- **THEN** each pass's rows take their absolute positions, attend to every earlier row of the prompt, and the generation matches the one-pass prefill's within the conformance tolerance

#### Scenario: Bounded activations
- **WHEN** Mistral 7B Q4_K_M generates after a 16,384-token prompt on one 12 GB GPU
- **THEN** the prefill no longer runs out of device memory in its MLP

### Requirement: One Graph For Every Pass

The Runtime SHALL right-pad the last pass to the chunk length and select its last real row's logits, end the KV history at the prompt's last row, and run every pass of every longer prompt with the chunk's graph and Prepared Execution Plan.

#### Scenario: Padding beyond the history
- **WHEN** the last pass holds fewer real rows than the chunk length
- **THEN** its padding rows are written beyond the prompt's rows, no real row reads them, and the first decode step writes over them

### Requirement: Chunking Eligibility

The Runtime SHALL prefill a prompt in one pass when its Provider keeps no KV history in place, under tensor parallelism, or when the prefill graph has an attention node reading K or V from an edge other than its KV history, or a KV-history edge read by a node other than attention.

#### Scenario: Latent attention
- **WHEN** a model whose prefill attends over keys and values expanded from its own rows while its history holds latent rows prefills a long prompt
- **THEN** the prompt is prefilled in one pass and chunking stays off for that model

### Requirement: Chunk Controls

The Runtime SHALL take the chunk length from `MAGNETAR_PREFILL_CHUNK` (2,048 when unset, `0` disabling chunking) and from `ProductionLoadedModel::set_prefill_chunk`, and SHALL report the chunk length a loaded model uses.

#### Scenario: Disabled
- **WHEN** the chunk length is 0
- **THEN** every prompt is prefilled in one pass
