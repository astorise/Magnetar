## ADDED Requirements

### Requirement: CUDA Provider Supports Multi-Step Decode

CUDA Provider SHALL support a generation request requiring more than one decode step. Historical KV concatenation across decode steps SHALL complete without downloading the historical KV tensor to host memory, and the resulting concatenated KV state SHALL remain Device-resident. Device memory held for a decode session's KV state SHALL NOT grow without bound across steps: at most one live allocation per layer, per K/V role, per KV identity (graph edge, pending, committed) SHALL exist at a time, with the previous step's allocation for that same identity released when superseded.

#### Scenario: Multi-token decode succeeds on real hardware

- **GIVEN** a generation request whose `max_tokens` requires more than one decode step, bound to CUDA Provider
- **WHEN** generation runs
- **THEN** it completes successfully and produces the requested number of real, decoded tokens, not an `Unsupported` rejection or an internal residency error

#### Scenario: Historical KV concatenation stays Device-resident

- **GIVEN** a decode step needs to concatenate this step's new K/V with the previous step's historical K/V, both held Device-resident by CUDA Provider
- **WHEN** the concatenation executes
- **THEN** it completes without CUDA Provider producing host-visible bytes for the historical KV tensor, and the concatenated result is itself Device-resident

#### Scenario: Device memory for KV state does not grow unboundedly across steps

- **GIVEN** a generation request decodes N tokens sequentially on CUDA Provider
- **WHEN** Memory Manager's active allocation count for this session's KV resources is inspected after each step
- **THEN** it remains constant across steps (one live allocation per layer per role per KV identity), not proportional to N

#### Scenario: Multi-token CUDA output matches Reference CPU

- **GIVEN** the same real ingested model, tokenizer, and prompt, decoded for the same number of tokens under greedy sampling
- **WHEN** compared between Reference CPU and CUDA Provider
- **THEN** the generated token sequence and decoded text are identical
