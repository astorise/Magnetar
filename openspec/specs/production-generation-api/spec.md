# production-generation-api Specification

## Purpose
TBD - created by archiving change expose-production-generation-parameters. Update Purpose after archive.
## Requirements
### Requirement: Production Generation Accepts Caller-Supplied Generation Parameters

The production generation entry point SHALL accept a caller-supplied `GenerationParameters` value and forward it, unmodified in meaning, to the same Generation Contract every other Runtime generation path uses -- not substitute an internally hardcoded value.

#### Scenario: Temperature and top_p reach sampling

- **WHEN** a caller requests production generation with `temperature` and `top_p` set to non-default, non-greedy values
- **THEN** the resulting `GenerationRequest.parameters` carries those exact values into the sampling contract, and the sampling decision is not forced to greedy selection

#### Scenario: Seed reaches the sampling contract

- **WHEN** a caller requests production generation with a `seed` set
- **THEN** the resulting `GenerationRequest.parameters.seed` carries that value into the sampling contract

#### Scenario: Legacy entry points preserve greedy defaults

- **WHEN** a caller uses `run_production_qwen_generation`, `run_production_qwen_generation_for_provider`, or `run_production_qwen_generation_for_provider_with_prompt`
- **THEN** generation runs exactly as it did before this requirement existed: greedy decoding with default stop conditions and the checkpoint manifest's own token budget

---

### Requirement: Production Generation Accepts Caller-Supplied Stop Conditions

The production generation entry point SHALL accept a caller-supplied `StopConditions` value, including plain-text stop sequences, and prepare those text sequences against the real tokenizer so generation actually stops on them -- a caller SHALL NOT be required to construct the tokenizer-aware prepared form itself.

#### Scenario: A text stop sequence stops generation early

- **WHEN** a caller requests production generation with a `stop_text_sequences` entry that the model is expected to produce before its natural token budget is exhausted
- **THEN** generation stops at that sequence, and the produced text does not extend past it

#### Scenario: Token-id stop conditions are still honored

- **WHEN** a caller requests production generation with `stop_conditions.stop_token_ids` set
- **THEN** generation stops when one of those token ids is produced, exactly as the underlying Generation Contract already does for any other caller

---

### Requirement: Production Generation Fails Closed On Unsupported Requests

When a caller-supplied generation parameter, stop condition, or token budget cannot be honored by the selected execution path, the production generation entry point SHALL return a structured `Unsupported` error naming the reason, and SHALL NOT silently ignore the request or proceed as if it had been honored.

#### Scenario: A token budget exceeding a non-multi-step Provider's capability is rejected

- **WHEN** a caller requests a resolved token budget (manifest default or caller override) greater than one, against a Provider that does not support multi-step decode
- **THEN** the entry point returns `InferenceApiError::Unsupported` naming the Provider and the unsupported decode-step count, and no generation is attempted

---

### Requirement: Production Generation Request Overrides Are Optional And Additive

`ProductionGenerationRequest`'s token-budget override SHALL be optional; when absent, the checkpoint manifest's own configured default SHALL apply exactly as it did before this capability existed.

#### Scenario: No override preserves manifest-driven token budget

- **WHEN** a caller supplies `ProductionGenerationRequest.max_new_tokens: None`
- **THEN** the resolved token budget is the checkpoint manifest's own configured default, unchanged from prior behavior

#### Scenario: An explicit override replaces the manifest default

- **WHEN** a caller supplies `ProductionGenerationRequest.max_new_tokens: Some(n)`
- **THEN** the resolved token budget is `n`, regardless of the checkpoint manifest's own configured default

