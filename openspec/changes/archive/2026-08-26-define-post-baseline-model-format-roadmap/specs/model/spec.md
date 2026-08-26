## ADDED Requirements

### Requirement: Normalized Model Artifact Manifest

Model Artifact SHALL support a normalized manifest that can be produced from
external formats.

#### Scenario: External format normalized

Given safetensors and config metadata are parsed

When normalization completes

Then Model Artifact manifest contains normalized identity, architecture,
weights, tokenizer, generation, trust, and integrity metadata.

---

### Requirement: Model Format Does Not Grant Trust

A model format SHALL not be trusted merely because it is recognized.

#### Scenario: Recognized untrusted format

Given safetensors file is parseable

But trust policy rejects its source

When Model Loading runs

Then loading fails.

---

### Requirement: Source Metadata Is Not Automatically Runtime Policy

Source metadata such as `torch_dtype` SHALL not silently become Runtime compute
policy.

#### Scenario: torch_dtype bf16

Given config declares `torch_dtype` as bf16

When Runtime loads model

Then compute dtype follows validated Runtime policy, not silent source metadata.