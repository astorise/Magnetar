## ADDED Requirements

### Requirement: Adapter Formats Normalize Into Adapter Artifact

Adapter formats such as LoRA safetensors and adapter_config SHALL normalize into
Adapter Artifact metadata.

#### Scenario: LoRA parsed

Given LoRA adapter files are parsed

When normalization completes

Then Adapter Artifact metadata includes method, target modules, rank, alpha,
scaling, tensor inventory, and base model compatibility.

---

### Requirement: Adapter Format Does Not Activate Adapter

Parsing an adapter format SHALL not activate the adapter.

#### Scenario: Adapter parsed

Given adapter artifact is normalized

When Model Loading completes

Then adapter activation still requires explicit Runtime policy.