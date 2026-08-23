## ADDED Requirements

### Requirement: Tokenizer Component Authority

A Tokenizer Component SHALL be limited to inference-scoped tokenizer authority.

It SHALL NOT receive filesystem, network, Git, secrets, workspace, or process
authority.

#### Scenario: Tokenizer Component reads tokenizer data

Given a Tokenizer Component requests tokenizer artifact access

When Runtime links it

Then access is mediated through Runtime-registered tokenizer artifacts.

---

### Requirement: Tokenizer Component Is Optional

Magnetar SHALL NOT require tokenizers to be implemented as Components.

Tokenizer implementations MAY be native, Component-based, browser-compatible, or
test fixtures.

#### Scenario: Native tokenizer

Given a native tokenizer implementation exists

When Runtime encodes text

Then Runtime may use it through the same Tokenizer Contract.
