## ADDED Requirements

### Requirement: CLI Owns User-Facing Source And Cache UX

`magnetar-cli` MAY own user-facing source and cache workflows such as pull, import, list, inspect, prune, pin, and unpin, and CLI SHALL not bypass Runtime artifact validation for these workflows.

#### Scenario: CLI cache list

Given user asks CLI to list cached models

When CLI displays entries

Then it uses redacted metadata and does not bypass Runtime validation for later
loads.

---

### Requirement: CLI Source Actions Do Not Grant Runtime Ambient Authority

CLI downloads, imports, aliases, or cache mutations SHALL not grant Runtime
ambient network, filesystem, credential, or cache mutation authority.

#### Scenario: CLI imported model

Given CLI imports local model

When Runtime later loads it

Then Runtime receives authorized artifact metadata and validates it.