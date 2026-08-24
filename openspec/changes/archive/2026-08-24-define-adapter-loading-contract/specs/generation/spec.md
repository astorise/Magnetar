## ADDED Requirements
### Requirement: Generation Uses Active Adapter Context

Generation SHALL use active adapter context when executing model forward.

#### Scenario: Adapter active

Given adapter A is active for a session

When generation runs in that session

Then Runtime executes model forward with adapter A according to Runtime plan.

---

### Requirement: Generation Does Not Implicitly Load Adapter

Generation SHALL NOT implicitly load or activate adapters unless explicit Runtime
policy allows it.

#### Scenario: Adapter referenced but not loaded

Given a generation request references unloaded adapter A

And implicit adapter loading is disabled

When Runtime validates generation

Then generation fails with adapter not ready or activation denied.

---

### Requirement: Adapter Changes Invalidate Generation State

Adapter activation or deactivation SHALL invalidate or rebuild generation state
where required.

#### Scenario: Adapter changes mid-session

Given a session has KV cache from adapter A

When adapter B becomes active

Then Runtime invalidates or rejects incompatible cached generation state.