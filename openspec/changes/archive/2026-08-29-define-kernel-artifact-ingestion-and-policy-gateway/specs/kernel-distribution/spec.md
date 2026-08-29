## ADDED Requirements

### Requirement: Distribution Source Terminates At Ingestion Gateway

External distribution mechanism SHALL deliver bytes/references to ingestion
boundary rather than directly to Kernel Registry or Provider.

#### Scenario: Tachyon distributes bundle

Given Tachyon delivers generated Kernel

When Magnetar receives it

Then bundle enters ingestion transaction before Runtime eligibility.

---

### Requirement: Distribution Does Not Grant Trust

Distribution channel identity SHALL not automatically imply artifact trust.

#### Scenario: Known registry

Given bundle arrives from configured registry

When trust requires artifact signature/digest policy

Then registry origin alone does not bypass that requirement unless explicit
authenticated-source policy says so.

---

### Requirement: Locator Is Not Authority

Manifest locator SHALL not create new distribution authority.

#### Scenario: Bundle links another domain

Given current Artifact Source policy does not allow domain

When resolving dependency

Then fetch is denied.