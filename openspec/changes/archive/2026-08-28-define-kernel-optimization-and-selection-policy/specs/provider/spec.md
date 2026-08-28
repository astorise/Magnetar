## ADDED Requirements

### Requirement: Provider Supplies Selection-Relevant Metadata

Exposed Kernel and Device metadata SHALL NOT misrepresent Kernel behavior; Provider SHOULD expose accurate metadata needed by Runtime selection.

#### Scenario: Device queue pressure

Given Provider can measure queue pressure

When Runtime ranks eligible candidates

Then pressure may be supplied as optimization input.

---

### Requirement: Provider Does Not Own Global Selection

Provider SHALL NOT choose between competing Providers for Runtime.

#### Scenario: CUDA Provider available

Given CPU is also available

When global selection occurs

Then CUDA Provider cannot force itself as selected.

---

### Requirement: Private Variant Selection Is Limited

Provider SHALL register a distinct Registry entry whenever any Runtime-visible property differs; Provider MAY otherwise privately choose implementation variants when Runtime-visible contract properties are unchanged.

#### Scenario: Two internal launch configurations

Given both preserve semantics, determinism, precision and resource contract

When Provider chooses internally

Then distinct Registry entries are not required.

---

### Requirement: Runtime-Relevant Differences Are Advertised

If private variants differ in Runtime-relevant properties they SHALL be
separately advertised.

#### Scenario: Different determinism

Given one internal kernel uses nondeterministic atomics

Then it cannot be hidden behind deterministic Kernel advertisement.