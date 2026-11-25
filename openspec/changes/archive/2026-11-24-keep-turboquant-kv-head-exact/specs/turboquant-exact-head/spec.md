## ADDED Requirements

### Requirement: Exact Head

A TurboQuant KV store SHALL keep its first `E` rows exact, `E` configured by `MAGNETAR_KV_EXACT_ROWS` (1,024 when unset, `0` for none) or `KvPrecisions::exact_rows` (with `ProductionLoadedModel::request_kv_precisions`), and attention SHALL read a row below `E` as written and a row from `E` on as its TurboQuant round trip with the store's offsets.

#### Scenario: Short history
- **WHEN** a history of at most `E` rows is kept in TurboQuant stores
- **THEN** generation matches the same history kept in f32 stores

#### Scenario: Long history
- **WHEN** a history grows past `E` rows
- **THEN** its first `E` rows are read exactly and the rest through their codes

### Requirement: Short Histories Decode As f32

A Provider SHALL decode attention over a history of at most `E` rows of a TurboQuant store from its exact head alone, without TurboQuant work, and SHALL NOT replay a captured step past the regime it was captured in.

#### Scenario: Crossing the head
- **WHEN** a captured decode step's history grows past `E` rows
- **THEN** the step is captured again for the longer regime and generation continues unchanged

### Requirement: Head Admission And Compatibility

The Runtime SHALL admit a TurboQuant store at its codes' size plus its exact head's, and SHALL record the head with the precisions in the KV compatibility and layout metadata, so that a history is only continued with the same head.

#### Scenario: Different head
- **WHEN** a history written with one head length is continued with another
- **THEN** the continuation is refused as for different precisions
