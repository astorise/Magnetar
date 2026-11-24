## ADDED Requirements

### Requirement: In-place KV history append

When the executing Provider supports in-place KV writes and bounded attention, first-native generation SHALL keep each layer and role's KV history in one store sized for the request, SHALL write only each step's new rows beyond the committed length, and SHALL let attention read only the committed rows plus the step's new rows.

#### Scenario: Decode step on a Provider with in-place KV support

- **WHEN** a decode step runs on a Provider that advertises `kv-write` and `key-count` attention
- **THEN** the step writes one row per layer and role into its store and attention reads only the committed rows plus that row

#### Scenario: Provider without in-place support

- **WHEN** the executing Provider does not advertise `kv-write` or `key-count` attention
- **THEN** the history is appended with the copy-based path, with identical results

### Requirement: Copy-free KV commit and rollback

Committing an in-place step SHALL advance the committed length without copying history. A step that is discarded or fails SHALL leave the committed length and every committed row unchanged.

#### Scenario: Commit

- **WHEN** a step that wrote its rows in place commits
- **THEN** the committed length grows by the step's rows and no KV history is copied

#### Scenario: Failed step

- **WHEN** a step fails or is discarded after writing its rows
- **THEN** the committed length and every committed row are unchanged, and the next step overwrites the uncommitted rows
