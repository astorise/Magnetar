## ADDED Requirements

### Requirement: CUDA Provider Reads Only The Looked-Up Rows Of A Packed Table

An embedding lookup from a packed table stored untransposed SHALL dequantize only the rows its ids name, in every packed format the CUDA Provider stores, and SHALL give the same values as those rows of the dequantized table. An id that is negative, not an integer or not below the table's row count SHALL fail the lookup.

#### Scenario: A Q6_K per-layer table
- **WHEN** a decode step looks up one token in a packed `Q6_K` `[262144, 256]` table
- **THEN** the lookup writes one 256-value row, not the dequantized table

#### Scenario: Every packed format
- **WHEN** a lookup reads rows of an untransposed packed table in any stored format
- **THEN** each row equals the same row of the table's dequantization, and an invalid id fails the lookup
