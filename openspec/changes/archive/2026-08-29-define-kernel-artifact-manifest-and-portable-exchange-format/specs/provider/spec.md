## ADDED Requirements

### Requirement: Provider Receives Normalized Artifact Metadata

Provider SHALL not be responsible for parsing portable Kernel Manifest.

#### Scenario: CUDA Provider prepares CUBIN

Given bundle has been validated

When Provider receives artifact

Then Runtime supplies normalized CompiledKernelArtifact-compatible data.

---

### Requirement: Provider Does Not Trust Filename

Provider compatibility SHALL use explicit artifact format and metadata.

#### Scenario: Blob named random hash

Given metadata says CUBIN

When CUDA Provider validates format

Then filename extension is irrelevant.

---

### Requirement: Provider Native State Not Exportable As Bundle

Provider SHALL not serialize native execution handles into portable
Kernel Exchange Bundle.

#### Scenario: Metal pipeline prepared

Given MTLComputePipelineState exists

When bundle exported

Then metallib/source artifact may be exported, but live pipeline object is not.

---

### Requirement: Unsupported Manifest Artifact Is Structured Failure

Provider incompatibility SHALL be reported explicitly.

#### Scenario: WGSL artifact sent toward CUDA Provider

Given Runtime evaluates candidate

Then unsupported format/compatibility is reported rather than reinterpretation.