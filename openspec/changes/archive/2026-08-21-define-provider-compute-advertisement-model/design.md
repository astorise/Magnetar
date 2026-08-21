## Context

`ProviderMetadata` already carried partial compute support maps for operation
families, operation schemas and data movement. Those maps were enough for local
validation but did not form one explicit Provider Compute Advertisement model.

The runtime also already has compute graph validation, Resource Affinity,
Resolution Policy evaluation and structured compute errors. The design must
therefore introduce the advertisement as a stable metadata model without
replacing native Provider execution APIs or exposing backend internals.

## Goals / Non-Goals

**Goals:**

- Add a first-class `ProviderComputeAdvertisement` runtime type.
- Represent capability versions, operation family support, operation schema
  support, dtype/layout/shape/precision constraints, data movement support and
  Device-specific support.
- Preserve compatibility with existing `ProviderMetadata` support maps.
- Use advertisements during operation, graph and data movement validation.
- Return stable structured errors for unsupported advertisement cases.
- Document the format and how it affects Provider selection.

**Non-Goals:**

- Do not expose native handles, queues, streams, kernel names or backend object
  references.
- Do not define live migration or implicit cross-Provider movement.
- Do not guarantee execution success after validation.
- Do not introduce a Scheduler, cost model or memory planner.

## Decisions

### First-class advertisement plus legacy compatibility

`ProviderMetadata` gains `compute_advertisement: ProviderComputeAdvertisement`.
Runtime validation reads an effective advertisement that merges the new
structure with the existing support maps.

Rationale: existing tests and Providers can continue to populate legacy maps
while new Providers can use the explicit model. This keeps the change additive.

Alternative considered: replace the legacy maps immediately. That would make
the model cleaner but create avoidable churn across existing validation tests
and Provider construction helpers.

### Layered support types

The advertisement is split into support types:

- `ComputeCapabilitySupport`
- `OperationFamilySupport`
- `OperationSchemaSupport`
- `DTypeSupport`
- `LayoutSupport`
- `ShapeLimitSupport`
- `PrecisionSupport`
- `DataMovementSupport`
- `DeviceComputeSupport`

Rationale: each support category maps directly to one validation concern and
can be reused for Provider-wide, schema-specific or Device-specific entries.

Alternative considered: one flat advertisement record. That would be simpler
to serialize but harder to validate and harder to extend per Device.

### Schema-specific support takes precedence

When an operation descriptor references a schema, the runtime first checks
explicit unsupported schema declarations, then schema-specific support, then
family support.

Rationale: Providers need a coarse family signal for broad capability matching
and a precise override for unsupported or constrained schemas.

Alternative considered: require every supported schema to be listed. That is
more precise but too verbose for Providers that support a whole family with
common constraints.

### Resource Affinity remains authoritative

Advertisements participate in validation but never override Provider, Device,
Capability, context, artifact or affinity group bindings on live resources.
Cross-Provider and cross-Device changes still require explicit data movement.

Rationale: advertisements describe declared support; they do not make existing
opaque resources portable.

Alternative considered: allow a compatible advertisement to relax affinity.
That would violate the existing resource ownership model and hide migration.

### Structured errors preserve portable contracts

Unsupported advertisement states map to `ComputeValidationError` and then to
stable `ComputeError` values with redacted diagnostics.

Rationale: Components and host adapters need stable failure categories without
depending on backend diagnostic strings.

Alternative considered: return generic validation failures. That would make
Provider selection and diagnostics less actionable.

## Risks / Trade-offs

[Risk] The merged effective advertisement can hide incomplete migration from
legacy support maps.
→ Mitigation: keep the merge localized in `effective_compute_advertisement` so
future changes can remove legacy fallback in one place.

[Risk] Device-specific support is modeled before full scheduling exists.
→ Mitigation: store Device-specific constraints as metadata only; selection and
memory planning can consume them later without changing the public model.

[Risk] Advertised support may still fail at execution time.
→ Mitigation: the model explicitly treats advertisements as validation claims,
not execution guarantees, and preserves structured execution errors.

[Risk] Provider-specific schemas could leak non-portable behavior into
portable Components.
→ Mitigation: Provider extension schemas are tracked separately from portable
schema support.

## Migration Plan

1. Add the advertisement model and keep legacy metadata maps working.
2. Update validation paths to use the effective advertisement.
3. Add focused tests for advertisement-only operation validation, rejection
   cases and data movement validation.
4. Document the advertisement contract and examples.
5. In a future cleanup, migrate Providers to populate only
   `compute_advertisement` and remove the legacy maps.

Rollback is straightforward: Providers that still populate legacy maps continue
to work because the new model is additive.

## Open Questions

- Which serialization format should plugin-loaded Providers use for
  advertisement manifests?
- Which scheduler policy will consume Device-specific memory and precision
  metadata first?
- When should legacy support maps be deprecated after Providers migrate?
