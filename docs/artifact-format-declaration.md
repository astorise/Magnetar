# Declaring a Model Artifact bundle's format

astorise/Magnetar#75. This is the minimal recipe an integrator (Tachyon or
otherwise) uses to make a production Model Artifact bundle loadable through
`inference-components`' `LoadedInferenceComponent`. See
`docs/audits/audit-magnetar-integration-tachyon-2026-09-22.md` (MAG-INT-01)
for the audit finding this closes, and `inference-components/src/lib.rs` for
the real, tested code this recipe is drawn from.

## Why this exists

Before this change, `LoadedInferenceComponent::load` picked between the
`GgufIngestor` and `HuggingFaceIngestor` by checking whether a file named
`model.gguf` existed next to the bundle -- a filesystem heuristic a caller
had no way to override, and no way to make explicit. A Model Component also
had no way to declare which bundle shapes it actually supports.

Format selection is now a **declarative compatibility property**, not an
inferred one:

- Every production Model Artifact bundle declares its own format explicitly,
  via a small sidecar file at the bundle root.
- Every registered Model Component may declare which format(s) it accepts,
  in its own `.magnetar-component.yaml`.
- Magnetar checks the two against each other and rejects a mismatch
  explicitly, the same way it already rejects an architecture-family
  mismatch (astorise/Magnetar#83).

Filesystem-structure detection (the old `model.gguf`-presence check) still
exists, but only as an explicit, separately-invoked **legacy migration**
step for bundles that predate this sidecar -- never as an automatic runtime
fallback. A bundle with no sidecar and no explicit migration step run
against it fails to load, with an error naming the missing file. Magnetar
never guesses.

## The sidecar file

A production bundle root must carry a file named:

```text
magnetar-artifact-format.yaml
```

with exactly one field:

```yaml
artifact_format: huggingface
```

or:

```yaml
artifact_format: gguf
```

These are the only two recognized values today (`ArtifactFormat::
HuggingFace` / `ArtifactFormat::Gguf` in `magnetar-runtime`'s `model`
module). A third real ingestor would add a third recognized value here, not
a new filesystem check. Any other value, or a file that is not valid YAML,
is rejected explicitly (`ProductionIngestionError::MalformedMetadata`); a
missing file is rejected explicitly too
(`ProductionIngestionError::RequiredPartMissing`). Neither case is silently
patched over.

This is symmetric to a Model Component's own `.magnetar-component.yaml`:
the bundle producer writes it explicitly, Magnetar never infers its content.

## Minimal examples

### A Hugging Face-shaped bundle

```text
my-qwen-bundle/
├── magnetar-artifact-format.yaml   # artifact_format: huggingface
├── config.json
├── tokenizer.json
├── tokenizer_config.json           # optional
└── model.safetensors               # or a sharded set + model.safetensors.index.json
```

```yaml
# my-qwen-bundle/magnetar-artifact-format.yaml
artifact_format: huggingface
```

### A GGUF-shaped bundle

```text
my-qwen-bundle/
├── magnetar-artifact-format.yaml   # artifact_format: gguf
└── model.gguf
```

```yaml
# my-qwen-bundle/magnetar-artifact-format.yaml
artifact_format: gguf
```

Note that `model.gguf`'s presence is no longer what selects the ingestor --
the sidecar is. `loaders/gguf`'s own ingestor still expects a file at that
same well-known name once it *has* been selected; only the selection
decision itself changed.

## Declaring a Component's accepted format(s)

A Model Component's `.magnetar-component.yaml` may declare which Artifact
format(s) it accepts, in the same `compatibility` block that already carries
`architecture_families` (astorise/Magnetar#83):

```yaml
compatibility:
  architecture_families:
    - qwen2
  artifact_formats:
    - huggingface
```

Semantics (identical in shape to `architecture_families`):

- No `compatibility` block, or a `compatibility` block with no
  `artifact_formats` key: **permissive** -- this Component accepts every
  format. Every manifest written before this field existed keeps behaving
  exactly as it always did.
- `artifact_formats` present with one or more values: only a bundle
  declaring one of those formats may load against this Component.
- `artifact_formats: []` (an explicit empty list): **rejected at manifest
  parse time**. An author who writes this is declaring *something* --
  treating it the same as omitting the field would let a typo'd or
  generated-empty list accidentally open a Component to every format.
- An unrecognized entry (anything other than `huggingface`/`gguf`) is
  rejected at manifest parse time for the same reason.

A Component manifest that declares `compatibility.artifact_formats` must
also declare `runtime.magnetar.min_version: "0.1.2"` or higher (see
`MAGNETAR_RUNTIME_VERSION` in `magnetar-runtime/src/component.rs`) --
an older Runtime would otherwise silently ignore this compatibility axis,
the same reasoning that drove `architecture_families`' own version bump.

When a registered Component's declared `artifact_formats` does not include
the loaded Artifact's own declared `artifact_format`,
`ProductionLoadedModel::load_with_component` rejects the pairing with
`ModelComponentError::ArtifactFormatUnsupported`, before any weight
materialization is attempted -- independent of, and checked alongside, the
pre-existing architecture-family gate.

## Loading a bundle

Nothing about the public `inference-components` entry points changed shape;
they simply no longer inspect the bundle's files to pick a format:

```rust
let source = magnetar_inference_component::InferenceComponentSource::authorized_local_bundle(
    "tachyon:some-provenance-string",
    "/path/to/my-qwen-bundle",
);
let component = magnetar_inference_component::LoadedInferenceComponent::load(
    "my-component-instance",
    magnetar_inference_component::InferenceComponentArtifact::from_bytes(
        component_bytes,
        component_manifest_bytes,
    ),
    source,
    trust_policy,
    magnetar_inference_component::InferenceComponentPlacement::ReferenceCpu,
)?;
```

If `my-qwen-bundle/magnetar-artifact-format.yaml` is missing, malformed, or
declares a format the registered Component's `compatibility.artifact_formats`
does not accept, `load` returns a structured `Err` instead of guessing --
check the error chain (`anyhow::Error::chain`) for
`magnetar-artifact-format.yaml` (missing/malformed sidecar) or
`artifact format unsupported` (Component/Artifact mismatch) to distinguish
the two.

`local_bundle_manifest_digest(root)` (the lighter-weight "just tell me this
bundle's real digest" helper, used e.g. to compute the value a caller passes
to `ArtifactTrustPolicy::trust_digest`) reads the same sidecar the same way.

## Migrating a legacy bundle

A bundle that predates this sidecar (produced before astorise/Magnetar#75)
is never auto-migrated by `load`/`local_bundle_manifest_digest` themselves.
An integrator with such a bundle calls the explicit, one-time migration
helper instead:

```rust
let format = magnetar_inference_component::migrate_legacy_bundle_artifact_format(
    "/path/to/legacy-bundle",
)?;
```

This derives a format using the exact filesystem heuristic Magnetar used to
apply automatically (does `model.gguf` exist at the bundle root?), then
writes `magnetar-artifact-format.yaml` once. It refuses to run again on a
bundle that already carries a sidecar, so it cannot silently overwrite an
operator's own explicit declaration. After migration, the bundle loads
through the normal explicit-declaration path like any other.

## What did not change

- `ProductionModelSource`, `ProductionModelArtifactIngestor`,
  `ProductionIngestionRegistry`, and every other part of the
  `production-model-ingestion` boundary are unchanged.
- `GgufIngestor`/`HuggingFaceIngestor`'s own bundle-content expectations
  (which files they read once selected) are unchanged.
- The opaque invocation entry points `invoke_payload_opaque`/
  `invoke_payload_streaming_opaque` (astorise/Magnetar#88) are unaffected --
  they operate after ingestion/loading has already succeeded and do not
  touch format selection at all.
