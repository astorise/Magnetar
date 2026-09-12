## Why

Reconciling the current codebase against a Tachyon-authored Magnetar scope charter (`perimetre-magnetar.md`) found that most of the charter is already real and verified, but two requirements it restates are not actually satisfied yet, and both have a real, observed impact:

1. **Chat template**: the charter requires "messages → chat template → tokens" to be a real, model-specific transformation. Today, `loaders/huggingface` parses a real `chat_template` string from `tokenizer_config.json` but never threads it into the produced `ModelManifest` (`chat_template: None` unconditionally), and the only formatter wired into `magnetar-cli`'s chat path (`CliChatTemplateFormatter`) is a generic `"{role}: {content}"` placeholder, not an interpreter of the model's own template. This is very likely why this session's own real-checkpoint smoke test (`Qwen/Qwen2.5-0.5B-Instruct`, an Instruct-tuned model) produced incoherent output (`" 1000000"`) for a plain-text prompt: an Instruct model given text without its `<|im_start|>`/`<|im_end|>` chat markup is operating far outside its trained distribution.
2. **CUDA multi-token decode signaling**: the charter requires an unsupported request to fail with an explicit `Unsupported` signal, not silently fall back to CPU (already true) and not leak an internal implementation detail. Today a CUDA generation request for more than one token fails deep inside KV-history resolution with `TensorError::ResidencyUnavailable` -- a real, correct, fail-closed outcome, but not the clear, checked-early signal the charter calls for, and not something a caller can distinguish from a genuine transient Provider fault.

Both are real, scoped, and independent of the device-resident multi-step CUDA decode work that remains separately tracked; closing them now removes two concrete correctness/UX gaps before that larger change begins.

## What Changes

- `loaders/huggingface` threads the real `chat_template` string (when present) from `tokenizer_config.json` into the ingested `ModelManifest.chat_template`, instead of discarding it.
- `loaders/huggingface` adds a real chat-template renderer implementing `magnetar_runtime`'s existing `ChatTemplateFormatter` trait, interpreting the Jinja2-subset templates real Hugging Face checkpoints declare (variable substitution, `{% for %}` over messages, `{% if %}` on loop/message state, and the small set of filters/tests real chat templates actually use) -- kept external to `magnetar-runtime`, matching the existing tokenizer/format externalization boundary. `magnetar-runtime` itself gains no new template-engine dependency.
- Production generation (`run_production_qwen_generation`/`_for_provider`, and `magnetar-cli`'s chat path) uses the artifact's own real chat template when the ingested manifest declares one, falling back to the existing generic formatter only when a bundle declares none -- never silently ignoring a real template that is present.
- `magnetar-runtime` adds an explicit, structured `Unsupported` failure (naming the real constraint: multi-step decode requires host-readable KV history, which a Device-resident-only Provider does not provide) raised before generation attempts a decode step, when the bound Provider cannot support it -- not left to surface as an internal residency error several layers down.
- `magnetar-runtime` records a real `tokens_per_second` value (already a contract field, currently always `None`) from actual prefill/decode wall-clock timing, for both Reference CPU and CUDA generation.
- Documents the reconciliation itself: which parts of the Tachyon scope charter are already real and verified (the large majority), which are correctly out of scope for this change and remain future work exactly as the charter itself frames them (multi-device, quantization, GGUF wired into Model Loading, native FP16/BF16 compute, additional Providers/Components), and which two gaps this change closes.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `tokenizer`: "Chat Template Is Artifact-Bound" is strengthened -- a real declared chat template SHALL be preserved through ingestion and SHALL be the one actually used to render messages, not merely a permitted-but-unused piece of metadata.
- `inference-api`: the existing "chat-template formatting occurs only through authorized Runtime prompt contracts" requirement is unchanged in spirit but its scenario is extended to require that an artifact-declared real template is what gets applied when one exists.
- `generation`: adds a fail-closed `Unsupported` requirement for a decode step a bound Provider structurally cannot perform (today: CUDA multi-token decode), and a requirement that real `tokens_per_second` usage metadata is populated from actual measured generation time rather than left `None`.

## Impact

- `loaders/huggingface`: `tokenizer.rs` (already parses `chat_template`, now actually returned), `lib.rs` (threads it into the manifest), new module for the real template renderer, new dependency (a minimal Jinja2-compatible crate, external to Core).
- `magnetar-runtime`: `generation.rs`/`inference_api.rs` (the new `Unsupported` fail-closed check and real `tokens_per_second` measurement); no change to the `ChatTemplateFormatter` trait itself, which already exists and already supports exactly this.
- `magnetar-cli`: `pipeline.rs` selects the real formatter when the loaded manifest declares a template, keeping `CliChatTemplateFormatter` as the fallback for bundles that declare none.
- No WIT contract changes; no Provider/Device capability changes; no changes to Resource Affinity, Memory Manager, or Model Loading's own lifecycle.
- Out of scope, explicitly deferred exactly as the Tachyon scope charter itself frames them: multi-device execution, quantization support, wiring `formats/gguf` into Model Loading, native CUDA FP16/BF16 compute, additional Providers (Metal/ROCm/NPU/TPU) or Model Components (Llama/Mistral/Gemma), and the device-resident multi-step CUDA decode implementation itself (tracked separately).
