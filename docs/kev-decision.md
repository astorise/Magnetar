# Kev-0.8B decision inference

The optional `kev` feature of `magnetar-inference-component` exposes a
structured decision path for the pinned Kev-0.8B profile. Supply a trusted
Qwen3.5 Component and the compatible, pre-merged checkpoint bundle when loading
`LoadedInferenceComponent`. Trust in the Component and model bundle is still
configured separately through `ArtifactTrustPolicy`.

The [checkpoint conversion tool](https://github.com/astorise/Magnetar-component-Kev/pull/8)
creates the derived Safetensors bundle from the pinned base and adapter. Its
provenance and format sidecars accompany the weights. The conversion tool is
proposed separately and is not yet in this repository's locked Kev revision.
Export the separately downloaded `head.pt`
with `integration-tests/production-loading/fixtures/export_kev_pointer_head.py`;
that script verifies the pinned input digest and uses PyTorch's restricted
weights-only loader. `load_pinned_head_safetensors` then reads the exported file
once, bounds its size, verifies the pinned export digest and applies the
checkpoint's saved temperature. The head and backbone must be the compatible
Kev-0.8B artifacts; the decision function does not authenticate that pairing.

```rust,ignore
use magnetar_inference_component::kev::{load_pinned_head_safetensors, TextQuestion};
use std::path::Path;

// `component` is a LoadedInferenceComponent built from the trusted Qwen
// Component and the compatible, pre-merged Kev bundle.
let head = load_pinned_head_safetensors(Path::new("kev-head.safetensors"))?;
let choices = ["yes", "no"];
let questions = [TextQuestion { instruction: "Choose an answer", options: &choices }];
let decisions = component.decide_kev(&head, "shared state", &questions)?;
```

Results preserve question and option order and contain option logits,
probabilities and the selected option. No confidence calibration, abstention
policy or action distribution is inferred. The text path uses the loaded
bundle's tokenizer, checks the five pinned control-token IDs and escapes
token-like user text as the upstream encoder does. Tokenized callers can use
`decide_tokenized`; their content must not contain delimiter IDs.

This initial profile permits 1–128 questions, 1–255 options per question,
at most 1 MiB of text per request and at most 512 tokens for each complete
state-plus-question row. It rejects overflow rather than truncating. Each row
is a separate prefill from position zero: prefix-cache reuse and isolated-option
mode are not implemented. The CPU numerical campaign does not establish GPU
qualification.
