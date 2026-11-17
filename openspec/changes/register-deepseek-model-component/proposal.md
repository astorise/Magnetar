## Why

Issue #198: the DeepSeek Model Component (`components/deepseek`) targets
`model-config` `1.7.0` and builds DeepSeekMoE and DeepSeek-V2 graphs
(astorise/Magnetar-component-DeepSeek#3, merged as `a82132f`). The Runtime side
of the contract is merged (#201), but nothing had run the two together. The
issue's first acceptance step is tiny random `deepseek_v2` bundles compared with
`transformers` on the Reference CPU. Running it found two Runtime problems.

1. **Model Loading refused every latent-attention checkpoint.** The generic
   architecture validation required `hidden_size == attention_head_count *
   head_dimension`. Latent attention breaks that rule: DeepSeek-V2-Lite has
   16 heads of 192 against a hidden size of 2048. So do other checkpoints that
   declare their head width, such as Gemma 4 E2B (8 × 256 against 1536) and
   31B, and Qwen3.
2. **Ambiguous routing was read silently.** For greedy and group-limited
   routing, DeepSeek-V2's own code renormalizes the chosen weights when
   `norm_topk_prob` is set, and scales them by `routed_scaling_factor`
   otherwise. transformers' port ignores `norm_topk_prob` and always scales.
   The Hugging Face loader followed the flag, so a checkpoint declaring both
   matched neither reference.

## What Changes

- **Architecture validation.** The check that the hidden size equals the head
  count times the head width is removed. Zero values and a KV head count that
  does not divide the attention heads are still refused. The query and output
  projections' expected shapes already carry both widths.
- **Hugging Face loader** (`loaders/huggingface`). Greedy or group-limited
  routing with `norm_topk_prob = true` is refused. Every published V2
  configuration declares `false`. V3's `noaux_tc` renormalizes and then
  scales in both implementations, and is unchanged.
- **The DeepSeek Component, registered.**
  - `components/deepseek` pins `a82132f`.
  - `magnetar-runtime/fixtures/components/deepseek-real.component.wasm` is
    built from that commit with the pinned toolchain (Rust 1.100, release
    `wasm32-unknown-unknown`, `wasm-tools component new`).
  - Its manifest declares:
    - WIT `1.7.0` imports, `graph-builder` and `model-config` only;
    - the families `deepseek`, `deepseek_v2`, `deepseek_v3` and `deepseek2`;
    - the formats `huggingface` and `gguf`.
- **Logit comparison with `transformers`.**
  - Three tiny random DeepSeek-V2 bundles and their `transformers` 5.23
    reference logits are checked in under
    `integration-tests/production-loading/fixtures/deepseek_v2/` (about 0.5 MB),
    with the scripts that made them.
  - Each is ingested by the Hugging Face loader and loaded with the Component
    its family selects. It then generates 12 greedy tokens on the Reference
    CPU, and every step's top-8 logits are compared with the reference.
  - The Gemma 4 comparison shares the same helper (`logit_comparison.rs`).

## Evidence

All three bundles have a dense first layer, routed and shared experts, and
latent attention:

| Bundle | Max abs logit difference |
|---|---|
| `lite`: no low-rank query, YaRN with `mscale == mscale_all_dim` | 6.676e-6 |
| `qlora`: low-rank query, YaRN scaling cos/sin, routed scaling 2.5 | 1.240e-5 |
| `plain`: no RoPE scaling | 3.338e-6 |

Tokens are identical in all three.

- **Before the validation change**, Model Loading refused all three
  ("hidden size must equal attention heads times head dimension").
- **Before the routing refusal**, a variant declaring `norm_topk_prob = true`
  had the same tokens as `transformers` but logits up to 0.13 apart. Isolating
  its features showed that the low-rank query alone (1.2e-5) and the scaled
  cosines and sines alone (4.8e-6) matched.
- **The comparison can fail.** Removing YaRN from the reference only makes the
  `lite` bundle differ by 1.02 at the first step, and its tokens differ from
  the second.

**Limits:**
- The checkpoints are tiny.
- Not run: a `deepseek2` GGUF, CUDA, and real DeepSeek-V2-Lite weights.
- V3 and V2-236B are still refused by the Component, since sigmoid and
  group-limited routing are Runtime milestone 3.

## Impact

- `magnetar-engine/src/model_component.rs`: the validation.
- `magnetar-runtime/src/qwen_model_component/tests.rs`: the test that expected
  the refusal.
- `loaders/huggingface`: `config.rs` (the refusal and its test).
- `magnetar-runtime/fixtures/components/`: the artifact and its manifest.
- `integration-tests/production-loading`: `fixtures/deepseek_v2/`,
  `logit_comparison.rs`, `tests_deepseek_v2_logits.rs`,
  `tests_gemma4_logits.rs`, `test_support.rs`.
- `components/deepseek`, `SUBMODULES.md`.
