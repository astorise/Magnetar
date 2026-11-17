## Context

The Component side asked two things to be checked against real runs on #198.
The first was the RoPE pairing: interleaved for latent attention, half-split
for the non-MLA DeepSeekMoE path. The second was the meaning of an absent
`mixture-of-experts-layers` or `expert-routing`. The tiny bundles exercise the
latent-attention path end to end.

## Decisions

- **Remove the hidden-size rule rather than exempt latent attention.**
  - The rule is not an invariant of the decoders the Runtime serves. Gemma 4
    and Qwen3 break it as well as DeepSeek-V2, because each declares a head
    width that differs from `hidden_size / heads`.
  - The loaders read a declared head width, and derive it only when the
    checkpoint declares none, in which case the rule holds by construction.
  - The projections' expected shapes catch a head width inconsistent with the
    weights.
- **Refuse, not guess, renormalized V2 routing.** The two reference
  implementations disagree, and no published V2 configuration declares it.
  Following the Hugging Face flag matched neither.
- **Interleaved pairing for latent attention** is confirmed by the logit
  match: a wrong pairing changes every rotated pair. The half-split
  DeepSeekMoE path is not covered by these bundles.
- **Fixture size.** The three bundles total about 0.5 MB, so all are kept.
  The generator and reference scripts are checked in beside them, since no
  other repository holds them.
