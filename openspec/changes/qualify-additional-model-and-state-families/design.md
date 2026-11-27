## Context

This draft is derived from roadmap proposal 1.0 (2026-10-29), whose reference commit is `09976e7792e610a04aa2b8af6c6991ba7bcbc225`. Repository inspection for this proposal used `acbb95cab732f01ace7dbc4203ebf698cd7a5fa6`. Canonical specs and implemented paths at the latter revision take precedence over historical roadmap wording. This is proposed work, not a stable new API or measured result.

## Goals / Non-Goals

Deliver the explicitly enumerated F01, F04 outcomes through Model Components and loaders; generic Runtime validation; Provider operators, preserving existing trust, resolution, lifetime and portable Component contracts. Reuse the baseline contracts linked from proposal.md. Broader roadmap work is split into independent changes; adopting an inspiration project or copying its implementation is not a goal.

## Decisions

### Implementation boundary and data flow

Qualify a second dense family and added Qwen versions before widening claims. Specialized MLA/GQA/SWA/sparse or KDA/recurrent families use explicit portable state schemas and operators; family semantics stay in Components, not Runtime Core.

### First additional dense checkpoints (F01, first increment)

Pinned real-checkpoint profiles: the real-checkpoint tests read `MAGNETAR_REAL_CHECKPOINT_PROFILE` from a table pinned by Hugging Face revision and `model.safetensors` sha256. The variable defaults to Qwen2.5-0.5B-Instruct, the logit corpus's checkpoint. The GPU workflow pins each checkpoint's JSON files by sha256.

| Profile | Revision | Shape |
|---|---|---|
| Qwen/Qwen2.5-1.5B-Instruct | `989aa79` | 28 layers, 1536 hidden, 12 query / 2 KV heads of 128 dimensions, tied LM head |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | `ad9f0ae` | the same Qwen2 shape, rope_theta 10000, untied LM head; served by the Qwen Component |

What qualifying them required:

- **128-dimension attention.** The Tensor Core prefill attention and the grouped split decode attention served only 64-dimension heads. Both are now templates on the head dimension:
  - The prefill attention uses 68 KB of dynamic shared memory for 128 dimensions, so the launcher raises the function's 48 KB limit.
  - The decode attention uses 32-key chunks for 128 dimensions.
  - Five 128-dimension cases joined the Reference CPU conformance (CUDA Provider `acb47db`).
- **Prefill activation memory.** Every graph edge's output stayed Provider-resident until the step ended. At 4096 prompt tokens, Qwen2.5-1.5B ran its 12 GB RTX 3060 out of memory at layer 24 (run 42127910391).
  - A prefill now releases each node output once its last reader has run. It keeps graph inputs, weights, KV stores and the edges it materializes.
  - Decode steps keep their resources, because a captured step replays against them.
- **Shared special tokens.** DeepSeek-R1-Distill declares its end-of-sequence token as its pad token. The tokenizer metadata now accepts one token in several roles. One role declared twice, or one id with two texts, is still a conflict.
- **Bounded verification memory.** The `model.safetensors` digest is computed in 8 MB reads. The 1.5B checkpoints in f32 (about 6 GB) do not fit the runner next to a Reference CPU run (exit 137, runs 42116467880 and 42116844824). Their check is therefore the same campaign with the Tensor Cores off, which must generate the same tokens.

Evidence (RTX 3060, GPU CI run 42137496729):

| Warm medians | 5 / 512 / 4096 prompt tokens: TTFT | TPOT |
|---|---|---|
| Qwen2.5-0.5B-Instruct | 14.6 / 53.6 / 478.3 ms | 4.4 / 4.4 / 5.0 ms |
| Qwen2.5-1.5B-Instruct | 33.4 / 171.7 / 1480.8 ms | 12.1 / 12.2 / 14.1 ms |
| Same, Tensor Cores off | 33.4 / 643.2 / 6086.4 ms | 12.2 / 12.2 / 14.2 ms |
| DeepSeek-R1-Distill-Qwen-1.5B | 33.2 / 172.2 / 1491.1 ms | 12.0 / 12.2 / 14.1 ms |
| Same, Tensor Cores off | 33.5 / 643.4 / 6083.6 ms | 12.1 / 12.2 / 14.1 ms |

- Both 1.5B checkpoints generate the same 128 greedy tokens with the Tensor Cores on and off, after 5-, 512- and 4096-token prompts.
- Qwen2.5-0.5B keeps its Reference CPU comparison, its logit corpus match and 13 of 16 replayed decode steps.
- Releasing prefill intermediates also lowered Qwen2.5-0.5B's warm TTFT at 512 prompt tokens from 62.7 to 53.6 ms, and its TPOT at 4096 from 5.8 to 5.0 ms (run 42106862830 before).
- A 1.5B decode step reads about 3.1 GB of bfloat16 weights. At 12.1 ms, that is about 70% of the RTX 3060's memory bandwidth.

Limits:

- The 1.5B checkpoints have no Reference CPU or logit-corpus comparison yet; a bfloat16 Reference CPU path would make one affordable.
- The Hugging Face loader ignores `add_bos_token`, which DeepSeek-R1-Distill sets. Tokens still match across Providers, but its prompts are not yet encoded exactly as upstream.

### First non-Qwen2 checkpoint: Llama 3.2 (F01, second increment)

`meta-llama/Llama-3.2-1B-Instruct` (revision `9213176`, gated) is the first checkpoint the Qwen Component cannot serve:

- Shape: 16 layers, 2048 hidden, 32 query / 8 KV heads of 64 dimensions, tied LM head.
- RoPE: rope_theta 500000 with Llama 3 per-frequency RoPE scaling (`rope_scaling.rope_type = "llama3"`, factor 32).

It is qualified through the Model Component boundary, so the Runtime gains no Llama-specific semantics:

- **Component selection.** The campaign loads each checkpoint with the registered Model Component whose manifest declares the ingested architecture family (`compatibility.architecture_families`). It never selects by model name. `llama` resolves to the Llama Component; `qwen2` (Qwen2.5 and DeepSeek-R1-Distill) resolves to the Qwen Component.
- **Contract.** `magnetar:model-component-graph@1.4.0` adds `model-config.rope-scaling`. It is a new function, not a new `architecture-config` field, so Components compiled against 1.2.0/1.3.0 still link. It returns the checkpoint's per-frequency scheme as normalized by the loader. The Hugging Face loader reads `rope_type` and refuses schemes it does not implement (`yarn`, `dynamic`, ...) instead of ingesting them as unscaled.
- **The Component declares, the Runtime forwards.** The Llama Component (`334d221`, which also implements the 1.3.0 segment exports) declares the scheme on its `rope` nodes as the portable `frequency_scaling` attributes. The Runtime forwards node attributes unchanged.
- **The Core names the scheme, not a family.** The operator, its Kernel feature and the WIT case call it `banded` (kept, interpolated and divided wavelength bands). Only the Hugging Face loader knows that `rope_type = "llama3"` maps onto it. The `quality / model-family isolation` guard enforces this.
  - It refuses a Component-built graph whose `rope` nodes do not declare exactly the bound configuration's scaling, for example the Qwen Component (1.3.0) given a Llama 3 checkpoint.
  - It refuses a Provider whose `rope` Kernels do not advertise `banded-frequency-scaling`.
- **Kernels.** Reference CPU, the CPU Provider and the CUDA Provider rescale each pair's inverse frequency with the same f32 operations:
  - Reference CPU matches Hugging Face's `_compute_llama3_parameters` within 1e-5 relative, across all three bands.
  - CUDA matches the CPU Provider in conformance.

Evidence (RTX 3060, GPU CI run 42186542821):

| Warm medians | 5 / 512 / 4096 prompt tokens: TTFT | TPOT |
|---|---|---|
| Llama-3.2-1B-Instruct | 27.2 / 127.9 / 1087.8 ms | 10.0 / 10.2 / 11.4 ms |
| Same, Tensor Cores off | 27.2 / 475.3 / 4502.5 ms | 10.0 / 10.2 / 11.5 ms |

- The same 128 greedy tokens with the Tensor Cores on and off, after all three prompts. The text is coherent at every length: " Paris. It is located on the Seine River and is known for its rich history, cultural landmarks, and iconic architecture..." after 4096 tokens.
- The Qwen2 checkpoints, now loaded through the same family-selected path, still pass every earlier check. DeepSeek-R1-Distill generates the same tokens as before the change (run 42144929355).
- Limits:
  - No Reference CPU or logit comparison yet. The checkpoint in f32 (about 5 GB) does not fit the runner next to the test.
  - The plain-text prompt is encoded without `<|begin_of_text|>`, the same `add_bos_token` gap as DeepSeek-R1-Distill.

### Experimental Olala expert network and hybrid state

The pinned Olala Mamba-3 reference returns its prefill angle state modulo
`2*pi`, while its incremental step retains the unwrapped next angle. A graph
must express that boundary explicitly: an additive `remainder-scalar@1`
operator accepts one finite contiguous f32 tensor and a finite positive f32
divisor, returning an equal-shaped tensor with the divisor-sign remainder.
The Component places it on the angle-state output at a prefill boundary and
omits it for cached decode. Missing or invalid divisors and non-finite inputs
are rejected. Because this is a distinct Operator identity, Providers without
an implementation cannot silently execute the graph. Reference CPU and the
CPU Provider are the initial targets; no GPU support is implied. The existing
unwrapped Mamba-3 operators and WIT signatures remain unchanged. Chunked
prefill and state transitions require independent numerical comparisons;
the operator alone does not qualify Olala inference.

At revision `b40f8165ee2254c18601220e19842600c6d91794`, Olala uses
selection-biased sigmoid routing and an **ungated** expert network
`W_out(relu(W_in(x))²)`, with optional routed input/output projections and a
separate shared ReLU² MLP. The existing gated `silu(gate) * up` expert form
cannot represent it. Extend portable `moe-experts@1` with
`activation = "relu_squared"`: the explicitly selected sigmoid form takes
`h`, logits, stacked `W_in`, stacked `W_out`, selection bias, and optionally
per-expert scale. The ungated form requires a new `moe-relu-squared` Kernel
feature as well as sigmoid routing and any other required feature; a Provider
advertising the older generic activation feature alone is ineligible. Preserve
the existing gated forms and input positions. The separate portable
`activation@1` kind `relu_squared` requires its own
`activation-relu-squared` Kernel feature. Other dtypes or Providers are not
qualified by a CPU-only fixture.

Selection uses sigmoid scores plus the learned bias, while expert mixture
weights use unbiased scores. The `multiple` normalization mode divides by
the selected-score sum plus `1e-20` only when more than one expert is used;
the routed scaling factor is applied afterwards. Exact ties across a selection
boundary are rejected because upstream does not specify their ordering.
The portable routing tests cover top-1, top-2 and the published top-6 of 256,
bias-driven selection changes, saturated scores, invalid values and ties.
Synthetic Olala MoE-to-logit fixtures execute pinned upstream PyTorch classes;
separately, tiny Kimi top-1/top-2 checkpoints exercise the same routing mode
end to end through Hugging Face loading, a Component and both CPU Providers.
Neither test qualifies Olala's full hybrid model or a GPU Provider.

An Olala-side f32 reference covers the preprocessed Mamba-3 MIMO recurrence:
rotary angles, the SSM, the previous rotated key and previous input are four
separate state tensors. The pure step preserves prior state on error and
accepts explicit grouped Q/K, MIMO projections, decay, dt and trap values.
Synthetic tests compare every output and state with the pinned upstream
PyTorch prefill and cached-step implementations, including chunk continuation
and zero-state reset. This standalone reference does not implement the
layer's learned projections or provide Runtime-managed state.

The additive portable `mamba3-mimo@1` operator covers the same f32,
single-sequence preprocessed recurrence without assuming a model family.
Its 18 inputs are Q/K `[T,R,G,N]`, input/gate `[T,H,P]`, decay/dt/trap
`[T,H]`, angle projection `[T,H,A]`, the four prior states (angle `[H,A]`,
SSM `[H,P,N]`, key `[R,H,N]`, input `[H,P]`), Q/K biases `[R,H,N]`,
three MIMO projections `[R,H,P]`, and skip `[H]`. Its five independent
outputs are `[T,H,P]` and the four next states with matching shapes. Positive
`H,G,R,P,N,A`, `H % G == 0`, even `N`, and `A <= N/2` are required; dt is
positive, preprocessed decay nonpositive and trap in `[0,1]`. Inputs and
outputs may not alias, every tensor must be finite, and unsupported dtypes,
geometries or missing Kernel forms fail closed. This pure operator does not
itself allocate, reset, commit or isolate a session's recurrent state.

For the pinned single-group, biasless-projection Mamba-3 profile, the
additive `mamba3-mixer@1` composes the learned input projections and Q/K
RMS normalizations with that recurrence. It receives hidden states, four
prior states, two input matrices, dt bias, two normalization weights, Q/K
biases, three MIMO projections and the skip weight. It returns the flattened
`[T,H*P]` activation and four next states. It requires `H*P = 2*hidden`,
the upstream rotary fraction, positive epsilon and decay floor, and an
explicit zero-centered normalization flag. More general grouping, optional
projection forms, post-gate normalization or auxiliary linear branches are
not assumed. A Runtime graph test binds the four state tensors to separate
recurrent slots, commits the wrapped prefill angle, and verifies cached steps,
discard/retry and reset against the pinned upstream fixture. This exercises
state ownership and isolation between two sessions for one mixer, not a
complete Olala Component. The Operator
catalog validates both forms' geometry, all input and output state shapes,
bounded intermediate sizes and the projected mixer's numerical attributes
before a graph is planned.
The Olala library now plans this f32 invocation for all 29 Mamba-3 layers
of the pinned mini configuration. Its eleven checkpoint names, shapes and
storage dtypes match the verified Safetensors header; matrix transposition and
the head/rank permutation are explicit after storage decoding. An opt-in test
decodes the first layer's real BF16/F32 weights through Runtime's loader and
compares the f32 mixer output and all four states with the pinned PyTorch
class, using synthetic hidden states. This does not qualify BF16 arithmetic,
full-model inference or state ownership in Runtime.

The portable `token-shift@1` operator expresses the previous-token mixing
used by the pinned Olala attention profile without equating it to KV history.
Its f32 inputs are raw values `[T,H,D]`, gate logits `[T,H]`, explicit
document-start flags `[T]` (exactly zero or one), and the previous raw token
`[H,D]`. The first output is `sigmoid(logit) * previous + (1-sigmoid(logit))
* current`; at a document start it is exactly the current value. The second
output is the last **raw**, not mixed, token. Both are fresh allocations.
This operator can be used independently for keys and values, but does not
manage their session state, KV attention history or a full attention layer.

The experimental `differential-attention@1` core accepts already-normalized
and position-scaled queries `[T,Hq,D]`, key/value history `[S,Hkv,D]` and
per-token mixing logits `[T,Hkv]`. `S >= T`, `Hq` is a multiple of `Hkv`, and
each KV head groups one noise head after one or more signal heads. It applies
causal attention with an explicit score scale, optional left-window and score
softcap, then subtracts sigmoid-weighted noise from each associated signal.
An explicit bfloat16-rounding flag models the upstream eager attention's Q/K/V
and attention-output conversions; the contract never silently assumes that
precision. This primitive does not project Q/K/V, update history, apply RoPE,
or manage token-shift state and therefore does not qualify a full V block.

For the pinned no-RoPE, biasless, Q/K-normalized and scalable-softmax profile,
the `projected-differential-attention@1` composes Q and low-rank K/V
projections, separate raw token shifts, Q/K RMS normalization, position scaling
and differential attention. It takes hidden states, previous raw K/V, prior
attention history (normalized K and shifted V), positions and eleven learned tensors, and returns the
attention output, two next raw states and two updated histories. A zero past
uses a single ignored history row because execution graphs require nonempty
shapes; `past_length` explicitly states the number of valid rows. Positions
must be contiguous from that length. A document reset is a new call with zero
past and position zero, which suppresses any supplied prior raw state. Packed
multi-document attention, RoPE, value embeddings, XSA and biased/CosNet
projections are outside this profile and must not silently use it. The graph
executor can bind raw previous K/V to Runtime-managed recurrent slots. The
original Olala test supplies attention history explicitly; a separate
hand-built graph now tests Runtime-owned copy-based history as described below.
The Olala profile can reorder decoded f32 weights into this operator's
layout. An opt-in integration test decodes the pinned checkpoint's first
differential-attention layer and compares its f32 projections and explicit
states with the upstream PyTorch class on synthetic hidden states, retaining
the class's bfloat16 attention-boundary rounding. This is not a qualification
of native BF16 projections or end-to-end hybrid inference.

The experimental attention-history increment is a Runtime-owned **tensor** input,
distinct from a Kernel-managed cache handle. A graph declares one input and
one complete-history output for each logical KV identity; the first call gets
a single ignored zero row, while later calls bind the exact committed or
pending history with a matching position, shape and compatibility key. The
Runtime stages each complete output and commits or discards it with the raw
recurrent state. This initial copy-based form must reject padded/in-place
stores and caller-supplied history bindings. It does not advertise device-side
KV access to a Kernel. The proposed additive graph-builder 1.16 Capability
exposes these bindings without changing older imports. A compiled test Component
builds paired prefill and decode histories through Wasmtime; the published
contract and the Runtime copy must be reconciled before release.
The hand-built projected-attention graph compares prefill, continuation,
cached decode, retry and separate sessions with the pinned PyTorch fixture.
The portable `position-range@1` operator generates a contiguous f32 vector
from a two-dimensional token anchor. A graph declares its nonnegative
`position_offset`; the Runtime supplies the actual step position on reuse.
The complete range is checked before allocation and stops at 16,777,215 so
every index and its successor remain exact in f32. The managed-history
attention test now builds its position input this way, including continuation
and decode, rather than receiving positions from its caller. Only Reference CPU
is qualified for this new operator.
Variable decode lengths still need Component graph/plan handling, and the
portable attention's element bound limits the supported window. Neither this
test nor the internal graph form qualifies full Olala generation.

The Component graph-construction API can request a decode graph at a different
committed-history length while retaining the original prefill length. A real
1.16 Component verifies that the history descriptors change with that length,
and a previously prepared plan rejects the new graph's different fingerprint.
This construction API does not replace the generation loop's published plan:
the separate experimental plan-advance API validates both graphs, growing KV
shapes and stable recurrent/KV schemas, the ready Model Instance revision,
the original canonical plan and its Kernel bindings, Provider, guards and
leases before publishing a new generation. It rejects custom plan policies
instead of discarding them. Tests execute several newly prepared plans with
committed KV and recurrent state and verify failed advances leave the old plan
unchanged. The caller still owns session/history validation, and the production
generation loop does not yet invoke this API for variable-length decoding.

Hybrid recurrent state uses an additive graph Capability version, not KV
aliases. `graph-builder@1.11.0` adds `recurrent-resource(layer, slot, shape)`,
`recurrent-input(resource-id)` and `recurrent-output(resource-id, edge-id)`.
The host issues identities; slot names are 1-64 ASCII letters, digits, `_` or
`-`, and fixed, nonempty f32 shapes are capped at 67,108,864 elements. Graph
validation requires one consumed input and one produced output per identity,
with matching descriptor, layer, slot and compatibility key. State edges cannot
alias or be KV/prefix edges. Fingerprints include state metadata and fusion
cannot remove a state edge. A real Wasmtime Component tests both graph phases;
older Components continue linking through their original versions. The current
first-native executor can stage and commit declared recurrent resources; this
does not by itself qualify an Olala or Nemotron model Component.

A conservative Mamba-2 prefill profile may continue through full-sized chunks
in production: every attention node reads causal KV history, both recurrent
states bind directly to each mixer, and the remaining graph uses the qualified
token-wise operators. The optimized graph may fuse a projection with its
residual. The real Component graph's two chunks, generated logits and decode
steps match the pinned synthetic PyTorch reference and a one-pass execution.
Unknown operators, other recurrent bindings, noncausal attention and a final
partial chunk retain the single-pass fallback; padding must not advance state.
Before a decoder graph pair is returned, the Runtime also checks that prefill
and decode declare the same state identities, layers, slots, compatibility
keys and tensor descriptors; mismatched phases, schemas or out-of-range layers
are rejected. Segment graphs may only read or produce recurrent states in
their own declared layer interval.
The graph contract extracts each validated input/output binding and its logical
layout in deterministic resource-id order. Cross-phase comparisons use the
layout, since edge names are local to each graph. Extraction checks the
recurrent boundary only and never replaces whole-graph validation; its logical
identity is not a session, model instance, Provider or allocation identity.
The existing first-native KV path retains the request's session, model,
tokenizer metadata and prompt fingerprint with both pending and committed
state. It checks them before decode, retry, commit or release, including when
an abandoned cache has already been forgotten. A colliding request id alone
does not authorize access to or cleanup of another request's state.
KV compatibility validation also compares declared component architecture,
model revision, tokenizer vocabulary size and attention implementation;
an absent declaration does not match a present one.
First-native generation records its Component identity and version together
with the supplied artifact digest (or its built-in default marker), and the
tokenizer vocabulary size. A changed Component cannot read or release an
existing request's state solely by reusing the same request id.
Each first-native KV generation also records its accepted token sequence.
Decode and commit require the exact previously accepted sequence, while a
second prefill cannot overwrite an existing committed generation. A rejected
history leaves both the prior cache and any pending step intact. Token IDs are
not printed by the state diagnostic representation.

For this execution path, initial state is zero; continuing calls read
pending state within an attempt or the last committed generation. State slots are keyed by session, model and
component compatibility, layer, slot and dtype/shape, and never contain
Provider handles in WIT. Outputs remain pending until the token/segment and
all matching KV changes commit together. Failed dispatch, sampling,
cancellation, incompatible shape/affinity or unsupported Provider features
must leave the prior committed generation unchanged. State reset and document
boundaries clear only the relevant sequence, while close/expiry releases all
its leases after in-flight work completes. Multi-output graph production,
old/new component linking, cross-segment isolation and completion-safe
retirement need explicit tests for each claimed profile. The standalone Mamba-2
reference is evidence for numeric equations, not a complete model Component.

Stateful scans naturally return their activation and several new state tensors
from the same computation. The additive `graph-builder@1.11.0` also exposes
`add-node-outputs` with the existing node identity, attributes and input list,
plus a list of 1-16 output shapes and a returned edge for each output.
Each output remains independently validated and may subsequently be bound to
a recurrent slot; this form does not attach KV metadata. The old single-output
`add-node` stays unchanged. Multi-output description alone does not establish
the state lifecycle or model-specific numerical behavior.

The initial experimental `mamba2-mixer@1` CPU operator is a pure function of
one sequence and two explicit f32 state tensors. Inputs are `[T,H]` hidden
states, `[C,K]` oldest-first convolution history, `[heads,head_dim,state_dim]`
SSM state, `[H,P]` input projection, `[C,K]` depthwise convolution weights,
`[heads]` dt bias, A-log and D, `[heads*head_dim]` gated-normalization weight,
and `[heads*head_dim,H]` output projection. Boolean attributes explicitly
declare an optional `[C]` convolution bias and optional `[P]`/`[H]` linear
biases, appended in that order. `C = heads*head_dim + 2*groups*state_dim` and
`P = heads*head_dim + C + heads`. Required attributes are integer `heads`,
`head_dim`, `groups`, `state_dim`, `conv_kernel`, float `epsilon`, `dt_min`,
and boolean `conv_bias`, `linear_bias`; optional float `dt_max` is finite when
present and absent means unbounded. Epsilon is positive and the minimum dt is
nonnegative. The three ordered outputs are `[T,H]` mixer output and
same-shaped next convolution and SSM states. The CPU form follows the pinned
Nemotron-H SiLU convolution, selective update and grouped gated RMS norm;
Mamba-3 is a different operator. Invalid shapes, values, unsupported bias
forms or missing Kernel support fail closed. A Runtime graph test binds the
convolution and SSM inputs and outputs to separate recurrent slots and compares
chunked prefill, cached steps, discarded attempts, failure, independent sessions
and reset against the pinned upstream synthetic fixture. A complete mixed
Nemotron Component, its attention history and checkpoint loading remain open.
The Operator catalog validates state, weight and output shapes, bias-dependent
arity and numerical attributes before a graph can be accepted for planning.

The pinned Nano remote model source has a separate whole-model qualification
ambiguity: `NemotronHBlock.forward` forwards `cache_position` to attention but
does not pass its `cache_params` as `past_key_value` or forward the attention
mask. Its Mamba call forwards `cache_params`, but not the Mamba mask. The
attention implementation updates its cache only when `past_key_value` is
supplied. `components/nemotron/tests/fixtures/check_nano_block_forwarding.py`
executes the SHA-256-verified upstream block with spies to reproduce this
wiring without weights or CUDA. The standalone Mamba-2 numerical fixtures do
not settle whether a complete cached hybrid graph should match this source or
the intended attention-history contract; neither is qualified by them.

The integrated Transformers 5.23.0 implementation is a possible separate
reference for that wiring: at commit `52b080b6416a7897e2af90febeb4dfad22b84d19`,
its `NemotronHBlock` forwards both the cache and masks to its attention and
Mamba mixers. It is not numerically interchangeable with the pinned NVIDIA
source merely for that reason. Its Mamba mixer sets the prefill scan's lower
dt bound to `config.time_step_min` (0.001 for the Nano configuration), whereas
the pinned NVIDIA profile uses zero; its PyTorch cached single-token scan does
not apply that lower bound. The hash-checked
`integration-tests/production-loading/fixtures/audit_nemotron_transformers_forwarding.py`
reproduces the forwarding with stand-ins and, optionally, runs the actual
Transformers PyTorch scan functions. With a deliberately small synthetic dt,
the latter returns approximately 0.001 on prefill and 2.06e-9 for a cached
step. This demonstrates a possible divergence, not that the real checkpoint
reaches that range. Whole-model qualification must select and document an
explicit reference and validate its prefill/decode behavior.

Reproduce with the unmodified [Transformers source at that commit](https://github.com/huggingface/transformers/blob/52b080b6416a7897e2af90febeb4dfad22b84d19/src/transformers/models/nemotron_h/modeling_nemotron_h.py)
(SHA-256 `d13b26d1adb49f54dd952b9ee11b3b008b79e55792c9e8d1cce9f46c213f6397`):

```sh
python integration-tests/production-loading/fixtures/audit_nemotron_transformers_forwarding.py \
  /path/to/modeling_nemotron_h.py --numeric
```

The numerical option requires PyTorch and Transformers 5.23.0 and verifies the
installed source hash before running the reference functions.

The Nemotron component also derives an invocation plan for the pinned Nano
mixer geometry. Checkpoint tensor names, shapes and dtypes for all 23 Mamba
layers were checked against 12 real Safetensors shard headers, including BF16 matrices and
F32 A-log/D vectors. The source configuration's `time_step_min` and
`time_step_max` initialize parameters; its default inference limit is zero to
unbounded. This is a shape and conversion plan for the f32 operator, not
checkpoint loading, BF16 numerical qualification or recurrent execution.

A separate synthetic CPU graph compares the pinned NVIDIA `NemotronHMOE`,
`NemotronHTopkRouter` and `NemotronHMLP` PyTorch classes with sigmoid routing,
correction bias, top-1 and top-2 normalization, ReLU-squared routed experts,
the shared expert and a logits projection. The fixture uses the pinned Nano
profile's one selected group and scaling semantics. It does not establish a
mixed Mamba/attention/MoE Component or cover other group-masking profiles.

An experimental Component now constructs mixed prefill and decode graphs
through `model-component-graph@1.15.0`; an actual Wasmtime test checks the
Mamba-2, non-RoPE attention, dense MLP and MoE nodes and separate
convolution/SSM slots, and rejects a missing weight or unsupported position
mode. The checked-in
Component fixture is built from `Magnetar-component-Nemotron` commit
`5b7370a6881a5f04c65deb4765e6852d749a54d1` and takes normalized f32 graph
weights. A synthetic mixed model built from pinned NVIDIA PyTorch primitives
compares the real graph's full prefill and prefix plus four cached decode
steps, including logits, convolution/SSM state and attention K/V. Releasing
the session and opening another starts both types of state afresh. This is not
checkpoint ingestion or full-checkpoint numerical qualification. Its explicit
causal attention history is a deliberate profile choice and does not claim
equivalence to the pinned NVIDIA block that fails to forward its cache.

### Experimental masked encoder attention

ModernBERT's attention is bidirectional; its local window is symmetric and
inclusive, unlike the decoder's causal window. Define a separate portable
`encoder-attention@1` with f32 `q`, `k`, `v` of shape `[T, heads * head_dim]`
and a `[T]` key-validity mask whose values are exactly 0 or 1. Output has the
same shape as `q`; padded queries are still evaluated as upstream does, but
only valid keys contribute. An optional nonnegative `local_radius` admits
keys with `abs(query_index-key_index) <= local_radius`; absent means global.
Every query must have at least one admitted key, otherwise fail explicitly.
The optional finite positive `scale` defaults to `1/sqrt(head_dim)`. Position
encoding is supplied by the Component before attention; this single-sequence
form does not infer positions, concatenate documents or silently repurpose a
decoder KV cache. ModernBERT's default position IDs are physical indices and
its local radius is `local_attention / 2` at the pinned Transformers profile.
This primitive does not by itself implement an encoder, padding-aware batching
or Laya's learned decision head. Unsupported Providers do not advertise it.

The f32 graph profile for an encoder also needs three additive, independently
negotiated operators. `affine-layernorm@1` accepts `[T,H]` input, `[H]` scale
and an optional trailing `[H]` bias, with a required positive finite `epsilon`;
it returns `[T,H]` using population variance and f32 accumulation. The
separate `gelu-exact@1` uses the erf form of GELU and preserves its input
shape, avoiding the existing tanh-approximate `gelu@1` semantics. CPU
implementations may numerically approximate erf but must match the independent
f32 reference within the declared test tolerance; they must not substitute
the distinct tanh GELU formula.
`slice-last@1` accepts `[T,H]`, nonnegative integer `start`, and positive
integer `width`, and returns `[T,width]` from the specified contiguous column
range. All three reject malformed, non-finite or unsupported inputs, and no
Provider advertises them without implementing their stated semantics. These
operators enable exact graph construction; they do not themselves supply an
encoder Component or a decision API.

The additive `model-component-graph@1.12.0` contract introduces an explicit
`encode` graph phase and a separate `encoder-component-graph-producer` world.
Its `build-encoder-graph(sequence-length)` exports one single-sequence graph
whose inputs are token IDs and a binary key mask, and whose output is the
per-token hidden-state tensor. This is neither decoder prefill nor generation.
An optional `model-config.encoder-config` supplies a finite positive LayerNorm
epsilon, the maximum position count, independent normalization/attention/MLP
bias flags and one ordered record per layer: its optional inclusive symmetric
local radius, positive RoPE theta and whether attention normalization exists.
Hidden/head/intermediate/vocabulary sizes still come from the existing
architecture record. Components must read any declared encoder configuration;
an older Component that cannot do so is rejected. Unbound, invalid, stale or
incompatible configurations fail closed. The first profile uses physical
zero-based positions and f32 tensors; arbitrary position IDs, batching,
decision-head execution and checkpoint weight mapping remain separate work.
The registered Component runtime can construct an encoder graph from an
explicit architecture, encoder profile and caller-provided weight inventory;
it checks input and output descriptors, requires both token IDs and the mask
to be connected to the hidden-state output, and releases temporary Component
state even on failure. This dataflow check cannot prove the numerical meaning
of a mask; independent reference comparisons are still necessary. A resident
encoder now loads trusted graph-facing weights and executes isolated,
single-sequence requests, releasing temporary resources on success and
failure. Checkpoint-specific source authentication and tensor conversion are
the responsibility of the caller that constructs those graph-facing weights.

### Typed decision results

The additive `magnetar:decision@1.0.0` types and their Rust counterpart
represent one question without padded options. Each option has its raw
finite score and probability in input order; the result distinguishes an
option index, a finite numeric prediction and explicit abstention. An optional
action distribution uses the checkpoint-declared order and does not imply an
abstention threshold. Confidence, when supplied, is a statistic in `[0,1]`,
not a calibration claim. Every present distribution must be nonempty, finite,
nonnegative, at most one per entry, and sum to one within `1e-6`; an option
prediction must be in range. Multiple questions can carry separate results
with different option counts. This contract is independent of any particular
decision checkpoint or calibration claim.

The standalone Laya f32 head can construct its parameters from decoded,
named tensors after validating geometry, shapes and finite values. Its 35
expected head tensors match the pinned checkpoint's F16 header. A separate
opt-in test compares the real head weights, converted to f32, with the pinned
PyTorch head using synthetic padded and variable-option encoder outputs. A
separate CPU comparison executes the complete checkpoint's encoder and head
against PyTorch. The pinned Laya decision path also authenticates the source
checkpoint and configurations, loads its graph-facing encoder weights into
the Runtime, and compares tokenized decisions and text requests using the real
tokenizer with independent references. This is f32 execution and does not
qualify native BF16 or GPU execution, nor establish statistical calibration.

### Identity, eligibility and recovery

Inputs retain validated artifact/model/Component identities, dtype/layout, phase, shape and version provenance where relevant. Runtime checks trust, ResourceAffinity, capability support and MemoryManager admission before policy ranks candidates. Plans and state reference immutable generations and opaque leases. Cancellation stops future submissions; in-flight work retains its leases until completion. On Device loss or incompatible state, invalidate affected work and replan or fail explicitly; never silently rebind resources or restore an untrusted generation.

### Gated-delta mixer profile

The experimental `gated-delta-mixer@1` is an additive, single-sequence f32
operator. It takes hidden states [T,H], convolution history [C,K], recurrence
[Vh,Kd,Vd], QKV/Z/A/B matrices, depthwise convolution weights, A_log, dt_bias,
normalization weights, the output projection, and a 0/1 mask. Matrices use
checkpoint order [output,input]; C is 2*Kh*Kd + Vh*Vd. It returns [T,H] and
the two next states, without mutating its inputs. Required attributes declare
Kh, Vh, Kd, Vd, K and epsilon; Vh must be a multiple of Kh. Only contiguous f32
tensors and finite values are supported. Invalid geometry, masks, aliases and
unsupported layouts fail before a result is published. Both CPU Providers
advertise this optional operator; an unqualified Provider must decline it.
The WIT graph-builder already transports operator identifiers and typed
attributes, so no WIT version is changed by this addition.

Synthetic fixtures compare the real CPU graph's prefill, chunked continuation,
single-token steps, grouped heads, mask handling, output and both explicit
states with independently executed pinned Transformers functions. The Qwen
Component also builds hybrid attention/recurrent graphs with Runtime-managed
state. An opt-in CPU test compares the merged Kev-0.8B backbone, pointer head
and selected logits with independent PyTorch outputs. This does not qualify
native BF16 arithmetic or a GPU Provider.

### Compatibility and WIT evolution

Preserve existing F32 and single-Device baseline fixtures and public request semantics. Reuse current WIT and Rust contracts first. Before implementation requires new graph operators, state envelopes or result types, record their exact versioned portable schema, negotiation, unsupported behavior and migration tests; native API types are prohibited. A new precision or model profile is opt-in until its validation evidence supports promotion. Vendor implementations stay in their repositories, followed by an explicit parent submodule pin update.

The additive `model-component-graph@1.15.0` configuration describes an
ordered sequence of Mamba-2 mixers, attention, dense MLP and routed MoE
residual blocks. Mixer head/group/state dimensions, convolution width,
normalization, bias flags and inference-time step limits are explicit and
bounded; checkpoint initialization limits are not inference limits. Attention
position handling, activation and expert positions and routing are declared
separately, never inferred from a model name. Existing Components continue to
link, but their graphs are refused if they have not read a declared hybrid
configuration. Each session must keep convolution and SSM state separate from
attention KV history. The mixer and MoE operators have independent CPU/PyTorch
comparisons; this contract alone does not qualify a complete hybrid decoder.
KV capacity estimates for this layout reserve rows only for attention blocks;
Mamba-2, dense MLP and expert blocks do not allocate attention history.

### Authority and isolation

Tools and service/transport adapters use caller-authorized inputs and bounded operations. Inference and Components gain no ambient filesystem, network, process, source-control or secret authority. Local source streaming checks artifact ranges and identity. External campaigns/artifacts are untrusted until verified. Snapshot or distributed data remains tenant-scoped and requires authorization in addition to checksums; signing keys and credentials never enter reports.

### Qualification and rollout

Use the shared real-checkpoint harness and baseline corpus rather than inventing isolated pass criteria. A profile records checkpoint revision/digests, code and Provider SHAs, hardware/driver, flags, dtype, shape, budgets and warmup. Exercise 1/4/16/32 sessions, 512/4096/16384-token contexts and 128-512 generated tokens only inside model/backend/capacity limits; report every skipped combination. Repeat at least five warm runs when affordable and document exceptions, uncertainty and cold costs. Check logits and downstream quality with declared thresholds; token identity alone is insufficient. Measure TTFT, TPOT/ITL p50/p95/p99, aggregate/per-request throughput, launch/phase/copy/I/O costs, RSS/VRAM, fragmentation, useful cache savings and energy only when valid telemetry exists.

Promotion requires appropriate real hardware and a real checkpoint, error/cancellation/memory-pressure/isolation tests, before/after evidence, a disable control and a qualified fallback or structured Unsupported result. Experimental features remain disabled by default until qualified. Run existing Rust, WIT, OpenSpec, license/SBOM/signature and artifact-reproducibility gates for affected code. Compare tail latency and total memory as well as throughput, under the same budget. Publish limitations and do not infer a blanket speedup from heterogeneous runs.

## Risks / Trade-offs

- Do not infer support from a shared family name, shared tensor dimensions or an attention label alone.
- A performance win can be below measurement noise or transfer cost; stop promotion until evidence meets the declared objective and uncertainty threshold.
- Wider hardware coverage is constrained by actual machines; record missing hardware as skipped, not passed, and deliver at most a contract or POC for that target.

## Migration and rollback

Land compatibility scaffolding and baseline tests before optimized execution. Keep new paths selectable and disableable by profile. Drain or fence old generations at safe boundaries; in-flight references remain alive. Roll back only to a still-trusted, resource-compatible qualified generation. If no safe fallback exists, return a structured error. Document any public schema migration before implementation and preserve supported old readers/linkers during the declared compatibility window.

## Open questions

Select exact hardware/checkpoint, numerical and downstream-quality thresholds, resource budgets and SLOs per qualification profile before promoting it. The roadmap's reference CPU/CUDA and Qwen checkpoint are the initial comparison, not evidence that other targets are available. Any Tachyon-facing interface additionally needs its independent repository audit and owner agreement. These decisions gate implementation promotion, not creation of this proposal.
