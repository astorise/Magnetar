# Production Qwen model loading: the minimal embedder integration recipe

This is the minimal recipe an embedder (Tachyon-Mesh or otherwise) uses to
load an authorized Qwen bundle and generate from it, using only Magnetar's
public API surface -- no Safetensors parsing, no Qwen graph construction, no
Tensor Resource identity fabrication, and no `Runtime` execution-engine
wiring of its own. See `implement-production-qwen-model-loading`'s proposal
("Public embedder / Tachyon loading surface") for the requirement this
satisfies, and `integration-tests/production-loading` for the real, tested
code this recipe is drawn from.

## Dependencies

An embedder crate depends on:

- `magnetar-runtime`, with the `wasmtime-component-engine` feature enabled
  (needed to produce graphs through the real compiled Qwen Component)
- A concrete production Model Artifact ingestor. The first one Magnetar
  pins is `loaders/huggingface` (`magnetar-loader-huggingface`,
  `astorise/Magnetar-loader-HuggingFace`), for Hugging Face-style bundles.
  `magnetar-runtime` never depends on this crate itself
  (`production-model-ingestion`'s externalization boundary) -- an embedder
  adds it directly.
- A Provider to execute on: `magnetar-provider-cpu`'s `ReferenceCpuProvider`
  (always available) or `magnetar-provider-cuda`'s `CudaProvider` for real
  GPU hardware.

## The recipe

1. **Authorize a source.** The embedder already knows where the bundle's
   bytes live (a local path it downloaded to, a path Tachyon staged, a
   client-provided directory) and constructs a
   `ProductionModelSource::authorized_local_bundle(kind, root)`, where `root`
   is the one directory the ingestor may read from and `kind` is a
   `ModelArtifactSource` identity (`Tachyon(..)`, `ClientProvided(..)`,
   `LocalPath(..)`, ...) -- provenance metadata only, never trust (Decision 2).

2. **Ingest.** Call the ingestor's `ingest(&source)`
   (`ProductionModelArtifactIngestor::ingest`), e.g.
   `HuggingFaceIngestor::new().ingest(&source)`. This parses real
   `config.json`, `tokenizer.json`/`tokenizer_config.json`, and
   Safetensors bytes and returns a `ProductionIngestionResult { manifest,
   payload_source }` -- a normalized `ModelManifest` plus bounded,
   on-demand tensor payload access. Parsing success grants no trust.

3. **Load a real tokenizer** from the same bundle's `tokenizer.json`
   (`HuggingFaceTokenizer::from_bytes`), wrapped behind the generic
   `Tokenizer` trait.

4. **Declare trust.** Build a `ModelTrustStore` naming the digests (or
   publishers) the embedder actually trusts --
   `ModelTrustStore::default().trust_digest(ingested.manifest.id.digest.value.clone())`
   for "trust exactly this bundle's declared digest", or a real deployment
   policy. An untrusted manifest fails inside `load_model` before any
   materialization; this step is the only place trust is decided, and the
   embedder never mints a trust decision or readiness evidence itself.

5. **Build a production fixture.** `production_qwen_fixture(manifest,
   tokenizer_metadata, tokenizer)` validates the manifest's architecture
   config and tensor inventory against Qwen Component compatibility and
   returns the value the next step needs. This is also where an
   incompatible bundle (wrong tensor shapes, missing required tensors)
   fails, before any Provider resource is touched.

6. **Load and generate.** One call:

   ```rust
   let outcome = magnetar_runtime::run_production_qwen_generation(
       fixture,
       ingested.payload_source.as_ref(),
       trust_store,
       prompt,
   )?;
   // outcome.text: decoded generated text
   // outcome.result: the full GenerationResult
   ```

   This single function builds its own `Runtime` internally (registering a
   `ReferenceCpuProvider`), loads the manifest into a ready `ModelInstance`
   via streaming transactional weight materialization, creates a session,
   tokenizes the prompt through the real tokenizer, produces prefill/decode
   graphs through the real compiled Qwen Component, prepares execution
   plans, and runs the generation loop -- the same primitives every other
   first-native caller uses. `Runtime`'s own execution-engine wiring is
   never exposed to the embedder (a guarded invariant, not an oversight);
   this function is the supported way to reach it.

   To target a Provider other than Reference CPU (e.g. CUDA), use
   `run_production_qwen_generation_for_provider(fixture, payload_source,
   trust_store, prompt, provider)` instead, passing an
   `Arc<dyn magnetar_runtime::Provider>` the embedder constructed itself
   (e.g. `Arc::new(magnetar_provider_cuda::CudaProvider::new())`) --
   `magnetar-runtime` never imports or references a concrete non-Reference-
   CPU Provider crate.

7. **Use the artifact's own real chat template (optional).** When the
   ingested `manifest.chat_template` is `Some` (the bundle's
   `tokenizer_config.json` declared one), build a real formatter and render
   `PromptInput::ChatMessages` through it instead of sending a bare
   plain-text prompt to an Instruct-tuned checkpoint (which is trained to
   expect its own chat markup, e.g. Qwen2.5-Instruct's
   `<|im_start|>`/`<|im_end|>`, and produces incoherent output otherwise):

   ```rust
   let chat_template = /* read the manifest.chat_template part's bytes
                           from the same source the embedder ingested
                           from -- e.g. the raw tokenizer_config.json's
                           chat_template string, already parsed once via
                           magnetar_loader_huggingface::parse_tokenizer_config */;
   let formatter = magnetar_loader_huggingface::HuggingFaceChatTemplateFormatter::new(
       chat_template,
   )?;
   let outcome = magnetar_runtime::run_production_qwen_generation_for_provider_with_prompt(
       fixture,
       ingested.payload_source.as_ref(),
       trust_store,
       magnetar_runtime::PromptInput::ChatMessages(vec![
           magnetar_runtime::ChatMessage::new("user", "What is the capital of France?"),
       ]),
       Some(&formatter),
       provider,
   )?;
   ```

   `HuggingFaceChatTemplateFormatter` renders the real subset of Jinja2
   real Hugging Face chat templates use (variable substitution, `{% for
   %}` over messages, `{% if %}`/`{% elif %}` on loop/message state, `{%
   set %}`, `tojson`, the `defined` test, whitespace-control tags, and both
   `message.role`/`message['role']` attribute-access forms) via `minijinja`
   -- kept entirely in `loaders/huggingface`, never a `magnetar-runtime`
   dependency. A syntactically invalid template is rejected at
   construction (`HuggingFaceChatTemplateFormatter::new` returns `Err`); a
   syntactically valid template referencing an unsupported construct (an
   unknown filter, for instance) fails closed at first render instead,
   with a structured `InferenceApiError::TokenizationFailed`, never a
   silently wrong render. `PromptInput::PlainText` and `None` reproduce
   `run_production_qwen_generation_for_provider`'s exact existing
   behavior unaffected -- this is an additive, opt-in path, not a
   behavior change for a caller not using it.

8. **Supply real generation parameters and stop conditions (optional).**
   Translating an OpenAI-shaped request (`temperature`, `top_p`, `seed`,
   `stop`, ...) requires reaching the real sampling/stop-matching contract,
   not reimplementing it. Use
   `run_production_qwen_generation_for_provider_with_request` instead of any
   of the entry points above:

   ```rust
   let outcome = magnetar_runtime::run_production_qwen_generation_for_provider_with_request(
       fixture,
       ingested.payload_source.as_ref(),
       trust_store,
       magnetar_runtime::ProductionGenerationRequest {
           prompt: magnetar_runtime::PromptInput::PlainText("The capital of France is".into()),
           parameters: magnetar_runtime::GenerationParameters {
               temperature: 0.7,
               top_p: Some(0.9),
               seed: Some(42),
               deterministic: true,
               greedy: false,
               sampling_enabled: true,
               ..Default::default()
           },
           stop_conditions: magnetar_runtime::StopConditions {
               stop_text_sequences: vec!["END".into()],
               ..Default::default()
           },
           max_new_tokens: Some(128),
       },
       None,
       provider,
   )?;
   ```

   `parameters` and `stop_conditions` reach the same `GenerationRequest`/
   sampling contract every other Runtime generation path already uses --
   temperature/top_p/top_k/min_p/typical_p/penalties/seed/banned-and-allowed
   tokens, and both token-id and text stop sequences (`stop_text_sequences`
   is prepared against the real tokenizer internally; the embedder never
   constructs the tokenizer-aware `prepared_stop_sequences` form itself).
   `max_new_tokens: None` preserves the checkpoint manifest's own configured
   token budget; `Some(n)` overrides it. Every entry point above
   (`run_production_qwen_generation(_for_provider)(_with_prompt)`) is a thin
   wrapper over this one, passing `GenerationParameters::greedy()`,
   `StopConditions::default()`, and `max_new_tokens: None` -- their behavior
   is unaffected by this option existing.

9. **Stream generation incrementally instead of waiting for the final
   result (optional).** For `"stream": true`-shaped requests, use
   `run_production_qwen_generation_for_provider_streaming` instead --
   otherwise identical to step 8's `_with_request`, plus a callback invoked
   once per produced token:

   ```rust
   let outcome = magnetar_runtime::run_production_qwen_generation_for_provider_streaming(
       fixture,
       ingested.payload_source.as_ref(),
       trust_store,
       production_generation_request, // same ProductionGenerationRequest shape as step 8
       None,
       provider,
       &mut |event| {
           match event {
               magnetar_runtime::GenerationStreamEvent::Token { token_id, text_delta } => {
                   // forward `text_delta` (an OpenAI chat.completion.chunk's
                   // `delta.content`, e.g.) to the embedder's own transport
                   // (SSE/WebSocket/gRPC) here, as it happens
                   let _ = token_id;
                   let _ = text_delta;
               }
               magnetar_runtime::GenerationStreamEvent::Finished { finish_reason, usage } => {
                   // send the terminal chunk/usage summary here
                   let _ = finish_reason;
                   let _ = usage;
               }
           }
           std::ops::ControlFlow::Continue(()) // or Break(()) to cancel, e.g. on client disconnect
       },
   )?;
   ```

   `Token` events are delivered in production order, one per generated
   token, with an incremental text delta already decoded through the real
   tokenizer -- concatenating every delivered `text_delta` reconstructs
   `outcome.text` exactly. Exactly one `Finished` event follows the last
   `Token`, carrying the same `finish_reason`/`usage` `outcome.result.output`
   itself reports, regardless of whether generation ended by a stop
   condition, the token budget, or the callback requesting cancellation.
   Returning `std::ops::ControlFlow::Break(())` from the callback (e.g. on
   a client disconnect Tachyon observes) stops decode immediately after
   the token just delivered and completes cleanly -- the same
   session-close/instance-unload cleanup any other completion runs, not a
   hard error. `outcome` is still returned afterward with the same shape
   step 8 returns, for a caller that also wants the aggregate result.

   The text delta is produced by re-decoding the accumulated token
   sequence each step and diffing against the previous step's decoded
   text, not `Tokenizer::streaming_decode`/`StreamingDecodeState` --
   `HuggingFaceTokenizer::decode` (the tokenizer backing every real
   production checkpoint) does not honor `streaming_state` at all, so a
   per-token `streaming_decode` call produces wrong text (loses the
   underlying BPE decoder's inter-token joining). See README's "Known gap"
   note under "Qwen production loading".

## What the embedder never does

- Parse Safetensors bytes itself (the ingestor does, via
  `magnetar-format-safetensors`)
- Construct a Qwen `ExecutionGraph` or call the Component's `graph-builder`
  host imports directly (`build_first_native_graphs_from_real_qwen_component`
  does, internally, from the real compiled Component)
- Fabricate a `TensorResourceId` or otherwise reach into Memory Manager /
  Provider resource identity
- Convert F16/BF16 storage bytes to F32 itself (Model Loading does this,
  transactionally, with the source-content digest verified first)
- Configure `Runtime`'s execution engine (`Runtime::builder().
  model_execution_engine(...)` stays `pub(crate)`)

## Current profile and limits

See README's "Qwen production loading" section for exactly what this
profile supports today and what it explicitly does not yet (native CUDA
F16/BF16 compute, GGUF/quantized/LoRA/non-Qwen architectures, remote model
hub download).

Multi-step CUDA decode is real: historical KV concatenation across decode
steps dispatches through a device-resident `concat` Kernel and a
`copy_tensor_admitted` Provider-side copy, never downloading KV history to
host. `run_production_qwen_generation_for_provider(_with_prompt)` needs no
different call shape for CUDA than for Reference CPU to reach this --
`PromptInput`/`max_tokens` are unchanged either way.

A Provider whose KV history is genuinely not host-readable in some other
way still has a real, generic escape hatch: `supports_multi_step_decode()`
(a `Provider` trait method, `true` by default) is checked once, before any
real execution work, when the request needs more than one decode step; a
Provider declaring `false` fails the request immediately with a structured
`InferenceApiError::Unsupported { reason }` naming the real constraint,
instead of the request failing deep inside decode after a real prefill
already ran. CUDA no longer declares `false` (it now supports this shape),
but the mechanism itself remains real for a future Provider that needs it.
An embedder calling `run_production_qwen_generation_for_provider
(_with_prompt)` with a Provider it does not control the implementation of
should match on this variant to distinguish "this Provider cannot do this"
from a transient fault (`ProviderUnavailable`) or a policy decision
(`PolicyDenied`).

`GenerationResult.output.usage.tokens_per_second` (and
`prefill_duration_millis`/`decode_duration_millis`) reflect real measured
wall-clock time for every Provider once at least one token was generated,
not an estimate or a fixed assumed rate -- `None` only when generation
produced no token at all (e.g. cancelled before the first step ran).
