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
profile supports today and what it explicitly does not yet (multi-step CUDA
decode, native CUDA F16/BF16 compute, GGUF/quantized/LoRA/non-Qwen
architectures, remote model hub download).
