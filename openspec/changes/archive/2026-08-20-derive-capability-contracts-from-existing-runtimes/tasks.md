# Tasks

## 1. Evidence Baseline

- [x] 1.1 Record the pinned Candle and Crane revisions, review method, and source-link convention in the architecture document

## 2. Candle Interface Review

- [x] 2.1 Document `Device` and `DeviceLocation` responsibilities and their Magnetar role
- [x] 2.2 Document `BackendDevice` allocation, transfer, random, and synchronization responsibilities
- [x] 2.3 Document `BackendStorage` ownership and kernel-dispatch responsibilities
- [x] 2.4 Group the `Tensor` surface into contract-relevant operation families
- [x] 2.5 Document the `Module` and `ModuleT` execution boundary

## 3. Crane Interface Review

- [x] 3.1 Document the `ModelForCausalLM` model and generation-session boundary
- [x] 3.2 Split `GenerationConfig` fields by portable policy and implementation concern
- [x] 3.3 Map `TokenStreamer` callbacks and channels to portable stream semantics
- [x] 3.4 Document tokenizer loading, encoding, decoding, and prompt-formatting boundaries
- [x] 3.5 Document implemented and placeholder high-level AI abilities without overstating source maturity

## 4. Capability Taxonomy

- [x] 4.1 Define low-level responsibility families and identify valid Capability candidates
- [x] 4.2 Define model-level responsibility families and identify valid Capability candidates
- [x] 4.3 Define application-level responsibility families and identify valid Capability candidates
- [x] 4.4 Document the dependency graph across all three layers

## 5. Contract Preparation

- [x] 5.1 Map Component-suitable and hybrid Capability candidates to provisional WIT packages
- [x] 5.2 Map native-only responsibilities to Provider, Device, or runtime services rather than Capabilities
- [x] 5.3 Identify coarse WIT interfaces that can cross a WASM Component boundary
- [x] 5.4 Classify fallback as transparent, restartable, or Provider-pinned for each Capability candidate

## 6. Publication and Validation

- [x] 6.1 Publish the complete taxonomy at `docs/architecture/capability-taxonomy.md`
- [x] 6.2 Link the taxonomy from the repository README
- [x] 6.3 Run strict OpenSpec validation and the repository test suite
