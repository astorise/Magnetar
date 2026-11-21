//! The first-native model execution engine: Component graph cache, decode step replay and the `RuntimeModelExecutionEngine` implementation.

use super::*;

#[derive(Clone)]
pub(super) struct E2eRuntimeModelExecutionEngine {
    pub(super) fixture: E2eFixture,
    pub(super) kv_states: Arc<Mutex<BTreeMap<String, FirstNativeExecutionKvState>>>,
    pub(super) pending_kv_states: Arc<Mutex<BTreeMap<String, FirstNativeExecutionKvState>>>,
    /// See [`ProductionLoadedModel::component_digest`]'s doc comment --
    /// must stay in lockstep with the same instance's `prepare_generation`
    /// digest so a published plan's fingerprint matches what dispatch here
    /// actually builds. `None` (every construction site but
    /// `ProductionLoadedModel::load`) preserves the pre-existing
    /// hardcoded-singleton behavior exactly.
    pub(super) component_digest: Option<ComponentDigest>,
    /// Test-only deterministic-token override, read by
    /// `execute_generation_step`'s forced-shortcut branch. Always `None` at
    /// every production construction site in this file; only
    /// `first_native_runtime::tests` ever constructs an instance with
    /// `Some(..)` (`build_runtime_with_model_execution_engine_and_forced_token`).
    /// Deliberately not `#[cfg(test)]`-gated: a conditionally-compiled field
    /// can't be relocated into a sibling `tests.rs` (Rust has no partial /
    /// extension structs), so the only extraction available for a
    /// struct-embedded test-only field is dropping the `cfg` and keeping the
    /// field, and its value, identical in every build.
    pub(super) forced_token: Option<TokenId>,
    /// Graphs already built for this model, shared with the
    /// `ProductionLoadedModel` that prepared their plans. `None` builds the
    /// graphs again for every step.
    pub(super) graph_cache: Option<ComponentGraphCache>,
    /// Keep KV history with the copy-based append even where the Provider
    /// supports in-place history (`append-kv-history-in-place`): for tests
    /// of the copy-based path.
    pub(super) copy_based_kv: bool,
    /// Decode-step capture and replay (`capture-and-replay-cuda-graphs`).
    pub(super) step_replay: StepReplay,
    /// Pooled in-place KV stores of bucketed requests
    /// (`replay-bucketed-prefill`).
    pub(super) kv_pool: KvStorePool,
}

/// How many prompt lengths' graphs one loaded model keeps.
pub(crate) const COMPONENT_GRAPH_CACHE_CAPACITY: usize = 8;

/// The execution graphs one loaded model's Component built, by prompt
/// length. A loaded model's Component, architecture and identity never
/// change, so the prompt length is the only input left that the graphs
/// depend on. Plans are prepared against these exact graphs and every step
/// dispatches them, so a step can never execute a graph other than the one
/// its plan was fingerprinted against. Bounded, least recently used first
/// out.
#[derive(Clone, Default)]
pub(crate) struct ComponentGraphCache {
    pub(super) entries: Arc<Mutex<VecDeque<CachedComponentGraphs>>>,
    /// The Provider graphs are fused for (`graph_fusion`), or `None` to
    /// keep every graph exactly as its Component built it.
    pub(super) fusion: Arc<Mutex<Option<ProviderBinding>>>,
    /// Whether the Provider splits the model by tensor parallelism, so
    /// row-parallel matmuls must keep their addends unfused.
    pub(super) tensor_parallel: Arc<std::sync::atomic::AtomicBool>,
    /// The prompt buckets of a bucketed prefill (`replay-bucketed-prefill`),
    /// ascending; empty when prefills run at the prompt's exact length.
    pub(super) prefill_buckets: Arc<Mutex<Vec<u64>>>,
    /// The chunk length of a chunked prefill (`chunk-long-prefills`): a
    /// longer prompt runs in passes of this many rows. `0` runs every
    /// prompt in one pass.
    pub(super) prefill_chunk: Arc<std::sync::atomic::AtomicU64>,
}

/// One graph length's graphs.
pub(super) type CachedComponentGraphs = (u64, Arc<CachedGraphs>);

/// Integer attributes that advance by one from one decode step to the next
/// (`capture-and-replay-cuda-graphs`): the RoPE position, the attention key
/// count and the in-place KV write offset.
pub(super) const STEP_ADVANCING_ATTRIBUTES: [&str; 3] =
    ["position_offset", "key_count", "row_offset"];

/// The decode step at which a request's step is captured: the first decode
/// step allocates every decode-shaped output, so the second one allocates
/// nothing and is the same as every later one.
pub(super) const STEP_CAPTURE_DECODE_INDEX: usize = 2;

/// Decode-step capture and replay (`capture-and-replay-cuda-graphs`), shared
/// by an engine and the `ProductionLoadedModel` that owns it.
#[derive(Clone, Default)]
pub(crate) struct StepReplay(pub(super) Arc<StepReplayState>);

pub(super) struct StepReplayState {
    pub(super) enabled: std::sync::atomic::AtomicBool,
    /// Each request's captured decode step, by KV state key.
    pub(super) recorded: Mutex<BTreeMap<String, RecordedDecodeStep>>,
    /// Requests whose capture failed: they run every step per node.
    pub(super) declined: Mutex<BTreeSet<String>>,
    /// Bucketed prefill capture and replay (`replay-bucketed-prefill`).
    pub(super) prefill_enabled: std::sync::atomic::AtomicBool,
    pub(super) prefills: Mutex<PrefillRecordings>,
    /// The bytes every bucket namespace's intermediates may hold together.
    pub(super) prefill_intermediate_bytes: std::sync::atomic::AtomicU64,
}

/// A prefill recording's key: the prefill graph, its bucket length and the
/// pooled store set its KV stores belong to.
pub(super) type PrefillRecordingKey = (ExecutionGraphSemanticFingerprint, u64, u64);

/// Recorded bucketed prefills, and where each key is in its
/// per-node / capture / replay sequence.
#[derive(Default)]
pub(super) struct PrefillRecordings {
    pub(super) recorded: BTreeMap<PrefillRecordingKey, RecordedDecodeStep>,
    /// Keys that ran per node, ready to capture.
    pub(super) warmed: BTreeSet<PrefillRecordingKey>,
    /// Keys whose capture failed: they run per node.
    pub(super) declined: BTreeSet<PrefillRecordingKey>,
    /// Each bucket namespace holding intermediates, with when it was last
    /// used.
    pub(super) namespaces: BTreeMap<u64, u64>,
    pub(super) tick: u64,
}

/// Where a bucketed prefill is in its key's sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum PrefillRecordingStage {
    /// Run per node, keeping its intermediates for a later capture.
    Warm,
    Capture,
    Replay(Box<RecordedDecodeStepHandle>),
    /// Run per node, as an unrecorded prefill.
    Declined,
}

/// A recorded prefill, compared by its capture.
#[derive(Clone)]
pub(super) struct RecordedDecodeStepHandle(pub(super) RecordedDecodeStep);

impl PartialEq for RecordedDecodeStepHandle {
    fn eq(&self, other: &Self) -> bool {
        self.0.capture == other.0.capture
    }
}

impl Eq for RecordedDecodeStepHandle {}

impl std::fmt::Debug for RecordedDecodeStepHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "RecordedDecodeStep({:?})", self.0.capture)
    }
}

impl Default for StepReplayState {
    fn default() -> Self {
        Self {
            enabled: std::sync::atomic::AtomicBool::new(true),
            recorded: Mutex::default(),
            declined: Mutex::default(),
            prefill_enabled: std::sync::atomic::AtomicBool::new(true),
            prefills: Mutex::default(),
            prefill_intermediate_bytes: std::sync::atomic::AtomicU64::new(
                std::env::var("MAGNETAR_PREFILL_REPLAY_BYTES")
                    .ok()
                    .and_then(|bytes| bytes.trim().parse().ok())
                    .unwrap_or(super::prefill_buckets::DEFAULT_PREFILL_INTERMEDIATE_BYTES),
            ),
        }
    }
}

/// Releases every capture still held, however the requests that made them
/// ended: a capture holds Provider resources (a CUDA graph executable).
impl Drop for StepReplayState {
    fn drop(&mut self) {
        if let Ok(recorded) = self.recorded.get_mut() {
            for step in recorded.values() {
                step.executor.release_step_capture(&step.capture);
            }
        }
        if let Ok(prefills) = self.prefills.get_mut() {
            for step in prefills.recorded.values() {
                step.executor.release_step_capture(&step.capture);
            }
        }
    }
}

impl StepReplay {
    pub(crate) fn set_enabled(&self, enabled: bool) {
        self.0.enabled.store(enabled, Ordering::Relaxed);
    }

    pub(super) fn is_enabled(&self) -> bool {
        self.0.enabled.load(Ordering::Relaxed)
    }

    pub(super) fn recorded(&self, key: &str) -> Option<RecordedDecodeStep> {
        self.0.recorded.lock().ok()?.get(key).cloned()
    }

    pub(super) fn record(&self, key: String, step: RecordedDecodeStep) {
        if let Ok(mut recorded) = self.0.recorded.lock()
            && let Some(replaced) = recorded.insert(key, step)
        {
            replaced.executor.release_step_capture(&replaced.capture);
        }
    }

    pub(super) fn decline(&self, key: String) {
        if let Ok(mut declined) = self.0.declined.lock() {
            declined.insert(key);
        }
    }

    pub(super) fn is_declined(&self, key: &str) -> bool {
        self.0
            .declined
            .lock()
            .map(|declined| declined.contains(key))
            .unwrap_or(true)
    }

    pub(crate) fn set_prefill_enabled(&self, enabled: bool) {
        self.0.prefill_enabled.store(enabled, Ordering::Relaxed);
    }

    pub(super) fn prefill_enabled(&self) -> bool {
        self.0.prefill_enabled.load(Ordering::Relaxed)
    }

    /// Where a bucketed prefill of `key` is in its sequence.
    pub(super) fn prefill_stage(&self, key: &PrefillRecordingKey) -> PrefillRecordingStage {
        let Ok(prefills) = self.0.prefills.lock() else {
            return PrefillRecordingStage::Declined;
        };
        if let Some(recorded) = prefills.recorded.get(key) {
            PrefillRecordingStage::Replay(Box::new(RecordedDecodeStepHandle(recorded.clone())))
        } else if prefills.declined.contains(key) {
            PrefillRecordingStage::Declined
        } else if prefills.warmed.contains(key) {
            PrefillRecordingStage::Capture
        } else {
            PrefillRecordingStage::Warm
        }
    }

    pub(super) fn prefill_warmed(&self, key: PrefillRecordingKey) {
        if let Ok(mut prefills) = self.0.prefills.lock() {
            prefills.warmed.insert(key);
        }
    }

    pub(super) fn record_prefill(&self, key: PrefillRecordingKey, step: RecordedDecodeStep) {
        if let Ok(mut prefills) = self.0.prefills.lock()
            && let Some(replaced) = prefills.recorded.insert(key, step)
        {
            replaced.executor.release_step_capture(&replaced.capture);
        }
    }

    pub(super) fn decline_prefill(&self, key: PrefillRecordingKey) {
        if let Ok(mut prefills) = self.0.prefills.lock() {
            prefills.warmed.remove(&key);
            prefills.declined.insert(key);
        }
    }

    /// Drops a recording that no longer replays: the key captures again
    /// after one per-node run.
    pub(super) fn invalidate_prefill(&self, key: &PrefillRecordingKey) {
        let step = self.0.prefills.lock().ok().and_then(|mut prefills| {
            prefills.warmed.remove(key);
            prefills.recorded.remove(key)
        });
        if let Some(step) = step {
            step.executor.release_step_capture(&step.capture);
        }
    }

    /// Forgets every prefill of a pooled store set, releasing its
    /// recordings: the set's stores are about to be released.
    pub(super) fn release_prefill_set(&self, set: u64) {
        let released: Vec<RecordedDecodeStep> = match self.0.prefills.lock() {
            Ok(mut prefills) => {
                prefills.warmed.retain(|key| key.2 != set);
                prefills.declined.retain(|key| key.2 != set);
                let keys: Vec<_> = prefills
                    .recorded
                    .keys()
                    .filter(|key| key.2 == set)
                    .cloned()
                    .collect();
                keys.iter()
                    .filter_map(|key| prefills.recorded.remove(key))
                    .collect()
            }
            Err(_) => Vec::new(),
        };
        for step in released {
            step.executor.release_step_capture(&step.capture);
        }
    }

    /// Bounds what every bucket namespace's intermediates hold together.
    #[cfg(test)]
    pub(super) fn set_prefill_intermediate_bytes(&self, bytes: u64) {
        self.0
            .prefill_intermediate_bytes
            .store(bytes, Ordering::Relaxed);
    }

    /// Marks `bucket`'s namespace as just used.
    pub(super) fn touch_prefill_namespace(&self, bucket: u64) {
        if let Ok(mut prefills) = self.0.prefills.lock() {
            prefills.tick += 1;
            let tick = prefills.tick;
            prefills.namespaces.insert(bucket, tick);
        }
    }

    /// The buckets whose namespaces hold intermediates, least recently
    /// used first.
    pub(super) fn prefill_namespaces(&self) -> Vec<u64> {
        let Ok(prefills) = self.0.prefills.lock() else {
            return Vec::new();
        };
        let mut buckets: Vec<(u64, u64)> = prefills
            .namespaces
            .iter()
            .map(|(bucket, tick)| (*tick, *bucket))
            .collect();
        buckets.sort_unstable();
        buckets.into_iter().map(|(_, bucket)| bucket).collect()
    }

    /// Forgets every prefill of `bucket`, releasing its recordings: its
    /// namespace's intermediates are about to be released.
    pub(super) fn release_prefill_bucket(&self, bucket: u64) {
        let released: Vec<RecordedDecodeStep> = match self.0.prefills.lock() {
            Ok(mut prefills) => {
                prefills.namespaces.remove(&bucket);
                prefills.warmed.retain(|key| key.1 != bucket);
                prefills.declined.retain(|key| key.1 != bucket);
                let keys: Vec<_> = prefills
                    .recorded
                    .keys()
                    .filter(|key| key.1 == bucket)
                    .cloned()
                    .collect();
                keys.iter()
                    .filter_map(|key| prefills.recorded.remove(key))
                    .collect()
            }
            Err(_) => Vec::new(),
        };
        for step in released {
            step.executor.release_step_capture(&step.capture);
        }
    }

    /// How many prefills are recorded (tests).
    #[cfg(test)]
    pub(super) fn prefill_recordings(&self) -> usize {
        self.0
            .prefills
            .lock()
            .map_or(0, |prefills| prefills.recorded.len())
    }

    /// Forgets a request, releasing its capture.
    pub(super) fn release(&self, key: &str) {
        if let Ok(mut declined) = self.0.declined.lock() {
            declined.remove(key);
        }
        let step = self
            .0
            .recorded
            .lock()
            .ok()
            .and_then(|mut recorded| recorded.remove(key));
        if let Some(step) = step {
            step.executor.release_step_capture(&step.capture);
        }
    }
}

/// What replaying a request's captured decode step needs.
#[derive(Clone)]
pub(super) struct RecordedDecodeStep {
    pub(super) capture: StepCaptureId,
    /// The Provider holding the capture.
    pub(super) executor: Arc<dyn ProviderExecutionApi>,
    pub(super) provider: ProviderBinding,
    /// The absolute position of the captured step.
    pub(super) position: u64,
    /// The inputs the captured step read, in the order a replay writes
    /// them: its token ids, then (a prefill) its logits row.
    pub(super) inputs: Vec<TensorResourceId>,
    /// The logits the captured step wrote, and their shape.
    pub(super) logits: (TensorResourceId, Vec<u64>),
    /// The KV state the captured step left, its row counts at `position`.
    pub(super) layer_kv: LayerKvMap,
    pub(super) dispatch: KernelDispatchResult,
}

/// Whether every node reading a KV-cache edge is an `attention` node: an
/// in-place KV store holds more rows than the history, and only attention
/// (through `key_count`) knows to read just the history
/// (`append-kv-history-in-place`).
pub(super) fn kv_edges_feed_only_attention(graph: &ExecutionGraph) -> bool {
    let kv_edges: BTreeSet<&TensorEdgeId> = graph
        .edges
        .iter()
        .filter(|(_, edge)| edge.kv_cache.is_some())
        .map(|(id, _)| id)
        .collect();
    graph.nodes.values().all(|node| {
        node.operator.name() == "attention"
            || !node.inputs.iter().any(|input| kv_edges.contains(input))
    })
}

/// Whether `provider` can keep KV history in place: an active prepared
/// `kv-write` Kernel, and every active `attention` Kernel of it honouring
/// `key_count` (`append-kv-history-in-place`).
pub(super) fn in_place_kv_supported(runtime: &Runtime, provider: &ProviderBinding) -> bool {
    let mut attention = false;
    for entry in runtime.kernel_registry().entries() {
        if !entry.active
            || &entry.advertisement.id.provider != provider
            || entry.advertisement.implemented_operator.name() != "attention"
        {
            continue;
        }
        if !entry
            .advertisement
            .id
            .features
            .contains(crate::reference_cpu::KEY_COUNT_FEATURE)
        {
            return false;
        }
        attention = true;
    }
    attention && provider_advertises_operator(runtime, provider, "kv-write")
}

/// Every `rope` node of a Component-built graph must declare exactly the
/// per-frequency scaling the bound configuration carries. A Component that
/// predates `model-config` 1.4.0, or does not understand the scheme, would
/// otherwise build an unscaled graph for a scaled checkpoint and degrade
/// every token silently; one that declares a scaling the checkpoint does
/// not have is just as wrong.
pub(super) fn validate_rope_frequency_scaling(
    graph: &ExecutionGraph,
    config: &ModelArchitectureConfig,
) -> Result<(), E2eConformanceError> {
    // `express-latent-attention-forms`: YaRN is checked the same way.
    #[derive(Debug, PartialEq)]
    enum Declared {
        Banded(crate::operator::RopeBandedFrequencyScaling),
        Yarn(crate::operator::RopeYarnFrequencyScaling),
    }
    let expected = config.rope_frequency_scaling.map(|scaling| match scaling {
        crate::RopeFrequencyScaling::Banded(banded) => Declared::Banded(banded),
        crate::RopeFrequencyScaling::Yarn { scaling, .. } => Declared::Yarn(scaling),
    });
    for (node_id, node) in &graph.nodes {
        if node.operator.name() != "rope" {
            continue;
        }
        let failed = |reason: String| E2eConformanceError::ModelComponentFailed {
            reason: format!("rope node '{node_id:?}': {reason}"),
        };
        let declared =
            match crate::operator::RopeYarnFrequencyScaling::from_attributes(&node.attributes)
                .map_err(failed)?
            {
                Some(yarn) => Some(Declared::Yarn(yarn)),
                None => {
                    crate::operator::RopeBandedFrequencyScaling::from_attributes(&node.attributes)
                        .map_err(failed)?
                        .map(Declared::Banded)
                }
            };
        if declared != expected {
            return Err(E2eConformanceError::ModelComponentFailed {
                reason: format!(
                    "rope node '{node_id:?}' declares frequency scaling {declared:?}, but the \
                     checkpoint's configuration requires {expected:?}; the Model Component \
                     does not implement this model's RoPE"
                ),
            });
        }
    }
    Ok(())
}

/// Whether every active `rope` Kernel of `provider` honours the banded
/// frequency-scaling attributes (and there is at least one).
pub(super) fn rope_banded_scaling_supported(runtime: &Runtime, provider: &ProviderBinding) -> bool {
    provider_kernels_declare(
        runtime,
        provider,
        "rope",
        crate::operator::ROPE_BANDED_FREQUENCY_SCALING_FEATURE,
    )
}

/// Whether `provider` has an active `operator` Kernel and every one of them
/// advertises `feature`: an attribute only some Kernels honour must not
/// reach the others.
pub(super) fn provider_kernels_declare(
    runtime: &Runtime,
    provider: &ProviderBinding,
    operator: &str,
    feature: &str,
) -> bool {
    let mut found = false;
    for entry in runtime.kernel_registry().entries() {
        if !entry.active
            || &entry.advertisement.id.provider != provider
            || entry.advertisement.implemented_operator.name() != operator
        {
            continue;
        }
        if !entry.advertisement.id.features.contains(feature) {
            return false;
        }
        found = true;
    }
    found
}

/// What every step derives from a graph alone: its plan fingerprint and
/// its execution order. A cached graph never changes, so each is computed
/// at most once instead of on every step.
#[derive(Default)]
pub(crate) struct GraphExecutionMemo {
    pub(super) fingerprint: std::sync::OnceLock<ExecutionGraphSemanticFingerprint>,
    pub(super) order: std::sync::OnceLock<Vec<ExecutionNodeId>>,
    pub(super) kv_edges_feed_only_attention: std::sync::OnceLock<bool>,
}

impl GraphExecutionMemo {
    pub(super) fn fingerprint(&self, graph: &ExecutionGraph) -> &ExecutionGraphSemanticFingerprint {
        self.fingerprint
            .get_or_init(|| semantic_graph_fingerprint(graph))
    }

    pub(super) fn kv_edges_feed_only_attention(&self, graph: &ExecutionGraph) -> bool {
        *self
            .kv_edges_feed_only_attention
            .get_or_init(|| kv_edges_feed_only_attention(graph))
    }

    pub(super) fn order(
        &self,
        graph: &ExecutionGraph,
    ) -> Result<&[ExecutionNodeId], InferenceApiError> {
        if let Some(order) = self.order.get() {
            return Ok(order);
        }
        let order = first_native_graph_execution_order(graph)?;
        Ok(self.order.get_or_init(|| order))
    }
}

/// A loaded model's graphs for one prompt length, with each graph's
/// [`GraphExecutionMemo`]. Only ever shared behind an `Arc`, so the graphs
/// cannot change after a memo was filled.
pub(crate) struct CachedGraphs {
    pub(super) graphs: FirstNativeComponentGraphs,
    /// Whether the prefill graph takes `input.logits_row` and produces the
    /// logits of that one row (`prefill_buckets::insert_logits_row_gather`).
    pub(crate) logits_row: bool,
    /// Whether the prefill graph is the chunk's (`chunk-long-prefills`): its
    /// plan serves every longer prompt, which runs in passes of it.
    pub(crate) serves_longer_prompts: bool,
    pub(crate) prefill_memo: GraphExecutionMemo,
    pub(crate) decode_memo: GraphExecutionMemo,
    /// The Prepared Execution Plans last prepared for these graphs, with
    /// what they were prepared against. A request whose key matches starts
    /// from a copy instead of selecting a Kernel for every node again.
    pub(super) plans: std::sync::Mutex<Option<(FirstNativePlanCacheKey, FirstNativePreparedPlans)>>,
    /// How many times `prepared_plans` had to prepare (tests).
    pub(super) preparations: std::sync::atomic::AtomicU64,
}

/// Everything a first-native plan preparation depends on besides its
/// graphs: the Model Instance and its revision, the Provider and the
/// Kernel Registry's state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FirstNativePlanCacheKey {
    pub(super) instance: ModelInstanceId,
    pub(super) mutation_version: u64,
    pub(super) provider: ProviderBinding,
    pub(super) registry_revision: u64,
}

impl CachedGraphs {
    pub(crate) fn new(graphs: FirstNativeComponentGraphs) -> Self {
        Self {
            graphs,
            logits_row: false,
            serves_longer_prompts: false,
            prefill_memo: GraphExecutionMemo::default(),
            decode_memo: GraphExecutionMemo::default(),
            plans: std::sync::Mutex::new(None),
            preparations: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Prepared plans for these graphs: a copy of the cached ones when
    /// they were prepared against exactly `runtime`'s current instance
    /// revision, `provider` and registry revision, freshly prepared (and
    /// cached) otherwise. A plan is never shared: every request mutates
    /// its own copy.
    pub(crate) fn prepared_plans(
        &self,
        runtime: &Runtime,
        instance: &ModelInstanceId,
        prompt_token_count: u64,
        provider: &ProviderBinding,
    ) -> Result<FirstNativePreparedPlans, E2eConformanceError> {
        let status = require_ready_first_native_instance(runtime, instance)?;
        let key = FirstNativePlanCacheKey {
            instance: instance.clone(),
            mutation_version: status.status().mutation_version,
            provider: provider.clone(),
            registry_revision: runtime.kernel_registry().revision(),
        };
        if let Some((cached_key, plans)) = self.plans.lock().unwrap().as_ref()
            && *cached_key == key
        {
            return Ok(plans.clone());
        }
        let plans = prepare_first_native_execution_plans_serving(
            runtime,
            instance,
            FirstNativeComponentGraphs::clone(&self.graphs),
            prompt_token_count,
            if self.serves_longer_prompts {
                u64::MAX
            } else {
                prompt_token_count
            },
            provider,
        )?;
        self.preparations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        *self.plans.lock().unwrap() = Some((key, plans.clone()));
        Ok(plans)
    }

    #[cfg(test)]
    pub(crate) fn preparations(&self) -> u64 {
        self.preparations.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl std::ops::Deref for CachedGraphs {
    type Target = FirstNativeComponentGraphs;

    fn deref(&self) -> &FirstNativeComponentGraphs {
        &self.graphs
    }
}

/// Whether `provider` has an active, prepared Kernel for `operator`.
pub(super) fn provider_advertises_operator(
    runtime: &Runtime,
    provider: &ProviderBinding,
    operator: &str,
) -> bool {
    let registry = runtime.kernel_registry();
    registry.entries().any(|entry| {
        entry.active
            && &entry.advertisement.id.provider == provider
            && entry.advertisement.implemented_operator.name() == operator
            && registry
                .active_prepared_kernel(&entry.advertisement.id)
                .is_some()
    })
}

impl ComponentGraphCache {
    pub(crate) fn get(&self, prompt_token_count: u64) -> Option<Arc<CachedGraphs>> {
        let mut entries = self.entries.lock().unwrap();
        let index = entries
            .iter()
            .position(|(count, _)| *count == prompt_token_count)?;
        let entry = entries.remove(index)?;
        let graphs = Arc::clone(&entry.1);
        entries.push_back(entry);
        Some(graphs)
    }

    pub(crate) fn insert(&self, prompt_token_count: u64, graphs: Arc<CachedGraphs>) {
        let mut entries = self.entries.lock().unwrap();
        entries.retain(|(count, _)| *count != prompt_token_count);
        if entries.len() == COMPONENT_GRAPH_CACHE_CAPACITY {
            entries.pop_front();
        }
        entries.push_back((prompt_token_count, graphs));
    }

    /// Fuses graphs for `provider` from now on, or stops fusing with
    /// `None`. Forgets every cached graph so plans and steps never mix
    /// fused and unfused graphs.
    pub(crate) fn set_fusion_provider(&self, provider: Option<ProviderBinding>) {
        *self.fusion.lock().unwrap() = provider;
        self.entries.lock().unwrap().clear();
    }

    pub(crate) fn tensor_parallel(&self) -> bool {
        self.tensor_parallel
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Records whether the Provider splits the model by tensor parallelism.
    /// Forgets every cached graph, like [`Self::set_fusion_provider`].
    pub(crate) fn set_tensor_parallel(&self, enabled: bool) {
        self.tensor_parallel
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
        self.entries.lock().unwrap().clear();
    }

    /// The prefill graph length for a prompt of `prompt_tokens` tokens:
    /// its bucket when bucketing applies, the prompt's own length
    /// otherwise. A bucket length is never the exact length of an
    /// unbucketed prompt (those are longer than every bucket), so graphs
    /// stay keyed by this length alone.
    pub(crate) fn prefill_graph_length(&self, prompt_tokens: u64) -> u64 {
        if self.tensor_parallel() {
            return prompt_tokens;
        }
        // A prompt longer than the chunk runs in passes of the chunk's
        // graph (`chunk-long-prefills`).
        let chunk = self.prefill_chunk();
        if chunk > 0 && prompt_tokens > chunk {
            return chunk;
        }
        crate::first_native_runtime::prefill_buckets::prefill_bucket(
            &self.prefill_buckets.lock().unwrap(),
            prompt_tokens,
        )
        .unwrap_or(prompt_tokens)
    }

    /// Whether a prefill graph of `graph_length` rows is a bucket's or the
    /// chunk's: one whose last rows may be padding.
    pub(crate) fn is_prefill_bucket(&self, graph_length: u64) -> bool {
        !self.tensor_parallel()
            && (self.prefill_buckets.lock().unwrap().contains(&graph_length)
                || self.prefill_chunk() == graph_length)
    }

    /// Whether `graph_length` is one of the prompt buckets
    /// (`replay-bucketed-prefill`), the chunk's length aside: only a
    /// bucket's prefills are pooled and recorded.
    pub(crate) fn is_bucket_length(&self, graph_length: u64) -> bool {
        !self.tensor_parallel() && self.prefill_buckets.lock().unwrap().contains(&graph_length)
    }

    /// The chunk length of a chunked prefill (`chunk-long-prefills`), `0`
    /// when every prompt runs in one pass.
    pub(crate) fn prefill_chunk(&self) -> u64 {
        self.prefill_chunk
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Sets the chunk length of later prefills; `0` turns chunking off.
    /// Forgets every cached graph, like [`Self::set_prefill_buckets`].
    pub(crate) fn set_prefill_chunk(&self, chunk: u64) {
        self.prefill_chunk
            .store(chunk, std::sync::atomic::Ordering::Relaxed);
        self.entries.lock().unwrap().clear();
    }

    /// Sets the prompt buckets; empty turns bucketing off. Forgets every
    /// cached graph, like [`Self::set_fusion_provider`].
    pub(crate) fn set_prefill_buckets(&self, buckets: Vec<u64>) {
        *self.prefill_buckets.lock().unwrap() = buckets;
        self.entries.lock().unwrap().clear();
    }

    /// Freshly built graphs of `graph_length` prefill rows, ready to cache:
    /// fused ([`Self::optimize`]) and with the logits row gather, outside
    /// tensor parallelism. Every prefill samples one row (a bucket's, a
    /// chunked prefill's passes, a prompt run in one pass at its own
    /// length), so the LM head runs on that row only: a long prompt's
    /// prefill never holds every row's logits. A graph whose LM head cannot
    /// take the gather keeps every row's logits; the engine then reads row
    /// `n - 1`.
    pub(crate) fn prepare(
        &self,
        runtime: &Runtime,
        graphs: FirstNativeComponentGraphs,
        graph_length: u64,
    ) -> CachedGraphs {
        let mut graphs = self.optimize(runtime, graphs);
        let gathered = (!self.tensor_parallel())
            .then(|| {
                crate::first_native_runtime::prefill_buckets::insert_logits_row_gather(
                    &graphs.prefill,
                )
            })
            .flatten();
        let logits_row = gathered.is_some();
        if let Some(prefill) = gathered {
            graphs.prefill = prefill;
            graphs.prefill_node_count += 1;
        }
        let mut cached = CachedGraphs::new(graphs);
        cached.logits_row = logits_row;
        cached.serves_longer_prompts =
            !self.tensor_parallel() && self.prefill_chunk() == graph_length;
        cached
    }

    /// Applies every fusion the fusion Provider currently advertises to
    /// freshly built graphs; graphs are returned unchanged otherwise.
    pub(crate) fn optimize(
        &self,
        runtime: &Runtime,
        mut graphs: FirstNativeComponentGraphs,
    ) -> FirstNativeComponentGraphs {
        let Some(provider) = self.fusion.lock().unwrap().clone() else {
            return graphs;
        };
        let is_weight = |edge: &TensorEdgeId| weight_tensor_name_from_edge(edge.as_str()).is_some();
        let tensor_parallel = self
            .tensor_parallel
            .load(std::sync::atomic::Ordering::Relaxed);
        type Pass<'a> = Box<dyn Fn(&ExecutionGraph) -> (ExecutionGraph, usize) + 'a>;
        let passes: [(&str, Pass<'_>); 4] = [
            (
                crate::graph_fusion::MATMUL_BIAS_OPERATOR,
                Box::new(move |graph| {
                    if tensor_parallel {
                        crate::graph_fusion::fuse_matmul_bias_tensor_parallel(graph, is_weight)
                    } else {
                        crate::graph_fusion::fuse_matmul_bias(graph, is_weight)
                    }
                }),
            ),
            (
                crate::graph_fusion::MATMUL_RESIDUAL_OPERATOR,
                Box::new(move |graph| {
                    if tensor_parallel {
                        crate::graph_fusion::fuse_matmul_residual_tensor_parallel(graph)
                    } else {
                        crate::graph_fusion::fuse_matmul_residual(graph)
                    }
                }),
            ),
            (
                crate::graph_fusion::SILU_MUL_OPERATOR,
                Box::new(crate::graph_fusion::fuse_silu_mul),
            ),
            // After `silu-mul`, whose nodes it folds with both matmuls.
            (
                crate::graph_fusion::GATED_MATMUL_OPERATOR,
                Box::new(crate::graph_fusion::fuse_gated_matmul),
            ),
        ];
        for (operator, pass) in passes {
            if !provider_advertises_operator(runtime, &provider, operator) {
                continue;
            }
            let (prefill, prefill_fused) = pass(&graphs.prefill);
            let (decode, decode_fused) = pass(&graphs.decode);
            graphs.prefill = prefill;
            graphs.prefill_node_count -= prefill_fused;
            graphs.decode = decode;
            graphs.decode_node_count -= decode_fused;
        }
        graphs
    }

    #[cfg(test)]
    pub(crate) fn prompt_lengths(&self) -> Vec<u64> {
        self.entries
            .lock()
            .unwrap()
            .iter()
            .map(|(count, _)| *count)
            .collect()
    }
}

#[derive(Clone, Debug)]
pub(super) struct FirstNativeExecutionKvState {
    pub(super) cache: KvCacheId,
    pub(super) compatibility: KvCacheCompatibility,
    pub(super) layer_kv: LayerKvMap,
    /// The Provider `execute_first_native_graph` actually resolved and wrote this
    /// step's pending K/V resources under. `None` before the first graph
    /// execution for this state (freshly created by
    /// `create_prefill_kv_state`); `KvUpdateTransaction::begin`/
    /// `discard_pending_kv_state` fall back to Reference CPU only in that
    /// case (`generalize-first-native-provider-dispatch`).
    pub(super) provider: Option<ProviderBinding>,
}

/// One layer's K/V tensor resource identities. Before commit, these name
/// *pending* resources this generation step wrote but sampling/token commit
/// has not yet accepted; after commit, they match the
/// `KvCache.layer_resources` bindings Runtime now owns (task 7.1/7.3).
///
/// `pub` (`add-real-multi-device-model-instance-placement`): reachable
/// through [`FirstNativeProviderRunOutcome::layer_kv`]
/// (`LayerKvMap`), which an external multi-step segment decode loop
/// must be able to hold onto and thread from one step's outcome into the
/// next step's `kv_history` argument.
#[derive(Clone, Debug)]
pub struct FirstNativeLayerKvState {
    pub(super) k: TensorResourceId,
    pub(super) v: TensorResourceId,
    /// In-place KV history (`append-kv-history-in-place`): `k` and `v` are
    /// the layer's stores and this many leading rows are valid. `None` for
    /// copy-based history, whose tensors are exactly the history.
    pub(super) rows: Option<u64>,
}

impl E2eRuntimeModelExecutionEngine {
    pub(super) fn create_prefill_kv_state(
        &self,
        runtime: &mut Runtime,
        request: &GenerationRequest,
    ) -> Result<FirstNativeExecutionKvState, InferenceApiError> {
        let compatibility = self.kv_compatibility(request);
        let mut cache = KvCache::new(
            KvCacheId::new("first-native-temporary-kv").map_err(InferenceApiError::from)?,
            if request.session.is_some() {
                KvCacheScope::Session
            } else {
                KvCacheScope::Operation
            },
            compatibility.clone(),
            KvCacheLayoutMetadata::contiguous(
                self.fixture.config.architecture.layer_count as u32,
                self.fixture.config.architecture.kv_head_count as u32,
                self.fixture.config.architecture.head_dimension as u32,
                request
                    .prompt_token_count
                    .saturating_add(request.max_new_tokens)
                    .max(self.prefill_graph_length(request.input_token_ids.len() as u64) as usize)
                    .max(1) as u32,
                ComputeDType::Float32,
            )
            .with_layer_row_widths(self.fixture.config.kv_row_widths())
            .with_layer_value_row_widths(self.fixture.config.kv_value_row_widths()),
        );
        if let Some(session) = &request.session {
            cache = cache.with_session(session.clone());
        }
        let cache_id = runtime.create_kv_cache(cache)?;
        Ok(FirstNativeExecutionKvState {
            cache: cache_id,
            compatibility,
            layer_kv: LayerKvMap::new(),
            provider: None,
        })
    }

    pub(super) fn kv_state_key(request: &GenerationRequest) -> String {
        request.request_id.as_str().to_string()
    }

    pub(super) fn store_kv_state(
        &self,
        request: &GenerationRequest,
        state: FirstNativeExecutionKvState,
    ) -> Result<(), InferenceApiError> {
        self.kv_states
            .lock()
            .map_err(|_| InferenceApiError::KvCacheUnavailable {
                reason: "first-native KV state lock poisoned".into(),
            })?
            .insert(Self::kv_state_key(request), state);
        Ok(())
    }

    pub(super) fn store_pending_kv_state(
        &self,
        request: &GenerationRequest,
        state: FirstNativeExecutionKvState,
    ) -> Result<(), InferenceApiError> {
        self.pending_kv_states
            .lock()
            .map_err(|_| InferenceApiError::KvCacheUnavailable {
                reason: "first-native pending KV state lock poisoned".into(),
            })?
            .insert(Self::kv_state_key(request), state);
        Ok(())
    }

    pub(super) fn take_pending_kv_state(
        &self,
        request: &GenerationRequest,
    ) -> Result<FirstNativeExecutionKvState, InferenceApiError> {
        self.pending_kv_states
            .lock()
            .map_err(|_| InferenceApiError::KvCacheUnavailable {
                reason: "first-native pending KV state lock poisoned".into(),
            })?
            .remove(&Self::kv_state_key(request))
            .ok_or_else(|| InferenceApiError::KvCacheUnavailable {
                reason: "generation step has no pending first-native KV state to commit".into(),
            })
    }

    /// Discards a generation step's pending (not yet committed) KV state,
    /// abandoning it (task group 9 / Correctif 11): releases each layer's
    /// pending K/V resource from the registered Provider's storage, not
    /// just this engine's own bookkeeping map, so an abandoned pending
    /// write does not linger as an orphaned Provider-owned resource
    /// forever. Idempotent -- discarding an already-discarded or
    /// never-pending request is not an error.
    pub(super) fn discard_pending_kv_state(
        &self,
        runtime: &mut Runtime,
        request: &GenerationRequest,
    ) -> Result<(), InferenceApiError> {
        let discarded = self
            .pending_kv_states
            .lock()
            .map_err(|_| InferenceApiError::KvCacheUnavailable {
                reason: "first-native pending KV state lock poisoned".into(),
            })?
            .remove(&Self::kv_state_key(request));
        let Some(discarded) = discarded else {
            return Ok(());
        };
        // `generalize-first-native-provider-dispatch`: resolve the Provider
        // this step's graph execution actually wrote the pending resources
        // under, not Reference CPU unconditionally -- discarding a non-CPU
        // step's pending state used to look for it in the wrong Provider's
        // storage.
        let provider_binding = discarded
            .provider
            .clone()
            .unwrap_or_else(|| ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME));
        if let Ok(provider) = resolve_kernel_execution_provider(runtime, &provider_binding) {
            // `release_admitted_tensor`, not `release_tensor`: the pending
            // write these resources came from was itself admitted through
            // `write_tensor_admitted` (Correctif 1), so discarding it must
            // release that allocation too, not just drop the Provider
            // storage entry.
            // In-place KV stores (`append-kv-history-in-place`) also hold
            // the committed history: they are released with it, unless no
            // step of this request ever committed (a failed prefill).
            let has_committed_state = self
                .kv_states
                .lock()
                .map(|states| states.contains_key(&Self::kv_state_key(request)))
                .unwrap_or(true);
            // A pooled store set's stores stay with the pool.
            let pooled = self.kv_pool.current(&Self::kv_state_key(request)).is_some();
            for layer in discarded.layer_kv.values() {
                if layer.rows.is_some() && (has_committed_state || pooled) {
                    continue;
                }
                // Best-effort: this pending state is being discarded
                // regardless of whether Provider-side release succeeds.
                release_pending_kv_resource(runtime, &provider, &layer.k);
                release_pending_kv_resource(runtime, &provider, &layer.v);
            }
        }
        Ok(())
    }

    pub(super) fn decode_kv_state(
        &self,
        runtime: &mut Runtime,
        request: &GenerationRequest,
    ) -> Result<FirstNativeExecutionKvState, InferenceApiError> {
        let state = self
            .kv_states
            .lock()
            .map_err(|_| InferenceApiError::KvCacheUnavailable {
                reason: "first-native KV state lock poisoned".into(),
            })?
            .get(&Self::kv_state_key(request))
            .cloned()
            .ok_or_else(|| InferenceApiError::KvCacheUnavailable {
                reason: "decode requires existing first-native KV state".into(),
            })?;
        runtime.validate_kv_cache_reuse(&state.cache, &state.compatibility, None)?;
        Ok(state)
    }

    pub(super) fn kv_compatibility(&self, request: &GenerationRequest) -> KvCacheCompatibility {
        KvCacheCompatibility::new(
            request.model.clone(),
            request.tokenizer.tokenizer_id.clone(),
        )
        .with_prefix_fingerprint(PrefixFingerprint::from_tokens(
            &request.input_token_ids,
            request.request_id.as_str(),
            &request.tokenizer.tokenizer_id,
        ))
    }
}

impl E2eRuntimeModelExecutionEngine {
    /// Runs a prefill: per node, or for a bucketed prefill whose stores
    /// are pooled, along its recording's sequence (`replay-bucketed-prefill`):
    /// the first prefill of a key runs per node keeping its intermediates,
    /// the second runs inside a step capture, later ones replay it. A
    /// replay may select a greedy token on the Device, returned with the
    /// output.
    #[allow(clippy::too_many_arguments)]
    fn execute_prefill_graph(
        &self,
        runtime: &mut Runtime,
        request: &GenerationRequest,
        kv_state: &FirstNativeExecutionKvState,
        execution_plan: &mut PreparedExecutionPlan,
        graphs: &CachedGraphs,
        model_instance: &ModelInstanceId,
        graph_length: u64,
        bucketed: bool,
        node_events: &mut Vec<PerNodeCausalEvent>,
        kv_stores: Option<KvStoreSpec>,
        greedy: Option<&DeviceGreedyPlan>,
    ) -> Result<(FirstNativeGraphExecutionOutput, Option<u64>), InferenceApiError> {
        let prompt_tokens = request.input_token_ids.len() as u64;
        let sampled_row = prompt_tokens.saturating_sub(1);
        // Right padding: the bucket's last rows repeat the last prompt
        // token; no real row reads them.
        let ids_tensor = HostTensor::new(
            [graph_length],
            super::prefill_buckets::pad_prompt_ids(&request.input_token_ids, graph_length as usize),
        )
        .map_err(runtime_generation_failed)?;
        let row_tensor =
            HostTensor::new([1], vec![sampled_row as f32]).map_err(runtime_generation_failed)?;
        let ids_edge = TensorEdgeId::new("input.token_ids");
        let row_edge = TensorEdgeId::new(super::prefill_buckets::LOGITS_ROW_INPUT);
        let recording = self.prefill_recording_key(
            runtime,
            execution_plan,
            graphs,
            graph_length,
            bucketed,
            kv_stores,
        )?;
        let namespace = super::prefill_buckets::bucket_namespace(graph_length);
        let stage = recording
            .as_ref()
            .map(|key| self.step_replay.prefill_stage(key));
        let mut keeps_intermediates = matches!(
            stage,
            Some(PrefillRecordingStage::Warm | PrefillRecordingStage::Capture)
        );
        match (&recording, stage) {
            (Some(key), Some(PrefillRecordingStage::Replay(recorded))) => {
                // The padding rows' outputs are never read: the packed
                // matmuls skip them (`PADDING_ROWS_ATTRIBUTE`).
                if let Some(output) = replay_recorded_step(
                    runtime,
                    &recorded.0,
                    vec![ids_tensor.clone(), row_tensor.clone()],
                    graph_length - prompt_tokens,
                    false,
                    greedy,
                )? {
                    self.step_replay.touch_prefill_namespace(graph_length);
                    return Ok(output);
                }
                // Invalidated: run per node, and capture again later.
                self.step_replay.invalidate_prefill(key);
                keeps_intermediates = true;
            }
            (Some(key), Some(PrefillRecordingStage::Capture)) => {
                let captured = self.capture_step(
                    runtime,
                    StepToCapture {
                        cache: &kv_state.cache,
                        graph: &graphs.prefill,
                        memo: &graphs.prefill_memo,
                        inputs: vec![
                            (
                                ids_edge.clone(),
                                TensorResourceId::new(format!("{namespace}.{ids_edge}")),
                                ids_tensor.clone(),
                            ),
                            (
                                row_edge.clone(),
                                TensorResourceId::new(format!("{namespace}.{row_edge}")),
                                row_tensor.clone(),
                            ),
                        ],
                        input_owner: MemoryAllocationOwner::Session(namespace.clone()),
                        kv_history: None,
                        position: 0,
                        resource_namespace: Some(&namespace),
                        advancing_attributes: &[super::prefill_buckets::PADDING_ROWS_ATTRIBUTE],
                    },
                    execution_plan,
                    model_instance,
                    node_events,
                    kv_stores,
                )?;
                match captured {
                    Some((recorded, output)) => {
                        trace_step_capture("captured", format!("prefill of {graph_length} rows"));
                        self.step_replay.record_prefill(key.clone(), recorded);
                        self.step_replay.touch_prefill_namespace(graph_length);
                        return Ok((output, None));
                    }
                    None => {
                        self.step_replay.decline_prefill(key.clone());
                        keeps_intermediates = false;
                    }
                }
            }
            _ => {}
        }
        let mut inputs = BTreeMap::from([(ids_edge, ids_tensor)]);
        if graphs.logits_row {
            inputs.insert(row_edge, row_tensor);
        }
        let output = execute_first_native_graph_namespaced(
            runtime,
            &self.fixture,
            model_instance,
            &kv_state.cache,
            &graphs.prefill,
            execution_plan,
            inputs,
            None,
            Some(0),
            node_events,
            // Only the last prompt row's logits are sampled (the graph's
            // one row with a logits row gather).
            if bucketed && !graphs.logits_row {
                MaterializedEdges::Row(&["logits"], sampled_row)
            } else {
                MaterializedEdges::LastRow(&["logits"])
            },
            Some(&graphs.prefill_memo),
            kv_stores,
            keeps_intermediates.then_some(namespace.as_str()),
        )?;
        if let Some(key) = recording.filter(|_| keeps_intermediates) {
            self.step_replay.prefill_warmed(key);
            self.step_replay.touch_prefill_namespace(graph_length);
            self.bound_prefill_intermediates(runtime, &output.3, graph_length);
        }
        Ok((output, None))
    }

    /// Keeps the bucket namespaces' intermediates within their budget
    /// (`MAGNETAR_PREFILL_REPLAY_BYTES`): releases the least recently used
    /// buckets' recordings and intermediates, never `current`'s.
    fn bound_prefill_intermediates(
        &self,
        runtime: &mut Runtime,
        provider: &ProviderBinding,
        current: u64,
    ) {
        let budget = self
            .step_replay
            .0
            .prefill_intermediate_bytes
            .load(Ordering::Relaxed);
        let held = |runtime: &Runtime, bucket: u64| -> u64 {
            let owner =
                MemoryAllocationOwner::Session(super::prefill_buckets::bucket_namespace(bucket));
            runtime
                .memory()
                .allocations()
                .filter(|allocation| {
                    allocation.state == MemoryAllocationState::Active
                        && allocation.request.owner == owner
                })
                .map(|allocation| allocation.request.size_bytes)
                .sum()
        };
        let buckets = self.step_replay.prefill_namespaces();
        let mut total: u64 = buckets.iter().map(|bucket| held(runtime, *bucket)).sum();
        let Ok(executor) = resolve_kernel_execution_provider(runtime, provider) else {
            return;
        };
        for bucket in buckets {
            if total <= budget {
                break;
            }
            if bucket == current {
                continue;
            }
            total = total.saturating_sub(held(runtime, bucket));
            self.step_replay.release_prefill_bucket(bucket);
            let prefix = format!("{}.", super::prefill_buckets::bucket_namespace(bucket));
            let resources: Vec<TensorResourceId> = runtime
                .memory()
                .tensor_residencies()
                .filter(|residency| residency.tensor.as_str().starts_with(&prefix))
                .map(|residency| residency.tensor.clone())
                .collect();
            for resource in &resources {
                release_pending_kv_resource(runtime, &executor, resource);
            }
            trace_step_capture(
                "released",
                format!("prefill intermediates of {bucket} rows"),
            );
        }
    }

    /// The recording key of a prefill that can be recorded: bucketed, with
    /// the logits row gather, pooled stores, prefill replay on, and a
    /// Provider that captures steps.
    fn prefill_recording_key(
        &self,
        runtime: &Runtime,
        execution_plan: &PreparedExecutionPlan,
        graphs: &CachedGraphs,
        graph_length: u64,
        bucketed: bool,
        kv_stores: Option<KvStoreSpec>,
    ) -> Result<Option<PrefillRecordingKey>, InferenceApiError> {
        let Some(set) = kv_stores.and_then(|spec| spec.pool_set) else {
            return Ok(None);
        };
        if !bucketed || !graphs.logits_row || !self.step_replay.prefill_enabled() {
            return Ok(None);
        }
        let Some(provider) = execution_plan
            .node_bindings
            .first()
            .map(|binding| binding.provider.clone())
        else {
            return Ok(None);
        };
        if !resolve_kernel_execution_provider(runtime, &provider)?.supports_step_capture() {
            return Ok(None);
        }
        Ok(Some((
            graphs.prefill_memo.fingerprint(&graphs.prefill).clone(),
            graph_length,
            set,
        )))
    }

    /// A prefill's in-place KV stores: a pooled store set of the request's
    /// capacity bucket for a bucketed prefill (`replay-bucketed-prefill`),
    /// stores of exactly `rows_needed` rows owned by the request otherwise,
    /// or when the pool cannot hold the set.
    fn prefill_kv_stores(
        &self,
        runtime: &mut Runtime,
        request: &GenerationRequest,
        bucketed: bool,
        rows_needed: u64,
        execution_plan: &PreparedExecutionPlan,
    ) -> Result<KvStoreSpec, InferenceApiError> {
        let key = Self::kv_state_key(request);
        let provider = execution_plan
            .node_bindings
            .first()
            .map(|binding| binding.provider.clone());
        let precisions = self.resolve_kv_precisions(runtime, provider.as_ref());
        let (set, capacity, dropped) = match (bucketed, provider) {
            (true, Some(provider)) => {
                // Never past the model's context, when that holds the request.
                let context = self.fixture.config.architecture.context_length;
                let capacity = match kv_capacity_bucket(rows_needed) {
                    bucket if context >= rows_needed => bucket.min(context),
                    bucket => bucket,
                };
                let (set, dropped) = self.kv_pool.acquire(
                    &key,
                    capacity,
                    precisions,
                    self.kv_set_bytes(capacity, precisions),
                    provider,
                );
                (set, capacity, dropped)
            }
            // Not pooled: a set the key still holds goes back to the pool.
            _ => (None, rows_needed, self.kv_pool.release(&key)),
        };
        self.release_pooled_kv_sets(runtime, &dropped);
        Ok(match set {
            Some(set) => KvStoreSpec {
                capacity,
                pool_set: Some(set),
                precisions,
            },
            None => KvStoreSpec {
                precisions,
                ..KvStoreSpec::owned(rows_needed)
            },
        })
    }

    /// The precisions this loaded model's KV stores are created in
    /// (`turboquant-kv-history`), resolved once: a requested TurboQuant
    /// precision is kept for a role only when `provider` advertises
    /// TurboQuant stores on `kv-write` and on every attention Kernel and
    /// every layer's head width for that role allows it; otherwise the role
    /// is f32 and the reason is recorded.
    pub(super) fn resolve_kv_precisions(
        &self,
        runtime: &Runtime,
        provider: Option<&ProviderBinding>,
    ) -> KvPrecisions {
        let head_dimensions = self.fixture.config.kv_head_dimensions();
        self.kv_pool.resolve_precisions(|requested| {
            let mut fallbacks = Vec::new();
            let mut resolve_role =
                |role: &str, precision: KvStorePrecision, pick: fn(&(u64, u64)) -> u64| {
                    if precision == KvStorePrecision::F32 {
                        return KvStorePrecision::F32;
                    }
                    let advertised = provider.is_some_and(|provider| {
                        provider_kernels_declare(
                            runtime,
                            provider,
                            "kv-write",
                            crate::operator::KV_WRITE_TURBOQUANT_FEATURE,
                        ) && provider_kernels_declare(
                            runtime,
                            provider,
                            "attention",
                            crate::operator::ATTENTION_TURBOQUANT_KV_FEATURE,
                        )
                    });
                    if !advertised {
                        fallbacks.push(format!(
                            "{role}: {} requested, but the Provider does not advertise TurboQuant \
                         KV stores on kv-write and every attention Kernel; f32 stores",
                            precision.as_str()
                        ));
                        return KvStorePrecision::F32;
                    }
                    if let Some(width) = head_dimensions
                        .iter()
                        .flatten()
                        .map(pick)
                        .find(|width| !precision.supports_head_dimension(*width))
                    {
                        fallbacks.push(format!(
                            "{role}: {} requested, but a layer's {role} heads are {width} wide \
                         (not a power of two of at least 8); f32 stores",
                            precision.as_str()
                        ));
                        return KvStorePrecision::F32;
                    }
                    precision
                };
            let mut resolved = KvPrecisions {
                k: resolve_role("K", requested.k, |(key, _)| *key),
                v: resolve_role("V", requested.v, |(_, value)| *value),
                exact_rows: 0,
            };
            // An exact head (`keep-turboquant-kv-head-exact`) only where a
            // TurboQuant store is, and the Provider keeps one.
            if requested.exact_rows > 0 && resolved.quantized() {
                let advertised = provider.is_some_and(|provider| {
                    provider_kernels_declare(
                        runtime,
                        provider,
                        "kv-write",
                        crate::operator::KV_WRITE_TURBOQUANT_EXACT_HEAD_FEATURE,
                    )
                });
                if advertised {
                    resolved.exact_rows = requested.exact_rows;
                } else {
                    fallbacks.push(format!(
                        "exact head of {} rows requested, but the Provider does not advertise \
                         exact heads on kv-write; every row quantized",
                        requested.exact_rows
                    ));
                }
            }
            (resolved, fallbacks)
        })
    }

    /// A KV history's precisions (`turboquant-kv-history`): a prefill
    /// records the precisions its stores are created in, in the history's
    /// compatibility and KV layout metadata; a later step only continues a
    /// history whose recorded precisions are those this model's stores now
    /// resolve to, and refuses any other as incompatible.
    fn check_kv_history_precisions(
        &self,
        runtime: &mut Runtime,
        kv_state: &mut FirstNativeExecutionKvState,
        prefill: bool,
        execution_plan: &PreparedExecutionPlan,
    ) -> Result<(), InferenceApiError> {
        let provider = execution_plan
            .node_bindings
            .first()
            .map(|binding| binding.provider.clone());
        let mode = kv_quantization_mode(self.resolve_kv_precisions(runtime, provider.as_ref()));
        if !prefill {
            let mut requested = kv_state.compatibility.clone();
            requested.quantization_mode = mode;
            runtime.validate_kv_cache_reuse(&kv_state.cache, &requested, None)?;
            return Ok(());
        }
        kv_state.compatibility.quantization_mode = mode.clone();
        let cache = runtime
            .kv_caches_mut()
            .cache_mut(&kv_state.cache)
            .map_err(InferenceApiError::from)?;
        cache.compatibility.quantization_mode = mode.clone();
        cache.layout.quantization = mode.map(|mode| KvCacheQuantization {
            mode,
            // One f32 norm per head row.
            group_size: None,
            scale_dtype: Some(DTypeDescriptor::portable(ComputeDType::Float32)),
        });
        Ok(())
    }

    /// The bytes of one store set of `capacity` rows: the K and V of every
    /// layer that keeps its own, at that layer's widths
    /// (`per-layer-kv-geometry`; V's own width under latent attention,
    /// `express-latent-attention-forms`).
    /// At their stored size in `precisions` (`turboquant-kv-history`).
    fn kv_set_bytes(&self, capacity: u64, precisions: KvPrecisions) -> u64 {
        kv_bytes_per_token(&self.fixture.config, precisions)
            .saturating_mul(capacity)
            .saturating_add(super::kv_pool::kv_exact_head_bytes(
                &self.fixture.config,
                precisions,
                capacity,
            ))
    }

    /// Releases the stores of sets the pool dropped.
    pub(super) fn release_pooled_kv_sets(&self, runtime: &mut Runtime, sets: &[PooledKvSet]) {
        if sets.is_empty() {
            return;
        }
        let Some(provider) = self
            .kv_pool
            .provider()
            .and_then(|binding| resolve_kernel_execution_provider(runtime, &binding).ok())
        else {
            return;
        };
        for set in sets {
            self.step_replay.release_prefill_set(set.id);
            // A layer that reuses an earlier layer's K/V has no store.
            for (layer, width) in self.fixture.config.kv_row_widths().iter().enumerate() {
                if width.is_none() {
                    continue;
                }
                for role in ["k", "v"] {
                    release_pending_kv_resource(
                        runtime,
                        &provider,
                        &pooled_kv_store_id(set.id, layer, role),
                    );
                }
            }
        }
    }

    /// [`Self::execute_step`]; a step refused Memory Manager admission is
    /// retried once after the pool's free store sets are released.
    pub(super) fn execute_step_reclaiming(
        &self,
        runtime: &mut Runtime,
        request: &GenerationRequest,
        generated_tokens: &[TokenId],
        mut execution_plan: Option<&mut PreparedExecutionPlan>,
        greedy: Option<&crate::SamplingRequest>,
    ) -> Result<RuntimeModelExecutionStep, InferenceApiError> {
        match self.execute_step(
            runtime,
            request,
            generated_tokens,
            execution_plan.as_deref_mut(),
            greedy,
        ) {
            Err(InferenceApiError::MemoryAdmissionFailed { reason }) => {
                let freed = self.kv_pool.drain_free();
                if freed.is_empty() {
                    return Err(InferenceApiError::MemoryAdmissionFailed { reason });
                }
                self.release_pooled_kv_sets(runtime, &freed);
                self.execute_step(runtime, request, generated_tokens, execution_plan, greedy)
            }
            result => result,
        }
    }

    /// The prefill graph length for a prompt of `prompt_tokens` tokens
    /// ([`ComponentGraphCache::prefill_graph_length`]). Copy-based KV
    /// history keeps every row it computes, so it never pads.
    pub(super) fn prefill_graph_length(&self, prompt_tokens: u64) -> u64 {
        match &self.graph_cache {
            Some(cache) if !self.copy_based_kv => cache.prefill_graph_length(prompt_tokens),
            _ => prompt_tokens,
        }
    }

    /// One generation step; with `greedy` (a plain greedy sampling
    /// request), a replayed decode step's token may be selected on the
    /// Device instead of returning its logits.
    pub(super) fn execute_step(
        &self,
        runtime: &mut Runtime,
        request: &GenerationRequest,
        generated_tokens: &[TokenId],
        execution_plan: Option<&mut PreparedExecutionPlan>,
        greedy: Option<&crate::SamplingRequest>,
    ) -> Result<RuntimeModelExecutionStep, InferenceApiError> {
        let forced_token = self.forced_token;

        let vocab = self.fixture.config.architecture.vocabulary_size as usize;
        let greedy_plan = greedy
            .filter(|_| forced_token.is_none())
            .and_then(|greedy| crate::sampling::device_greedy_plan(greedy, vocab));
        let mut device_greedy_column = None;
        let kv_clock = runtime.phase_ledger().clock();
        let mut kv_state = if generated_tokens.is_empty() {
            self.create_prefill_kv_state(runtime, request)?
        } else {
            self.decode_kv_state(runtime, request)?
        };
        let model_input_token_count = if generated_tokens.is_empty() {
            request.input_token_ids.len()
        } else {
            1
        };
        let absolute_position = if generated_tokens.is_empty() {
            request.input_token_ids.len()
        } else {
            generation_decode_absolute_position(
                request.input_token_ids.len(),
                generated_tokens.len(),
            )
            .ok_or_else(|| InferenceApiError::GenerationFailed {
                reason: "decode requires a newly admitted token position".into(),
            })?
        };
        self.discard_pending_kv_state(runtime, request)?;
        runtime
            .phase_ledger_mut()
            .record(kv_clock, StepPhase::KvStatePrepare);
        // Both flags below (task 8.1) are only ever assigned immediately
        // after the check they attest to actually ran and passed for *this*
        // step -- never assumed from an earlier, one-time check (Model
        // Instance readiness is otherwise confirmed once, at plan
        // preparation time, before the generation loop starts) or from a
        // bare literal disconnected from any check.
        let model_instance_ready;
        let graph_validated;
        let mut node_events: Vec<PerNodeCausalEvent> = Vec::new();
        let (dispatch_result, logits) = if let Some(token) = forced_token {
            // Test-only deterministic shortcut, never exercised by
            // `validate_e2e_no_shortcuts` (the success path this gate
            // checks is always built without a forced token -- see
            // `build_runtime_with_model_execution_engine`). The returned
            // logits are synthetic by construction; a trivial matmul is
            // still dispatched so the Provider/observability path is
            // exercised, but its output is deliberately not what is
            // returned.
            let one = HostTensor::new([1, 1], vec![1.0]).map_err(|error| {
                InferenceApiError::GenerationFailed {
                    reason: error.to_string(),
                }
            })?;
            let (dispatch_result, _proof_output) = dispatch_matmul(runtime, &one, &one)?;
            let mut logits = vec![0.0_f32; vocab];
            logits[token as usize] = 10.0;
            model_instance_ready = true;
            graph_validated = true;
            (dispatch_result, logits)
        } else {
            // The first-native hot path must always execute a published,
            // ready `PreparedExecutionPlan` -- never fall back to ad hoc
            // Kernel Registry selection here. A missing plan is a structured
            // failure, not a silent per-node reselection.
            let execution_plan = execution_plan.ok_or_else(|| InferenceApiError::GraphPlanningFailed {
                reason: "first-native execution requires a published prepared execution plan; none was bound for this generation step".into(),
            })?;
            // Task 11.5: must be the exact same graph-production recipe
            // `prepare_first_native_execution_plans` used to compute
            // `execution_plan`'s `graph_fingerprint` -- a graph built any
            // other way (even one that is logically equivalent to a human
            // reader) fails `PreparedExecutionPlanExecutor::
            // prepare_node_execution`'s fingerprint check below with
            // `PlanValidationFailed`, since that check is deliberately
            // exact-match, not semantic.
            // `first_native_component_graphs_for_prompt` (or, when this
            // instance carries a caller-registered Component digest,
            // `build_first_native_graphs_from_named_component`) is that
            // same recipe (real Qwen Component under the strict, default
            // build; a structured fail-closed error in production
            // otherwise; the Rust-synthesized recipe only in a test build
            // without a strict engine) -- deterministic for a given
            // `(fixture, prompt_token_count)` whenever it succeeds at all,
            // so calling it again here reproduces the identical graph the
            // plan was prepared against. Must match `prepare_generation`'s
            // own choice of digest exactly -- see
            // `ProductionLoadedModel::component_digest`'s doc comment.
            let prompt_token_count = request.input_token_ids.len() as u64;
            // A bucketed prefill (`replay-bucketed-prefill`) runs its
            // bucket's graph, which the plan was prepared against.
            let graph_length = self.prefill_graph_length(prompt_token_count);
            let graph_build_clock = runtime.phase_ledger().clock();
            let cached_graphs = self
                .graph_cache
                .as_ref()
                .and_then(|cache| cache.get(graph_length));
            let component_graphs = match cached_graphs {
                Some(graphs) => graphs,
                None => Arc::new(
                    match &self.component_digest {
                        Some(digest) => {
                            build_first_native_graphs_from_named_component_for_artifact(
                                digest,
                                &architecture_config_from_first_native_model_config(
                                    &self.fixture.config,
                                ),
                                &self.fixture.manifest.tensors,
                                &self.fixture.identity,
                                graph_length,
                            )
                            .map(|(graphs, _definition, _instance)| graphs)
                        }
                        None => {
                            first_native_component_graphs_for_prompt(&self.fixture, graph_length)
                        }
                    }
                    .map_err(|error| InferenceApiError::GraphPlanningFailed {
                        reason: error.to_string(),
                    })
                    .map(|graphs| match &self.graph_cache {
                        Some(cache) => cache.prepare(runtime, graphs, graph_length),
                        None => CachedGraphs::new(graphs),
                    })?,
                ),
            };
            runtime
                .phase_ledger_mut()
                .record(graph_build_clock, StepPhase::GraphBuild);
            // Weight resources are bound to an active Model Instance (task
            // 6.3), so graph execution needs its id to look them up.
            let GenerationModelReference::ModelInstance(model_instance) = &request.model else {
                return Err(InferenceApiError::ModelLoadingFailed {
                    reason: "first-native execution requires an active Model Instance to resolve weight resources".into(),
                });
            };
            // Confirm this Model Instance is genuinely ready to generate
            // *right now* -- not merely inferred from the one-time check
            // `prepare_first_native_execution_plans` performed before the
            // generation loop started, which cannot see an instance
            // unloaded or invalidated between steps.
            let instance_clock = runtime.phase_ledger().clock();
            require_ready_first_native_instance(&*runtime, model_instance)?;
            runtime
                .phase_ledger_mut()
                .record(instance_clock, StepPhase::InstanceCheck);
            model_instance_ready = true;
            // The graph, not a hand-written Rust sequence, is the
            // authoritative recipe for prefill/decode execution (see
            // `execute_first_native_graph`'s doc comment). Its output edge
            // `"logits"` carries logits for every model-input row; only the
            // last row is a newly admitted token's distribution.
            // Every row this request's KV history can reach: its prompt and
            // every token it may generate (`append-kv-history-in-place`).
            // A bucketed prefill also writes its padding rows, and so does
            // the last pass of a chunked one (`chunk-long-prefills`).
            let chunked = generated_tokens.is_empty() && graph_length < prompt_token_count;
            // A bucket's prefill: padded, or of a bucket's length exactly.
            // A prompt of exactly the chunk's length runs the chunk's graph
            // in one pass, unpooled and unrecorded.
            let bucketed = !chunked
                && (graph_length != prompt_token_count
                    || (component_graphs.logits_row
                        && self
                            .graph_cache
                            .as_ref()
                            .is_some_and(|cache| cache.is_bucket_length(graph_length))));
            let rows_needed = (request.input_token_ids.len() as u64)
                .saturating_add(request.max_new_tokens as u64)
                .max(if chunked {
                    super::prefill_chunks::chunked_rows(prompt_token_count, graph_length)
                } else {
                    graph_length
                })
                .max(1);
            let kv_stores = if self.copy_based_kv {
                None
            } else if generated_tokens.is_empty() {
                Some(self.prefill_kv_stores(
                    runtime,
                    request,
                    bucketed,
                    rows_needed,
                    execution_plan,
                )?)
            } else {
                // The stores its prefill wrote.
                Some(
                    self.kv_pool
                        .current(&Self::kv_state_key(request))
                        .unwrap_or(KvStoreSpec::owned(rows_needed)),
                )
            };
            if kv_stores.is_some() {
                self.check_kv_history_precisions(
                    runtime,
                    &mut kv_state,
                    generated_tokens.is_empty(),
                    execution_plan,
                )?;
            }
            let (dispatch, mut bindings, layer_kv, resolved_provider) = if chunked {
                let kv_stores = kv_stores.ok_or_else(|| InferenceApiError::KvCacheUnavailable {
                    reason: "a chunked prefill continues KV history kept in place".into(),
                })?;
                self.execute_chunked_prefill(
                    runtime,
                    request,
                    &kv_state,
                    execution_plan,
                    &component_graphs,
                    model_instance,
                    graph_length,
                    &mut node_events,
                    kv_stores,
                )?
            } else if generated_tokens.is_empty() {
                let (mut output, selected) = self.execute_prefill_graph(
                    runtime,
                    request,
                    &kv_state,
                    execution_plan,
                    &component_graphs,
                    model_instance,
                    graph_length,
                    bucketed,
                    &mut node_events,
                    kv_stores,
                    greedy_plan.as_ref(),
                )?;
                device_greedy_column = selected;
                if bucketed {
                    // The padding rows lie beyond the committed length:
                    // the first decode step writes row `n` over them.
                    for layer in output.2.values_mut() {
                        if layer.rows.is_some() {
                            layer.rows = Some(prompt_token_count);
                        }
                    }
                }
                output
            } else {
                let token = *generated_tokens.last().ok_or_else(|| {
                    InferenceApiError::GenerationFailed {
                        reason: "decode requires a newly admitted token".into(),
                    }
                })?;
                let (output, selected) = self.execute_decode_graph(
                    runtime,
                    request,
                    &kv_state,
                    execution_plan,
                    &component_graphs,
                    model_instance,
                    token,
                    absolute_position as u64,
                    generated_tokens.len(),
                    &mut node_events,
                    kv_stores,
                    greedy_plan.as_ref(),
                )?;
                device_greedy_column = selected;
                output
            };
            // `execute_first_native_graph` topologically validates the graph (via
            // `first_native_graph_execution_order`) as its first action -- reaching
            // this point with an `Ok` result means that validation
            // genuinely ran and passed for this step's graph, not that it
            // is merely assumed.
            graph_validated = true;
            kv_state.layer_kv = layer_kv;
            // Carries the Provider this step's graph execution actually
            // resolved through to KV commit/discard
            // (`generalize-first-native-provider-dispatch`): those used to
            // hardcode Reference CPU unconditionally regardless of which
            // Provider the pending K/V resources were actually written
            // under.
            kv_state.provider = Some(resolved_provider);
            let logits_clock = runtime.phase_ledger().clock();
            // A token selected on the Device leaves its logits there.
            let logits = if device_greedy_column.is_some() {
                Vec::new()
            } else {
                let logits_tensor =
                    bindings
                        .remove(&TensorEdgeId::new("logits"))
                        .ok_or_else(|| InferenceApiError::GenerationFailed {
                            reason: "first-native graph produced no logits output".into(),
                        })?;
                let output_rows = logits_tensor.data.len() / vocab.max(1);
                let last_row_start = output_rows.saturating_sub(1) * vocab;
                logits_tensor.data[last_row_start..last_row_start + vocab].to_vec()
            };
            runtime
                .phase_ledger_mut()
                .record(logits_clock, StepPhase::LogitsExtract);
            (dispatch, logits)
        };
        let kv_commit = if generated_tokens.is_empty() {
            RuntimeKvCacheCommit::PrefillCompleted {
                cache: kv_state.cache.clone(),
                tokens: request.prompt_token_count as u32,
            }
        } else {
            RuntimeKvCacheCommit::DecodeAppended {
                cache: kv_state.cache.clone(),
                tokens: 1,
            }
        };
        let store_clock = runtime.phase_ledger().clock();
        self.store_pending_kv_state(request, kv_state.clone())?;
        runtime
            .phase_ledger_mut()
            .record(store_clock, StepPhase::KvStateStore);
        let evidence_clock = runtime.phase_ledger().clock();
        let mut evidence = RuntimeGenerationExecutionEvidence::from_dispatch_result(
            &dispatch_result,
            model_instance_ready,
            graph_validated,
        )
        .with_context(format!("request={}", request.request_id))
        .with_context(format!("decode_step={}", generated_tokens.len()))
        .with_context(format!("model_input_tokens={model_input_token_count}"))
        .with_context(format!("kv_cache={}", kv_state.cache))
        .with_context(format!("absolute_position={absolute_position}"));
        if let GenerationModelReference::ModelInstance(instance) = &request.model {
            evidence = evidence.with_context(format!("model_instance={instance}"));
        }
        if let Some(session) = &request.session {
            evidence = evidence.with_context(format!("session={session}"));
        }
        runtime
            .phase_ledger_mut()
            .record(evidence_clock, StepPhase::EvidenceBuild);
        let mut step = RuntimeModelExecutionStep::new(logits, evidence)
            .with_kv_commit(kv_commit)
            .with_node_events(node_events);
        step.device_greedy_column = device_greedy_column;
        Ok(step)
    }
}

/// [`materialize_first_native_graph_bindings`] for only the last row of
/// each rank-2 binding.
/// The nodes that may run on one row of their first input only: the LM
/// head behind each edge of which only one row is materialized (a
/// prefill's logits), when the head is a row-wise matmul reached directly
/// or through row-wise unary nodes ([`super::prefill_buckets::logits_head`]).
/// Each output row of those nodes depends on the same input row alone, so
/// the selected row is unchanged, and the nodes after the head then
/// process one row too.
pub(super) fn last_row_heads(
    graph: &ExecutionGraph,
    materialized_rows: &[&str],
) -> BTreeSet<ExecutionNodeId> {
    materialized_rows
        .iter()
        .filter_map(|edge| {
            super::prefill_buckets::logits_head(graph, &TensorEdgeId::new(*edge))
                .map(|(head, _)| head)
        })
        .collect()
}

/// Replaces a rank-2 `value` of more than one row by its last row. A
/// resident value's row is read back (one row, a few KB) and goes to the
/// Kernel as a fresh input.
pub(super) fn last_row_only(
    provider: &dyn ProviderExecutionApi,
    value: &mut NodeValue,
    row: Option<u64>,
) -> Result<(), InferenceApiError> {
    let (rows, cols) = value.rows_cols()?;
    if rows <= 1 {
        return Ok(());
    }
    let row = row.unwrap_or(rows - 1);
    let row = match value {
        NodeValue::Host(tensor) => {
            let start = usize::try_from(row * cols).ok();
            let end = usize::try_from((row + 1) * cols).ok();
            start
                .zip(end)
                .and_then(|(start, end)| tensor.data.get(start..end))
                .and_then(|row| HostTensor::new([1, cols], row.to_vec()).ok())
        }
        NodeValue::Resident { id, .. } => (row < rows)
            .then(|| provider.read_tensor_rows(id, row, 1))
            .flatten(),
    }
    .ok_or_else(|| InferenceApiError::GraphPlanningFailed {
        reason: "the last row of a matmul input could not be read".into(),
    })?;
    *value = NodeValue::Host(row);
    Ok(())
}

pub(super) fn materialize_last_rows(
    provider: &dyn ProviderExecutionApi,
    bindings: &BTreeMap<TensorEdgeId, (TensorResourceId, Vec<u64>)>,
    row: Option<u64>,
) -> Result<BTreeMap<TensorEdgeId, HostTensor>, InferenceApiError> {
    let mut materialized = BTreeMap::new();
    for (edge_id, (resource_id, shape)) in bindings {
        let rows = shape.first().copied().unwrap_or(0);
        // A single row is the selected one (the LM head already read it).
        let row = if rows == 1 {
            0
        } else {
            row.unwrap_or(rows.saturating_sub(1))
        };
        let tensor = (shape.len() == 2 && row < rows)
            .then(|| provider.read_tensor_rows(resource_id, row, 1))
            .flatten()
            .ok_or_else(|| InferenceApiError::GraphPlanningFailed {
                reason: format!(
                    "no last row materialized for graph edge '{edge_id}' (resource '{resource_id}', shape {shape:?})"
                ),
            })?;
        materialized.insert(edge_id.clone(), tensor);
    }
    Ok(materialized)
}

impl RuntimeModelExecutionEngine for E2eRuntimeModelExecutionEngine {
    fn execute_generation_step(
        &self,
        runtime: &mut Runtime,
        request: &GenerationRequest,
        generated_tokens: &[TokenId],
        execution_plan: Option<&mut PreparedExecutionPlan>,
    ) -> Result<RuntimeModelExecutionStep, InferenceApiError> {
        self.execute_step_reclaiming(runtime, request, generated_tokens, execution_plan, None)
    }

    fn execute_generation_step_selecting(
        &self,
        runtime: &mut Runtime,
        request: &GenerationRequest,
        generated_tokens: &[TokenId],
        execution_plan: Option<&mut PreparedExecutionPlan>,
        greedy: &crate::SamplingRequest,
    ) -> Result<RuntimeModelExecutionStep, InferenceApiError> {
        self.execute_step_reclaiming(
            runtime,
            request,
            generated_tokens,
            execution_plan,
            Some(greedy),
        )
    }

    fn commit_generation_step(
        &self,
        runtime: &mut Runtime,
        request: &GenerationRequest,
        generated_tokens_before_step: &[TokenId],
        _accepted_token: TokenId,
        _step: &RuntimeModelExecutionStep,
    ) -> Result<(), InferenceApiError> {
        let mut state = self.take_pending_kv_state(request)?;
        // Validate the cache transition before promoting resources. A sealed
        // or full cache must not replace the preceding committed generation.
        // The same cache methods are used here and by the actual commit.
        let validation = (|| -> Result<(), InferenceApiError> {
            let mut candidate = runtime.kv_cache(&state.cache)?.clone();
            if generated_tokens_before_step.is_empty() {
                candidate.complete_prefill(request.prompt_token_count as u32)?;
            } else {
                candidate.append_tokens(1)?;
            }
            Ok(())
        })();
        if let Err(error) = validation {
            self.store_pending_kv_state(request, state)?;
            return Err(error);
        }
        // Promote this step's pending KV resources to Runtime-owned,
        // committed bindings on the KvCache itself (task 7.1/7.3/7.4
        // commit) only now that sampling and token commit have succeeded --
        // never before. A failure anywhere before this point leaves the
        // pending writes unpromoted and the cache's committed bindings
        // untouched (task 7.4 abort/7.5 rollback).
        state.layer_kv = match self.promote_pending_kv_resources(runtime, &state) {
            Ok(layers) => layers,
            Err(error) => {
                // Promotion did not publish a new generation. Keep the
                // original pending resources reachable for retry or request
                // cleanup, including when the first prefill never committed.
                self.store_pending_kv_state(request, state)?;
                return Err(error);
            }
        };
        if generated_tokens_before_step.is_empty() {
            runtime.prefill_kv_cache_completed(&state.cache, request.prompt_token_count as u32)?;
        } else {
            runtime.append_decode_kv_cache(&state.cache, 1)?;
        }
        self.store_kv_state(request, state)
    }

    fn release_request_state(
        &self,
        runtime: &mut Runtime,
        request: &GenerationRequest,
    ) -> Result<(), InferenceApiError> {
        self.step_replay.release(&Self::kv_state_key(request));
        self.discard_pending_kv_state(runtime, request)?;
        // A pooled store set outlives its request (`replay-bucketed-prefill`).
        let pooled = self.kv_pool.current(&Self::kv_state_key(request)).is_some();
        let dropped = self.kv_pool.release(&Self::kv_state_key(request));
        self.release_pooled_kv_sets(runtime, &dropped);
        let committed = self
            .kv_states
            .lock()
            .map_err(|_| InferenceApiError::KvCacheUnavailable {
                reason: "first-native KV state lock poisoned".into(),
            })?
            .remove(&Self::kv_state_key(request));
        let Some(committed) = committed else {
            return Ok(());
        };
        // The pooled stores' admissions belong to the pool: detach them
        // from the request's KV cache, whose release (when its session
        // closes) frees every allocation its layers name.
        if pooled && let Ok(cache) = runtime.kv_caches_mut().cache_mut(&committed.cache) {
            cache
                .layer_resources
                .retain(|_, binding| !is_pooled_kv_store(&binding.k));
        }
        let provider_binding = committed
            .provider
            .clone()
            .unwrap_or_else(|| ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME));
        let Ok(provider) = resolve_kernel_execution_provider(runtime, &provider_binding) else {
            return Ok(());
        };
        // The committed bindings, plus each layer's `.pending` staging
        // resource that every commit copies from and leaves behind
        // (`execute_first_native_graph_nodes`'s KV pending writes).
        for (layer, binding) in &committed.layer_kv {
            // Copy-based generations are Provider-admitted; in-place stores
            // are Runtime pre-admissions: release either, unless pooled.
            if !(pooled && binding.rows.is_some()) {
                release_pending_kv_resource(runtime, &provider, &binding.k);
                release_pending_kv_resource(runtime, &provider, &binding.v);
            }
            for role in ["k", "v"] {
                release_pending_kv_resource(
                    runtime,
                    &provider,
                    &TensorResourceId::new(format!(
                        "kv.{}.layer{layer}.{role}.pending",
                        committed.cache
                    )),
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
