//! Unit tests for the parent module.
//!
//! Shared fixtures live in `first_native_runtime/tests.rs`.

use super::super::tests::*;
use super::*;

fn provider() -> ProviderBinding {
    ProviderBinding::new(crate::REFERENCE_CPU_PROVIDER_NAME)
}

#[test]
fn capacity_buckets_are_powers_of_two_from_the_minimum() {
    assert_eq!(kv_capacity_bucket(1), 256);
    assert_eq!(kv_capacity_bucket(256), 256);
    assert_eq!(kv_capacity_bucket(257), 512);
    assert_eq!(kv_capacity_bucket(3000), 4096);
}

#[test]
fn a_released_set_serves_the_next_request_of_its_capacity() {
    let pool = KvStorePool::with_max_bytes(1 << 20);
    let (a, dropped) = pool.acquire("a", 256, KvPrecisions::default(), 100, provider());
    assert!(dropped.is_empty());
    let (b, _) = pool.acquire("b", 256, KvPrecisions::default(), 100, provider());
    assert_ne!(a, b, "live requests never share a set");
    assert!(pool.release("a").is_empty());
    let (c, _) = pool.acquire("c", 256, KvPrecisions::default(), 100, provider());
    assert_eq!(c, a);
    // Another capacity takes a new set.
    let (d, _) = pool.acquire("d", 512, KvPrecisions::default(), 200, provider());
    assert!(d.is_some() && d != a && d != b);
    assert_eq!(pool.held_bytes(), 400);
    // A key's own set is kept across its prefills.
    assert_eq!(
        pool.acquire("c", 256, KvPrecisions::default(), 100, provider())
            .0,
        c
    );
    assert_eq!(
        pool.current("c"),
        Some(KvStoreSpec {
            capacity: 256,
            pool_set: c,
            precisions: KvPrecisions::default(),
        })
    );
}

#[test]
fn the_pool_keeps_at_most_four_sets_per_capacity_bucket() {
    let pool = KvStorePool::with_max_bytes(1 << 20);
    let keys: Vec<String> = (0..6).map(|key| key.to_string()).collect();
    let sets: Vec<u64> = keys
        .iter()
        .map(|key| {
            pool.acquire(key, 256, KvPrecisions::default(), 10, provider())
                .0
                .unwrap()
        })
        .collect();
    // Six live requests hold six sets; freeing them keeps four.
    let mut dropped = Vec::new();
    for key in &keys {
        dropped.extend(pool.release(key));
    }
    assert_eq!(pool.free_sets(), KV_POOL_SETS_PER_BUCKET);
    // The least recently used go first.
    assert_eq!(
        dropped.iter().map(|set| set.id).collect::<Vec<_>>(),
        sets[..2].to_vec()
    );
}

#[test]
fn the_pool_stays_within_its_bytes() {
    let pool = KvStorePool::with_max_bytes(250);
    let (a, _) = pool.acquire("a", 256, KvPrecisions::default(), 100, provider());
    let (b, _) = pool.acquire("b", 512, KvPrecisions::default(), 100, provider());
    pool.release("a");
    pool.release("b");
    // A third set needs room: the least recently used free one goes.
    let (c, dropped) = pool.acquire("c", 1024, KvPrecisions::default(), 100, provider());
    assert!(c.is_some());
    assert_eq!(
        dropped.iter().map(|set| set.id).collect::<Vec<_>>(),
        vec![a.unwrap()]
    );
    assert_eq!(pool.held_bytes(), 200);
    // Larger than the whole pool: never pooled.
    assert_eq!(
        pool.acquire("d", 2048, KvPrecisions::default(), 300, provider())
            .0,
        None
    );
    // Only free sets are drained.
    let drained = pool.drain_free();
    assert_eq!(
        drained.iter().map(|set| set.id).collect::<Vec<_>>(),
        vec![b.unwrap()]
    );
    assert_eq!(pool.held_bytes(), 100);
}

#[test]
fn a_set_never_moves_to_another_provider() {
    let pool = KvStorePool::with_max_bytes(1 << 20);
    assert!(
        pool.acquire("a", 256, KvPrecisions::default(), 10, provider())
            .0
            .is_some()
    );
    assert_eq!(
        pool.acquire(
            "b",
            256,
            KvPrecisions::default(),
            10,
            ProviderBinding::new("other")
        )
        .0,
        None
    );
}

/// One Runtime with the fixture loaded and a bucketing engine.
struct Harness {
    fixture: E2eFixture,
    runtime: Runtime,
    instance: ModelInstanceId,
    engine: E2eRuntimeModelExecutionEngine,
}

impl Harness {
    fn new(max_runtime_bytes: Option<u64>) -> Result<Self, String> {
        let fixture = e2e_fixture().expect("fixture builds");
        let mut config = crate::RuntimeConfig::default();
        config.memory.max_runtime_bytes = max_runtime_bytes;
        let mut runtime = Runtime::builder()
            .config(config)
            .register_provider(Arc::new(ReferenceCpuProvider::new()))
            .trust_store(
                ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
            )
            .build()
            .map_err(|error| error.to_string())?;
        register_reference_cpu_prepared_kernels(&mut runtime);
        let (instance, _memory) =
            load_fixture_instance(&fixture, &mut runtime).map_err(|error| error.to_string())?;
        let mut engine = new_kv_lifecycle_engine(&fixture);
        let cache = ComponentGraphCache::default();
        cache.set_prefill_buckets(vec![8, 16]);
        engine.graph_cache = Some(cache);
        Ok(Self {
            fixture,
            runtime,
            instance,
            engine,
        })
    }

    fn request(&self, id: &str, prompt: &[TokenId], max_new_tokens: usize) -> GenerationRequest {
        let tokenized = tokenize_prompt_input(
            &self.fixture.tokenizer,
            TokenizationRequest::new(PromptInput::TokenIds(prompt.to_vec())),
            None,
        )
        .unwrap();
        prepare_generation(
            &self.runtime,
            build_generation_request(
                GenerationRequestId::new(id).unwrap(),
                None,
                GenerationModelReference::ModelInstance(self.instance.clone()),
                generation_tokenizer_reference(&self.fixture),
                tokenized,
                max_new_tokens,
                GenerationParameters::greedy(),
                StopConditions::default(),
                StreamingMode::Disabled,
            ),
        )
        .unwrap()
    }

    fn plans(&mut self, request: &GenerationRequest) -> FirstNativePreparedPlans {
        let cache = self.engine.graph_cache.clone().unwrap();
        let length = cache.prefill_graph_length(request.input_token_ids.len() as u64);
        let graphs = match cache.get(length) {
            Some(graphs) => graphs,
            None => {
                let graphs =
                    first_native_component_graphs_for_prompt(&self.fixture, length).unwrap();
                let graphs = Arc::new(cache.prepare(&self.runtime, graphs, length));
                cache.insert(length, Arc::clone(&graphs));
                graphs
            }
        };
        graphs
            .prepared_plans(&self.runtime, &self.instance, length, &provider())
            .unwrap()
    }

    /// The prefill, committed, with or without the pool's reclaim.
    fn prefill(
        &mut self,
        request: &GenerationRequest,
        reclaim: bool,
    ) -> Result<Vec<f32>, InferenceApiError> {
        let mut plans = self.plans(request);
        let step = if reclaim {
            self.engine.execute_generation_step(
                &mut self.runtime,
                request,
                &[],
                Some(&mut plans.prefill),
            )
        } else {
            self.engine.execute_step(
                &mut self.runtime,
                request,
                &[],
                Some(&mut plans.prefill),
                None,
            )
        }?;
        self.engine
            .commit_generation_step(&mut self.runtime, request, &[], 1, &step)?;
        Ok(step.logits)
    }

    /// A prefill and `decode_steps` decode steps, committed; released with
    /// `release`.
    fn generate(
        &mut self,
        request: &GenerationRequest,
        decode_steps: usize,
        release: bool,
    ) -> Vec<Vec<f32>> {
        let mut steps = vec![self.prefill(request, true).unwrap()];
        let mut plans = self.plans(request);
        let mut generated = vec![1];
        for index in 1..=decode_steps {
            let step = self
                .engine
                .execute_generation_step(
                    &mut self.runtime,
                    request,
                    &generated,
                    Some(&mut plans.decode),
                )
                .unwrap();
            steps.push(step.logits.clone());
            let token = (index % 3 + 1) as TokenId;
            self.engine
                .commit_generation_step(&mut self.runtime, request, &generated, token, &step)
                .unwrap();
            generated.push(token);
        }
        if release {
            self.release(request);
        }
        steps
    }

    fn release(&mut self, request: &GenerationRequest) {
        self.engine
            .release_request_state(&mut self.runtime, request)
            .unwrap();
    }

    /// The Memory Manager allocation of a pooled store.
    fn store_allocation(&self, set: u64) -> Option<MemoryAllocationId> {
        self.runtime
            .memory()
            .tensor_residency(&pooled_kv_store_id(set, 0, "k"))
            .and_then(|residency| residency.allocation)
    }
}

/// The logits of `prompt` on a fresh Runtime.
fn fresh(prompt: &[TokenId], decode_steps: usize) -> Vec<Vec<f32>> {
    let mut harness = Harness::new(None).unwrap();
    let request = harness.request("fresh", prompt, decode_steps + 1);
    harness.generate(&request, decode_steps, true)
}

/// A request reuses the stores a released one left in the pool: the same
/// resources and Memory Manager allocations, and its own results, as if
/// the stores were new. Its history starts empty, and no row it reads was
/// written by the earlier request.
#[test]
fn a_reused_store_set_keeps_its_allocation_and_holds_no_earlier_rows() {
    let decode_steps = 4;
    let mut harness = Harness::new(None).unwrap();
    let first = harness.request("first", &[3, 9, 4, 7, 7, 1], decode_steps + 1);
    harness.generate(&first, decode_steps, true);
    let allocation = harness.store_allocation(1).expect("the pool kept set 1");
    let held = harness.engine.kv_pool.held_bytes();
    assert!(held > 0);

    let prompt = [5, 2];
    let second = harness.request("second", &prompt, decode_steps + 1);
    let steps = harness.generate(&second, decode_steps, false);
    assert_eq!(
        harness
            .engine
            .kv_pool
            .current("second")
            .and_then(|spec| spec.pool_set),
        Some(1)
    );
    assert_eq!(harness.store_allocation(1), Some(allocation));
    assert_eq!(harness.engine.kv_pool.held_bytes(), held);
    assert_eq!(steps, fresh(&prompt, decode_steps));
    harness.release(&second);
    assert_eq!(harness.store_allocation(1), Some(allocation));
}

/// Two live requests never share a set, and each gets its own results.
#[test]
fn live_requests_use_different_store_sets() {
    let mut harness = Harness::new(None).unwrap();
    let a = harness.request("a", &[3, 9, 4], 3);
    let b = harness.request("b", &[6, 1, 2, 8], 3);
    let a_logits = harness.prefill(&a, true).unwrap();
    let b_logits = harness.prefill(&b, true).unwrap();
    let set = |key: &str| harness.engine.kv_pool.current(key).unwrap().pool_set;
    assert_ne!(set("a"), set("b"));
    assert_eq!(a_logits, fresh(&[3, 9, 4], 0)[0]);
    assert_eq!(b_logits, fresh(&[6, 1, 2, 8], 0)[0]);
}

/// An exact-length prefill owns its stores and releases them with itself.
#[test]
fn an_unbucketed_request_is_not_pooled() {
    let mut harness = Harness::new(None).unwrap();
    let request = harness.request("long", &[4; 17], 2);
    harness.generate(&request, 1, true);
    assert_eq!(harness.engine.kv_pool.held_bytes(), 0);
}

/// Under admission pressure, the pool's free sets are released and the
/// step runs again: a budget that refuses a prefill while the pool holds a
/// free set admits it once the pool lets go.
#[test]
fn a_refused_admission_reclaims_the_pool_s_free_sets() {
    // Two released requests leave two free sets; a third takes one, and
    // the other is all reclaim can release.
    let attempt = |budget: u64, reclaim: bool| -> Result<KvStorePool, String> {
        let mut harness = Harness::new(Some(budget))?;
        let a = harness.request("a", &[3, 9, 4], 4);
        let b = harness.request("b", &[6, 1], 4);
        for request in [&a, &b] {
            harness
                .prefill(request, true)
                .map_err(|error| format!("{error:?}"))?;
        }
        harness.release(&a);
        harness.release(&b);
        let c = harness.request("c", &[5, 2, 2, 8, 1, 3, 3, 9, 7, 7], 4);
        harness
            .prefill(&c, reclaim)
            .map_err(|error| format!("{error:?}"))?;
        Ok(harness.engine.kv_pool.clone())
    };
    // The smallest budget the third request fits in with reclaim.
    let (mut low, mut high) = (1_u64, 1 << 30);
    assert!(attempt(high, true).is_ok());
    while low + 1 < high {
        let middle = low + (high - low) / 2;
        if attempt(middle, true).is_ok() {
            high = middle;
        } else {
            low = middle;
        }
    }
    let pool = attempt(high, true).unwrap();
    assert_eq!(pool.free_sets(), 0, "the free set was reclaimed");
    assert!(pool.current("c").is_some());
    let refused = attempt(high, false).expect_err("without reclaim the budget is too small");
    assert!(refused.contains("MemoryAdmissionFailed"), "{refused}");
}

/// A set serves only requests whose stores use the precisions it was
/// created in (`turboquant-kv-history`), and reports them.
#[test]
fn a_set_serves_only_requests_in_its_precisions() {
    let tq4 = KvPrecisions::uniform(KvStorePrecision::Tq4);
    let pool = KvStorePool::with_max_bytes(1 << 20);
    let (a, _) = pool.acquire("a", 256, tq4, 30, provider());
    assert_eq!(pool.current("a").map(|spec| spec.precisions), Some(tq4));
    // The same key in other precisions takes another set.
    let (b, _) = pool.acquire("a", 256, KvPrecisions::default(), 100, provider());
    assert!(b.is_some() && b != a);
    assert!(pool.release("a").is_empty());
    // A free set is not reused in other precisions...
    let (c, _) = pool.acquire(
        "c",
        256,
        KvPrecisions::uniform(KvStorePrecision::Tq3),
        25,
        provider(),
    );
    assert!(c.is_some() && c != a && c != b);
    // ...but is in its own.
    assert_eq!(pool.acquire("d", 256, tq4, 30, provider()).0, a);
}

/// f32 histories keep the compatibility they always had; TurboQuant ones
/// name both precisions.
#[test]
fn kv_quantization_modes_name_turboquant_precisions_only() {
    assert_eq!(kv_quantization_mode(KvPrecisions::default()), None);
    assert_eq!(
        kv_quantization_mode(KvPrecisions {
            k: KvStorePrecision::Tq4,
            v: KvStorePrecision::Tq3,
            exact_rows: 0,
        })
        .as_deref(),
        Some("turboquant-kv:k=tq4,v=tq3")
    );
    // An exact head is part of the compatibility: a history with one is
    // read differently from one without (`keep-turboquant-kv-head-exact`).
    assert_eq!(
        kv_quantization_mode(KvPrecisions::uniform(KvStorePrecision::Tq4)).as_deref(),
        Some("turboquant-kv:k=tq4,v=tq4,exact=1024")
    );
    // An f32 history has no head to keep exact.
    assert_eq!(KvPrecisions::uniform(KvStorePrecision::F32).exact_rows, 0);
    assert_eq!(
        KvPrecisions::uniform(KvStorePrecision::F32).with_exact_rows(64),
        KvPrecisions::default()
    );
}
