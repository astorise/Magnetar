//! Pooled in-place KV stores (`replay-bucketed-prefill`): a bucketed
//! request's stores outlive it, so the next request of the same capacity
//! bucket writes into the same Device buffers, which is what lets a
//! recording that references them replay.

use super::*;

/// The smallest capacity bucket, in rows.
pub(crate) const KV_POOL_MIN_CAPACITY: u64 = 256;

/// At most this many store sets are kept per capacity bucket.
pub(crate) const KV_POOL_SETS_PER_BUCKET: usize = 4;

/// The bytes the pool may hold when `MAGNETAR_KV_POOL_BYTES` is unset.
pub(crate) const DEFAULT_KV_POOL_BYTES: u64 = 1 << 30;

/// The Memory Manager owner of pooled stores: they belong to no request.
pub(crate) const KV_STORE_POOL_OWNER: &str = "kv-store-pool";

/// How a graph execution keeps its KV history in place
/// (`append-kv-history-in-place`): each store's rows, and the pooled store
/// set its stores belong to, if any.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct KvStoreSpec {
    pub(crate) capacity: u64,
    pub(crate) pool_set: Option<u64>,
}

impl KvStoreSpec {
    /// Stores of `capacity` rows owned by the request.
    pub(crate) fn owned(capacity: u64) -> Self {
        Self {
            capacity,
            pool_set: None,
        }
    }
}

/// The capacity bucket for a request needing `rows` rows: the next power
/// of two, at least [`KV_POOL_MIN_CAPACITY`].
pub(crate) fn kv_capacity_bucket(rows: u64) -> u64 {
    rows.max(KV_POOL_MIN_CAPACITY).next_power_of_two()
}

/// A pooled store's resource id.
pub(crate) fn pooled_kv_store_id(set: u64, layer: usize, role: &str) -> TensorResourceId {
    TensorResourceId::new(format!("kv.pool{set}.layer{layer}.{role}.store"))
}

/// Whether `id` names a pooled store.
pub(crate) fn is_pooled_kv_store(id: &TensorResourceId) -> bool {
    id.as_str().starts_with("kv.pool")
}

/// One set of stores (every layer's K and V) of one capacity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PooledKvSet {
    pub(crate) id: u64,
    pub(crate) capacity: u64,
    pub(crate) bytes: u64,
    last_used: u64,
}

#[derive(Debug)]
pub(crate) struct KvStorePoolState {
    max_bytes: u64,
    next_set: u64,
    tick: u64,
    /// Sets no live request uses, least recently used first out.
    free: Vec<PooledKvSet>,
    /// Each live request's set, by KV state key.
    in_use: BTreeMap<String, PooledKvSet>,
    /// The Provider every set's stores live in.
    provider: Option<ProviderBinding>,
}

/// The pool of one loaded model's stores, shared by its engine and its
/// `ProductionLoadedModel`. Bounded by bytes and by sets per capacity
/// bucket. Methods that drop sets return them: the caller releases their
/// stores, which this type cannot reach.
#[derive(Clone, Debug)]
pub(crate) struct KvStorePool(pub(super) Arc<Mutex<KvStorePoolState>>);

impl Default for KvStorePool {
    fn default() -> Self {
        let max_bytes = std::env::var("MAGNETAR_KV_POOL_BYTES")
            .ok()
            .and_then(|bytes| bytes.trim().parse().ok())
            .unwrap_or(DEFAULT_KV_POOL_BYTES);
        Self::with_max_bytes(max_bytes)
    }
}

impl KvStorePool {
    pub(crate) fn with_max_bytes(max_bytes: u64) -> Self {
        Self(Arc::new(Mutex::new(KvStorePoolState {
            max_bytes,
            next_set: 0,
            tick: 0,
            free: Vec::new(),
            in_use: BTreeMap::new(),
            provider: None,
        })))
    }

    /// A store set of `capacity` rows (`bytes` in all) for the request
    /// keyed `key`: its current one when it has one of that capacity, else
    /// the most recently used free one, else a new one. `None` when the
    /// pool cannot hold a set that large; the request then owns its stores.
    /// Also returns the sets dropped to make room.
    pub(crate) fn acquire(
        &self,
        key: &str,
        capacity: u64,
        bytes: u64,
        provider: ProviderBinding,
    ) -> (Option<u64>, Vec<PooledKvSet>) {
        let mut state = self.0.lock().unwrap();
        // One loaded model executes on one Provider; a set never moves.
        if state
            .provider
            .as_ref()
            .is_some_and(|held| *held != provider)
        {
            return (None, Vec::new());
        }
        state.provider = Some(provider);
        state.tick += 1;
        let tick = state.tick;
        let mut dropped = Vec::new();
        if let Some(current) = state.in_use.remove(key) {
            if current.capacity == capacity {
                let id = current.id;
                state.in_use.insert(
                    key.into(),
                    PooledKvSet {
                        last_used: tick,
                        ..current
                    },
                );
                return (Some(id), dropped);
            }
            // The key's earlier request is over: its set is free again.
            state.free.push(current);
        }
        let reused = state
            .free
            .iter()
            .enumerate()
            .filter(|(_, set)| set.capacity == capacity)
            .max_by_key(|(_, set)| set.last_used)
            .map(|(index, _)| index);
        let set = match reused {
            Some(index) => state.free.remove(index),
            None => {
                if bytes > state.max_bytes {
                    dropped.extend(state.enforce_bounds());
                    return (None, dropped);
                }
                while state.held_bytes() + bytes > state.max_bytes {
                    let Some(set) = state.pop_least_recently_used(|_| true) else {
                        return (None, dropped);
                    };
                    dropped.push(set);
                }
                state.next_set += 1;
                PooledKvSet {
                    id: state.next_set,
                    capacity,
                    bytes,
                    last_used: tick,
                }
            }
        };
        let id = set.id;
        state.in_use.insert(
            key.into(),
            PooledKvSet {
                last_used: tick,
                ..set
            },
        );
        dropped.extend(state.enforce_bounds());
        (Some(id), dropped)
    }

    /// The set the request keyed `key` uses: its id and capacity.
    pub(crate) fn current(&self, key: &str) -> Option<KvStoreSpec> {
        self.0
            .lock()
            .unwrap()
            .in_use
            .get(key)
            .map(|set| KvStoreSpec {
                capacity: set.capacity,
                pool_set: Some(set.id),
            })
    }

    /// Returns the request's set to the pool, then drops sets beyond the
    /// bounds, least recently used first.
    pub(crate) fn release(&self, key: &str) -> Vec<PooledKvSet> {
        let mut state = self.0.lock().unwrap();
        if let Some(set) = state.in_use.remove(key) {
            state.free.push(set);
        }
        state.enforce_bounds()
    }

    /// The Provider the sets' stores live in.
    pub(crate) fn provider(&self) -> Option<ProviderBinding> {
        self.0.lock().unwrap().provider.clone()
    }

    /// Drops every free set (admission pressure).
    pub(crate) fn drain_free(&self) -> Vec<PooledKvSet> {
        std::mem::take(&mut self.0.lock().unwrap().free)
    }

    /// Bytes held by every set, free or in use.
    pub(crate) fn held_bytes(&self) -> u64 {
        self.0.lock().unwrap().held_bytes()
    }

    /// How many sets are free (tests).
    #[cfg(test)]
    pub(crate) fn free_sets(&self) -> usize {
        self.0.lock().unwrap().free.len()
    }
}

impl KvStorePoolState {
    fn held_bytes(&self) -> u64 {
        self.free
            .iter()
            .chain(self.in_use.values())
            .map(|set| set.bytes)
            .sum()
    }

    fn pop_least_recently_used(
        &mut self,
        mut eligible: impl FnMut(&PooledKvSet) -> bool,
    ) -> Option<PooledKvSet> {
        let index = self
            .free
            .iter()
            .enumerate()
            .filter(|(_, set)| eligible(set))
            .min_by_key(|(_, set)| set.last_used)
            .map(|(index, _)| index)?;
        Some(self.free.remove(index))
    }

    /// Drops free sets until each capacity bucket holds at most
    /// [`KV_POOL_SETS_PER_BUCKET`] sets and the pool at most its bytes.
    fn enforce_bounds(&mut self) -> Vec<PooledKvSet> {
        let mut dropped = Vec::new();
        loop {
            let crowded = self.free.iter().map(|set| set.capacity).find(|capacity| {
                self.free
                    .iter()
                    .chain(self.in_use.values())
                    .filter(|set| set.capacity == *capacity)
                    .count()
                    > KV_POOL_SETS_PER_BUCKET
            });
            let next = match crowded {
                Some(capacity) => self.pop_least_recently_used(|set| set.capacity == capacity),
                None if self.held_bytes() > self.max_bytes => {
                    self.pop_least_recently_used(|_| true)
                }
                None => None,
            };
            match next {
                Some(set) => dropped.push(set),
                None => return dropped,
            }
        }
    }
}

#[cfg(test)]
mod tests;
