//! Unit tests for the parent module.
//!
//! Shared mock executors and fixtures live in `first_native_runtime/tests.rs`.

use super::super::tests::*;
use super::*;

/// `generalize-first-native-provider-dispatch` P0 fix regression:
/// `KvUpdateTransaction::begin` resolves the Provider carried on
/// `FirstNativeExecutionKvState.provider` (what `execute_first_native_graph`
/// resolved and wrote this step's pending K/V resources under), not
/// Reference CPU unconditionally. Tests `begin`'s resolution directly
/// (rather than driving a full non-CPU generation step, which would need a
/// mock Provider implementing every Qwen graph Operator, not just this
/// contract) since `begin` is the entire fix -- everything downstream of it
/// already receives whichever `provider_binding` it resolves.
#[test]
fn kv_update_transaction_resolves_the_states_bound_provider() {
    let mock_provider = ProviderBinding::new("magnetar:provider/mock-kernel");
    let runtime = Runtime::builder()
        .register_provider(Arc::new(ReferenceCpuProvider::new()))
        .register_provider(Arc::new(MockKernelProvider::new()))
        .build()
        .expect("both providers register cleanly");

    let state = FirstNativeExecutionKvState {
        cache: KvCacheId::new("test-kv-transaction-binding-cache").unwrap(),
        compatibility: KvCacheCompatibility::new(
            GenerationModelReference::LoadedModelContext("qwen-test".into()),
            TokenizerId::new("qwen-test-tokenizer").unwrap(),
        ),
        layer_kv: LayerKvMap::new(),
        provider: Some(mock_provider.clone()),
    };

    let transaction = KvUpdateTransaction::begin(&runtime, &state).expect("mock provider resolves");
    assert_eq!(
        transaction.provider_binding, mock_provider,
        "must resolve the Provider bound on the KV state, not Reference CPU"
    );
}
