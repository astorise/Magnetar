//! Mixed Nemotron graph production through the real Component engine.

use super::*;
use std::collections::BTreeMap;

#[derive(serde::Deserialize)]
struct ReferenceTensor {
    shape: Vec<u64>,
    data: Vec<f32>,
}

#[derive(serde::Deserialize)]
struct Reference {
    ids: Vec<u32>,
    weights: BTreeMap<String, ReferenceTensor>,
    prefix: ReferenceTensor,
    prefix_state: BTreeMap<String, ReferenceTensor>,
    full: ReferenceTensor,
    steps: Vec<BTreeMap<String, ReferenceTensor>>,
}

fn close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() <= 4e-6 + 2e-6 * b.abs(),
            "{label}[{index}]: {a} != {b}"
        );
    }
}

#[test]
fn nemotron_component_builds_mixed_prefill_and_decode_graphs() {
    // Built from components/nemotron with cargo build --locked --release
    // --target wasm32-unknown-unknown, followed by wasm-tools component new.
    let bytes = include_bytes!("../../../fixtures/components/nemotron-hybrid.component.wasm");
    let digest = ComponentDigest::sha256(bytes);
    let manifest = include_str!(
        "../../../fixtures/components/qwen-real.component.wasm.magnetar-component.yaml"
    )
    .replace(
        "sha256:321b42b83773544e1ba553cc7196d3d742cd9ef1e2af3f04da72b66e402a730f",
        &digest.value,
    )
    .replace("magnetar.qwen.real", "magnetar.nemotron.hybrid-test")
    .replace("qwen-model-component", "nemotron-model-component")
    .replace("1.14.0", "1.15.0");
    register_inference_component_artifact(
        bytes.to_vec(),
        manifest.into_bytes(),
        &ComponentTrustStore::default().trust_digest(&digest.value),
    )
    .unwrap();

    let fixture = e2e_fixture().unwrap();
    let mut config = architecture_config_from_first_native_model_config(&fixture.config);
    config.hidden_size = 5;
    config.intermediate_size = 3;
    config.num_hidden_layers = 4;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.head_dim = 5;
    config.vocab_size = 24;
    config.rms_norm_eps = 1e-5;
    config.attention_bias = false;
    config.tie_word_embeddings = false;
    config.hidden_activation = Some("relu_squared".into());
    config.rope_layers = Some(vec![false; 4]);
    config.mixture_of_experts = Some(crate::MixtureOfExpertsConfig {
        experts: 4,
        experts_per_token: 2,
        expert_intermediate_size: 3,
        shared_expert_intermediate_size: Some(6),
        normalize_top_k: true,
    });
    config.moe_layers = Some(vec![false, false, false, true]);
    config.expert_routing = Some(crate::ExpertRoutingConfig {
        scoring: crate::ExpertScoring::Sigmoid,
        groups: Some(crate::ExpertGroups {
            count: 1,
            per_token: 1,
            score: crate::ExpertGroupScore::TopTwoSum,
        }),
        routed_scaling_factor: 2.5,
        selection_bias: true,
        normalization: Some(crate::ExpertNormalization::Always),
        shared_expert_gated: false,
    });
    config.decoder_extensions.mamba2_hybrid = Some(crate::Mamba2HybridConfig {
        layers: vec![
            crate::HybridDecoderBlock::Mamba2,
            crate::HybridDecoderBlock::Attention,
            crate::HybridDecoderBlock::DenseMlp,
            crate::HybridDecoderBlock::RoutedMoe,
        ],
        heads: 4,
        head_dim: 2,
        groups: 2,
        state_dim: 3,
        conv_kernel: 3,
        epsilon: 1e-5,
        dt_min: 0.0,
        dt_max: None,
        conv_bias: true,
        linear_bias: false,
    });
    config.validate().unwrap();
    let weights: BTreeMap<String, Vec<u64>> = [
        ("token_embedding", vec![24, 5]),
        ("final_norm", vec![5]),
        ("lm_head", vec![5, 24]),
        ("layers.0.norm", vec![5]),
        ("layers.0.mamba2.in_proj", vec![5, 32]),
        ("layers.0.mamba2.conv", vec![20, 3]),
        ("layers.0.mamba2.dt_bias", vec![4]),
        ("layers.0.mamba2.a_log", vec![4]),
        ("layers.0.mamba2.d", vec![4]),
        ("layers.0.mamba2.norm", vec![8]),
        ("layers.0.mamba2.out_proj", vec![8, 5]),
        ("layers.0.mamba2.conv_bias", vec![20]),
        ("layers.1.norm", vec![5]),
        ("layers.1.attention.q_proj", vec![5, 5]),
        ("layers.1.attention.k_proj", vec![5, 5]),
        ("layers.1.attention.v_proj", vec![5, 5]),
        ("layers.1.attention.o_proj", vec![5, 5]),
        ("layers.2.norm", vec![5]),
        ("layers.2.mlp.in", vec![5, 3]),
        ("layers.2.mlp.out", vec![3, 5]),
        ("layers.3.norm", vec![5]),
        ("layers.3.moe.router", vec![5, 4]),
        ("layers.3.moe.experts_in", vec![5, 12]),
        ("layers.3.moe.experts_out", vec![3, 20]),
        ("layers.3.moe.selection_bias", vec![4]),
        ("layers.3.moe.shared.in", vec![5, 6]),
        ("layers.3.moe.shared.out", vec![6, 5]),
    ]
    .into_iter()
    .map(|(name, shape)| (name.to_string(), shape))
    .collect();
    let runtime = named_component_runtime(&digest).unwrap();
    let build = |config: &ModelArchitectureConfig, weights: BTreeMap<String, Vec<u64>>, tokens| {
        build_first_native_graphs_with_runtime(
            &runtime.manager,
            &runtime.capability,
            &runtime.model_config_capability,
            runtime.definition,
            config,
            weights,
            &fixture.identity,
            tokens,
        )
    };
    let (graphs, _, _) = build(&config, weights.clone(), 4).unwrap();
    for graph in [&graphs.prefill, &graphs.decode] {
        graph.validate(&initial_operator_catalog()).unwrap();
        for operator in ["mamba2-mixer", "attention", "moe-experts"] {
            assert!(
                graph
                    .nodes
                    .values()
                    .any(|node| node.operator.name() == operator),
                "{operator}"
            );
        }
        let slots: Vec<_> = graph
            .edges
            .values()
            .filter_map(|edge| edge.recurrent_state.as_ref())
            .map(|state| state.slot.as_str())
            .collect();
        assert_eq!(slots.iter().filter(|slot| **slot == "conv").count(), 2);
        assert_eq!(slots.iter().filter(|slot| **slot == "ssm").count(), 2);
    }
    let reference: Reference = serde_json::from_str(include_str!(
        "../../../fixtures/nemotron-hybrid-reference.json"
    ))
    .unwrap();
    assert_eq!(weights.len(), reference.weights.len());
    for (name, tensor) in &reference.weights {
        assert_eq!(weights.get(name), Some(&tensor.shape), "{name}");
    }
    let mut execution_fixture = fixture.clone();
    execution_fixture.config.architecture =
        first_native_architecture_metadata(5, 4, 1, 1, 5, 3, 24, 32);
    let mut execution_runtime = build_runtime_trusting_fixture(&fixture);
    let (instance, _) = load_fixture_instance(&fixture, &mut execution_runtime).unwrap();
    let revision = require_ready_first_native_instance(&execution_runtime, &instance)
        .unwrap()
        .status()
        .mutation_version;
    let provider = ProviderBinding::new(crate::REFERENCE_CPU_PROVIDER_NAME);
    let executor = resolve_kernel_execution_provider(&execution_runtime, &provider).unwrap();
    let cache = execution_runtime
        .create_kv_cache(KvCache::new(
            KvCacheId::new("nemotron-hybrid-reference").unwrap(),
            KvCacheScope::Session,
            KvCacheCompatibility::new(
                GenerationModelReference::ModelInstance(instance.clone()),
                TokenizerId::new("tokenizer").unwrap(),
            ),
            KvCacheLayoutMetadata::contiguous(4, 1, 5, 32, ComputeDType::Float32),
        ))
        .unwrap();
    let bindings = |ids: &[u32]| {
        let mut result: BTreeMap<_, _> = reference
            .weights
            .iter()
            .map(|(name, tensor)| {
                (
                    TensorEdgeId::new(format!("weight.{name}")),
                    HostTensor::new(tensor.shape.clone(), tensor.data.clone()).unwrap(),
                )
            })
            .collect();
        result.insert(
            TensorEdgeId::new("input.token_ids"),
            HostTensor::new(
                [ids.len() as u64],
                ids.iter().map(|&id| id as f32).collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        result
    };
    let mut plan = prepare_first_native_plan_for_graph(
        &execution_runtime,
        &graphs.prefill,
        &instance,
        revision,
        4,
        PreparedExecutionPlanGeneration::new(1),
        &provider,
    )
    .unwrap();
    let (_, outputs, pending, _) = execute_first_native_graph_materializing(
        &mut execution_runtime,
        &execution_fixture,
        &instance,
        &cache,
        &graphs.prefill,
        &mut plan,
        bindings(&reference.ids[..4]),
        None,
        Some(0),
        &mut Vec::new(),
        MaterializedEdges::All,
        None,
        None,
    )
    .unwrap();
    close(
        &outputs[&TensorEdgeId::new("logits")].data,
        &reference.prefix.data,
        "prefix logits",
    );
    for slot in ["conv", "ssm"] {
        let (edge, _) = graphs
            .prefill
            .edges
            .iter()
            .find(|(_, edge)| {
                edge.recurrent_state.as_ref().is_some_and(|state| {
                    state.slot == slot && state.behavior == GraphRecurrentStateBehavior::Output
                })
            })
            .unwrap();
        close(
            &outputs[edge].data,
            &reference.prefix_state[slot].data,
            slot,
        );
    }
    for (role, resource) in [("keys", &pending[&1].k), ("values", &pending[&1].v)] {
        close(
            &executor.read_tensor(resource).unwrap().data,
            &reference.prefix_state[role].data,
            role,
        );
    }
    let mut transaction = KvUpdateTransaction {
        provider_binding: provider.clone(),
        executor: executor.clone(),
        promoted: Vec::new(),
    };
    for (&layer, state) in &pending {
        transaction
            .promote_layer(&mut execution_runtime, &cache, layer, state)
            .unwrap();
    }
    let mut history = transaction.commit(&mut execution_runtime, &cache).unwrap();
    let mut decode_plan = prepare_first_native_plan_for_graph(
        &execution_runtime,
        &graphs.decode,
        &instance,
        revision,
        1,
        PreparedExecutionPlanGeneration::new(2),
        &provider,
    )
    .unwrap();
    for (index, expected) in reference.steps.iter().enumerate() {
        let position = index + 4;
        let (_, outputs, pending, _) = execute_first_native_graph_materializing(
            &mut execution_runtime,
            &execution_fixture,
            &instance,
            &cache,
            &graphs.decode,
            &mut decode_plan,
            bindings(&reference.ids[position..position + 1]),
            Some(&history),
            Some(position as u64),
            &mut Vec::new(),
            MaterializedEdges::All,
            None,
            None,
        )
        .unwrap();
        close(
            &outputs[&TensorEdgeId::new("logits")].data,
            &expected["logits"].data,
            "decode logits",
        );
        for slot in ["conv", "ssm"] {
            let (edge, _) = graphs
                .decode
                .edges
                .iter()
                .find(|(_, edge)| {
                    edge.recurrent_state.as_ref().is_some_and(|state| {
                        state.slot == slot && state.behavior == GraphRecurrentStateBehavior::Output
                    })
                })
                .unwrap();
            close(&outputs[edge].data, &expected[slot].data, slot);
        }
        for (role, resource) in [("keys", &pending[&1].k), ("values", &pending[&1].v)] {
            close(
                &executor.read_tensor(resource).unwrap().data,
                &expected[role].data,
                role,
            );
        }
        let mut transaction = KvUpdateTransaction {
            provider_binding: provider.clone(),
            executor: executor.clone(),
            promoted: Vec::new(),
        };
        for (&layer, state) in &pending {
            transaction
                .promote_layer(&mut execution_runtime, &cache, layer, state)
                .unwrap();
        }
        history = transaction.commit(&mut execution_runtime, &cache).unwrap();
    }
    // Closing the old session and opening a fresh one resets both recurrent
    // state and attention history. Its full prefill must match the separately
    // computed PyTorch full sequence, including the final state tensors.
    execution_runtime.release_kv_cache(&cache).unwrap();
    let reset_cache = execution_runtime
        .create_kv_cache(KvCache::new(
            KvCacheId::new("nemotron-hybrid-reset").unwrap(),
            KvCacheScope::Session,
            KvCacheCompatibility::new(
                GenerationModelReference::ModelInstance(instance.clone()),
                TokenizerId::new("tokenizer").unwrap(),
            ),
            KvCacheLayoutMetadata::contiguous(4, 1, 5, 32, ComputeDType::Float32),
        ))
        .unwrap();
    let (full_graphs, _, _) = build(&config, weights.clone(), 8).unwrap();
    let mut full_plan = prepare_first_native_plan_for_graph(
        &execution_runtime,
        &full_graphs.prefill,
        &instance,
        revision,
        8,
        PreparedExecutionPlanGeneration::new(3),
        &provider,
    )
    .unwrap();
    let (_, outputs, pending, _) = execute_first_native_graph_materializing(
        &mut execution_runtime,
        &execution_fixture,
        &instance,
        &reset_cache,
        &full_graphs.prefill,
        &mut full_plan,
        bindings(&reference.ids),
        None,
        Some(0),
        &mut Vec::new(),
        MaterializedEdges::All,
        None,
        None,
    )
    .unwrap();
    close(
        &outputs[&TensorEdgeId::new("logits")].data,
        &reference.full.data,
        "full prefill logits",
    );
    let final_state = reference.steps.last().unwrap();
    for slot in ["conv", "ssm"] {
        let (edge, _) = full_graphs
            .prefill
            .edges
            .iter()
            .find(|(_, edge)| {
                edge.recurrent_state.as_ref().is_some_and(|state| {
                    state.slot == slot && state.behavior == GraphRecurrentStateBehavior::Output
                })
            })
            .unwrap();
        close(&outputs[edge].data, &final_state[slot].data, slot);
    }
    for (role, resource) in [("keys", &pending[&1].k), ("values", &pending[&1].v)] {
        close(
            &executor.read_tensor(resource).unwrap().data,
            &final_state[role].data,
            role,
        );
    }
    execution_runtime.release_kv_cache(&reset_cache).unwrap();
    // Run the mixed graph as two prefill passes, like production chunking.
    // The KV stores contain all prior rows while the recurrent inputs come
    // from the previous pass's pending state, not the caller's bindings.
    let chunk_cache = execution_runtime
        .create_kv_cache(KvCache::new(
            KvCacheId::new("nemotron-hybrid-chunks").unwrap(),
            KvCacheScope::Session,
            KvCacheCompatibility::new(
                GenerationModelReference::ModelInstance(instance.clone()),
                TokenizerId::new("tokenizer").unwrap(),
            ),
            KvCacheLayoutMetadata::contiguous(4, 1, 5, 32, ComputeDType::Float32),
        ))
        .unwrap();
    let (chunk_graphs, _, _) = build(&config, weights.clone(), 4).unwrap();
    assert!(super::super::prefill_chunks::prefill_continues_history(
        &chunk_graphs.prefill
    ));
    assert!(
        super::super::prefill_chunks::recurrent_prefill_continues_history(&chunk_graphs.prefill)
    );
    let mut noncausal = chunk_graphs.prefill.clone();
    noncausal
        .nodes
        .values_mut()
        .find(|node| node.operator.name() == "attention")
        .unwrap()
        .attributes
        .insert("causal".into(), OperatorAttributeValue::Boolean(false));
    assert!(!super::super::prefill_chunks::recurrent_prefill_continues_history(&noncausal));
    let mut unknown = chunk_graphs.prefill.clone();
    unknown.nodes.values_mut().next().unwrap().operator =
        OperatorId::magnetar("unknown", 1, OperatorFamily::Tensor);
    assert!(!super::super::prefill_chunks::recurrent_prefill_continues_history(&unknown));
    let mut chunk_plan = prepare_first_native_plan_for_graph(
        &execution_runtime,
        &chunk_graphs.prefill,
        &instance,
        revision,
        4,
        PreparedExecutionPlanGeneration::new(4),
        &provider,
    )
    .unwrap();
    let mut chunks_history = None;
    for start in [0, 4] {
        let (_, outputs, pending, _) = execute_first_native_graph_materializing(
            &mut execution_runtime,
            &execution_fixture,
            &instance,
            &chunk_cache,
            &chunk_graphs.prefill,
            &mut chunk_plan,
            bindings(&reference.ids[start..start + 4]),
            chunks_history.as_ref(),
            Some(start as u64),
            &mut Vec::new(),
            MaterializedEdges::All,
            None,
            Some(KvStoreSpec::owned(8)),
        )
        .unwrap();
        assert_eq!(pending[&1].rows, Some((start + 4) as u64));
        close(
            &outputs[&TensorEdgeId::new("logits")].data,
            &reference.full.data[start * 24..(start + 4) * 24],
            "chunk logits",
        );
        if start == 4 {
            for slot in ["conv", "ssm"] {
                let (edge, _) = chunk_graphs
                    .prefill
                    .edges
                    .iter()
                    .find(|(_, edge)| {
                        edge.recurrent_state.as_ref().is_some_and(|state| {
                            state.slot == slot
                                && state.behavior == GraphRecurrentStateBehavior::Output
                        })
                    })
                    .unwrap();
                close(&outputs[edge].data, &final_state[slot].data, slot);
            }
            for (role, resource) in [("keys", &pending[&1].k), ("values", &pending[&1].v)] {
                close(
                    &executor.read_tensor(resource).unwrap().data,
                    &final_state[role].data,
                    role,
                );
            }
        }
        chunks_history = Some(pending);
    }
    super::super::recurrent_state::release_recurrent_state(
        &mut execution_runtime,
        &executor,
        &chunk_cache,
    );
    execution_runtime.release_kv_cache(&chunk_cache).unwrap();

    // Exercise the same Component and weights through production loading and
    // generation, including the non-divisible prompt's no-padding fallback.
    let (mut production, _) = super::super::tests::production_shaped_canonical_fixture();
    production.config = first_native_model_config_from_architecture_config(
        &config,
        32,
        production.manifest.architecture.family.clone(),
    );
    production.manifest.architecture_config = Some(config.clone());
    let template = production.manifest.tensors[0].clone();
    let mut offset = 0;
    let mut bytes_by_name = BTreeMap::new();
    production.manifest.tensors = reference
        .weights
        .iter()
        .map(|(name, value)| {
            let bytes: Vec<u8> = value.data.iter().flat_map(|v| v.to_le_bytes()).collect();
            let mut tensor = template.clone();
            tensor.name = name.clone();
            tensor.shape = value.shape.clone();
            tensor.offset_bytes = Some(offset);
            tensor.size_bytes = Some(bytes.len() as u64);
            tensor.digest = Some(ModelDigest::sha256(&bytes));
            offset += bytes.len() as u64;
            bytes_by_name.insert(name.clone(), bytes);
            tensor
        })
        .collect();
    let mut tokenizer = production.tokenizer.metadata().clone();
    tokenizer.vocabulary_size = 24;
    tokenizer.token_id_range = TokenIdRange::new(0, 23);
    tokenizer
        .special_tokens
        .retain(|token| token.kind == SpecialTokenKind::Eos);
    tokenizer.added_token_count = 1;
    production.tokenizer = FixtureTokenizer::new(tokenizer);
    let payload = super::super::tests::ProductionIntegrationPayloadSource { bytes_by_name };
    let trust =
        ModelTrustStore::default().trust_digest(production.manifest.id.digest.value.clone());
    let mut model = ProductionLoadedModel::load_with_component(
        production,
        &payload,
        trust,
        Arc::new(ReferenceCpuProvider::new()),
        Some(digest.clone()),
    )
    .unwrap();
    model.set_prefill_buckets(Vec::new());
    let generate = |model: &mut ProductionLoadedModel, ids: &[u32]| {
        let mut logits = Vec::new();
        let result = model
            .generate_streaming_with_logits(
                ProductionGenerationRequest {
                    prompt: PromptInput::TokenIds(ids.to_vec()),
                    parameters: GenerationParameters::greedy(),
                    stop_conditions: StopConditions::default(),
                    max_new_tokens: Some(4),
                    max_generation_millis: None,
                },
                None,
                &mut |_| std::ops::ControlFlow::Continue(()),
                &mut |step| logits.push(step.to_vec()),
            )
            .unwrap();
        (result.result.output.generated_token_ids, logits)
    };
    model.set_prefill_chunk(0);
    let unchunked_full = generate(&mut model, &reference.ids);
    let unchunked_short = generate(&mut model, &reference.ids[..7]);
    close(
        &unchunked_full.1[0],
        &reference.full.data[7 * 24..8 * 24],
        "production logits",
    );
    model.set_prefill_chunk(4);
    let (probe_graphs, _, _) = build(&config, weights.clone(), 4).unwrap();
    let probe = model.graph_cache.prepare(&model.runtime, probe_graphs, 4);
    assert!(
        super::super::prefill_chunks::recurrent_prefill_continues_history(&probe.prefill),
        "the optimized prefill must preserve the qualified continuation profile"
    );
    let chunked_full = generate(&mut model, &reference.ids);
    assert_eq!(chunked_full.0, unchunked_full.0);
    for (actual, expected) in chunked_full.1.iter().zip(&unchunked_full.1) {
        close(actual, expected, "production chunked generation");
    }
    assert_eq!(chunked_full.1.len(), unchunked_full.1.len());
    assert_eq!(model.prefill_chunk(), 4);
    assert_eq!(model.graph_cache.prompt_lengths(), vec![4]);
    let unaligned = generate(&mut model, &reference.ids[..7]);
    assert_eq!(unaligned.0, unchunked_short.0);
    for (actual, expected) in unaligned.1.iter().zip(&unchunked_short.1) {
        close(actual, expected, "production unaligned generation");
    }
    assert_eq!(unaligned.1.len(), unchunked_short.1.len());
    assert_eq!(model.prefill_chunk(), 0);
    assert_eq!(model.graph_cache.prompt_lengths(), vec![7]);

    let mut invalid = config.clone();
    invalid.rope_layers.as_mut().unwrap()[1] = true;
    assert!(build(&invalid, weights.clone(), 4).is_err());
    let mut missing = weights;
    missing.remove("layers.0.mamba2.conv_bias");
    assert!(build(&config, missing, 4).is_err());
}
