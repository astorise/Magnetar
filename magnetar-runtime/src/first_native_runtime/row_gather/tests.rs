//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than Runtime implementation source.

use super::*;
use crate::production_model_ingestion::MappedPayload;
use crate::qwen4_exp_semantics_tests::{PleCase, fixture};
use crate::{ProductionIngestionError, RowTableDType};
use magnetar_provider_api::HostTensor;
use std::sync::Mutex;

const TABLE_NAME: &str = "layers.0.ple.table";

/// A mapping over memory that records the prefetch hints it receives.
#[derive(Debug)]
struct MemoryPayload {
    bytes: Vec<u8>,
    hints: Mutex<Vec<Vec<(u64, u64)>>>,
}

impl MappedPayload for MemoryPayload {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<(), ProductionIngestionError> {
        let start = offset as usize;
        let slice = self.bytes.get(start..start + out.len()).ok_or_else(|| {
            ProductionIngestionError::PayloadOutOfBounds {
                identity: "memory".into(),
            }
        })?;
        out.copy_from_slice(slice);
        Ok(())
    }

    fn prefetch(&self, spans: &[(u64, u64)]) {
        self.hints.lock().unwrap().push(spans.to_vec());
    }
}

/// The fixture's table as a row table over memory, and its mapping.
fn row_table(case: &PleCase) -> (BTreeMap<String, RowTableHandle>, Arc<MemoryPayload>) {
    let bytes = case
        .table
        .data
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let payload = Arc::new(MemoryPayload {
        bytes,
        hints: Mutex::new(Vec::new()),
    });
    let table = RowTable::new(
        payload.clone(),
        case.table.shape[0],
        case.table.shape[1],
        RowTableDType::F32,
    )
    .unwrap();
    (
        BTreeMap::from([(TABLE_NAME.to_string(), RowTableHandle(Arc::new(table)))]),
        payload,
    )
}

fn attribute(value: u64) -> OperatorAttributeValue {
    OperatorAttributeValue::Integer(value as i64)
}

/// A graph of one `ngram-embedding` node reading `table_edge`.
fn graph_reading(case: &PleCase, table_edge: &str) -> ExecutionGraph {
    let tokens = case.token_ids.len() as u64;
    let heads = ((case.ngram_size - 1) * case.heads_per_ngram) as u64;
    let context = case.ngram_size as u64 - 1;
    let edge = |name: &str, shape: &[u64]| {
        TensorEdge::new(
            TensorEdgeId::new(name),
            f32_tensor_descriptor_from_shape(shape),
        )
    };
    let node = ExecutionNode::new(
        ExecutionNodeId::new("ngram"),
        OperatorId::magnetar("ngram-embedding", 1, OperatorFamily::Tensor),
    )
    .with_attribute("ngram_size", attribute(case.ngram_size as u64))
    .with_attribute("heads_per_ngram", attribute(case.heads_per_ngram as u64))
    .with_attribute("vocab_size", attribute(case.vocab_size))
    .with_attribute("vocab_base", attribute(case.vocab_base))
    .with_attribute("layer_index", attribute(case.layer_index))
    .with_attribute("seed", attribute(case.seed))
    .with_attribute("eos_token_id", attribute(case.eos_token_id as u64))
    .with_input(TensorEdgeId::new("input.token_ids"))
    .with_input(TensorEdgeId::new("state.context"))
    .with_input(TensorEdgeId::new(table_edge))
    .with_output(TensorEdgeId::new("embeddings"));
    ExecutionGraph::new(
        ExecutionGraphId::new("row-gather"),
        ExecutionGraphPhase::Prefill,
    )
    .with_edge(edge("input.token_ids", &[tokens]))
    .with_edge(edge("state.context", &[context]))
    .with_edge(edge(table_edge, &case.table.shape))
    .with_edge(edge("embeddings", &[tokens, heads * case.table.shape[1]]))
    .with_node(node)
}

fn hash_of(case: &PleCase) -> NgramEmbedding {
    NgramEmbedding {
        ngram_size: case.ngram_size,
        heads_per_ngram: case.heads_per_ngram,
        vocab_size: case.vocab_size,
        vocab_base: case.vocab_base,
        layer_index: case.layer_index,
        seed: case.seed,
        eos_token_id: case.eos_token_id,
        gathered_rows: false,
    }
}

fn step_of(case: &PleCase, context: Vec<i64>) -> StepTokens {
    StepTokens {
        tokens: case.token_ids.iter().map(|&token| token as i64).collect(),
        context,
    }
}

/// The rows a node reads are the ones the operator's own hash names, copied
/// out of the table in (token, head) order, and every one is hinted to the
/// mapping before any is read.
#[test]
fn a_node_gathers_the_rows_its_tokens_name() {
    let fixture = fixture();
    let case = &fixture.ngram_ple;
    let (tables, payload) = row_table(case);
    let graph = graph_reading(case, &format!("weight.{TABLE_NAME}"));
    let hash = hash_of(case);
    let context = vec![case.eos_token_id; hash.context_len()];
    let gather = RowGather::plan(&graph, &tables, |_, _| Ok(step_of(case, context.clone())))
        .unwrap()
        .expect("the node reads a row table");
    let node = ExecutionNodeId::new("ngram");
    assert!(gather.reads(&node));

    gather.prefetch();
    let hints = payload.hints.lock().unwrap().clone();
    assert_eq!(hints.len(), 1, "one hint carries every row of the step");
    assert!(!hints[0].is_empty());

    let rows = gather.gather(&node).unwrap();
    let head_dim = case.table.shape[1] as usize;
    let tokens: Vec<i64> = case.token_ids.iter().map(|&token| token as i64).collect();
    let mut expected = Vec::new();
    for row_ids in hash.row_ids(&tokens, &context) {
        for row in row_ids {
            let from = row as usize * head_dim;
            expected.extend_from_slice(&case.table.data[from..from + head_dim]);
        }
    }
    assert_eq!(
        rows.shape,
        [(tokens.len() * hash.heads()) as u64, head_dim as u64]
    );
    assert_eq!(rows.data, expected);

    // Hinted pages hold every row read, nothing more than their pages.
    for row in hash.row_ids(&tokens, &context).into_iter().flatten() {
        let start = row * head_dim as u64 * 4;
        assert!(
            hints[0]
                .iter()
                .any(|&(from, length)| from <= start && start < from + length),
            "row {row} lies in a hinted span"
        );
    }
}

/// The gathered form of the node is the node with its table input read as
/// rows; nothing else about it changes.
#[test]
fn the_gathered_form_only_sets_the_gathered_rows_attribute() {
    let fixture = fixture();
    let case = &fixture.ngram_ple;
    let graph = graph_reading(case, &format!("weight.{TABLE_NAME}"));
    let node = &graph.nodes[&ExecutionNodeId::new("ngram")];
    let gathered = gathered_form(node);
    assert_eq!(
        gathered.attributes.get(GATHERED_ROWS),
        Some(&OperatorAttributeValue::Boolean(true))
    );
    let mut without = gathered.attributes.clone();
    without.remove(GATHERED_ROWS);
    assert_eq!(without, node.attributes);
    assert_eq!(gathered.inputs, node.inputs);
    assert_eq!(gathered.outputs, node.outputs);
    assert!(
        !node.attributes.contains_key(GATHERED_ROWS),
        "the graph's own node is untouched"
    );
}

/// A step with no row table is the dense one, and nobody is asked for its
/// tokens: an instance without row tables, or a node whose table is a
/// weight, plans nothing.
#[test]
fn a_step_without_a_row_table_plans_nothing() {
    let fixture = fixture();
    let case = &fixture.ngram_ple;
    let (tables, _) = row_table(case);
    let ask = |_: &ExecutionNode, _: &NgramEmbedding| -> Result<StepTokens, InferenceApiError> {
        panic!("a dense step describes no tokens")
    };

    let graph = graph_reading(case, &format!("weight.{TABLE_NAME}"));
    assert!(
        RowGather::plan(&graph, &BTreeMap::new(), ask)
            .unwrap()
            .is_none()
    );

    let other = graph_reading(case, "weight.layers.0.ple.key");
    assert!(RowGather::plan(&other, &tables, ask).unwrap().is_none());

    let not_a_weight = graph_reading(case, "edge.table");
    assert!(
        RowGather::plan(&not_a_weight, &tables, ask)
            .unwrap()
            .is_none()
    );
}

/// A context shorter than the hash reaches back is a defect of the step's
/// description, refused before any row is named.
#[test]
fn a_context_of_the_wrong_length_is_refused() {
    let fixture = fixture();
    let case = &fixture.ngram_ple;
    let (tables, _) = row_table(case);
    let graph = graph_reading(case, &format!("weight.{TABLE_NAME}"));
    let error = RowGather::plan(&graph, &tables, |_, _| Ok(step_of(case, vec![0])))
        .err()
        .expect("the context is too short");
    assert!(
        matches!(error, InferenceApiError::GraphPlanningFailed { ref reason } if reason.contains("tokens of context")),
        "{error:?}"
    );
}

/// A row past the table is a failed read, not a wrapped one: the table's own
/// row count is the bound.
#[test]
fn a_table_shorter_than_the_hash_needs_fails_the_read() {
    let fixture = fixture();
    let case = &fixture.ngram_ple;
    // A table of two rows: the hash names rows far past them.
    let bytes: Vec<u8> = vec![0; 2 * case.table.shape[1] as usize * 4];
    let payload = Arc::new(MemoryPayload {
        bytes,
        hints: Mutex::new(Vec::new()),
    });
    let table = RowTable::new(payload, 2, case.table.shape[1], RowTableDType::F32).unwrap();
    let tables = BTreeMap::from([(TABLE_NAME.to_string(), RowTableHandle(Arc::new(table)))]);
    let graph = graph_reading(case, &format!("weight.{TABLE_NAME}"));
    let context = vec![case.eos_token_id; case.ngram_size - 1];
    let gather = RowGather::plan(&graph, &tables, |_, _| Ok(step_of(case, context.clone())))
        .unwrap()
        .unwrap();
    gather.prefetch();
    let error = gather
        .gather(&ExecutionNodeId::new("ngram"))
        .err()
        .expect("the rows are past the table");
    assert!(
        matches!(error, InferenceApiError::GenerationFailed { ref reason } if reason.contains("past the table")),
        "{error:?}"
    );
}

/// A step's tokens come from its host inputs: the ids, and the context as
/// the tokens before the step (held relative to the end token).
#[test]
fn a_steps_tokens_and_context_come_from_its_host_inputs() {
    let fixture = fixture();
    let case = &fixture.ngram_ple;
    let graph = graph_reading(case, &format!("weight.{TABLE_NAME}"));
    let node = &graph.nodes[&ExecutionNodeId::new("ngram")];
    let hash = hash_of(case);
    let provider: Arc<dyn ProviderExecutionApi> =
        Arc::new(crate::reference_cpu::ReferenceCpuExecutor::new());
    let ids = HostTensor::new([2], vec![5.0, 9.0]).unwrap();
    // Two tokens before the step: the end token plus 3 and the end token
    // plus 7 (the state holds each as its id minus the end token).
    let context = HostTensor::new([2], vec![3.0, 7.0]).unwrap();
    let initial = BTreeMap::from([
        (TensorEdgeId::new("input.token_ids"), ids.clone()),
        (TensorEdgeId::new("state.context"), context),
    ]);
    let step = step_tokens(node, &hash, &initial, &BTreeMap::new(), &provider).unwrap();
    assert_eq!(step.tokens, [5, 9]);
    assert_eq!(step.context, [case.eos_token_id + 3, case.eos_token_id + 7]);

    // Without the ids on the host (a captured step's resident input) the
    // step cannot name its rows.
    let only_state = BTreeMap::from([(
        TensorEdgeId::new("state.context"),
        HostTensor::new([2], vec![0.0, 0.0]).unwrap(),
    )]);
    let error = step_tokens(node, &hash, &only_state, &BTreeMap::new(), &provider)
        .err()
        .expect("the ids are not on the host");
    assert!(
        matches!(error, InferenceApiError::GraphPlanningFailed { ref reason } if reason.contains("token ids on the host")),
        "{error:?}"
    );

    // A state that is not a token is refused rather than hashed.
    let bad = BTreeMap::from([
        (TensorEdgeId::new("input.token_ids"), ids),
        (
            TensorEdgeId::new("state.context"),
            HostTensor::new([2], vec![0.5, 0.0]).unwrap(),
        ),
    ]);
    assert!(step_tokens(node, &hash, &bad, &BTreeMap::new(), &provider).is_err());
}
