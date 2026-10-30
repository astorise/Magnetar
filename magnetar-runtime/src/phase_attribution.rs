//! Redacted per-step phase attribution (A06).
//!
//! When enabled, the Runtime times the leaves of every generation step on
//! the host monotonic clock and keeps one record per step. A leaf records
//! only its exclusive time: a leaf timed inside another is subtracted from
//! the outer one, so their sum never exceeds the step's wall time; whatever the
//! Runtime did that no leaf covers is reported as `unattributed` instead of
//! being folded into a neighbour. Device-side timing is reported as
//! unavailable until a Provider exposes real device timestamps, so host
//! time spent waiting on the Device is not mistaken for Device time.
//!
//! Records carry only a closed vocabulary of phase and operator labels and
//! counters: no request, session, cache, resource or node identifier, no
//! prompt text, token id, tensor value or path can reach a record.
//! Attribution is disabled by default and costs one branch per leaf then.

use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// How many step records a ledger keeps before dropping the oldest.
pub const PHASE_ATTRIBUTION_STEP_CAPACITY: usize = 4096;

/// The step a record belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepKind {
    /// Everything a request does before its first Runtime step: session,
    /// tokenization, graphs and plans.
    RequestPreparation,
    Prefill,
    Decode,
}

/// One leaf of a generation step, timed exclusively of any leaf nested in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepPhase {
    /// Opening the request's inference session.
    SessionCreate,
    /// Tokenizing the prompt and resolving stop sequences.
    Tokenize,
    /// Preparing (or revalidating cached) execution plans.
    PlanPrepare,
    /// Validating the generation request against the Runtime.
    RequestValidate,
    /// Building the step's execution graphs through the model Component.
    GraphBuild,
    /// Re-checking that the Model Instance can still generate.
    InstanceCheck,
    /// Resolving or creating the step's KV state and discarding stale
    /// pending KV writes.
    KvStatePrepare,
    /// Per graph execution: execution order, executing Provider and the
    /// instance's weight bindings.
    GraphPrepare,
    /// Writing host-supplied graph inputs into Provider storage.
    InputUpload,
    /// Reading node inputs back from Provider storage, including any
    /// host download of a non-resident input.
    NodeInputResolve,
    /// One node's dispatch, labelled by operator, exclusive of the
    /// dispatch leaves nested in it.
    NodeDispatch,
    /// Inside a dispatch: resolving the node's plan binding, Kernel
    /// advertisement and dispatch plan.
    KernelResolve,
    /// Inside a dispatch: looking up the Operator specification.
    OperatorCatalog,
    /// Inside a dispatch: submitting the Kernel to the Provider and waiting
    /// for its completion, labelled by operator.
    KernelExecute,
    /// Appending the step's K/V row to the history (the concat dispatch),
    /// exclusive of the dispatch it contains.
    KvAppend,
    /// Copying or writing the step's K/V into its pending resource.
    KvPendingWrite,
    /// Writing a node output whose Kernel could not write it in place.
    EdgeWrite,
    /// Binding a node's outputs to their graph edges, exclusive of the KV
    /// and edge-write leaves nested in it.
    NodeOutputBind,
    /// Downloading the graph's live edges to the host at the end of the step.
    OutputMaterialize,
    /// Copying the last row of logits out of the materialized output.
    LogitsExtract,
    /// Recording the step's pending KV state for commit.
    KvStateStore,
    /// Handing the step's logits to a caller-supplied observer.
    LogitsObserve,
    /// Publishing the step's Runtime observations (per-node causal events,
    /// KV commits, token events) and validating its execution evidence.
    StepObservation,
    /// Building the step's execution evidence.
    EvidenceBuild,
    /// Sampling the next token.
    Sampling,
    /// Promoting the step's pending KV state and committing the token.
    Commit,
}

/// Where a duration was measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClockDomain {
    HostMonotonic,
}

/// Whether a counter was measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CounterAvailability {
    Measured,
    Unavailable,
}

/// The operator labels a record may carry. Any operator name outside this
/// list is reported as `other`, so a Component cannot smuggle text into a
/// record through an operator name.
const OPERATOR_LABELS: &[&str] = &[
    "add",
    "attention",
    "concat",
    "embedding",
    "matmul",
    "mul",
    "residual-add",
    "rmsnorm",
    "rope",
    "silu",
    "softmax",
    "split",
];

/// Maps an operator name to its closed-vocabulary label.
pub fn operator_label(name: &str) -> &'static str {
    OPERATOR_LABELS
        .iter()
        .find(|label| **label == name)
        .copied()
        .unwrap_or("other")
}

/// Accumulated host time of one `(phase, operator)` leaf within a step.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PhaseTotal {
    pub count: u64,
    pub host_micros: u64,
}

impl PhaseTotal {
    fn add(&mut self, other: &PhaseTotal) {
        self.count += other.count;
        self.host_micros += other.host_micros;
    }
}

/// The key of one leaf: its phase and, for dispatches, its operator label.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PhaseKey {
    pub phase: StepPhase,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operator: Option<String>,
}

/// One generation step's attribution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StepAttribution {
    /// Position of this step in the ledger since it was enabled. Not a
    /// request or token identifier.
    pub ordinal: u64,
    pub kind: StepKind,
    pub wall_micros: u64,
    /// Leaf totals, sorted by key.
    pub phases: Vec<(PhaseKey, PhaseTotal)>,
}

impl StepAttribution {
    /// Host time covered by leaves.
    pub fn attributed_micros(&self) -> u64 {
        self.phases.iter().map(|(_, total)| total.host_micros).sum()
    }

    /// Wall time no leaf covers.
    pub fn unattributed_micros(&self) -> u64 {
        self.wall_micros.saturating_sub(self.attributed_micros())
    }
}

#[derive(Debug)]
struct OpenStep {
    kind: StepKind,
    started_at: Instant,
    phases: BTreeMap<PhaseKey, PhaseTotal>,
    /// Host time recorded so far in this step; lets a leaf that contains
    /// other leaves record only its own, exclusive time.
    attributed_micros: u64,
}

/// The Runtime-owned ledger. Disabled until [`PhaseLedger::enable`].
#[derive(Debug, Default)]
pub struct PhaseLedger {
    enabled: bool,
    open: Option<OpenStep>,
    steps: VecDeque<StepAttribution>,
    next_ordinal: u64,
    dropped_steps: u64,
}

/// A started leaf timer; `None` while attribution is disabled.
#[derive(Clone, Copy, Debug)]
pub struct PhaseClock(Option<(Instant, u64)>);

impl PhaseLedger {
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Stops recording and forgets every record and any open step.
    pub fn disable(&mut self) {
        *self = Self::default();
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Starts a leaf timer.
    pub fn clock(&self) -> PhaseClock {
        PhaseClock(
            self.open
                .as_ref()
                .filter(|_| self.enabled)
                .map(|open| (Instant::now(), open.attributed_micros)),
        )
    }

    /// Opens a step. A step left open by a failed predecessor is discarded.
    pub fn begin_step(&mut self, kind: StepKind) {
        if self.enabled {
            self.open = Some(OpenStep {
                kind,
                started_at: Instant::now(),
                phases: BTreeMap::new(),
                attributed_micros: 0,
            });
        }
    }

    /// Records the time since `clock` started under `phase`.
    pub fn record(&mut self, clock: PhaseClock, phase: StepPhase) {
        self.record_key(
            clock,
            PhaseKey {
                phase,
                operator: None,
            },
        );
    }

    /// Records the time since `clock` started under `phase`, labelled by
    /// the operator's closed-vocabulary label.
    pub fn record_operator(&mut self, clock: PhaseClock, phase: StepPhase, operator: &str) {
        self.record_key(
            clock,
            PhaseKey {
                phase,
                operator: Some(operator_label(operator).to_string()),
            },
        );
    }

    fn record_key(&mut self, clock: PhaseClock, key: PhaseKey) {
        let (Some((started_at, attributed_before)), Some(open)) = (clock.0, self.open.as_mut())
        else {
            return;
        };
        // Leaves recorded while this one was running are excluded, so a
        // leaf may contain others without any time being counted twice.
        let nested = open.attributed_micros.saturating_sub(attributed_before);
        let exclusive = micros(started_at.elapsed()).saturating_sub(nested);
        open.attributed_micros += exclusive;
        let total = open.phases.entry(key).or_default();
        total.count += 1;
        total.host_micros += exclusive;
    }

    /// Closes the open step, if any, and keeps its record.
    pub fn end_step(&mut self) {
        let Some(open) = self.open.take() else {
            return;
        };
        if self.steps.len() == PHASE_ATTRIBUTION_STEP_CAPACITY {
            self.steps.pop_front();
            self.dropped_steps += 1;
        }
        self.steps.push_back(StepAttribution {
            ordinal: self.next_ordinal,
            kind: open.kind,
            wall_micros: micros(open.started_at.elapsed()),
            phases: open.phases.into_iter().collect(),
        });
        self.next_ordinal += 1;
    }

    /// Drains the closed steps into a report. Recording stays enabled.
    pub fn take_report(&mut self) -> PhaseAttributionReport {
        let steps: Vec<StepAttribution> = self.steps.drain(..).collect();
        let dropped_steps = std::mem::take(&mut self.dropped_steps);
        PhaseAttributionReport::from_steps(steps, dropped_steps)
    }
}

fn micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

/// Totals for every step of one kind.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StepKindSummary {
    pub kind: StepKind,
    pub steps: u64,
    pub wall_micros: u64,
    pub unattributed_micros: u64,
    pub phases: Vec<(PhaseKey, PhaseTotal)>,
}

/// What [`PhaseLedger::take_report`] returns.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhaseAttributionReport {
    pub clock_domain: ClockDomain,
    /// Every duration here is host time. Host time of a leaf that waits on
    /// the Device includes that wait.
    pub host_time: CounterAvailability,
    pub device_time: CounterAvailability,
    pub host_device_overlap: CounterAvailability,
    /// Steps dropped because the ledger was full.
    pub dropped_steps: u64,
    pub summaries: Vec<StepKindSummary>,
    pub steps: Vec<StepAttribution>,
}

impl PhaseAttributionReport {
    fn from_steps(steps: Vec<StepAttribution>, dropped_steps: u64) -> Self {
        let mut by_kind: BTreeMap<StepKind, (u64, u64, u64, BTreeMap<PhaseKey, PhaseTotal>)> =
            BTreeMap::new();
        for step in &steps {
            let (count, wall, unattributed, phases) = by_kind.entry(step.kind).or_default();
            *count += 1;
            *wall += step.wall_micros;
            *unattributed += step.unattributed_micros();
            for (key, total) in &step.phases {
                phases.entry(key.clone()).or_default().add(total);
            }
        }
        Self {
            clock_domain: ClockDomain::HostMonotonic,
            host_time: CounterAvailability::Measured,
            device_time: CounterAvailability::Unavailable,
            host_device_overlap: CounterAvailability::Unavailable,
            dropped_steps,
            summaries: by_kind
                .into_iter()
                .map(
                    |(kind, (steps, wall_micros, unattributed_micros, phases))| StepKindSummary {
                        kind,
                        steps,
                        wall_micros,
                        unattributed_micros,
                        phases: phases.into_iter().collect(),
                    },
                )
                .collect(),
            steps,
        }
    }

    pub fn summary(&self, kind: StepKind) -> Option<&StepKindSummary> {
        self.summaries.iter().find(|summary| summary.kind == kind)
    }
}

#[cfg(test)]
#[path = "phase_attribution_tests.rs"]
mod tests;
