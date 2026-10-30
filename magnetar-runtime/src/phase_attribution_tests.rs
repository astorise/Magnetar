use super::*;

fn spin(duration: Duration) {
    let started = Instant::now();
    while started.elapsed() < duration {}
}

#[test]
fn a_disabled_ledger_records_nothing() {
    let mut ledger = PhaseLedger::default();
    ledger.begin_step(StepKind::Decode);
    let clock = ledger.clock();
    ledger.record(clock, StepPhase::Sampling);
    ledger.end_step();
    let report = ledger.take_report();
    assert!(report.steps.is_empty());
    assert!(report.summaries.is_empty());
}

#[test]
fn leaves_partition_the_step_without_double_counting() {
    let mut ledger = PhaseLedger::default();
    ledger.enable();
    ledger.begin_step(StepKind::Decode);
    for _ in 0..3 {
        let clock = ledger.clock();
        spin(Duration::from_millis(2));
        ledger.record_operator(clock, StepPhase::NodeDispatch, "matmul");
    }
    let clock = ledger.clock();
    spin(Duration::from_millis(2));
    ledger.record(clock, StepPhase::OutputMaterialize);
    spin(Duration::from_millis(2));
    ledger.end_step();

    let report = ledger.take_report();
    assert_eq!(report.clock_domain, ClockDomain::HostMonotonic);
    assert_eq!(report.device_time, CounterAvailability::Unavailable);
    assert_eq!(report.host_device_overlap, CounterAvailability::Unavailable);
    let step = &report.steps[0];
    let dispatch = &step.phases[0];
    assert_eq!(dispatch.0.phase, StepPhase::NodeDispatch);
    assert_eq!(dispatch.0.operator.as_deref(), Some("matmul"));
    assert_eq!(dispatch.1.count, 3);
    assert!(dispatch.1.host_micros >= 6_000);
    assert!(step.attributed_micros() <= step.wall_micros);
    assert!(step.unattributed_micros() >= 2_000);
    assert_eq!(
        step.attributed_micros() + step.unattributed_micros(),
        step.wall_micros
    );
    let summary = report.summary(StepKind::Decode).unwrap();
    assert_eq!(summary.steps, 1);
    assert_eq!(summary.unattributed_micros, step.unattributed_micros());
}

#[test]
fn unknown_operator_names_are_reported_as_other() {
    let sentinel = "MAGNETAR-SENTINEL-7f3a";
    assert_eq!(operator_label("rmsnorm"), "rmsnorm");
    assert_eq!(operator_label(sentinel), "other");
    let mut ledger = PhaseLedger::default();
    ledger.enable();
    ledger.begin_step(StepKind::Prefill);
    let clock = ledger.clock();
    ledger.record_operator(clock, StepPhase::NodeDispatch, sentinel);
    ledger.end_step();
    let json = serde_json::to_string(&ledger.take_report()).unwrap();
    assert!(!json.contains(sentinel));
    assert!(json.contains("\"other\""));
}

#[test]
fn a_full_ledger_drops_its_oldest_steps_and_counts_them() {
    let mut ledger = PhaseLedger::default();
    ledger.enable();
    for _ in 0..PHASE_ATTRIBUTION_STEP_CAPACITY + 2 {
        ledger.begin_step(StepKind::Decode);
        ledger.end_step();
    }
    let report = ledger.take_report();
    assert_eq!(report.steps.len(), PHASE_ATTRIBUTION_STEP_CAPACITY);
    assert_eq!(report.dropped_steps, 2);
    assert_eq!(report.steps[0].ordinal, 2);
    assert!(ledger.take_report().steps.is_empty());
}

#[test]
fn disabling_forgets_records_and_an_open_step() {
    let mut ledger = PhaseLedger::default();
    ledger.enable();
    ledger.begin_step(StepKind::Decode);
    ledger.end_step();
    ledger.begin_step(StepKind::Decode);
    ledger.disable();
    assert!(!ledger.is_enabled());
    ledger.end_step();
    assert!(ledger.take_report().steps.is_empty());
}

#[test]
fn a_nested_leaf_is_subtracted_from_the_leaf_containing_it() {
    let mut ledger = PhaseLedger::default();
    ledger.enable();
    ledger.begin_step(StepKind::Decode);
    let outer = ledger.clock();
    spin(Duration::from_millis(3));
    let inner = ledger.clock();
    spin(Duration::from_millis(6));
    ledger.record_operator(inner, StepPhase::KernelExecute, "matmul");
    ledger.record_operator(outer, StepPhase::NodeDispatch, "matmul");
    ledger.end_step();
    let step = &ledger.take_report().steps[0];
    let time = |phase| {
        step.phases
            .iter()
            .find(|(key, _)| key.phase == phase)
            .unwrap()
            .1
            .host_micros
    };
    assert!(time(StepPhase::KernelExecute) >= 6_000);
    let dispatch = time(StepPhase::NodeDispatch);
    assert!((3_000..6_000).contains(&dispatch), "{dispatch}");
    assert!(step.attributed_micros() <= step.wall_micros);
}
