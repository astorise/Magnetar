use super::*;

fn checkpoint() -> CheckpointIdentity {
    CheckpointIdentity {
        name: "example".into(),
        revision: Some("abc".into()),
        weights_digest: Some("sha256:00".into()),
        digest_verified: true,
    }
}

fn logits(step: usize) -> Vec<f32> {
    (0..64)
        .map(|token| ((token * 7 + step * 13) % 64) as f32 / 4.0 - 8.0)
        .collect()
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
        .unwrap()
        .0 as u32
}

fn reference(provider: &str, perturb: impl Fn(usize, &mut Vec<f32>)) -> LogitReference {
    let mut recorder = LogitRecorder::default();
    let mut tokens = Vec::new();
    for step in 0..3 {
        let mut step_logits = logits(step);
        perturb(step, &mut step_logits);
        recorder.observe(&step_logits);
        tokens.push(argmax(&step_logits));
    }
    recorder.finish(checkpoint(), provider, "f32", vec![1, 2, 3], tokens)
}

#[test]
fn snapshot_keeps_ranked_top_logits_and_a_full_digest() {
    let step_logits = logits(0);
    let snapshot = LogitStepSnapshot::capture(0, &step_logits, argmax(&step_logits));
    assert_eq!(snapshot.vocab_size, 64);
    assert_eq!(snapshot.top.len(), SNAPSHOT_TOP_K);
    assert_eq!(snapshot.top[0].token_id, snapshot.sampled_token_id);
    assert!(
        snapshot
            .top
            .windows(2)
            .all(|pair| pair[0].logit >= pair[1].logit)
    );
    assert!(snapshot.full_digest.starts_with("sha256:"));
    assert_eq!(snapshot.non_finite_count, 0);

    let tiny = LogitStepSnapshot::capture(1, &[1.0, f32::NAN, 3.0], 2);
    assert_eq!(tiny.top.len(), 2, "non-finite logits are never ranked");
    assert_eq!(tiny.top[0].token_id, 2);
    assert_eq!(tiny.non_finite_count, 1);
}

#[test]
fn identical_runs_compare_bit_identical() {
    let comparison = compare_logit_references(
        &reference("reference-cpu", |_, _| {}),
        &reference("reference-cpu", |_, _| {}),
        &NumericalProfile::f32_cross_provider(),
    );
    assert!(comparison.passed(), "{:?}", comparison.failures);
    assert_eq!(comparison.steps.len(), 3);
    assert!(comparison.steps.iter().all(|step| step.bit_identical));
    assert_eq!(comparison.max_abs_difference, 0.0);
}

#[test]
fn small_cross_provider_rounding_passes_within_tolerance() {
    let candidate = reference("cuda", |_, values| {
        for value in values.iter_mut() {
            *value += 1e-3;
        }
    });
    let comparison = compare_logit_references(
        &reference("reference-cpu", |_, _| {}),
        &candidate,
        &NumericalProfile::f32_cross_provider(),
    );
    assert!(comparison.passed(), "{:?}", comparison.failures);
    assert!(comparison.steps.iter().all(|step| !step.bit_identical));
    assert!(comparison.max_abs_difference > 0.0 && comparison.max_abs_difference < 2e-3);
}

#[test]
fn drift_beyond_tolerance_fails_despite_identical_tokens_and_keeps_evidence() {
    // The argmax token gains 0.5 at step 1: still the argmax, so the
    // sampled tokens (and decoded text) are identical -- but the logits are
    // not within tolerance.
    let candidate = reference("cuda", |step, values| {
        if step == 1 {
            let top = argmax(values) as usize;
            values[top] += 0.5;
        }
    });
    let baseline = reference("reference-cpu", |_, _| {});
    assert_eq!(baseline.generated_token_ids, candidate.generated_token_ids);
    let comparison = compare_logit_references(
        &baseline,
        &candidate,
        &NumericalProfile::f32_cross_provider(),
    );
    assert!(!comparison.passed());
    assert_eq!(comparison.failures.len(), 1, "{:?}", comparison.failures);
    assert!(comparison.failures[0].contains("step 1"));
    assert_eq!(
        comparison.steps.len(),
        3,
        "every step's evidence is retained"
    );
    assert_eq!(comparison.steps[1].max_abs_difference, 0.5);
    assert_eq!(comparison.steps[0].max_abs_difference, 0.0);
    let evidence = serde_json::to_value(&comparison).unwrap();
    assert_eq!(evidence["steps"][1]["max_abs_difference"], 0.5);
}

#[test]
fn token_divergence_missing_ids_and_identity_mismatches_fail() {
    let baseline = reference("reference-cpu", |_, _| {});

    let mut swapped = reference("cuda", |_, _| {});
    swapped.steps[0].sampled_token_id += 1;
    swapped.steps[2].top.truncate(1);
    swapped.steps[2].non_finite_count = 2;
    swapped.prompt_token_ids = vec![9];
    swapped.checkpoint.weights_digest = Some("sha256:ff".into());
    swapped.steps.push(swapped.steps[2].clone());

    let comparison =
        compare_logit_references(&baseline, &swapped, &NumericalProfile::f32_cross_provider());
    let failures = comparison.failures.join("\n");
    assert!(
        failures.contains("different checkpoint digests"),
        "{failures}"
    );
    assert!(failures.contains("different prompt token ids"));
    assert!(failures.contains("step count differs"));
    assert!(failures.contains("step 0: sampled token"));
    assert!(failures.contains("step 2: reference top tokens"));
    assert!(failures.contains("non-finite"));
    assert!(!comparison.steps[2].missing_token_ids.is_empty());
}

#[test]
fn references_round_trip_through_json() {
    let reference = reference("reference-cpu", |_, _| {});
    assert_eq!(reference.schema_version, LOGIT_REFERENCE_SCHEMA_VERSION);
    let decoded: LogitReference =
        serde_json::from_str(&serde_json::to_string(&reference).unwrap()).unwrap();
    assert_eq!(decoded, reference);
}

#[test]
fn unpaired_observed_steps_are_visible() {
    let mut recorder = LogitRecorder::default();
    recorder.observe(&[1.0, 2.0]);
    recorder.observe(&[2.0, 1.0]);
    let reference = recorder.finish(checkpoint(), "cpu", "f32", vec![], vec![1]);
    assert_eq!(reference.steps[1].sampled_token_id, u32::MAX);
}
