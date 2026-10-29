use super::*;

#[test]
fn summary_reports_order_statistics_and_dispersion() {
    let summary = summarize(&[5.0, 1.0, 3.0, 2.0, 4.0]).unwrap();
    assert_eq!(summary.count, 5);
    assert_eq!(summary.min, 1.0);
    assert_eq!(summary.max, 5.0);
    assert_eq!(summary.mean, 3.0);
    assert_eq!(summary.median, 3.0);
    assert_eq!(summary.p50, 3.0);
    assert_eq!(summary.p95, 5.0);
    assert_eq!(summary.p99, 5.0);
    assert_eq!(summary.median_absolute_deviation, 1.0);
    assert!((summary.standard_deviation - 2.5_f64.sqrt()).abs() < 1e-12);
    assert_eq!(summary.relative_dispersion, Some(1.0 / 3.0));
}

#[test]
fn even_sample_counts_average_the_middle_pair_but_percentiles_stay_observed() {
    let summary = summarize(&[4.0, 1.0, 3.0, 2.0]).unwrap();
    assert_eq!(summary.median, 2.5);
    assert_eq!(summary.p50, 2.0);
    assert_eq!(summary.p95, 4.0);
}

#[test]
fn single_sample_has_zero_dispersion() {
    let summary = summarize(&[7.0]).unwrap();
    assert_eq!(summary.standard_deviation, 0.0);
    assert_eq!(summary.median_absolute_deviation, 0.0);
    assert_eq!(summary.p99, 7.0);
}

#[test]
fn empty_or_non_finite_samples_produce_no_summary() {
    assert!(summarize(&[]).is_none());
    assert!(summarize(&[1.0, f64::NAN]).is_none());
    assert!(summarize(&[1.0, f64::INFINITY]).is_none());
}

#[test]
fn zero_median_has_no_relative_dispersion() {
    assert_eq!(
        summarize(&[0.0, 0.0, 0.0]).unwrap().relative_dispersion,
        None
    );
}

fn tight(center: f64) -> SampleSummary {
    summarize(&[center - 0.1, center, center, center, center + 0.1]).unwrap()
}

#[test]
fn clear_latency_reduction_is_an_improvement() {
    let verdict = compare(&tight(10.0), &tight(5.0), MetricDirection::LowerIsBetter, 5);
    assert_eq!(verdict, ComparisonVerdict::Improvement { ratio: 2.0 });
    assert!(verdict.supports_promotion());
}

#[test]
fn clear_throughput_drop_is_a_regression() {
    let verdict = compare(
        &tight(10.0),
        &tight(5.0),
        MetricDirection::HigherIsBetter,
        5,
    );
    assert_eq!(verdict, ComparisonVerdict::Regression { ratio: 0.5 });
    assert!(!verdict.supports_promotion());
}

#[test]
fn overlapping_noise_is_never_promoted_as_a_speedup() {
    let baseline = summarize(&[8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let candidate = summarize(&[7.5, 8.5, 9.5, 10.5, 11.5]).unwrap();
    let verdict = compare(&baseline, &candidate, MetricDirection::LowerIsBetter, 5);
    assert_eq!(verdict, ComparisonVerdict::WithinNoise);
    assert!(!verdict.supports_promotion());
}

#[test]
fn too_few_repetitions_cannot_support_any_claim() {
    let few = summarize(&[1.0, 1.0]).unwrap();
    let verdict = compare(&tight(10.0), &few, MetricDirection::LowerIsBetter, 5);
    assert_eq!(
        verdict,
        ComparisonVerdict::InsufficientSamples {
            baseline: 5,
            candidate: 2,
            minimum: 5
        }
    );
    assert!(!verdict.supports_promotion());
}
