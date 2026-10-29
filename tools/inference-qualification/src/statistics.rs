//! Aggregate statistics over raw campaign samples (A04): median,
//! percentiles, dispersion and a noise-aware comparison that refuses to
//! promote a speedup the samples cannot support.

use serde::{Deserialize, Serialize};

/// Scale factor turning a median absolute deviation into a consistent
/// estimate of a normal distribution's standard deviation.
const MAD_NORMAL_CONSISTENCY: f64 = 1.4826;

/// Half-width of the robust noise band, in scaled MADs, that two medians
/// must clear before a difference counts as a measured change.
const NOISE_BAND_SCALED_MADS: f64 = 2.0;

/// Order statistics and dispersion of one metric's raw samples. The raw
/// samples themselves are retained separately by the campaign report;
/// this summary is derived from them and never replaces them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SampleSummary {
    pub count: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub standard_deviation: f64,
    pub median_absolute_deviation: f64,
    /// `median_absolute_deviation / median`, the scale-free dispersion
    /// used to flag results too noisy to compare. `None` when the median
    /// is zero.
    pub relative_dispersion: Option<f64>,
}

/// Summarizes `samples`, ignoring nothing: a non-finite sample is a
/// measurement bug, so it makes the whole summary unavailable rather than
/// being silently dropped. Returns `None` for an empty slice.
pub fn summarize(samples: &[f64]) -> Option<SampleSummary> {
    if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let count = sorted.len();
    let mean = sorted.iter().sum::<f64>() / count as f64;
    let variance = if count > 1 {
        sorted
            .iter()
            .map(|sample| (sample - mean).powi(2))
            .sum::<f64>()
            / (count - 1) as f64
    } else {
        0.0
    };
    let median = median_of_sorted(&sorted);
    let mut deviations: Vec<f64> = sorted
        .iter()
        .map(|sample| (sample - median).abs())
        .collect();
    deviations.sort_by(f64::total_cmp);
    let median_absolute_deviation = median_of_sorted(&deviations);
    Some(SampleSummary {
        count,
        min: sorted[0],
        max: sorted[count - 1],
        mean,
        median,
        p50: nearest_rank_percentile(&sorted, 50.0),
        p95: nearest_rank_percentile(&sorted, 95.0),
        p99: nearest_rank_percentile(&sorted, 99.0),
        standard_deviation: variance.sqrt(),
        median_absolute_deviation,
        relative_dispersion: (median != 0.0).then(|| median_absolute_deviation / median.abs()),
    })
}

fn median_of_sorted(sorted: &[f64]) -> f64 {
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

/// Nearest-rank percentile: always one of the observed samples, never an
/// interpolated value no run actually produced.
fn nearest_rank_percentile(sorted: &[f64], percentile: f64) -> f64 {
    let rank = ((percentile / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// Whether lower or higher values of a metric are better.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricDirection {
    LowerIsBetter,
    HigherIsBetter,
}

/// Outcome of comparing a candidate against a baseline for one metric.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum ComparisonVerdict {
    /// The candidate's median clears the combined noise band in the
    /// better direction. `ratio` is baseline/candidate for lower-is-better
    /// metrics and candidate/baseline otherwise, so `> 1.0` is a gain.
    Improvement { ratio: f64 },
    /// The candidate's median clears the noise band in the worse
    /// direction.
    Regression { ratio: f64 },
    /// The medians differ by less than the measurement noise; no change
    /// may be claimed either way.
    WithinNoise,
    /// Too few samples on at least one side to make any claim.
    InsufficientSamples {
        baseline: usize,
        candidate: usize,
        minimum: usize,
    },
}

impl ComparisonVerdict {
    /// Only a measured improvement may be promoted as a speedup.
    pub fn supports_promotion(&self) -> bool {
        matches!(self, Self::Improvement { .. })
    }
}

/// Compares two metric summaries. Requires `minimum_samples` on both
/// sides, then requires the medians to differ by more than
/// [`NOISE_BAND_SCALED_MADS`] scaled MADs of each side combined; anything
/// less is reported as noise rather than as a speedup or regression.
pub fn compare(
    baseline: &SampleSummary,
    candidate: &SampleSummary,
    direction: MetricDirection,
    minimum_samples: usize,
) -> ComparisonVerdict {
    if baseline.count < minimum_samples || candidate.count < minimum_samples {
        return ComparisonVerdict::InsufficientSamples {
            baseline: baseline.count,
            candidate: candidate.count,
            minimum: minimum_samples,
        };
    }
    let noise_band = NOISE_BAND_SCALED_MADS
        * MAD_NORMAL_CONSISTENCY
        * (baseline.median_absolute_deviation + candidate.median_absolute_deviation);
    let difference = candidate.median - baseline.median;
    if difference.abs() <= noise_band || baseline.median == 0.0 || candidate.median == 0.0 {
        return ComparisonVerdict::WithinNoise;
    }
    let (ratio, improved) = match direction {
        MetricDirection::LowerIsBetter => (baseline.median / candidate.median, difference < 0.0),
        MetricDirection::HigherIsBetter => (candidate.median / baseline.median, difference > 0.0),
    };
    if improved {
        ComparisonVerdict::Improvement { ratio }
    } else {
        ComparisonVerdict::Regression { ratio }
    }
}

#[cfg(test)]
#[path = "statistics_tests.rs"]
mod tests;
