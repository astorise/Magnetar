//! Per-request generation timing (A01): wall-clock instants recorded at
//! the public streaming boundary, from which TTFT, inter-token latency,
//! TPOT and throughput are derived. Raw instants are kept so every derived
//! number can be recomputed from the report.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Records one generation request as it streams.
#[derive(Debug)]
pub struct TraceRecorder {
    started_at: Instant,
    token_offsets: Vec<Duration>,
}

impl TraceRecorder {
    /// Starts timing immediately before the request is submitted.
    pub fn start() -> Self {
        Self {
            started_at: Instant::now(),
            token_offsets: Vec::new(),
        }
    }

    /// Records that one token was delivered now.
    pub fn token(&mut self) {
        self.token_offsets.push(self.started_at.elapsed());
    }

    /// Stops timing once the request has fully completed (including its
    /// final event and resource release).
    pub fn finish(self, prompt_tokens: usize) -> GenerationTrace {
        GenerationTrace {
            prompt_tokens,
            token_offsets_micros: self
                .token_offsets
                .iter()
                .map(|offset| duration_micros(*offset))
                .collect(),
            completed_micros: duration_micros(self.started_at.elapsed()),
        }
    }
}

fn duration_micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

/// The raw timing of one request: microsecond offsets from submission of
/// every delivered token and of completion.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationTrace {
    pub prompt_tokens: usize,
    pub token_offsets_micros: Vec<u64>,
    pub completed_micros: u64,
}

/// Metrics derived from one [`GenerationTrace`]. Latencies are in
/// milliseconds, throughput in tokens per second.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationMetrics {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    /// Submission to first delivered token: prompt processing plus the
    /// first sampled token, including any per-request session setup.
    pub time_to_first_token_ms: Option<f64>,
    /// Mean time per output token after the first:
    /// `(last - first) / (generated - 1)`.
    pub time_per_output_token_ms: Option<f64>,
    /// Gaps between consecutive delivered tokens (inter-token latency).
    pub inter_token_latencies_ms: Vec<f64>,
    /// Submission to completion.
    pub end_to_end_latency_ms: f64,
    /// Generated tokens over end-to-end latency.
    pub request_throughput_tokens_per_second: Option<f64>,
    /// Tokens after the first over the decode span.
    pub decode_throughput_tokens_per_second: Option<f64>,
    /// Least-squares slope of inter-token latency against token position:
    /// how much each further decode step costs than the one before.
    /// Near zero when a step's cost does not depend on history length.
    pub inter_token_latency_slope_ms_per_token: Option<f64>,
}

/// Least-squares slope of `values` against their index. `None` for fewer
/// than two values.
pub fn latency_slope(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let n = values.len() as f64;
    let mean_x = (n - 1.0) / 2.0;
    let mean_y = values.iter().sum::<f64>() / n;
    let (covariance, variance) =
        values
            .iter()
            .enumerate()
            .fold((0.0, 0.0), |(covariance, variance), (index, value)| {
                let dx = index as f64 - mean_x;
                (covariance + dx * (value - mean_y), variance + dx * dx)
            });
    Some(covariance / variance)
}

impl GenerationTrace {
    pub fn metrics(&self) -> GenerationMetrics {
        let to_ms = |micros: u64| micros as f64 / 1_000.0;
        let offsets = &self.token_offsets_micros;
        let generated_tokens = offsets.len();
        let inter_token_latencies_ms: Vec<f64> = offsets
            .windows(2)
            .map(|pair| to_ms(pair[1].saturating_sub(pair[0])))
            .collect();
        let decode_span_ms = match (offsets.first(), offsets.last()) {
            (Some(first), Some(last)) if generated_tokens > 1 => Some(to_ms(last - first)),
            _ => None,
        };
        let end_to_end_latency_ms = to_ms(self.completed_micros);
        GenerationMetrics {
            prompt_tokens: self.prompt_tokens,
            generated_tokens,
            time_to_first_token_ms: offsets.first().copied().map(to_ms),
            time_per_output_token_ms: decode_span_ms
                .map(|span| span / (generated_tokens - 1) as f64),
            inter_token_latency_slope_ms_per_token: latency_slope(&inter_token_latencies_ms),
            inter_token_latencies_ms,
            end_to_end_latency_ms,
            request_throughput_tokens_per_second: (generated_tokens > 0
                && end_to_end_latency_ms > 0.0)
                .then(|| generated_tokens as f64 / (end_to_end_latency_ms / 1_000.0)),
            decode_throughput_tokens_per_second: decode_span_ms
                .filter(|span| *span > 0.0)
                .map(|span| (generated_tokens - 1) as f64 / (span / 1_000.0)),
        }
    }
}

#[cfg(test)]
#[path = "trace_tests.rs"]
mod tests;
