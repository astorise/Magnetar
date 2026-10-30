use super::*;

#[test]
fn metrics_derive_ttft_tpot_and_throughput_from_raw_offsets() {
    let trace = GenerationTrace {
        prompt_tokens: 7,
        token_offsets_micros: vec![100_000, 150_000, 210_000, 250_000],
        completed_micros: 500_000,
    };
    let metrics = trace.metrics();
    assert_eq!(metrics.prompt_tokens, 7);
    assert_eq!(metrics.generated_tokens, 4);
    assert_eq!(metrics.time_to_first_token_ms, Some(100.0));
    assert_eq!(metrics.inter_token_latencies_ms, vec![50.0, 60.0, 40.0]);
    assert_eq!(metrics.time_per_output_token_ms, Some(50.0));
    assert_eq!(metrics.end_to_end_latency_ms, 500.0);
    assert_eq!(metrics.request_throughput_tokens_per_second, Some(8.0));
    assert_eq!(metrics.decode_throughput_tokens_per_second, Some(20.0));
}

#[test]
fn a_single_token_has_ttft_but_no_decode_metrics() {
    let metrics = GenerationTrace {
        prompt_tokens: 1,
        token_offsets_micros: vec![20_000],
        completed_micros: 40_000,
    }
    .metrics();
    assert_eq!(metrics.time_to_first_token_ms, Some(20.0));
    assert_eq!(metrics.time_per_output_token_ms, None);
    assert_eq!(metrics.decode_throughput_tokens_per_second, None);
    assert!(metrics.inter_token_latencies_ms.is_empty());
}

#[test]
fn no_tokens_reports_only_end_to_end_latency() {
    let metrics = GenerationTrace {
        prompt_tokens: 3,
        token_offsets_micros: vec![],
        completed_micros: 1_000,
    }
    .metrics();
    assert_eq!(metrics.time_to_first_token_ms, None);
    assert_eq!(metrics.request_throughput_tokens_per_second, None);
    assert_eq!(metrics.end_to_end_latency_ms, 1.0);
}

#[test]
fn recorder_offsets_are_monotonic_and_bounded_by_completion() {
    let mut recorder = TraceRecorder::start();
    recorder.token();
    recorder.token();
    let trace = recorder.finish(2);
    assert_eq!(trace.token_offsets_micros.len(), 2);
    assert!(trace.token_offsets_micros[0] <= trace.token_offsets_micros[1]);
    assert!(trace.token_offsets_micros[1] <= trace.completed_micros);
}

#[test]
fn latency_slope_measures_per_token_cost_growth() {
    assert_eq!(latency_slope(&[]), None);
    assert_eq!(latency_slope(&[5.0]), None);
    assert_eq!(latency_slope(&[5.0, 5.0, 5.0]), Some(0.0));
    assert!((latency_slope(&[10.0, 12.0, 14.0, 16.0]).unwrap() - 2.0).abs() < 1e-12);
    let metrics = GenerationTrace {
        prompt_tokens: 1,
        token_offsets_micros: vec![0, 10_000, 30_000, 60_000],
        completed_micros: 60_000,
    }
    .metrics();
    assert_eq!(metrics.inter_token_latencies_ms, vec![10.0, 20.0, 30.0]);
    assert!((metrics.inter_token_latency_slope_ms_per_token.unwrap() - 10.0).abs() < 1e-12);
}
