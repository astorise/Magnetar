use magnetar_runtime::{FinishReason, GenerationMemoryEstimate, GenerationUsage, UsageReport};

#[test]
fn inference_api_usage_report_never_carries_raw_prompt_text() {
    let usage = GenerationUsage::new(3, 4, FinishReason::EosToken);
    let memory = GenerationMemoryEstimate::default();

    let report = UsageReport::from_generation(&usage, &memory, Some(true), Some(12));
    assert_eq!(report.prompt_token_count, 3);
    assert_eq!(report.generated_token_count, 4);
    assert_eq!(report.cache_hit, Some(true));
    assert!(!report.cancelled);
}
