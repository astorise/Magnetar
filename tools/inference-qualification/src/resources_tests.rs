use super::*;

#[test]
fn parses_procfs_status_io_and_stat() {
    let status = "Name:\tmagnetar\nVmHWM:\t  2048 kB\nVmRSS:\t  1024 kB\n";
    assert_eq!(parse_status_kib(status, "VmHWM"), Some(2048));
    assert_eq!(parse_status_kib(status, "VmRSS"), Some(1024));
    assert_eq!(parse_status_kib(status, "VmSwap"), None);

    let io = "rchar: 100\nwchar: 5\nsyscr: 1\nsyscw: 1\nread_bytes: 40\nwrite_bytes: 0\n";
    assert_eq!(
        parse_process_io(io),
        Some(ProcessIo {
            rchar: 100,
            read_bytes: 40
        })
    );
    assert_eq!(parse_process_io("rchar: 1\n"), None);

    // Field 2 deliberately contains spaces and a parenthesis.
    let stat = "42 (my (odd) cmd) R 1 2 3 4 5 6 7 8 9 10 250 50 0 0 20 0";
    assert_eq!(parse_cpu_ticks(stat), Some(300));
    assert_eq!(parse_cpu_ticks("no parenthesis"), None);
}

#[test]
fn interval_probe_measures_real_reads_on_linux() {
    let probe = IntervalProbe::begin();
    let bytes = fs::read("/proc/self/status").unwrap_or_default();
    let usage = probe.end();
    assert!(
        usage.device_memory_peak_bytes.value().is_none(),
        "device memory is never fabricated by the process probe"
    );
    if cfg!(target_os = "linux") {
        assert!(
            usage
                .peak_resident_bytes
                .value()
                .is_some_and(|peak| *peak > 0)
        );
        assert!(
            usage
                .read_bytes
                .value()
                .is_some_and(|read| *read >= bytes.len() as u64)
        );
        assert!(usage.cpu_utilization_cores.value().is_some());
    } else {
        assert!(usage.peak_resident_bytes.value().is_none());
    }
}

#[test]
fn not_measured_carries_its_reason() {
    let measurement: Measurement<u64> = Measurement::not_measured("no telemetry");
    assert_eq!(measurement.value(), None);
    assert_eq!(
        serde_json::to_value(&measurement).unwrap(),
        serde_json::json!({"status": "not_measured", "reason": "no telemetry"})
    );
}
