//! Process resource accounting for a measured interval (A01): peak RSS,
//! I/O bytes and CPU utilization, read from Linux procfs. Anything the
//! host cannot measure is recorded as [`Measurement::NotMeasured`] with
//! its reason, never as zero.

use std::{fs, time::Instant};

use serde::{Deserialize, Serialize};

/// Linux's userspace clock-tick rate for `/proc/<pid>/stat` CPU times.
/// `USER_HZ` is fixed at 100 by the kernel ABI on every architecture
/// Magnetar targets, independent of the kernel's internal `HZ`.
const LINUX_USER_HZ: f64 = 100.0;

/// A measured value or the explicit reason it is absent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Measurement<T> {
    Measured { value: T },
    NotMeasured { reason: String },
}

impl<T> Measurement<T> {
    pub fn not_measured(reason: impl Into<String>) -> Self {
        Self::NotMeasured {
            reason: reason.into(),
        }
    }

    pub fn value(&self) -> Option<&T> {
        match self {
            Self::Measured { value } => Some(value),
            Self::NotMeasured { .. } => None,
        }
    }
}

/// Whether a peak-RSS reading covers only the measured interval or the
/// whole process lifetime (when the kernel refused to reset the peak).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PeakScope {
    Interval,
    ProcessLifetime,
}

/// Resource usage attributed to one measured interval.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub peak_resident_bytes: Measurement<u64>,
    pub peak_scope: PeakScope,
    pub resident_bytes_at_end: Measurement<u64>,
    /// Bytes the process read through any read syscall (`rchar`),
    /// including page-cache hits.
    pub read_bytes: Measurement<u64>,
    /// Bytes actually fetched from storage (`read_bytes`).
    pub storage_read_bytes: Measurement<u64>,
    /// Process CPU time over wall time; `1.0` is one fully busy core.
    pub cpu_utilization_cores: Measurement<f64>,
    /// Device memory. Supplied by the caller from Provider telemetry when
    /// one exists; the process-level probe cannot see it.
    pub device_memory_peak_bytes: Measurement<u64>,
}

/// Measures one interval. Call [`IntervalProbe::begin`] immediately
/// before the work and [`IntervalProbe::end`] immediately after.
#[derive(Debug)]
pub struct IntervalProbe {
    started_at: Instant,
    peak_scope: PeakScope,
    start_io: Option<ProcessIo>,
    start_cpu_ticks: Option<u64>,
}

impl IntervalProbe {
    pub fn begin() -> Self {
        // Writing "5" to clear_refs resets VmHWM to the current RSS
        // (Linux >= 4.0), making the next peak reading interval-scoped.
        let peak_scope = if fs::write("/proc/self/clear_refs", "5").is_ok() {
            PeakScope::Interval
        } else {
            PeakScope::ProcessLifetime
        };
        Self {
            started_at: Instant::now(),
            peak_scope,
            start_io: read_process_io(),
            start_cpu_ticks: read_cpu_ticks(),
        }
    }

    pub fn end(self) -> ResourceUsage {
        let wall_seconds = self.started_at.elapsed().as_secs_f64();
        let status = fs::read_to_string("/proc/self/status").ok();
        let status_kib = |key: &str| {
            status
                .as_deref()
                .and_then(|status| parse_status_kib(status, key))
                .map_or_else(
                    || Measurement::not_measured(format!("/proc/self/status {key} unavailable")),
                    |kib| Measurement::Measured { value: kib * 1024 },
                )
        };
        let io_delta = |select: fn(&ProcessIo) -> u64, name: &str| match (
            self.start_io.as_ref(),
            read_process_io().as_ref(),
        ) {
            (Some(start), Some(end)) => Measurement::Measured {
                value: select(end).saturating_sub(select(start)),
            },
            _ => Measurement::not_measured(format!("/proc/self/io {name} unavailable")),
        };
        let cpu_utilization_cores = match (self.start_cpu_ticks, read_cpu_ticks()) {
            (Some(start), Some(end)) if wall_seconds > 0.0 => Measurement::Measured {
                value: (end.saturating_sub(start) as f64 / LINUX_USER_HZ) / wall_seconds,
            },
            _ => Measurement::not_measured("/proc/self/stat CPU times unavailable"),
        };
        ResourceUsage {
            peak_resident_bytes: status_kib("VmHWM"),
            peak_scope: self.peak_scope,
            resident_bytes_at_end: status_kib("VmRSS"),
            read_bytes: io_delta(|io| io.rchar, "rchar"),
            storage_read_bytes: io_delta(|io| io.read_bytes, "read_bytes"),
            cpu_utilization_cores,
            device_memory_peak_bytes: Measurement::not_measured(
                "no Provider device-memory telemetry is exposed at the public generation boundary",
            ),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ProcessIo {
    rchar: u64,
    read_bytes: u64,
}

fn read_process_io() -> Option<ProcessIo> {
    parse_process_io(&fs::read_to_string("/proc/self/io").ok()?)
}

fn parse_process_io(io: &str) -> Option<ProcessIo> {
    let field = |key: &str| {
        io.lines()
            .find_map(|line| line.strip_prefix(key)?.strip_prefix(':'))
            .and_then(|value| value.trim().parse().ok())
    };
    Some(ProcessIo {
        rchar: field("rchar")?,
        read_bytes: field("read_bytes")?,
    })
}

fn parse_status_kib(status: &str, key: &str) -> Option<u64> {
    status
        .lines()
        .find_map(|line| line.strip_prefix(key)?.strip_prefix(':'))
        .and_then(|value| value.trim().strip_suffix("kB"))
        .and_then(|value| value.trim().parse().ok())
}

fn read_cpu_ticks() -> Option<u64> {
    parse_cpu_ticks(&fs::read_to_string("/proc/self/stat").ok()?)
}

/// `utime + stime` (fields 14 and 15) from `/proc/<pid>/stat`. The
/// command name (field 2) may contain spaces, so fields are counted from
/// after its closing parenthesis.
fn parse_cpu_ticks(stat: &str) -> Option<u64> {
    let after_command = &stat[stat.rfind(')')? + 1..];
    let mut fields = after_command.split_whitespace().skip(11);
    let user: u64 = fields.next()?.parse().ok()?;
    let system: u64 = fields.next()?.parse().ok()?;
    Some(user + system)
}

#[cfg(test)]
#[path = "resources_tests.rs"]
mod tests;
