## 1. Memory Manager

- [x] 1.1 Split live and retired allocations; bound the retired history and observation logs; keep `allocations()` observable in id order; add `is_live_allocation`.
- [x] 1.2 Treat non-live allocations as already released when a KV cache is released.

## 2. Providers and KV lifecycle

- [x] 2.1 Decide pre-admission by residency ownership in Reference CPU, `providers/cpu` and `providers/cuda`.
- [x] 2.2 Release superseded KV generations with `release_admitted_tensor` on commit and abort.
- [x] 2.3 Release each finished request's KV state and forget its closed session and released caches on `ProductionLoadedModel`.

## 3. Verification

- [x] 3.1 Regression test: six requests on one resident instance keep every bookkeeping table bounded and no live allocation orphaned; unit tests for the live/retired split and bounds.
- [x] 3.2 Tiny-bundle probe: TPOT flat (6.4-8.0 ms) over 12 requests of 32 tokens, versus 7 to 45 ms before.
- [x] 3.3 GPU CI: the CUDA campaign's per-request TPOT stays flat on real hardware. Run 41051599580 (RTX 3060): TPOT 1310-1342 ms on all seven requests (was 1.48 -> 5.34 s), ITL growth within +/-1 ms/token, decode profile 1309-1340 ms per 8-position bucket over 64 tokens (slope 0.14 ms/token, was 18.8); no high-dispersion exception left in the campaign report.
