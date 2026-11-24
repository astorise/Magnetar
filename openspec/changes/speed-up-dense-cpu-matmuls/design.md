## Context

`NativeCpuExecutor` wraps a `ReferenceCpuExecutor`. The wrapped executor consults the optimized Provider's `PackedWeights` hooks:

- `run`, for a Kernel the Provider computes itself;
- `materialize`, for reads;
- `shape`, for resident shapes.

Only packed weights had a `run` path. The Runtime writes every unpacked weight through `ProviderExecutionApi::write_weight_tensor`, a write that is never repeated while the instance lives, so the Provider may keep any representation that reproduces every value.

## Decisions

### Compact storage, as on CUDA

A weight of at least `COMPACT_WEIGHT_MIN_ELEMENTS` (65,536) values whose every value has zero low 16 bits is stored as its high 16 bits. This is the CUDA Provider's rule. Biases and norm weights stay in f32.

A compact weight lives beside the packed ones. It is forgotten by every write that replaces its resource, as a packed weight is. It reads back unchanged through `materialize`, which counts the expansion like a packed weight's.

### Kernels by weight layout

Every weight of the Qwen2.5 Components is used input-major (`transpose_b` false): the input-major kernel matters most.

A matmul's weight is stored either output-major (`[n, k]` used with `transpose_b`, a checkpoint's linear weights) or input-major (`[k, n]`).

- **Output-major.** Each output value is a dot product over a stored row. Rows of `a` are taken four at a time, so one weight load serves four rows. A single row (a decode step) uses four independent accumulators, so consecutive fused multiply-adds do not wait on each other.
- **Input-major.** Each stored row scales and adds into a block of output columns, four rows per read and write of the block.
  - A weight narrower than 2,048 columns per thread is shared out by blocks of whole rows: each task reads its rows contiguously into a partial product, which the caller sums. Column blocks of a few hundred columns read a few hundred bytes per row between strides of the row width, at about 15 GB/s.
  - A wider one (the 151,936-column LM head) is shared out by column blocks (see the splits below).

Both are written once per weight encoding (f32, bfloat16), so a loop never dispatches per load. The pool shares output columns, in blocks that are multiples of 16.

### Stored orientation and decode kernels

A decode step reads each weight once, a few stored rows side by side. Locally (4 threads, bfloat16 weights streamed from memory), the bandwidth followed the stored row length more than the layout: rows of 1,792 bytes (a `[4864, 896]` down projection used input-major, or any 896-value row read one output at a time) reached 15 to 21 GB/s, rows of 9,728 bytes 34 to 46 GB/s. Short runs leave the hardware prefetchers behind.

- **Orientation.** A compact weight with more rows than columns is held transposed (its stored rows are its longer dimension) and used the other way round: the down projection becomes output-major over rows of 4,864 values, the `[151936, 896]` embedding table an input-major table whose lookup gathers a column. The choice depends on the stored shape alone, whichever way the graph uses the weight. Reads, shapes and lookups return the logical values.
- **Output-major, one row.** Sixteen outputs at a time (eight on AVX2), each activation vector loaded once for all of them. The down projection went from 20.5 GB/s (input-major) to 44 GB/s.
- **Input-major, one row.** Eight weight rows per pass over the output block, so one read and write of the block serves eight rows.
- **AVX-512 for one row too.** With these kernels the 16-lane variants were faster for a single row as well (gate/up 38.9 against 34.7 GB/s, down 44.0 against 37.9).
- **Splits.** An input-major weight narrower than 4,096 columns per thread is shared out by blocks of whole rows (gate/up on 2 threads used to split into 2,048 + 2,048 + 768 columns, one thread doing twice the other's share). A wider one (the LM head) takes one column block per thread, two when they stay wider than 16,384 columns: 2,000-column blocks read 4 KB runs between 300 KB strides (27 against 47 GB/s in a standalone measurement). Output-major tasks are whole groups of sixteen outputs, about eight tasks per thread.

### Thread count

The pool takes one thread per physical core the process may run on (Linux `thread_siblings_list` over the CPUs in `Cpus_allowed_list`, never more than `available_parallelism`), every logical CPU where the topology is unknown; `MAGNETAR_CPU_THREADS` overrides it. The benchmark runners have 2 cores with 2 hardware threads each. A decode step streams weights, which a core's second hardware thread does not speed up, and a pool thread polling between Kernels takes issue slots from its sibling. llama.cpp (Ollama's backend) defaults to the physical core count too (`n_threads = 2` on those runners). The CPU benchmark reruns Magnetar on every logical CPU of the same host after the campaign, so this default stays measured.

### Numerics

Products are accumulated in f32 with fused multiply-adds, in another order than the Reference CPU kernel's. Results agree to rounding, not bit for bit, as the packed kernels' do. Tests compare them within the existing tolerance (`2e-5` of the largest value).

### Polling workers

Pool workers that finish a job, and a caller waiting for the last tasks, poll for 500 microseconds before sleeping. A decode step runs a pooled Kernel every few hundred microseconds, and a sleeping worker's wake-up left the caller running most tasks of a sub-millisecond matmul alone.

## Risks / Trade-offs

- **Bfloat16 dot products.** The runners' Zen 4 and Zen 5 cores have AVX-512 bfloat16 dot products (`vdpbf16ps`), which llama.cpp may use. A decode step is bound by memory bandwidth, so they would mostly help prefills.
- **The transposed embedding table.** A lookup gathers one value per stored row (896 cache lines per token): about 0.05 ms per lookup locally, against 0.013 ms for a row.
- **Non-matmul Kernels.** Norms, RoPE and element-wise Kernels still run on the Reference CPU kernels. They are small against the matmuls at these sizes.
