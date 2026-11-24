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

### Prefill tiles

Several rows of `a` (a bucketed prefill computes its live rows) are compute-bound, not bandwidth-bound. The decode loops, applied to them, loaded and stored the output once per row of `a` and broadcast each scale from memory: about 29 loads for 20 fused multiply-adds.

The prefill kernels are register tiles of up to six rows of `a`:
- by four vectors of outputs on AVX-512, two on AVX2: 24 or 12 accumulators, plus the weight vectors and one broadcast, within the register file;
- accumulated across the whole reduction, so each weight vector is loaded and widened once for all the tile's rows, and each output is stored once.

Input-major tiles read their strip of columns down the weight rows at a stride of a whole row, so they walk the rows in blocks of 32. A block bounds the pages and streams a tile touches. Unblocked, a 151,936-column weight read a new page on every load.

### Thread count

The pool keeps one thread per logical CPU (`available_parallelism`); `MAGNETAR_CPU_THREADS` overrides it.

- The runners have 2 cores with 2 hardware threads each, and llama.cpp (Ollama's backend) defaults to the physical core count (`n_threads = 2`).
- One thread per physical core was tried and measured on the same host (run 45343428878, AMD EPYC 9V74). It was slower than every logical CPU:
  - TTFT 81.3 against 71.5 ms, since the prefill is compute-bound and the second hardware threads help;
  - TPOT 35.8 against 34.8 ms.

  So the default stayed on logical CPUs.
- On Intel Xeon 6 the prefill first favored one thread per core (TTFT 60.7 against 80.2 ms, run 45383670846), until the input-major tiles read their weights in blocks of 8 rows. With the blocks, every logical CPU wins there too: TTFT 49.6 against 57.0 ms, TPOT 34.3 against 37.5 ms (run 45387956437).
- The CPU benchmark reruns Magnetar on one thread per physical core after the campaign, so the choice stays measured.
- Once the one-row kernels read their next rows ahead ("Decode step streaming"), one thread per core decoded faster than every logical CPU on Intel Xeon 6 (32.5 against 34.2 ms per token, run 45406355532), while the prefill still preferred every logical CPU (46.1 against 52.5 ms). `MAGNETAR_CPU_DECODE_THREADS` runs one-row Kernels on a pool of their own, so a host can decode on one thread per core and prefill on all; the benchmark measures it, and the default stays every logical CPU until a Zen 4 sample agrees.

### Decode step streaming

A local 64-token profile (4 cores, no SMT, a 9 GB/s per-thread sequential read ceiling, 33 GB/s on 4 threads) put a replayed decode step at about 31 ms of dense products and 6 ms of everything else. A traced pool run showed why the small products were slow: the query projection's four tasks of 400 KB each ran at 6-7 GB/s per thread, its rows of 1,792 bytes ending before the hardware prefetchers caught up.

- **Software prefetch.** A one-row input-major pass reads the next pass's eight rows ahead, one prefetch per cache line, when its rows are up to 16 KiB. An output-major step reads the next rows ahead only when they are up to 4 KiB: the down projection's 9,728-byte rows, sixteen at a time, did not gain. Over a step's real matmul sequence (in-process A/B, alternating step by step): the o projection 25-35% faster, query/key/value 20-28%, gate/up 5-15%.
- **Grouped products at replay.** The query, key and value projections run in one pool run. The Runtime's order puts the rope and KV write of the key between the query and value projections, so the replay moves the value projection ahead of them when nothing conflicts (requirement "Grouped dense products in a replayed step"). Each product keeps the tasks it has alone, so its values do not change.
- **Gated products.** Both products in one pool run, then their partial sums and the activation across the pool, instead of a single-threaded sum and activation.
- **Smaller costs.** Bias and residual sums in place; a vectorizable greedy scan of the 151,936 logits (0.41 ms per step before); invocations without an advancing attribute borrowed at replay.
- **Tried and dropped.** Reading the next product's weights ahead on the workers, in the background, while the caller runs the small Kernels: no measurable gain on top of the above. Storing every weight output-major and reading one row at a time, as llama.cpp does: slower here (35.2 against 28.6 ms for a step's matmuls).

### Numerics

Products are accumulated in f32 with fused multiply-adds, in another order than the Reference CPU kernel's. Results agree to rounding, not bit for bit, as the packed kernels' do. Tests compare them within the existing tolerance (`2e-5` of the largest value).

### Polling workers

Pool workers that finish a job, and a caller waiting for the last tasks, poll for 500 microseconds before sleeping. A decode step runs a pooled Kernel every few hundred microseconds, and a sleeping worker's wake-up left the caller running most tasks of a sub-millisecond matmul alone.

## Risks / Trade-offs

- **Bfloat16 dot products.** The runners' Zen 4 and Zen 5 cores have AVX-512 bfloat16 dot products (`vdpbf16ps`), which llama.cpp may use. A decode step is bound by memory bandwidth, so they would mostly help prefills.
- **The transposed embedding table.** A lookup gathers one value per stored row (896 cache lines per token): about 0.05 ms per lookup locally, against 0.013 ms for a row.
- **Non-matmul Kernels.** Norms, RoPE and element-wise Kernels still run on the Reference CPU kernels. They are small against the matmuls at these sizes.
