## ADDED Requirements

### Requirement: GGUF Ingestion Reads deepseek2 Files

GGUF ingestion SHALL read a `deepseek2` file's `attention.kv_lora_rank` as multi-head latent attention, its YaRN keys as llama.cpp applies them, its leading dense layers and routed experts with their routing, and SHALL split its latent-attention projections into the tensors the Model Component requests, from either form of the key and value projection llama.cpp's converter has written.

The expanded head widths come from `key_length_mla` and `value_length_mla` when present (files converted after llama.cpp split `attn_kv_b`, which declare the latent's widths in `key_length` and `value_length` and one K/V head), from `key_length` and `value_length` otherwise; the rotated width from `rope.dimension_count`. YaRN keeps cosines and sines unscaled and takes `mscale_all_dim` as `yarn_log_multiplier / 0.1`, with `beta_fast` 32 and `beta_slow` 1 unless declared. Scoring is softmax when `expert_gating_func` is absent or `1`, sigmoid when `2`; another value is refused. With `expert_group_count` above 1, a group is scored by its best expert after a softmax and by its two best after a sigmoid. `exp_probs_b.bias` is the selection bias. `attn_q`/`attn_q_b`, `attn_kv_a_mqa` and `attn_kv_b` are split by rows like the Hugging Face checkpoint's; `attn_v_b` is already the value projection; `attn_k_b`'s heads are transposed back, in the stored float type or in `f32` from a block-quantized one.

#### Scenario: A DeepSeek-V2-Lite file from before the split
- **WHEN** the loader reads the metadata of `DeepSeek-V2-Lite-Chat.Q4_K_M.gguf` (converted in 2024)
- **THEN** it declares the same latent attention, head widths, experts and dense first layer as the Hugging Face configuration, YaRN with factor 40 over 4096 positions and `mscale_all_dim` 0.707, and every tensor name in the file has a canonical name

#### Scenario: Either key and value form
- **WHEN** a file stores `attn_kv_b`, or `attn_k_b` and `attn_v_b` written from the same `kv_b` as the converter writes them
- **THEN** both yield the same split key and value projections, equal to the rows of `kv_b`

#### Scenario: A block-quantized split key projection
- **WHEN** `attn_k_b` is stored as `Q8_0`
- **THEN** the key projection reaches the Component in `f32`, each head transposed back
