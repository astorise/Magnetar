## ADDED Requirements

### Requirement: Hugging Face Ingestion Reads Latent Attention, YaRN And Routed Expert Layers

Hugging Face ingestion SHALL read a `config.json` that declares a `kv_lora_rank` as multi-head latent attention, a `yarn` rope scaling as YaRN, and routed experts after `first_k_dense_replace` dense layers with their scoring, groups, scaling factor and selection bias, and SHALL split the checkpoint's fused latent-attention projections by rows and stack its experts stored one by one into the tensors the Model Component requests.

The query projection (`q_proj`, or `q_b_proj` after a low-rank query) stores each head's position-independent rows before its rotated rows; `kv_a_proj_with_mqa` stores the latent's rows before the shared rotated key's; `kv_b_proj` stores each head's key rows before its value rows. Row splits copy bytes. A layer missing a projection, a projection of the wrong shape, latent projections without a declared `kv_lora_rank`, and experts that are missing or differ in shape are refused.

#### Scenario: A DeepSeek-V2-Lite configuration
- **WHEN** the loader parses DeepSeek-V2-Lite's `config.json`
- **THEN** it declares latent attention with no low-rank query, a 512-wide latent, 128 + 64-wide query and key heads and 128-wide values, 16 K/V heads of 192, YaRN with factor 40 over 4096 positions and `mscale_all_dim` 0.707, a dense first layer, 64 experts of width 1408 with 6 per token and two shared experts summed to 2816, and softmax routing without groups

#### Scenario: Split projections reproduce the checkpoint's own computation
- **WHEN** a random tiny `DeepseekV2ForCausalLM` saved by transformers 5.23, with or without a low-rank query, is ingested
- **THEN** each derived projection gives the model's own query, latent, key, value, output, expert, shared-expert and router results on the same input (equal, or within 1e-10 for the summed experts)
