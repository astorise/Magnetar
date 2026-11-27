# Laya decision inference

The optional `laya` feature of `magnetar-inference-component` exposes a
single-sequence decision path for the pinned Laya checkpoint. Register and
trust a ModernBERT encoder Component first, and independently trust the model
digest when constructing `LoadedLaya`. The Component and the weights are
separate trust decisions.

`LoadedLaya::load_pinned` takes the checkpoint path, the encoder and decision
configuration bytes, the registered Component digest, a model trust store and
a Provider. It verifies the checkpoint's SHA-256 and the two configurations,
then authenticates every tensor read before converting the encoder weights to
the Component's graph layout. The normalized configurations used for this
profile are available in `components/modernbert/tests/fixtures/laya-encoder.config.json`
and `components/laya/tests/fixtures/rl_agent_config.json`. CRLF and LF line
endings are accepted; other configuration changes are rejected.

`decide_text` takes the checkpoint's tokenizer, serialized state text and a
typed choice, score or noul question. It verifies the tokenizer digest and its
control-token IDs before applying the pinned text layout. `decide_tokenized`
accepts token IDs, a validity mask, the option-marker positions and a question
type; marker positions must refer to distinct, valid `[MASK]` tokens in order.
Both methods return `DecisionResult`, preserving option scores and
probabilities, the typed prediction and action scores and probabilities.
The choice/score confidence is the model's entropy statistic and does not
establish statistical calibration; no abstention threshold is inferred.

This profile processes one question per call, supports at most 512 tokens and
between 2 and 255 options, and uses the checkpoint's right-truncation rule for text. Text
requests are limited to 1 MiB. It uses the pinned checkpoint and tokenizer,
not arbitrary ModernBERT or Laya variants. Call `unload` to release resident
encoder weights. GPU execution is not qualified by the CPU reference tests.
