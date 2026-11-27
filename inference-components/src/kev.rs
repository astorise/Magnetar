//! Explicit Kev-0.8B decision integration on a caller-selected Qwen model.
//!
//! The caller must bind a compatible trusted backbone and authenticated head.
//! Rows run independently from position zero; prefix reuse is not implemented.

use super::{LoadedInferenceComponent, invalid_request};
use anyhow::{Context, Result, anyhow, ensure};
use magnetar_component_kev::encoding::{Delimiters, encode_rows};
pub use magnetar_component_kev::{PointerHead, encoding::Question as TokenizedQuestion};
use magnetar_runtime::{
    DecisionPrediction, DecisionResult, DecisionScore, HostTensor, ModelDType, ModelDigest,
    ProductionLoadedModel, host_tensors_from_artifact_bytes,
    tokenizer::{EncodeInput, Tokenizer},
};
use std::{collections::BTreeMap, io::Read, path::Path};

const MAX_ROW_TOKENS: usize = 512;
const MAX_QUESTIONS: usize = 128;
const MAX_TEXT_BYTES: usize = 1_048_576;
const DELIMITERS: Delimiters = Delimiters {
    state: 248_060,
    question: 248_061,
    option_start: 248_049,
    option_end: 248_050,
    decide: 248_062,
};
const SPECIAL_TOKENS: [(&str, u32); 5] = [
    ("<|fim_prefix|>", DELIMITERS.state),
    ("<|fim_middle|>", DELIMITERS.question),
    ("<|box_start|>", DELIMITERS.option_start),
    ("<|box_end|>", DELIMITERS.option_end),
    ("<|fim_suffix|>", DELIMITERS.decide),
];

/// One question and its choices, in the order returned by the model.
#[derive(Clone, Debug)]
pub struct TextQuestion<'a> {
    pub instruction: &'a str,
    pub options: &'a [&'a str],
}

/// Load the pinned Kev-0.8B head after its restricted PyTorch export to
/// Safetensors. Only the known, hash-verified export and its checkpoint
/// temperature are accepted; the file is bounded and read only once.
pub fn load_pinned_head_safetensors(path: &Path) -> Result<PointerHead> {
    const MAX_HEAD_BYTES: u64 = 4 * 1024 * 1024;
    const EXPECTED_DIGEST: &str =
        "sha256:ab7c75921e79ec27ca97d8bf7b928e277dac843a6f4a528f9e8b7ab0fe7cd47d";
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .context("could not open Kev pointer head")?
        .take(MAX_HEAD_BYTES + 1)
        .read_to_end(&mut bytes)
        .context("could not read Kev pointer head")?;
    ensure!(
        bytes.len() as u64 <= MAX_HEAD_BYTES,
        "Kev pointer head exceeds its size limit"
    );
    ensure!(
        ModelDigest::sha256(&bytes).value == EXPECTED_DIGEST,
        "Kev pointer head digest does not match the pinned export"
    );
    let artifact =
        magnetar_format_safetensors::parse(&bytes).context("invalid Kev Safetensors head")?;
    ensure!(
        artifact
            .tensors
            .iter()
            .all(|tensor| tensor.storage_dtype == ModelDType::F32),
        "Kev head must use f32 tensors"
    );
    let header: [u8; 8] = bytes
        .get(..8)
        .ok_or_else(|| anyhow!("missing Kev Safetensors header"))?
        .try_into()?;
    let data_start = 8_u64
        .checked_add(u64::from_le_bytes(header))
        .ok_or_else(|| anyhow!("Kev Safetensors header is too large"))?;
    let decoded = host_tensors_from_artifact_bytes(&artifact.tensors, &bytes, data_start)
        .context("could not decode Kev pointer head")?;
    let tensors: BTreeMap<_, _> = decoded
        .into_iter()
        .map(|(name, tensor)| {
            let shape = tensor
                .shape
                .into_iter()
                .map(usize::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            Ok((
                name,
                magnetar_component_kev::CheckpointTensor {
                    shape,
                    values: tensor.data,
                },
            ))
        })
        .collect::<Result<_>>()?;
    PointerHead::from_checkpoint_tensors(tensors, 2.351_096).context("invalid Kev pointer head")
}

/// Evaluate tokenized questions with the pinned Kev-0.8B delimiters.
///
/// The supplied head and Qwen Component must come from the same compatible
/// checkpoint. The caller owns their authentication. Each full state + row
/// must fit within 512 tokens and is executed independently; no prefix cache,
/// option isolation, abstention threshold or confidence calibration is implied.
pub fn decide_tokenized(
    model: &mut ProductionLoadedModel,
    head: &PointerHead,
    state: &[u32],
    questions: &[TokenizedQuestion<'_>],
) -> Result<Vec<DecisionResult>> {
    score_with(head, state, questions, |tokens| {
        model
            .observe_prefill_edge(tokens, "edge.final_norm")
            .map_err(Into::into)
    })
}

fn score_with(
    head: &PointerHead,
    state: &[u32],
    questions: &[TokenizedQuestion<'_>],
    mut observe: impl FnMut(&[u32]) -> Result<HostTensor>,
) -> Result<Vec<DecisionResult>> {
    if !(1..=MAX_QUESTIONS).contains(&questions.len()) {
        return Err(invalid_request("Kev requires between 1 and 128 questions"));
    }
    // Validate every branch before running any of them.
    let request = encode_rows(state, questions, DELIMITERS, MAX_ROW_TOKENS, MAX_ROW_TOKENS)
        .map_err(|error| invalid_request(format!("invalid Kev request: {error}")))?;
    let mut results = Vec::with_capacity(request.rows.len());
    for row in &request.rows {
        let tokens: Vec<u32> = request
            .prefix
            .iter()
            .chain(&row.token_ids)
            .copied()
            .collect();
        let hidden = observe(&tokens).context("Kev backbone prefill failed")?;
        let [rows, width] = hidden.shape.as_slice() else {
            return Err(anyhow!("Kev backbone returned a non-matrix representation"));
        };
        ensure!(
            *rows == tokens.len() as u64 && *width > 0,
            "Kev backbone returned an invalid representation shape"
        );
        let width = usize::try_from(*width).context("Kev hidden width is too large")?;
        let prefix_elements = request
            .prefix
            .len()
            .checked_mul(width)
            .ok_or_else(|| anyhow!("Kev representation size overflow"))?;
        let row_hidden = hidden
            .data
            .get(prefix_elements..)
            .ok_or_else(|| anyhow!("Kev backbone returned incomplete representations"))?;
        let scores = head
            .score_row(row, row_hidden)
            .context("Kev pointer head failed")?;
        let mut selected = 0;
        for index in 1..scores.logits.len() {
            if scores.logits[index] > scores.logits[selected] {
                selected = index;
            }
        }
        let result = DecisionResult {
            options: scores
                .logits
                .into_iter()
                .zip(scores.probabilities)
                .map(|(score, probability)| DecisionScore {
                    score: f64::from(score),
                    probability: f64::from(probability),
                })
                .collect(),
            prediction: DecisionPrediction::SelectedOption(selected as u32),
            confidence: None,
            actions: None,
        };
        result
            .validate()
            .context("Kev produced an invalid decision")?;
        results.push(result);
    }
    Ok(results)
}

/// Match the pinned upstream encoder's escaping of token-like user text.
fn escape_controls(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<|") {
        result.push_str(&rest[..start]);
        let candidate = &rest[start + 2..];
        let name_len = candidate
            .bytes()
            .take_while(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
            .count();
        if name_len > 0 && candidate[name_len..].starts_with("|>") {
            result.push_str("<¦");
            result.push_str(&candidate[..name_len]);
            result.push_str("¦>");
            rest = &candidate[name_len + 2..];
        } else {
            result.push_str("<|");
            rest = candidate;
        }
    }
    result.push_str(rest);
    result
}

fn tokenize(tokenizer: &dyn Tokenizer, text: &str) -> Result<Vec<u32>> {
    tokenizer
        .encode(EncodeInput::new(escape_controls(text)))
        .map(|encoded| encoded.token_ids)
        .map_err(|error| invalid_request(format!("Kev tokenization failed: {error}")))
}

/// Run text questions with a compatible model, tokenizer and authenticated
/// Kev head. The entire request is rejected if any branch exceeds 512 tokens.
pub fn decide_text(
    model: &mut ProductionLoadedModel,
    tokenizer: &dyn Tokenizer,
    head: &PointerHead,
    state: &str,
    questions: &[TextQuestion<'_>],
) -> Result<Vec<DecisionResult>> {
    if !(1..=MAX_QUESTIONS).contains(&questions.len()) {
        return Err(invalid_request("Kev requires between 1 and 128 questions"));
    }
    let mut bytes = state.len();
    for question in questions {
        if !(1..=255).contains(&question.options.len()) {
            return Err(invalid_request(
                "each Kev question requires 1 to 255 options",
            ));
        }
        bytes = bytes
            .checked_add(question.instruction.len())
            .ok_or_else(|| invalid_request("Kev text is too large"))?;
        for option in question.options {
            bytes = bytes
                .checked_add(option.len())
                .ok_or_else(|| invalid_request("Kev text is too large"))?;
        }
    }
    if bytes > MAX_TEXT_BYTES {
        return Err(invalid_request("Kev text exceeds the 1 MiB input limit"));
    }
    for (special, expected) in SPECIAL_TOKENS {
        let actual = tokenizer
            .encode(EncodeInput::new(special))
            .context("could not verify Kev tokenizer delimiters")?;
        ensure!(
            actual.token_ids == [expected],
            "loaded tokenizer does not match Kev-0.8B delimiters"
        );
    }
    let state = tokenize(tokenizer, state)?;
    let encoded: Vec<_> = questions
        .iter()
        .map(|question| {
            let instruction = tokenize(tokenizer, question.instruction)?;
            let options = question
                .options
                .iter()
                .map(|option| tokenize(tokenizer, option))
                .collect::<Result<Vec<_>>>()?;
            Ok((instruction, options))
        })
        .collect::<Result<_>>()?;
    let questions: Vec<_> = encoded
        .iter()
        .map(|(instruction, options)| TokenizedQuestion {
            instruction,
            options,
        })
        .collect();
    decide_tokenized(model, head, &state, &questions)
}

impl LoadedInferenceComponent {
    /// Run text questions with the bundle's tokenizer and a caller-supplied,
    /// authenticated Kev head. The caller must select a compatible Qwen3.5
    /// Component, converted checkpoint and head before loading this instance.
    /// The entire request is rejected if any branch exceeds 512 tokens.
    pub fn decide_kev(
        &self,
        head: &PointerHead,
        state: &str,
        questions: &[TextQuestion<'_>],
    ) -> Result<Vec<DecisionResult>> {
        let mut model = self
            .loaded_model
            .lock()
            .map_err(|_| anyhow!("resident Magnetar inference Component lock poisoned"))?;
        decide_text(&mut model, self.tokenizer.as_ref(), head, state, questions)
    }
}

#[cfg(test)]
mod tests;
