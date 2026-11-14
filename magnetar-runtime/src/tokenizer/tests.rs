//! Unit tests for the parent module.

use super::*;
use crate::tests::*;

/// One token may serve several special roles (DeepSeek-R1-Distill's pad
/// token is its end-of-sequence token); one id carrying two texts, or one
/// role declared twice, is still a conflict.
#[test]
fn tokenizer_metadata_accepts_one_token_in_several_roles_but_not_one_id_with_two_texts() {
    let mut shared = generation_tokenizer_metadata();
    shared
        .special_tokens
        .push(SpecialToken::new(SpecialTokenKind::Pad, "<eos>", 299));
    shared.validate().unwrap();

    let mut clash = generation_tokenizer_metadata();
    clash
        .special_tokens
        .push(SpecialToken::new(SpecialTokenKind::Pad, "<pad>", 299));
    assert_eq!(clash.validate(), Err(TokenizerError::SpecialTokenConflict));

    let mut twice = generation_tokenizer_metadata();
    twice
        .special_tokens
        .push(SpecialToken::new(SpecialTokenKind::Eos, "<eos>", 297));
    assert_eq!(twice.validate(), Err(TokenizerError::SpecialTokenConflict));
}
