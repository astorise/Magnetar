//! Unit tests for the sampling candidate cuts.
//!
//! Kept in its own file so coverage tooling classifies it as test source
//! rather than Runtime implementation source.
//!
//! The cuts were rewritten from full sorts to partial selection for the
//! per-token hot path, so these tests compare them against straightforward
//! sort-based references over randomized inputs. An optimization that changes
//! which tokens survive is a correctness bug, not a speedup.

use super::*;
use crate::{
    ModelDigest, SpecialToken, TokenIdRange, TokenizerArtifactId, TokenizerFamily, TokenizerId,
    TokenizerRevision,
};

/// Deterministic pseudorandom source, so a failure reproduces exactly.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        splitmix64(&mut self.0)
    }

    /// Scores in roughly [-8, 8), including exact duplicates often enough to
    /// exercise tie-breaking.
    fn score(&mut self) -> f32 {
        (self.next_u64() % 320) as f32 / 20.0 - 8.0
    }
}

fn candidates(rng: &mut Rng, count: u32) -> Vec<Candidate> {
    (0..count)
        .map(|token_id| Candidate {
            token_id,
            score: rng.score(),
            eligible: true,
        })
        .collect()
}

fn eligible_ids(candidates: &[Candidate]) -> Vec<TokenId> {
    candidates
        .iter()
        .filter(|candidate| candidate.eligible)
        .map(|candidate| candidate.token_id)
        .collect()
}

/// Sort-based reference for top-k: rank everything, keep the first k.
fn reference_top_k(k: usize, candidates: &[Candidate]) -> Vec<TokenId> {
    let mut eligible = candidates
        .iter()
        .filter(|candidate| candidate.eligible)
        .copied()
        .collect::<Vec<_>>();
    if k >= eligible.len() {
        return eligible_ids(candidates);
    }
    eligible.sort_by(|left, right| {
        candidate_order((left.score, left.token_id), (right.score, right.token_id))
    });
    let mut kept = eligible
        .into_iter()
        .take(k)
        .map(|candidate| candidate.token_id)
        .collect::<Vec<_>>();
    kept.sort_unstable();
    kept
}

/// Sort-based reference for top-p: rank everything, keep the prefix whose
/// cumulative probability first reaches `top_p`.
fn reference_top_p(top_p: f32, candidates: &[Candidate]) -> Vec<TokenId> {
    let mut eligible = candidates
        .iter()
        .filter(|candidate| candidate.eligible)
        .copied()
        .collect::<Vec<_>>();
    eligible.sort_by(|left, right| {
        candidate_order((left.score, left.token_id), (right.score, right.token_id))
    });
    let probabilities = softmax(&eligible.iter().collect::<Vec<_>>());
    let mut cumulative = 0.0;
    let mut kept = Vec::new();
    for candidate in probabilities {
        cumulative += candidate.probability;
        kept.push(candidate.token_id);
        if cumulative >= top_p {
            break;
        }
    }
    kept.sort_unstable();
    kept
}

#[test]
fn top_k_matches_the_sort_based_reference() {
    let mut rng = Rng(0x5eed);
    for trial in 0..500 {
        let count = 1 + (trial % 40) as u32;
        let k = (trial % 45) as usize;
        let base = candidates(&mut rng, count);

        let expected = reference_top_k(k, &base);
        let mut actual = base.clone();
        apply_top_k(Some(k as u32), &mut actual);

        assert_eq!(
            eligible_ids(&actual),
            expected,
            "trial {trial}: count {count}, k {k}"
        );
    }
}

#[test]
fn top_k_keeps_exactly_k_candidates_when_it_cuts() {
    let mut rng = Rng(0xc0ffee);
    for k in 1..24_usize {
        let mut set = candidates(&mut rng, 24);
        apply_top_k(Some(k as u32), &mut set);
        assert_eq!(eligible_ids(&set).len(), k, "k {k}");
    }
}

#[test]
fn top_k_is_a_no_op_when_k_covers_every_candidate() {
    let mut rng = Rng(7);
    let base = candidates(&mut rng, 12);
    for k in [12_u32, 13, 100] {
        let mut set = base.clone();
        apply_top_k(Some(k), &mut set);
        assert_eq!(eligible_ids(&set), eligible_ids(&base), "k {k}");
    }
}

#[test]
fn top_k_respects_candidates_already_ruled_out() {
    let mut rng = Rng(99);
    let mut base = candidates(&mut rng, 20);
    for candidate in base.iter_mut().filter(|c| c.token_id % 3 == 0) {
        candidate.eligible = false;
    }

    let expected = reference_top_k(4, &base);
    let mut actual = base.clone();
    apply_top_k(Some(4), &mut actual);

    assert_eq!(eligible_ids(&actual), expected);
    assert!(
        eligible_ids(&actual).iter().all(|id| id % 3 != 0),
        "an already-ineligible candidate was revived"
    );
}

#[test]
fn top_p_matches_the_sort_based_reference() {
    let mut rng = Rng(0xbeef);
    for trial in 0..500 {
        let count = 1 + (trial % 30) as u32;
        let top_p = [0.05_f32, 0.3, 0.5, 0.9, 0.999, 1.0][trial % 6];
        let base = candidates(&mut rng, count);

        let expected = reference_top_p(top_p, &base);
        let mut actual = base.clone();
        apply_top_p(Some(top_p), &mut actual).unwrap();

        assert_eq!(
            eligible_ids(&actual),
            expected,
            "trial {trial}: count {count}, top_p {top_p}"
        );
    }
}

#[test]
fn top_k_then_top_p_matches_the_reference_composition() {
    let mut rng = Rng(0xd00d);
    for trial in 0..300 {
        let count = 2 + (trial % 25) as u32;
        let k = 1 + (trial % 12);
        let top_p = [0.2_f32, 0.6, 0.95][trial % 3];
        let base = candidates(&mut rng, count);

        let mut expected_stage = base.clone();
        let kept = reference_top_k(k, &expected_stage);
        for candidate in expected_stage.iter_mut() {
            if !kept.contains(&candidate.token_id) {
                candidate.eligible = false;
            }
        }
        let expected = reference_top_p(top_p, &expected_stage);

        let mut actual = base.clone();
        apply_top_k(Some(k as u32), &mut actual);
        apply_top_p(Some(top_p), &mut actual).unwrap();

        assert_eq!(
            eligible_ids(&actual),
            expected,
            "trial {trial}: count {count}, k {k}, top_p {top_p}"
        );
    }
}

#[test]
fn rank_matches_the_sort_based_reference() {
    let mut rng = Rng(0xfeed);
    for trial in 0..300 {
        let count = 1 + (trial % 30) as u32;
        let set = candidates(&mut rng, count);
        let probabilities = softmax(&set.iter().collect::<Vec<_>>());

        let mut ranked = probabilities.clone();
        ranked.sort_by(|left, right| {
            candidate_order((left.score, left.token_id), (right.score, right.token_id))
        });

        for (index, candidate) in ranked.iter().enumerate() {
            assert_eq!(
                rank_for(candidate.token_id, &probabilities),
                index + 1,
                "trial {trial}: token {}",
                candidate.token_id
            );
        }
    }
}

#[test]
fn rank_is_one_for_a_token_outside_the_candidate_set() {
    let mut rng = Rng(3);
    let set = candidates(&mut rng, 4);
    let probabilities = softmax(&set.iter().collect::<Vec<_>>());
    assert_eq!(rank_for(999, &probabilities), 1);
}

#[test]
fn greedy_selection_matches_softmax_and_rank_over_the_eligible_candidates() {
    // Ties (resolved to the lower token id), ineligible candidates, a
    // non-finite score and a pseudo-random spread of scores.
    let mut candidates: Vec<Candidate> = (0..2048u32)
        .map(|index| {
            let mixed = index.wrapping_mul(2_654_435_761) % 1000;
            Candidate {
                token_id: index + 7,
                score: mixed as f32 / 37.0 - 13.0,
                eligible: index % 11 != 3,
            }
        })
        .collect();
    candidates[5].score = f32::NEG_INFINITY;
    candidates[5].eligible = false;
    let top = candidates
        .iter()
        .filter(|candidate| candidate.eligible)
        .map(|candidate| candidate.score)
        .fold(f32::NEG_INFINITY, f32::max);
    candidates[100].score = top;
    candidates[900].score = top;
    let eligible: Vec<&Candidate> = candidates.iter().filter(|c| c.eligible).collect();
    let probabilities = softmax(&eligible);
    let expected = probabilities
        .iter()
        .min_by(|left, right| {
            candidate_order((left.score, left.token_id), (right.score, right.token_id))
        })
        .unwrap();
    let selected = greedy_selection(&candidates, true).unwrap();
    assert_eq!(selected.token_id, expected.token_id);
    assert_eq!(
        selected.probability.to_bits(),
        expected.probability.to_bits()
    );
    assert_eq!(selected.rank, rank_for(expected.token_id, &probabilities));
    let without = greedy_selection(&candidates, false).unwrap();
    assert_eq!(
        (without.token_id, without.rank),
        (selected.token_id, selected.rank)
    );

    for candidate in &mut candidates {
        candidate.eligible = false;
    }
    assert!(matches!(
        greedy_selection(&candidates, true),
        Err(SamplingError::NoEligibleToken)
    ));
}

fn fast_path_tokenizer(vocabulary: u32) -> TokenizerMetadata {
    TokenizerMetadata {
        id: TokenizerId::new("fast-path").unwrap(),
        artifact: TokenizerArtifactId::new("fast-path-tokenizer").unwrap(),
        digest: ModelDigest::sha256(b"tokenizer"),
        family: TokenizerFamily::new("fast-path").unwrap(),
        revision: TokenizerRevision::new("1.0.0").unwrap(),
        vocabulary_size: vocabulary,
        added_token_count: 0,
        token_id_range: TokenIdRange::new(0, vocabulary),
        model_max_length: None,
        special_tokens: vec![
            SpecialToken::new(SpecialTokenKind::Eos, "<eos>", vocabulary - 1),
            SpecialToken::new(SpecialTokenKind::Pad, "<pad>", 3),
        ],
        additional_special_tokens: vec![SpecialToken::new(
            SpecialTokenKind::Additional,
            "<extra>",
            vocabulary - 2,
        )],
        byte_fallback: false,
        normalization: None,
        pre_tokenizer: None,
        supports_offsets: false,
        supports_token_type_ids: false,
        supports_browser: false,
    }
}

/// The greedy fast path selects the same token, with the same rank, as the
/// general candidate path, which an allowed list naming every token forces
/// without changing which tokens are eligible.
#[test]
fn fast_greedy_selection_matches_the_general_path() {
    let mut rng = Rng(0x5eed_fa57);
    let mut selected = 0;
    for round in 0..200u32 {
        let vocabulary = 64 + (rng.next_u64() % 64) as u32;
        let mut scores: Vec<f32> = (0..vocabulary).map(|_| rng.score()).collect();
        for _ in 0..3 {
            let index = (rng.next_u64() % u64::from(vocabulary)) as usize;
            scores[index] = match rng.next_u64() % 3 {
                0 => f32::NAN,
                1 => f32::INFINITY,
                _ => f32::NEG_INFINITY,
            };
        }
        let mut request = SamplingRequest::host_scores(
            SamplingRequestId::new(format!("fast-{round}")).unwrap(),
            scores,
            fast_path_tokenizer(vocabulary),
        );
        request.parameters.greedy = true;
        request.parameters.sampling_enabled = false;
        // Temperature rounding can tie scores; a tiny temperature overflows
        // finite scores to infinity.
        request.parameters.temperature = [1.0, 0.7, 0.0, 1.3, 1e-38][(round % 5) as usize];
        request.step_index = (rng.next_u64() % 3) as usize;
        request.policy.suppress_pad = rng.next_u64().is_multiple_of(2);
        request.policy.suppress_additional_special_tokens = rng.next_u64().is_multiple_of(2);
        request.stop.minimum_generated_tokens = Some(2);
        request.stop.mask_stop_tokens_before_minimum = rng.next_u64().is_multiple_of(2);
        request.stop.stop_token_ids = vec![(rng.next_u64() % u64::from(vocabulary)) as u32];
        request.banned_token_ids = (0..(rng.next_u64() % 4))
            .map(|_| (rng.next_u64() % u64::from(vocabulary)) as u32)
            .collect();
        if round.is_multiple_of(5) {
            // A banned run past the vocabulary's end.
            request.banned_token_ids.push(TokenId::MAX);
        }
        let fast = select_next_token(&request);
        let mut general = request.clone();
        general.parameters.allowed_token_ids = Some((0..vocabulary).collect());
        let general = select_next_token(&general);
        match (fast, general) {
            (Ok(fast), Ok(general)) => {
                selected += 1;
                assert_eq!(
                    fast.selected_token_id, general.selected_token_id,
                    "round {round}"
                );
                assert_eq!(fast.token_rank, general.token_rank, "round {round}");
                assert_eq!(fast.finish_hint, general.finish_hint, "round {round}");
            }
            (Err(fast), Err(general)) => {
                assert_eq!(format!("{fast:?}"), format!("{general:?}"), "round {round}");
            }
            (fast, general) => panic!("round {round}: {fast:?} vs {general:?}"),
        }
    }
    // Most rounds select a token rather than agree on an error.
    assert!(selected > 150, "only {selected} rounds selected a token");
}

#[test]
fn fast_greedy_selection_breaks_temperature_ties_like_the_general_path() {
    // Tokens 4 and 5 divide by 0.7 to the same f32; the lower id wins.
    let mut scores = vec![-1.0_f32; 16];
    scores[4] = 1.400_000_1;
    scores[5] = 1.400_000_2;
    let mut request = SamplingRequest::host_scores(
        SamplingRequestId::new("fast-tie").unwrap(),
        scores,
        fast_path_tokenizer(16),
    );
    request.parameters.greedy = true;
    request.parameters.sampling_enabled = false;
    request.parameters.temperature = 0.7;
    assert_ne!(1.400_000_1_f32, 1.400_000_2_f32);
    assert_eq!(1.400_000_1_f32 / 0.7, 1.400_000_2_f32 / 0.7);
    assert_eq!(select_next_token(&request).unwrap().selected_token_id, 4);
}
