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
