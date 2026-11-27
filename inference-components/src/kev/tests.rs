use super::*;
use magnetar_component_kev::LinearProjection;

fn head() -> PointerHead {
    let projection =
        || LinearProjection::new(2, 2, vec![1.0, 0.0, 0.0, 1.0], vec![0.0; 2]).unwrap();
    PointerHead::new(projection(), projection(), 1.0).unwrap()
}

#[test]
fn scores_independent_question_rows_and_returns_typed_results() {
    let first_options = vec![vec![20], vec![21]];
    let second_options = vec![vec![22], vec![23], vec![24]];
    let questions = [
        TokenizedQuestion {
            instruction: &[10],
            options: &first_options,
        },
        TokenizedQuestion {
            instruction: &[11],
            options: &second_options,
        },
    ];
    let mut observed = Vec::new();
    let results = score_with(&head(), &[271], &questions, |tokens| {
        observed.push(tokens.to_vec());
        let data = tokens
            .iter()
            .enumerate()
            .flat_map(|(position, &id)| [id as f32 / 100_000.0, position as f32])
            .collect::<Vec<_>>();
        Ok(HostTensor::new([tokens.len() as u64, 2], data).unwrap())
    })
    .unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].options.len(), 2);
    assert_eq!(results[1].options.len(), 3);
    assert!(results.iter().all(|result| result.validate().is_ok()));
    assert_eq!(results[0].prediction, DecisionPrediction::SelectedOption(1));
    assert_eq!(results[1].prediction, DecisionPrediction::SelectedOption(2));
    assert!(
        results
            .iter()
            .all(|result| result.confidence.is_none() && result.actions.is_none())
    );
    assert_eq!(&observed[0][..2], &[DELIMITERS.state, 271]);
    assert_eq!(&observed[1][..2], &[DELIMITERS.state, 271]);
    assert!(!observed[1].contains(&20));
}

#[test]
fn rejects_invalid_request_before_running_any_question() {
    let good = vec![vec![20]];
    let injected = vec![vec![DELIMITERS.decide]];
    let questions = [
        TokenizedQuestion {
            instruction: &[10],
            options: &good,
        },
        TokenizedQuestion {
            instruction: &[11],
            options: &injected,
        },
    ];
    let mut calls = 0;
    let error = score_with(&head(), &[271], &questions, |_| {
        calls += 1;
        unreachable!()
    })
    .unwrap_err();
    assert!(crate::is_invalid_component_invocation(&error));
    assert_eq!(calls, 0);
    assert!(score_with(&head(), &[], &[], |_| unreachable!()).is_err());
    assert!(score_with(&head(), &vec![1; 512], &questions[..1], |_| unreachable!()).is_err());
}

#[test]
fn escapes_only_upstream_token_like_sequences() {
    assert_eq!(
        escape_controls("é <|fim_prefix|> <|a_2|> <|bad-name|> <||> <|unfinished"),
        "é <¦fim_prefix¦> <¦a_2¦> <|bad-name|> <||> <|unfinished"
    );
}

#[test]
fn pinned_head_loader_rejects_unverified_bytes() {
    use std::io::Write;
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(b"not the pinned pointer head").unwrap();
    assert!(load_pinned_head_safetensors(file.path()).is_err());
}
