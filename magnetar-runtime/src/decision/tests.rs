use super::*;

fn result() -> DecisionResult {
    DecisionResult {
        options: vec![
            DecisionScore {
                score: -2.0,
                probability: 0.25,
            },
            DecisionScore {
                score: 3.0,
                probability: 0.75,
            },
        ],
        prediction: DecisionPrediction::SelectedOption(1),
        confidence: Some(0.8),
        actions: None,
    }
}

#[test]
fn variable_options_numeric_answers_and_explicit_abstention_are_distinct() {
    let mut choice = result();
    assert_eq!(choice.validate(), Ok(()));
    choice.options.push(DecisionScore {
        score: 7.0,
        probability: 0.0,
    });
    choice.prediction = DecisionPrediction::NumericValue(1.75);
    choice.actions = Some(vec![
        DecisionScore {
            score: 0.3,
            probability: 0.4,
        },
        DecisionScore {
            score: -0.3,
            probability: 0.6,
        },
    ]);
    assert_eq!(choice.validate(), Ok(()));
    choice.prediction = DecisionPrediction::Abstain;
    assert_eq!(choice.validate(), Ok(()));
    assert!(choice.actions.is_some());
}

#[test]
fn rejects_malformed_distributions_and_out_of_range_predictions() {
    let mut invalid = result();
    invalid.options.clear();
    assert_eq!(
        invalid.validate(),
        Err(DecisionResultError::EmptyDistribution(
            DecisionDistribution::Options
        ))
    );
    invalid = result();
    invalid.options[0].score = f64::NAN;
    assert_eq!(
        invalid.validate(),
        Err(DecisionResultError::NonFiniteScore(
            DecisionDistribution::Options
        ))
    );
    invalid = result();
    invalid.options[0].probability = f64::INFINITY;
    assert_eq!(
        invalid.validate(),
        Err(DecisionResultError::InvalidProbability(
            DecisionDistribution::Options
        ))
    );
    invalid = result();
    invalid.options[0].probability = -0.1;
    assert_eq!(
        invalid.validate(),
        Err(DecisionResultError::InvalidProbability(
            DecisionDistribution::Options
        ))
    );
    invalid = result();
    invalid.options[0].probability = 0.1;
    assert_eq!(
        invalid.validate(),
        Err(DecisionResultError::ProbabilitiesDoNotSumToOne(
            DecisionDistribution::Options
        ))
    );
    invalid = result();
    invalid.prediction = DecisionPrediction::SelectedOption(2);
    assert_eq!(
        invalid.validate(),
        Err(DecisionResultError::OptionIndexOutOfRange)
    );
    invalid.prediction = DecisionPrediction::NumericValue(f64::NAN);
    assert_eq!(
        invalid.validate(),
        Err(DecisionResultError::NonFinitePrediction)
    );
    invalid.prediction = DecisionPrediction::Abstain;
    invalid.confidence = Some(1.01);
    assert_eq!(
        invalid.validate(),
        Err(DecisionResultError::InvalidConfidence)
    );
    invalid.confidence = None;
    invalid.actions = Some(vec![]);
    assert_eq!(
        invalid.validate(),
        Err(DecisionResultError::EmptyDistribution(
            DecisionDistribution::Actions
        ))
    );
    invalid.actions = Some(vec![DecisionScore {
        score: 0.0,
        probability: 0.8,
    }]);
    assert_eq!(
        invalid.validate(),
        Err(DecisionResultError::ProbabilitiesDoNotSumToOne(
            DecisionDistribution::Actions
        ))
    );
}
