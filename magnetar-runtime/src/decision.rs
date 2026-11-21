//! Portable, validated decision results, independent of any model family.
//!
//! These types describe a result; they do not provide an inference engine or
//! imply that a model's probabilities or confidence are calibrated.

use std::{error::Error, fmt};

/// An unnormalized model score and the corresponding probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecisionScore {
    pub score: f64,
    pub probability: f64,
}

/// The meaning of a decision, including an explicit abstention.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DecisionPrediction {
    SelectedOption(u32),
    NumericValue(f64),
    Abstain,
}

/// The result for one question. Options contain no padding and retain the
/// input order. Optional actions retain the checkpoint's declared order;
/// their presence does not impose an action threshold or an abstention.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionResult {
    pub options: Vec<DecisionScore>,
    pub prediction: DecisionPrediction,
    /// A model-supplied statistic, not a claim of statistical calibration.
    pub confidence: Option<f64>,
    pub actions: Option<Vec<DecisionScore>>,
}

/// The distribution that failed result validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionDistribution {
    Options,
    Actions,
}

/// Invalid or inconsistent output at the decision-result boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionResultError {
    EmptyDistribution(DecisionDistribution),
    NonFiniteScore(DecisionDistribution),
    InvalidProbability(DecisionDistribution),
    ProbabilitiesDoNotSumToOne(DecisionDistribution),
    OptionIndexOutOfRange,
    NonFinitePrediction,
    InvalidConfidence,
}

impl fmt::Display for DecisionResultError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDistribution(kind) => write!(formatter, "{kind:?} distribution is empty"),
            Self::NonFiniteScore(kind) => write!(formatter, "{kind:?} score is non-finite"),
            Self::InvalidProbability(kind) => write!(formatter, "{kind:?} probability is invalid"),
            Self::ProbabilitiesDoNotSumToOne(kind) => {
                write!(formatter, "{kind:?} probabilities do not sum to one")
            }
            Self::OptionIndexOutOfRange => formatter.write_str("selected option is out of range"),
            Self::NonFinitePrediction => formatter.write_str("numeric prediction is non-finite"),
            Self::InvalidConfidence => formatter.write_str("confidence is outside [0, 1]"),
        }
    }
}

impl Error for DecisionResultError {}

impl DecisionResult {
    /// Reject malformed output before it crosses an inference boundary.
    /// Each distribution must sum to one within an absolute `1e-6` to allow
    /// normal floating-point rounding from a model's softmax.
    pub fn validate(&self) -> Result<(), DecisionResultError> {
        validate_distribution(&self.options, DecisionDistribution::Options)?;
        if let Some(actions) = &self.actions {
            validate_distribution(actions, DecisionDistribution::Actions)?;
        }
        match self.prediction {
            DecisionPrediction::SelectedOption(index) if index as usize >= self.options.len() => {
                return Err(DecisionResultError::OptionIndexOutOfRange);
            }
            DecisionPrediction::NumericValue(value) if !value.is_finite() => {
                return Err(DecisionResultError::NonFinitePrediction);
            }
            _ => {}
        }
        if self
            .confidence
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(DecisionResultError::InvalidConfidence);
        }
        Ok(())
    }
}

fn validate_distribution(
    items: &[DecisionScore],
    kind: DecisionDistribution,
) -> Result<(), DecisionResultError> {
    if items.is_empty() {
        return Err(DecisionResultError::EmptyDistribution(kind));
    }
    let mut total = 0.0;
    for item in items {
        if !item.score.is_finite() {
            return Err(DecisionResultError::NonFiniteScore(kind));
        }
        if !item.probability.is_finite() || !(0.0..=1.0).contains(&item.probability) {
            return Err(DecisionResultError::InvalidProbability(kind));
        }
        total += item.probability;
    }
    if !total.is_finite() || (total - 1.0).abs() > 1e-6 {
        return Err(DecisionResultError::ProbabilitiesDoNotSumToOne(kind));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
