//! Dispatcher for fitting AFT models.

use crate::aft::exponential::fit_exponential;
use crate::aft::generalized_gamma::fit_generalized_gamma;
use crate::aft::log_logistic::fit_log_logistic;
use crate::aft::log_normal::fit_log_normal;
use crate::aft::weibull::fit_weibull;
use crate::data::Dataset;
use crate::error::SurvivalResult;

/// AFT family identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AftFamily {
    Exponential,
    Weibull,
    LogNormal,
    LogLogistic,
    GeneralizedGamma,
}

/// Unified AFT fit result.
#[derive(Debug, Clone)]
pub enum AftFit {
    Exponential(crate::aft::exponential::ExponentialFit),
    Weibull(crate::aft::weibull::WeibullFit),
    LogNormal(crate::aft::log_normal::LogNormalFit),
    LogLogistic(crate::aft::log_logistic::LogLogisticFit),
    GeneralizedGamma(crate::aft::generalized_gamma::GeneralizedGammaFit),
}

impl AftFit {
    /// Final log-likelihood.
    #[must_use]
    pub fn log_likelihood(&self) -> f64 {
        match self {
            Self::Exponential(f) => f.log_likelihood,
            Self::Weibull(f) => f.log_likelihood,
            Self::LogNormal(f) => f.log_likelihood,
            Self::LogLogistic(f) => f.log_likelihood,
            Self::GeneralizedGamma(f) => f.log_likelihood,
        }
    }
}

/// Fit an AFT model by family.
pub fn fit_aft(data: &Dataset, family: AftFamily) -> SurvivalResult<AftFit> {
    Ok(match family {
        AftFamily::Exponential => AftFit::Exponential(fit_exponential(data)?),
        AftFamily::Weibull => AftFit::Weibull(fit_weibull(data)?),
        AftFamily::LogNormal => AftFit::LogNormal(fit_log_normal(data)?),
        AftFamily::LogLogistic => AftFit::LogLogistic(fit_log_logistic(data)?),
        AftFamily::GeneralizedGamma => AftFit::GeneralizedGamma(fit_generalized_gamma(data)?),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Observation;

    fn synth() -> Dataset {
        Dataset::new(
            (1..20)
                .map(|i| Observation::new(i as f64, true).expect("ok"))
                .collect(),
            None,
            None,
        )
        .expect("ok")
    }

    #[test]
    fn dispatch_exponential() {
        let d = synth();
        let f = fit_aft(&d, AftFamily::Exponential).expect("ok");
        assert!(matches!(f, AftFit::Exponential(_)));
    }

    #[test]
    fn dispatch_weibull() {
        let d = synth();
        let f = fit_aft(&d, AftFamily::Weibull).expect("ok");
        assert!(matches!(f, AftFit::Weibull(_)));
    }

    #[test]
    fn dispatch_log_normal() {
        let d = synth();
        let f = fit_aft(&d, AftFamily::LogNormal).expect("ok");
        assert!(matches!(f, AftFit::LogNormal(_)));
    }

    #[test]
    fn dispatch_log_logistic() {
        let d = synth();
        let f = fit_aft(&d, AftFamily::LogLogistic).expect("ok");
        assert!(matches!(f, AftFit::LogLogistic(_)));
    }

    #[test]
    fn dispatch_gen_gamma() {
        let d = synth();
        let f = fit_aft(&d, AftFamily::GeneralizedGamma).expect("ok");
        assert!(matches!(f, AftFit::GeneralizedGamma(_)));
    }

    #[test]
    fn ll_finite() {
        let d = synth();
        let f = fit_aft(&d, AftFamily::Weibull).expect("ok");
        assert!(f.log_likelihood().is_finite());
    }
}
