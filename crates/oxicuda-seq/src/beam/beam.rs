//! Generic beam search with length-normalisation and diversity penalty.

use crate::error::{SeqError, SeqResult};

/// Beam-search configuration.
#[derive(Debug, Clone, Copy)]
pub struct BeamConfig {
    pub beam_width: usize,
    pub max_steps: usize,
    /// Exponent applied to `length^alpha` denominator for length normalisation.
    pub length_alpha: f64,
    /// Penalty subtracted from each candidate proportional to its position in
    /// the sorted beam (diversity / Diverse Beam Search).
    pub diversity: f64,
}

impl Default for BeamConfig {
    fn default() -> Self {
        Self {
            beam_width: 4,
            max_steps: 32,
            length_alpha: 0.0,
            diversity: 0.0,
        }
    }
}

/// Generic beam search over a score-function callback.
///
/// * `init_token` is the starting token used to seed all beam items.
/// * `is_terminal(token) -> bool` halts a beam if the most recent token is terminal.
/// * `successors(path) -> Vec<(token, log_prob)>` yields candidate extensions.
#[derive(Debug, Clone)]
pub struct BeamSearch {
    pub cfg: BeamConfig,
}

impl BeamSearch {
    pub fn new(cfg: BeamConfig) -> SeqResult<Self> {
        if cfg.beam_width == 0 {
            return Err(SeqError::InvalidConfiguration(
                "beam_width must be > 0".to_string(),
            ));
        }
        Ok(Self { cfg })
    }

    /// Run beam search.  `Token = usize` for simplicity.  Returns the top-1 path.
    pub fn search<F, G>(
        &self,
        init_token: usize,
        mut successors: F,
        mut is_terminal: G,
    ) -> SeqResult<(Vec<usize>, f64)>
    where
        F: FnMut(&[usize]) -> Vec<(usize, f64)>,
        G: FnMut(usize) -> bool,
    {
        let mut beam: Vec<(Vec<usize>, f64, bool)> = vec![(vec![init_token], 0.0, false)];
        for _step in 0..self.cfg.max_steps {
            if beam.iter().all(|(_, _, done)| *done) {
                break;
            }
            let mut new_beam: Vec<(Vec<usize>, f64, bool)> = Vec::with_capacity(beam.len() * 4);
            for (path, score, done) in &beam {
                if *done {
                    new_beam.push((path.clone(), *score, true));
                    continue;
                }
                let next = successors(path);
                if next.is_empty() {
                    new_beam.push((path.clone(), *score, true));
                    continue;
                }
                for (tok, logp) in next {
                    let mut p = path.clone();
                    p.push(tok);
                    let term = is_terminal(tok);
                    new_beam.push((p, score + logp, term));
                }
            }
            // Apply length normalisation when ranking.
            new_beam.sort_by(|a, b| {
                let sa = norm_score(a.1, a.0.len(), self.cfg.length_alpha);
                let sb = norm_score(b.1, b.0.len(), self.cfg.length_alpha);
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });
            // Diversity penalty: subtract `i * diversity` from rank-i candidate.
            if self.cfg.diversity > 0.0 {
                for (i, item) in new_beam.iter_mut().enumerate() {
                    item.1 -= self.cfg.diversity * i as f64;
                }
                new_beam.sort_by(|a, b| {
                    let sa = norm_score(a.1, a.0.len(), self.cfg.length_alpha);
                    let sb = norm_score(b.1, b.0.len(), self.cfg.length_alpha);
                    sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            new_beam.truncate(self.cfg.beam_width);
            beam = new_beam;
        }
        let (path, score, _) = beam
            .into_iter()
            .next()
            .ok_or_else(|| SeqError::NumericalInstability("empty beam".to_string()))?;
        Ok((path, score))
    }
}

#[inline]
fn norm_score(score: f64, length: usize, alpha: f64) -> f64 {
    if alpha == 0.0 || length == 0 {
        score
    } else {
        score / (length as f64).powf(alpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beam_trivial_chain() {
        // Generative model that always appends token 1 with log-prob -0.1
        // and token 2 with log-prob -0.5.  Beam width 1 should pick token 1
        // every step.
        let bs = BeamSearch::new(BeamConfig {
            beam_width: 1,
            max_steps: 3,
            length_alpha: 0.0,
            diversity: 0.0,
        })
        .expect("ok");
        let (path, score) = bs
            .search(0, |_path| vec![(1, -0.1), (2, -0.5)], |t| t == 9)
            .expect("ok");
        assert_eq!(path, vec![0, 1, 1, 1]);
        assert!((score - (-0.3)).abs() < 1e-9);
    }

    #[test]
    fn beam_terminates_on_end_token() {
        let bs = BeamSearch::new(BeamConfig::default()).expect("ok");
        let mut step = 0;
        let (path, _score) = bs
            .search(
                0,
                |_path| {
                    step += 1;
                    if step >= 3 {
                        vec![(9, 0.0)]
                    } else {
                        vec![(1, 0.0), (2, -0.1)]
                    }
                },
                |t| t == 9,
            )
            .expect("ok");
        assert!(path.contains(&9));
    }

    #[test]
    fn beam_zero_width_errors() {
        assert!(
            BeamSearch::new(BeamConfig {
                beam_width: 0,
                max_steps: 1,
                length_alpha: 0.0,
                diversity: 0.0,
            })
            .is_err()
        );
    }
}
