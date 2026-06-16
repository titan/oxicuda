//! Hough transform for straight-line detection.
//!
//! Detects lines in a **single-channel** `[h × w]` binary edge image (foreground
//! = value `> 0`, e.g. the output of [`crate::imgproc::edges::canny`]). Each edge
//! pixel votes for every line passing through it in the polar parameter space
//!
//! ```text
//! ρ = x·cos(θ) + y·sin(θ),   θ ∈ [0, π)
//! ```
//!
//! Votes accumulate in a discrete `(ρ, θ)` grid; collinear edge pixels reinforce
//! a single cell, producing a peak whose vote count equals the number of pixels
//! on the line.
//!
//! * [`hough_accumulate`] — build the `(ρ, θ)` vote accumulator.
//! * [`HoughAccumulator::peaks`] — extract local-maximum cells above a vote
//!   threshold, with neighbourhood non-maximum suppression.
//! * [`hough_lines`] — convenience wrapper: accumulate then extract peaks,
//!   returning lines sorted by descending vote count.

use crate::error::{VisionError, VisionResult};
use std::f32::consts::PI;

/// A detected line in polar form together with its supporting vote count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoughLine {
    /// Signed distance from the origin to the line (pixels).
    pub rho: f32,
    /// Orientation of the line normal in radians, `θ ∈ [0, π)`.
    pub theta: f32,
    /// Number of edge pixels that voted for this line.
    pub votes: u32,
}

/// Discrete `(ρ, θ)` vote accumulator.
#[derive(Debug, Clone)]
pub struct HoughAccumulator {
    /// Row-major `[n_rho × n_thetas]` vote counts (ρ-major).
    pub votes: Vec<u32>,
    /// Number of ρ bins.
    pub n_rho: usize,
    /// Number of θ bins spanning `[0, π)`.
    pub n_thetas: usize,
    /// ρ value of bin `0`.
    pub rho_min: f32,
    /// ρ quantisation step (pixels per bin).
    pub rho_step: f32,
    /// θ quantisation step (radians per bin), `π / n_thetas`.
    pub theta_step: f32,
}

impl HoughAccumulator {
    /// Vote count of bin `(rho_idx, theta_idx)`.
    #[must_use]
    pub fn vote(&self, rho_idx: usize, theta_idx: usize) -> u32 {
        self.votes[rho_idx * self.n_thetas + theta_idx]
    }

    /// ρ value at the centre of bin `rho_idx`.
    #[must_use]
    pub fn rho_of(&self, rho_idx: usize) -> f32 {
        self.rho_min + rho_idx as f32 * self.rho_step
    }

    /// θ value (radians) of bin `theta_idx`.
    #[must_use]
    pub fn theta_of(&self, theta_idx: usize) -> f32 {
        theta_idx as f32 * self.theta_step
    }

    /// Extract peak lines: accumulator cells with `votes >= threshold` that are
    /// local maxima within a `(2·neighborhood + 1)²` window, followed by greedy
    /// non-maximum suppression so that no two reported peaks lie within
    /// `neighborhood` bins of each other.
    ///
    /// The returned lines are sorted by descending vote count (ties broken by
    /// `(ρ, θ)` index for determinism).
    #[must_use]
    pub fn peaks(&self, threshold: u32, neighborhood: usize) -> Vec<HoughLine> {
        let reach = neighborhood as isize;
        let mut candidates: Vec<(usize, usize, u32)> = Vec::new();
        for ri in 0..self.n_rho {
            for ti in 0..self.n_thetas {
                let v = self.vote(ri, ti);
                if v == 0 || v < threshold {
                    continue;
                }
                if self.is_local_max(ri, ti, v, reach) {
                    candidates.push((ri, ti, v));
                }
            }
        }

        // Highest votes first; deterministic tie-break by bin index.
        candidates.sort_by(|a, b| {
            b.2.cmp(&a.2)
                .then_with(|| a.0.cmp(&b.0))
                .then_with(|| a.1.cmp(&b.1))
        });

        let mut accepted: Vec<(usize, usize, u32)> = Vec::new();
        for &(ri, ti, v) in &candidates {
            let suppressed = accepted.iter().any(|&(ar, at, _)| {
                ri.abs_diff(ar) <= neighborhood && ti.abs_diff(at) <= neighborhood
            });
            if !suppressed {
                accepted.push((ri, ti, v));
            }
        }

        accepted
            .into_iter()
            .map(|(ri, ti, v)| HoughLine {
                rho: self.rho_of(ri),
                theta: self.theta_of(ti),
                votes: v,
            })
            .collect()
    }

    /// Whether bin `(ri, ti)` with value `v` dominates its neighbourhood.
    fn is_local_max(&self, ri: usize, ti: usize, v: u32, reach: isize) -> bool {
        for dr in -reach..=reach {
            for dt in -reach..=reach {
                if dr == 0 && dt == 0 {
                    continue;
                }
                let nr = ri as isize + dr;
                let nt = ti as isize + dt;
                if nr < 0 || nt < 0 || nr as usize >= self.n_rho || nt as usize >= self.n_thetas {
                    continue;
                }
                if self.vote(nr as usize, nt as usize) > v {
                    return false;
                }
            }
        }
        true
    }
}

/// Parameters controlling [`hough_lines`].
#[derive(Debug, Clone)]
pub struct HoughConfig {
    /// Number of θ bins spanning `[0, π)`.
    pub n_thetas: usize,
    /// ρ quantisation step in pixels.
    pub rho_step: f32,
    /// Minimum votes for a cell to be reported as a line.
    pub threshold: u32,
    /// Non-maximum suppression neighbourhood radius (in accumulator bins).
    pub nms_neighborhood: usize,
}

impl HoughConfig {
    /// Create a configuration with a default NMS neighbourhood of `1`.
    ///
    /// # Errors
    /// Returns [`VisionError::Internal`] if `n_thetas == 0` or `rho_step` is not
    /// strictly positive and finite.
    pub fn new(n_thetas: usize, rho_step: f32, threshold: u32) -> VisionResult<Self> {
        if n_thetas == 0 {
            return Err(VisionError::Internal(
                "hough: n_thetas must be > 0".to_string(),
            ));
        }
        if rho_step <= 0.0 || !rho_step.is_finite() {
            return Err(VisionError::Internal(format!(
                "hough: rho_step must be > 0 and finite (got {rho_step})"
            )));
        }
        Ok(Self {
            n_thetas,
            rho_step,
            threshold,
            nms_neighborhood: 1,
        })
    }

    /// Override the NMS neighbourhood radius (builder).
    #[must_use]
    pub fn with_nms_neighborhood(mut self, neighborhood: usize) -> Self {
        self.nms_neighborhood = neighborhood;
        self
    }
}

/// Validate a single-channel image buffer.
#[inline]
fn validate_gray(img: &[f32], h: usize, w: usize) -> VisionResult<()> {
    if h == 0 || w == 0 {
        return Err(VisionError::InvalidImageSize {
            height: h,
            width: w,
            channels: 1,
        });
    }
    if img.len() != h * w {
        return Err(VisionError::DimensionMismatch {
            expected: h * w,
            got: img.len(),
        });
    }
    Ok(())
}

/// Build the `(ρ, θ)` Hough accumulator for a binary edge image.
///
/// `n_thetas` θ bins span `[0, π)`; ρ is quantised at `rho_step` pixels across
/// `[−diag, +diag]` where `diag = √(w² + h²)`.
///
/// # Errors
/// Returns [`VisionError::InvalidImageSize`] / [`VisionError::DimensionMismatch`]
/// for shape problems and [`VisionError::Internal`] if `n_thetas == 0` or
/// `rho_step` is not strictly positive and finite.
pub fn hough_accumulate(
    img: &[f32],
    h: usize,
    w: usize,
    n_thetas: usize,
    rho_step: f32,
) -> VisionResult<HoughAccumulator> {
    validate_gray(img, h, w)?;
    if n_thetas == 0 {
        return Err(VisionError::Internal(
            "hough: n_thetas must be > 0".to_string(),
        ));
    }
    if rho_step <= 0.0 || !rho_step.is_finite() {
        return Err(VisionError::Internal(format!(
            "hough: rho_step must be > 0 and finite (got {rho_step})"
        )));
    }

    let diag = (w as f32).hypot(h as f32);
    let rho_min = -diag;
    let n_rho = (2.0 * diag / rho_step).ceil() as usize + 1;
    let theta_step = PI / n_thetas as f32;

    // Precompute (cos θ, sin θ) per θ bin to avoid recomputation per pixel.
    let trig: Vec<(f32, f32)> = (0..n_thetas)
        .map(|t| {
            let theta = t as f32 * theta_step;
            (theta.cos(), theta.sin())
        })
        .collect();

    let mut votes = vec![0u32; n_rho * n_thetas];
    for y in 0..h {
        for x in 0..w {
            if img[y * w + x] <= 0.0 {
                continue;
            }
            let xf = x as f32;
            let yf = y as f32;
            for (ti, &(cos_t, sin_t)) in trig.iter().enumerate() {
                let rho = xf * cos_t + yf * sin_t;
                let ri = (((rho - rho_min) / rho_step).round() as isize)
                    .clamp(0, n_rho as isize - 1) as usize;
                votes[ri * n_thetas + ti] += 1;
            }
        }
    }

    Ok(HoughAccumulator {
        votes,
        n_rho,
        n_thetas,
        rho_min,
        rho_step,
        theta_step,
    })
}

/// Detect straight lines in a binary edge image.
///
/// Builds the accumulator via [`hough_accumulate`] and extracts peak lines via
/// [`HoughAccumulator::peaks`], returning them sorted by descending vote count.
///
/// # Errors
/// Propagates the errors of [`hough_accumulate`].
pub fn hough_lines(
    img: &[f32],
    h: usize,
    w: usize,
    config: &HoughConfig,
) -> VisionResult<Vec<HoughLine>> {
    let acc = hough_accumulate(img, h, w, config.n_thetas, config.rho_step)?;
    Ok(acc.peaks(config.threshold, config.nms_neighborhood))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Vertical line: foreground at every row of column `col`.
    fn vertical_line(h: usize, w: usize, col: usize) -> Vec<f32> {
        let mut img = vec![0.0_f32; h * w];
        for y in 0..h {
            img[y * w + col] = 1.0;
        }
        img
    }

    /// Horizontal line: foreground at every column of row `row`.
    fn horizontal_line(h: usize, w: usize, row: usize) -> Vec<f32> {
        let mut img = vec![0.0_f32; h * w];
        for x in 0..w {
            img[row * w + x] = 1.0;
        }
        img
    }

    #[test]
    fn accumulator_dimensions() {
        let img = vec![0.0_f32; 16 * 16];
        let acc = hough_accumulate(&img, 16, 16, 18, 1.0).expect("acc");
        assert_eq!(acc.n_thetas, 18);
        assert_eq!(acc.votes.len(), acc.n_rho * acc.n_thetas);
        assert!((acc.theta_step - PI / 18.0).abs() < 1e-6);
    }

    #[test]
    fn vertical_line_peak() {
        // Column 8 of a 16×16 image: 16 collinear edge pixels.
        let col = 8;
        let img = vertical_line(16, 16, col);
        let acc = hough_accumulate(&img, 16, 16, 18, 1.0).expect("acc");
        // θ bin 0 (θ = 0): ρ = x = col for all pixels → 16 votes in one cell.
        let lines = acc.peaks(12, 1);
        assert!(!lines.is_empty(), "vertical line must produce a peak");
        let top = &lines[0];
        assert_eq!(top.votes, 16, "all 16 pixels vote for the same cell");
        // Vertical line: θ ≈ 0 with ρ ≈ +col.
        assert!(top.theta < acc.theta_step * 1.5, "θ should be ≈ 0");
        assert!(
            (top.rho - col as f32).abs() <= acc.rho_step,
            "ρ should be ≈ {col}, got {}",
            top.rho
        );
    }

    #[test]
    fn horizontal_line_peak() {
        let row = 8;
        let img = horizontal_line(16, 16, row);
        let acc = hough_accumulate(&img, 16, 16, 18, 1.0).expect("acc");
        let lines = acc.peaks(12, 1);
        assert!(!lines.is_empty(), "horizontal line must produce a peak");
        let top = &lines[0];
        assert_eq!(top.votes, 16);
        // Horizontal line: θ ≈ π/2 (bin 9 of 18) with ρ ≈ +row.
        assert!(
            (top.theta - PI / 2.0).abs() < acc.theta_step * 1.5,
            "θ should be ≈ π/2, got {}",
            top.theta
        );
        assert!(
            (top.rho - row as f32).abs() <= acc.rho_step,
            "ρ should be ≈ {row}, got {}",
            top.rho
        );
    }

    #[test]
    fn accumulator_counts_line_pixels() {
        // The maximum vote in the accumulator equals the number of edge pixels
        // on the (axis-aligned) line.
        let img = vertical_line(16, 16, 5);
        let acc = hough_accumulate(&img, 16, 16, 18, 1.0).expect("acc");
        let max_vote = acc.votes.iter().copied().max().unwrap_or(0);
        assert_eq!(max_vote, 16);
    }

    #[test]
    fn empty_image_no_lines() {
        let img = vec![0.0_f32; 12 * 12];
        let acc = hough_accumulate(&img, 12, 12, 18, 1.0).expect("acc");
        assert!(acc.votes.iter().all(|&v| v == 0));
        let lines = acc.peaks(1, 1);
        assert!(lines.is_empty(), "blank image yields no lines");
    }

    #[test]
    fn threshold_filters_short_lines() {
        // A short 5-pixel segment.
        let mut img = vec![0.0_f32; 16 * 16];
        for y in 4..9 {
            img[y * 16 + 6] = 1.0;
        }
        let acc = hough_accumulate(&img, 16, 16, 18, 1.0).expect("acc");
        // High threshold rejects the 5-vote line.
        assert!(acc.peaks(12, 1).is_empty(), "threshold 12 rejects 5 votes");
        // Low threshold accepts it.
        let lines = acc.peaks(3, 1);
        assert!(!lines.is_empty(), "threshold 3 accepts 5 votes");
        assert_eq!(lines[0].votes, 5);
    }

    #[test]
    fn hough_lines_wrapper_matches() {
        let img = vertical_line(16, 16, 8);
        let cfg = HoughConfig::new(18, 1.0, 12).expect("cfg");
        let lines = hough_lines(&img, 16, 16, &cfg).expect("lines");
        assert!(!lines.is_empty());
        assert_eq!(lines[0].votes, 16);
    }

    #[test]
    fn two_separated_lines_detected() {
        // Two parallel vertical lines, far apart in ρ.
        let mut img = vertical_line(16, 16, 3);
        let other = vertical_line(16, 16, 12);
        for (a, b) in img.iter_mut().zip(other.iter()) {
            *a = a.max(*b);
        }
        let cfg = HoughConfig::new(18, 1.0, 12).expect("cfg");
        let lines = hough_lines(&img, 16, 16, &cfg).expect("lines");
        assert_eq!(lines.len(), 2, "two distinct ρ peaks expected");
        for line in &lines {
            assert_eq!(line.votes, 16);
        }
    }

    #[test]
    fn invalid_config_errors() {
        assert!(matches!(
            HoughConfig::new(0, 1.0, 1),
            Err(VisionError::Internal(_))
        ));
        assert!(matches!(
            HoughConfig::new(18, 0.0, 1),
            Err(VisionError::Internal(_))
        ));
        assert!(matches!(
            HoughConfig::new(18, -1.0, 1),
            Err(VisionError::Internal(_))
        ));
    }

    #[test]
    fn accumulate_invalid_args_error() {
        let img = vec![0.0_f32; 8 * 8];
        assert!(matches!(
            hough_accumulate(&img, 8, 8, 0, 1.0),
            Err(VisionError::Internal(_))
        ));
        assert!(matches!(
            hough_accumulate(&img, 8, 8, 18, 0.0),
            Err(VisionError::Internal(_))
        ));
    }

    #[test]
    fn accumulate_wrong_size_errors() {
        let img = vec![0.0_f32; 10];
        assert!(matches!(
            hough_accumulate(&img, 8, 8, 18, 1.0),
            Err(VisionError::DimensionMismatch { .. })
        ));
    }
}
