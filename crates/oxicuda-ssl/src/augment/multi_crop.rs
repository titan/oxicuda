//! Multi-crop augmentation strategy used by SwAV and DINO.
//!
//! Each image generates 2 *global* crops (e.g. 224×224) and `n_local` *local*
//! crops (e.g. 96×96) at different aspect ratios. Local crops are passed only
//! to the student network; global crops are passed to both student and teacher.
//! This module computes the deterministic crop sizes only — pixel-level
//! cropping is the caller's responsibility.

use crate::error::{SslError, SslResult};

/// Multi-crop configuration.
#[derive(Debug, Clone)]
pub struct MultiCropConfig {
    /// Target spatial size of each global crop (e.g. 224).
    pub global_crop_size: usize,
    /// Target spatial size of each local crop (e.g. 96).
    pub local_crop_size: usize,
    /// Number of additional local crops per image (default 6).
    pub n_local_crops: usize,
}

impl Default for MultiCropConfig {
    fn default() -> Self {
        Self {
            global_crop_size: 224,
            local_crop_size: 96,
            n_local_crops: 6,
        }
    }
}

impl MultiCropConfig {
    /// Validated config.
    ///
    /// # Errors
    /// - [`SslError::InvalidNumCrops`] if `n_local_crops < 1`.
    /// - [`SslError::EmptyInput`] if either crop size is zero.
    pub fn new(
        global_crop_size: usize,
        local_crop_size: usize,
        n_local_crops: usize,
    ) -> SslResult<Self> {
        if n_local_crops == 0 {
            return Err(SslError::InvalidNumCrops);
        }
        if global_crop_size == 0 || local_crop_size == 0 {
            return Err(SslError::EmptyInput);
        }
        Ok(Self {
            global_crop_size,
            local_crop_size,
            n_local_crops,
        })
    }

    /// Total number of crops produced per image (`2 globals + n_local`).
    #[must_use]
    pub fn n_crops(&self) -> usize {
        2 + self.n_local_crops
    }
}

/// Per-crop spec returned by [`multi_crop`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CropSpec {
    /// Crop size (assumed square).
    pub size: usize,
    /// True if this is a global crop, false if local.
    pub is_global: bool,
}

/// Generate the deterministic list of crop specs given a configuration.
///
/// The pixel-level cropping (random window selection within the original image)
/// is the caller's responsibility — this helper only enumerates the desired
/// `(size, is_global)` tuples.
///
/// # Errors
/// Propagates errors from [`MultiCropConfig`].
pub fn multi_crop(cfg: &MultiCropConfig) -> SslResult<Vec<CropSpec>> {
    let n = cfg.n_crops();
    let mut crops = Vec::with_capacity(n);
    crops.push(CropSpec {
        size: cfg.global_crop_size,
        is_global: true,
    });
    crops.push(CropSpec {
        size: cfg.global_crop_size,
        is_global: true,
    });
    for _ in 0..cfg.n_local_crops {
        crops.push(CropSpec {
            size: cfg.local_crop_size,
            is_global: false,
        });
    }
    Ok(crops)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_dino_canonical() {
        let cfg = MultiCropConfig::default();
        assert_eq!(cfg.global_crop_size, 224);
        assert_eq!(cfg.local_crop_size, 96);
        assert_eq!(cfg.n_local_crops, 6);
        assert_eq!(cfg.n_crops(), 8);
    }

    #[test]
    fn rejects_zero_local_crops() {
        assert!(MultiCropConfig::new(224, 96, 0).is_err());
    }

    #[test]
    fn rejects_zero_crop_sizes() {
        assert!(MultiCropConfig::new(0, 96, 6).is_err());
        assert!(MultiCropConfig::new(224, 0, 6).is_err());
    }

    #[test]
    fn multi_crop_returns_two_globals_then_n_local() {
        let cfg = MultiCropConfig::new(160, 64, 4).unwrap();
        let crops = multi_crop(&cfg).unwrap();
        assert_eq!(crops.len(), 6);
        assert_eq!(crops[0].size, 160);
        assert!(crops[0].is_global);
        assert_eq!(crops[1].size, 160);
        assert!(crops[1].is_global);
        for c in &crops[2..] {
            assert_eq!(c.size, 64);
            assert!(!c.is_global);
        }
    }

    #[test]
    fn multi_crop_default_yields_8_specs() {
        let cfg = MultiCropConfig::default();
        let crops = multi_crop(&cfg).unwrap();
        assert_eq!(crops.len(), 8);
    }
}
