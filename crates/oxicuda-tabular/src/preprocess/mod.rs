//! Preprocessing primitives: normalisation, feature embedding, SMOTE oversampling,
//! target encoding, quantile feature transformation, data augmentation,
//! self-supervised pretraining, and concept-drift detection.

pub mod augment;
pub mod concept_drift;
pub mod embed;
pub mod normalize;
pub mod quantile_feat;
pub mod smote;
pub mod ssl_pretrain;
pub mod target_encode;
pub mod woe;

pub use augment::{CutMixConfig, MixupConfig, cutmix_batch, cutmix_pair, mixup_batch, mixup_pair};
pub use concept_drift::{
    AdwinTabular, AdwinTabularConfig, DriftStatus, KsDriftDetector, PageHinkleyTabular,
    PageHinkleyTabularConfig,
};
pub use quantile_feat::{QuantileDist, QuantileTransformer, probit, std_normal_cdf};
pub use smote::{
    SmoteConfig, SmoteResult, SmoteVariant, class_imbalance_ratio, smote, smote_augment,
};
pub use ssl_pretrain::{ScarfConfig, ScarfModel, VimeConfig, VimeModel, VimeSample};
pub use target_encode::{TargetEncoder, TargetEncoderConfig};
pub use woe::{WoeEncoder, information_value};
