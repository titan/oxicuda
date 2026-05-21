//! Masked SSL losses: Masked Autoencoder (MAE) random-patch dropping +
//! reconstruction MSE; SimMIM L1/L2 reconstruction with block & random masking;
//! data2vec joint-embedding masked prediction (Baevski et al. 2022);
//! BEiT discrete-token prediction (Bao et al. 2021).

pub mod beit;
pub mod data2vec;
pub mod mae;
pub mod simmim;

pub use beit::{
    BeitConfig, BeitResult, VqCodebook, beit_block_mask, beit_loss, vq_codebook_init, vq_encode,
    vq_update_codebook,
};
pub use data2vec::{
    Data2VecConfig, Data2VecResult, Data2VecState, data2vec_batch_loss, data2vec_loss,
    data2vec_mask, huber_loss, normalize_teacher_targets,
};
pub use simmim::{
    SimMimConfig, simmim_block_mask, simmim_l1_loss, simmim_l2_loss, simmim_random_mask,
    simmim_reconstruction_loss,
};
