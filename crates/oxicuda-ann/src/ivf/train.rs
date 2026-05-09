use crate::error::AnnResult;
use crate::handle::LcgRng;
use crate::kmeans::kmeans::KMeans;

/// Train k-means coarse quantizer centroids for IVF.
pub fn train_coarse(
    data: &[f32],
    n: usize,
    dim: usize,
    n_lists: usize,
    n_epochs: usize,
    rng: &mut LcgRng,
) -> AnnResult<Vec<f32>> {
    let km = KMeans::fit(data, n, dim, n_lists, n_epochs, rng)?;
    Ok(km.centroids().to_vec())
}
