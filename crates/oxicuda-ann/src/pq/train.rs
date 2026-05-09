use super::codebook::PqCodebook;
use crate::error::{AnnError, AnnResult};
use crate::handle::LcgRng;
use crate::kmeans::kmeans::KMeans;

/// Train a PQ codebook by splitting `dim` into `m` equal subspaces and running
/// independent k-means per subspace.
pub fn train_pq(
    data: &[f32],
    n: usize,
    dim: usize,
    m: usize,
    ksub: usize,
    n_epochs: usize,
    rng: &mut LcgRng,
) -> AnnResult<PqCodebook> {
    if dim == 0 || !dim.is_multiple_of(m) {
        return Err(AnnError::InvalidNumSubspaces { m, dim });
    }
    if n == 0 {
        return Err(AnnError::EmptyInput);
    }
    if ksub == 0 || ksub > n {
        return Err(AnnError::InvalidK { k: ksub, n });
    }

    let dsub = dim / m;
    let mut cb = PqCodebook::new(m, ksub, dsub);

    // Extract subspace data and train per subspace
    let mut sub_data = vec![0.0_f32; n * dsub];
    for s in 0..m {
        for i in 0..n {
            for d in 0..dsub {
                sub_data[i * dsub + d] = data[i * dim + s * dsub + d];
            }
        }

        let km = KMeans::fit(&sub_data, n, dsub, ksub, n_epochs, rng)?;
        let centers = km.centroids();

        for c in 0..ksub {
            let dst = cb.centroid_mut(s, c);
            dst.copy_from_slice(&centers[c * dsub..(c + 1) * dsub]);
        }
    }

    Ok(cb)
}
