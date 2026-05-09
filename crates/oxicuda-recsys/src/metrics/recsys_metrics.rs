use std::collections::HashSet;

pub fn precision_at_k(recommended: &[usize], relevant: &HashSet<usize>, k: usize) -> f32 {
    if k == 0 {
        return 0.0;
    }
    let hits = recommended
        .iter()
        .take(k)
        .filter(|id| relevant.contains(id))
        .count();
    hits as f32 / k as f32
}

pub fn recall_at_k(recommended: &[usize], relevant: &HashSet<usize>, k: usize) -> f32 {
    if relevant.is_empty() || k == 0 {
        return 0.0;
    }
    let hits = recommended
        .iter()
        .take(k)
        .filter(|id| relevant.contains(id))
        .count();
    hits as f32 / relevant.len() as f32
}

pub fn ndcg_at_k(recommended: &[usize], relevant: &HashSet<usize>, k: usize) -> f32 {
    if k == 0 || relevant.is_empty() {
        return 0.0;
    }
    let dcg: f32 = recommended
        .iter()
        .take(k)
        .enumerate()
        .map(|(pos, id)| {
            if relevant.contains(id) {
                1.0 / (pos as f32 + 2.0).log2()
            } else {
                0.0
            }
        })
        .sum();

    let ideal_k = k.min(relevant.len());
    let idcg: f32 = (0..ideal_k)
        .map(|pos| 1.0 / (pos as f32 + 2.0).log2())
        .sum();

    if idcg < 1e-12 {
        return 0.0;
    }
    dcg / idcg
}

pub fn map_at_k(recommended: &[usize], relevant: &HashSet<usize>, k: usize) -> f32 {
    if relevant.is_empty() || k == 0 {
        return 0.0;
    }
    let mut hits = 0usize;
    let mut sum_prec = 0.0_f32;

    for (pos, id) in recommended.iter().take(k).enumerate() {
        if relevant.contains(id) {
            hits += 1;
            sum_prec += hits as f32 / (pos + 1) as f32;
        }
    }

    if hits == 0 {
        return 0.0;
    }
    sum_prec / relevant.len().min(k) as f32
}

pub fn mrr(recommended: &[usize], relevant: &HashSet<usize>) -> f32 {
    for (pos, id) in recommended.iter().enumerate() {
        if relevant.contains(id) {
            return 1.0 / (pos + 1) as f32;
        }
    }
    0.0
}

pub fn hit_rate_at_k(recommended: &[usize], relevant: &HashSet<usize>, k: usize) -> f32 {
    let hit = recommended.iter().take(k).any(|id| relevant.contains(id));
    if hit { 1.0 } else { 0.0 }
}

/// AUC via Wilcoxon-Mann-Whitney statistic.
pub fn auc_score(scores: &[(f32, bool)]) -> f32 {
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let n_pos = sorted.iter().filter(|&&(_, label)| label).count();
    let n_neg = sorted.len() - n_pos;

    if n_pos == 0 || n_neg == 0 {
        return 0.5;
    }

    // Assign ranks (1-based), handle ties by averaging
    let n = sorted.len();
    let mut ranks = vec![0.0_f32; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (sorted[j].0 - sorted[i].0).abs() < 1e-9 {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f32 / 2.0;
        for rank in ranks.iter_mut().skip(i).take(j - i) {
            *rank = avg_rank;
        }
        i = j;
    }

    let rank_sum_pos: f32 = sorted
        .iter()
        .zip(ranks.iter())
        .filter(|&(&(_, label), _)| label)
        .map(|(_, &r)| r)
        .sum();

    let u_pos = rank_sum_pos - (n_pos * (n_pos + 1)) as f32 / 2.0;
    u_pos / (n_pos * n_neg) as f32
}
