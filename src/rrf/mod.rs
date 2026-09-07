use std::collections::HashMap;

pub fn rrf(bm25_results: &[usize], vector_results: &[usize], k: usize) -> Vec<usize> {
    let mut scores = HashMap::new();

    for (rank, chunk_id) in bm25_results.iter().enumerate() {
        *scores.entry(*chunk_id).or_insert(0.0) += 1.0 / (k + rank + 1) as f64;
    }

    for (rank, chunk_id) in vector_results.iter().enumerate() {
        *scores.entry(*chunk_id).or_insert(0.0) += 1.0 / (k + rank + 1) as f64;
    }

    let mut results: Vec<(usize, f64)> = scores.into_iter().collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    results
        .into_iter()
        .map(|(chunk_id, _score)| chunk_id)
        .collect()
}
