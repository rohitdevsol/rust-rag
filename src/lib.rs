// use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
pub mod db;
pub mod embed;
pub mod llm;
pub mod models;
pub mod rrf;
pub mod schema;
pub mod utils;

// THIS CODE WAS USED BEFORE PGVECTOR

// pub fn query_to_embeddings(query: &String) -> Result<Vec<f32>, fastembed::Error> {
//     TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::AllMiniLML6V2))
//         .and_then(|mut m| Ok(m.embed(vec![query], None)?.remove(0)))
// }

// pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
//     assert_eq!(a.len(), b.len());

//     let mut dot_product = 0.0;
//     let mut magnitude_a = 0.0;
//     let mut magnitude_b = 0.0;

//     for i in 0..a.len() {
//         dot_product += a[i] * b[i];
//         magnitude_a += a[i] * a[i];
//         magnitude_b += b[i] * b[i];
//     }

//     dot_product / (magnitude_a.sqrt()) * (magnitude_b.sqrt())
// }

// #[cfg(test)]
// mod test {

//     use anyhow::Ok;

//     use super::*;
//     use crate::chunks::{chunks_to_embeddings, make_chunks};

//     #[test]
//     pub fn test_string_chunking() -> anyhow::Result<()> {
//         let st = String::from("Hello how are you doing hope you are doing fine yo");

//         let chunks = make_chunks(st, 5, 2)?;

//         for chunk in chunks {
//             println!("{:?}", chunk);
//         }
//         Ok(())
//     }

//     #[test]
//     pub fn test_embeddings_from_chunks() -> anyhow::Result<()> {
//         let st = String::from("Hello how are you doing hope you are doing fine");
//         let chunks = make_chunks(st, 12, 2)?;

//         let embeddings = chunks_to_embeddings(&chunks);

//         for embedding in embeddings.unwrap() {
//             println!("Dimensions: {}", embedding.len());
//         }
//         Ok(())
//     }

//     #[test]
//     pub fn test_cosine_smimilarity() {
//         let a = vec![1.0, 0.0];
//         let b = vec![0.0, 0.0];

//         let score = cosine_similarity(&a, &b);
//         println!("Score is: {score}");
//     }
// }
