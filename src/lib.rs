use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

pub fn make_chunks(file: String, chunk_size: usize, overlap: usize) -> Vec<String> {
    assert!(chunk_size > 0);

    let mut vec = Vec::new();

    let chunks: Vec<String> = file.split_whitespace().map(|it| it.to_string()).collect();

    let mut i = 0;

    while i < chunks.len() {
        let words = {
            if i + chunk_size >= chunks.len() {
                chunks[i..].join(" ")
            } else {
                chunks[i..i + chunk_size].join(" ")
            }
        };

        vec.push(words);
        i += chunk_size - overlap
    }

    vec
}

pub fn chunks_to_embeddings(chunks: &Vec<String>) -> Result<Vec<Vec<f32>>, fastembed::Error> {
    println!("Loading embedding model...");

    TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::AllMiniLML6V2))
        .and_then(|mut m| m.embed(chunks, None))
}

pub fn query_to_embeddings(query: &String) -> Result<Vec<f32>, fastembed::Error> {
    TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::AllMiniLML6V2))
        .and_then(|mut m| Ok(m.embed(vec![query], None)?.remove(0)))
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let mut dot_product = 0.0;
    let mut magnitude_a = 0.0;
    let mut magnitude_b = 0.0;

    for i in 0..a.len() {
        dot_product += a[i] * b[i];
        magnitude_a += a[i] * a[i];
        magnitude_b += b[i] * b[i];
    }

    dot_product / (magnitude_a.sqrt()) * (magnitude_b.sqrt())
}

#[cfg(test)]
#[test]
pub fn test_string_chunking() {
    let st = String::from("Hello how are you doing hope you are doing fine yo");

    let chunks = make_chunks(st, 5, 2);

    for chunk in chunks {
        println!("{}", chunk);
    }
}

#[test]
pub fn test_embeddings_from_chunks() {
    let st = String::from("Hello how are you doing hope you are doing fine");
    let chunks = make_chunks(st, 12, 2);

    let embeddings = chunks_to_embeddings(&chunks);

    for embedding in embeddings.unwrap() {
        println!("Dimensions: {}", embedding.len());
    }
}

#[test]
pub fn test_cosine_smimilarity() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 0.0];

    let score = cosine_similarity(&a, &b);
    println!("Score is: {score}");
}
