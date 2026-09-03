use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

pub fn make_chunks(file: String, chunk_size: usize) -> Vec<String> {
    let mut vec = Vec::new();

    let chunks: Vec<String> = file.split_whitespace().map(|it| it.to_string()).collect();

    let mut prev = 0;
    let mut next = 0;

    while chunks[prev..].len() > 0 {
        if next >= chunk_size {
            vec.push(chunks[prev..prev + next].join(" "));
            prev = prev + next;
            next = 0;
        } else if chunks[prev..].len() < chunk_size {
            vec.push(chunks[prev..].join(" "));
            break;
        } else {
            next = next + 1;
        }
    }
    vec
}

pub fn chunks_to_embeddings(chunks: &Vec<String>) -> Result<Vec<Vec<f32>>, fastembed::Error> {
    println!("Loading embedding model...");

    TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::AllMiniLML6V2))
        .and_then(|mut m| m.embed(chunks, None))
}

pub fn query_to_embeddings(query: String) -> Result<Vec<f32>, fastembed::Error> {
    TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::AllMiniLML6V2))
        .and_then(|mut m| Ok(m.embed(vec![query], None)?.swap_remove(0)))
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

    let chunks = make_chunks(st, 5);

    for chunk in chunks {
        println!("{}", chunk);
    }
}

#[test]
pub fn test_embeddings_from_chunks() {
    let st = String::from("Hello how are you doing hope you are doing fine");
    let chunks = make_chunks(st, 12);

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
