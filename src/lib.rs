use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

pub fn make_chunks(file: String, chunk_size: usize) -> Vec<String> {
    let mut vec = Vec::new();

    let mut prev = 0;
    let mut next = 0;
    for _ in file.as_bytes() {
        if next >= chunk_size {
            vec.push(file[prev..prev + next].to_string());
            prev = prev + next;
            next = 0;
        } else if file[prev..].len() < chunk_size {
            vec.push(file[prev..].to_string());
            break;
        } else {
            next = next + 1;
        }
    }
    vec
}

pub fn chunks_to_embeddings(chunks: &Vec<String>) -> Vec<Vec<f32>> {
    println!("Loading embedding model...");

    let mut model =
        TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::AllMiniLML6V2)).unwrap();

    println!("Embedding model loaded!");
    let embeddings = model.embed(chunks, None).unwrap();

    println!("Embeddings generated!");
    embeddings
}

pub fn query_to_embeddings(query: String) -> Vec<f32> {
    let mut model =
        TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::AllMiniLML6V2)).unwrap();

    model.embed(vec![query], None).unwrap().swap_remove(0)
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
    let st = String::from("Hello how are you doing hope you are doing fine");

    let chunks = make_chunks(st, 12);

    for chunk in chunks {
        println!("{}", chunk);
    }
}

#[test]
pub fn test_embeddings_from_chunks() {
    let st = String::from("Hello how are you doing hope you are doing fine");
    let chunks = make_chunks(st, 12);

    let embeddings = chunks_to_embeddings(&chunks);

    for embedding in embeddings {
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
