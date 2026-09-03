use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

pub fn make_chunks(file: String, chunk_size: usize) -> Vec<String> {
    // itertate over this file and every 10 chars split and store that much owned string in the Vec
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

pub fn chunks_to_embeddings(chunks: Vec<String>) -> Vec<Vec<f32>> {
    let mut model =
        TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::AllMiniLML6V2)).unwrap();

    let embeddings = model.embed(chunks, None).unwrap();
    embeddings
}

pub fn query_to_embeddings(query: String) -> Vec<Vec<f32>> {
    let mut v = Vec::new();
    v.push(query);

    chunks_to_embeddings(v)
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

    let embeddings = chunks_to_embeddings(chunks);

    for embedding in embeddings {
        println!("Dimensions: {}", embedding.len());
    }
}
