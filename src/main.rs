use naive_rag::{chunks_to_embeddings, make_chunks};
fn main() {
    let file = std::fs::read_to_string("./document.txt").unwrap();

    let chunks = make_chunks(file, 20);

    let embeddings = chunks_to_embeddings(chunks);

    for embedding in embeddings {
        println!("Dimensions: {}", embedding.len());
    }
}
