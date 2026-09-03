use std::io::{self, Write};

use naive_rag::{chunks_to_embeddings, cosine_similarity, make_chunks, query_to_embeddings};
fn main() {
    let file = std::fs::read_to_string("./document.txt").unwrap();

    let chunks = make_chunks(file, 50);

    let embeddings = chunks_to_embeddings(&chunks);

    print!("Enter your query: ");
    io::stdout().flush().unwrap();

    let mut query = String::new();

    io::stdin().read_line(&mut query).unwrap();

    let query_embedding = query_to_embeddings(query);

    let mut res: Vec<(f32, String)> = Vec::new();

    for (chunk, embedding) in chunks.iter().zip(embeddings.iter()) {
        let score = cosine_similarity(&query_embedding, &embedding);
        res.push((score, chunk.clone()));
        // println!("Score is {}", score)
    }

    res.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    for record in res.iter().take(3) {
        println!("Score: {}", record.0);
        println!("Chunk: {}", record.1);
        println!("----------------");
        println!("                ");
    }
}
