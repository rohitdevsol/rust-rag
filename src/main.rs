use naive_rag::{
    chunks::{Chunk, chunks_to_embeddings, make_chunks},
    cosine_similarity, llm, query_to_embeddings,
};
use reqwest::header::{HeaderMap, HeaderValue};
use std::io::{self, Write};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let file = std::fs::read_to_string("./document.txt").unwrap();

    dotenvy::dotenv().ok();
    let api_key = std::env::var("GEMINI_API_KEY").unwrap();

    let chunks = make_chunks(file, 50, 2)?;

    let embeddings = match chunks_to_embeddings(&chunks) {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    print!("Enter your query: ");
    io::stdout().flush().unwrap();

    let mut query = String::new();

    io::stdin().read_line(&mut query).unwrap();

    let query_embedding = match query_to_embeddings(&query) {
        Ok(v) => v,
        Err(e) => return Err(e.into()),
    };

    let mut res: Vec<(f32, Chunk)> = Vec::new();

    for (chunk, embedding) in chunks.into_iter().zip(embeddings.iter()) {
        let score = cosine_similarity(&query_embedding, &embedding);
        res.push((score, chunk));
    }

    res.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    for record in res.iter().take(3) {
        println!("Score: {}", record.0);
        println!("Chunk ID: {}", record.1.id);
        println!("Text: {}", record.1.text);
        println!("----------------");
        println!("                ");
    }

    let client = llm::build_req_client().unwrap();

    let context = res
        .iter()
        .take(3)
        .map(|(_, chunk)| chunk.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");

    let input = format!(
        r#"
            Answer the user's question using ONLY the provided context and do not think much.
            Context:
            {context}
            Question:
            {query}
            Answer briefly.
            "#
    );

    let mut headers: HeaderMap<HeaderValue> = HeaderMap::new();
    headers.insert("x-goog-api-key", HeaderValue::from_str(&api_key).unwrap());
    headers.insert("Content-Type", HeaderValue::from_static("application/json"));
    headers.insert("Api-Revision", HeaderValue::from_static("2026-05-20"));

    let response = llm::new_request(
        client,
        llm::LLMProvider::GEMINI,
        HeaderMap::from(headers),
        &input,
    )
    .await
    .unwrap();

    println!("Status: {}", response.status());

    let body = response.text().await?;
    eprintln!("{:#}", body);

    Ok(())
}
