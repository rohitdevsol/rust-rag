use diesel::prelude::*;
use naive_rag::llm;
use naive_rag::models::ChunkRow;
use naive_rag::{
    db::establish_connection,
    embed::{FastEmbedLocal, make_chunks},
    schema::chunks,
};
use pgvector::VectorExpressionMethods;
use reqwest::header::{HeaderMap, HeaderValue};
use std::io::{self, Write};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let file = std::fs::read_to_string("./document.txt").unwrap();

    dotenvy::dotenv().ok();

    let mut connection = establish_connection();
    let _ = &mut connection;

    println!("Connected to the database");

    let api_key = std::env::var("GEMINI_API_KEY").unwrap();

    let chunks = make_chunks(file, 50, 2)?;

    let mut embedder = FastEmbedLocal::new().unwrap();
    let new_chunks = embedder.embed_multi(&chunks).unwrap();

    diesel::insert_into(chunks::table)
        .values(&new_chunks)
        .execute(&mut connection)?;

    print!("Enter your query: ");
    io::stdout().flush().unwrap();

    let mut query = String::new();

    io::stdin().read_line(&mut query).unwrap();

    let query_vec = embedder.embed_single(&query).unwrap();

    let results = chunks::table
        .select((
            ChunkRow::as_select(),
            chunks::embedding.cosine_distance(query_vec.clone()),
        ))
        .order(chunks::embedding.cosine_distance(query_vec))
        .limit(3)
        .load::<(ChunkRow, f64)>(&mut connection)?;

    for (chunk, distance) in results.iter() {
        println!("Distance: {}", distance);
        println!("ID: {}", chunk.id);
        println!("Text: {}", chunk.text);
        println!("----------------");
    }

    let client = llm::build_req_client().unwrap();

    let context = results
        .iter()
        .map(|(chunk, _)| chunk.text.as_str())
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
    eprintln!("{:#?}", body);

    Ok(())
}
