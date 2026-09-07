use diesel::prelude::*;
use naive_rag::llm;
use naive_rag::utils::{get_gemini_headers, retrive};
use naive_rag::{
    db::establish_connection,
    embed::{FastEmbedLocal, make_chunks},
    schema::chunks,
};

use std::io::{self, Write};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let mut connection = establish_connection();
    println!("Connected to the database");

    let file = std::fs::read_to_string("./document.txt").unwrap();

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

    let results = retrive(&mut connection, query_vec)?;

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

    let response = llm::new_request(
        client,
        llm::LLMProvider::GEMINI,
        get_gemini_headers(&api_key),
        &input,
    )
    .await
    .unwrap();

    println!("Status: {}", response.status());

    let body = response.text().await?;
    eprintln!("{:#?}", body);

    Ok(())
}
