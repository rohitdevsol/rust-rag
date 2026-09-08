#![allow(dead_code, unused)]
use diesel::prelude::*;
use naive_rag::{
    db::establish_connection,
    embed::{BM25Retriever, FastEmbedLocal, make_chunks},
    llm,
    rrf::rrf,
    schema::chunks,
    utils::{get_gemini_headers, retrive},
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

    let bm25 = BM25Retriever::new(&chunks);
    let bm25_ids = bm25.search(&query, 3);

    let vector_results = retrive(&mut connection, query_vec)?;

    let vector_ids: Vec<usize> = vector_results
        .iter()
        .map(|(chunk, _distance)| chunk.id as usize - 1)
        .collect();

    let fused_ids = rrf(&bm25_ids, &vector_ids, 60);

    println!("BM25 IDs: {:?}", bm25_ids);
    println!("Vector IDs: {:?}", vector_ids);
    println!("Fused IDs: {:?}", fused_ids);

    for &id in fused_ids.iter().take(3) {
        println!("\nCHUNK {id}:\n{}", chunks[id].text);
    }

    let results = fused_ids
        .iter()
        .take(3)
        .map(|&id| &chunks[id])
        .collect::<Vec<_>>();

    let client = llm::build_req_client().unwrap();

    let context = results
        .iter()
        .map(|chunk| chunk.text.as_str())
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
    println!("{:#?}", body);

    Ok(())
}
