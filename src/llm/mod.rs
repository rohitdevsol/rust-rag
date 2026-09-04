use anyhow::Ok;
use reqwest::{Client, Response, header::HeaderMap};
use serde_json::json;

#[derive(PartialEq)]
pub enum LLMProvider {
    GEMINI,
    OPENAI,
}

pub async fn new_request(
    client: Client,
    provider: LLMProvider,
    headers: HeaderMap,
    input: &String,
) -> anyhow::Result<Response> {
    let model = {
        if provider == LLMProvider::GEMINI {
            "gemini-3.1-flash-lite"
        } else {
            "none"
        }
    };

    //will handle url based config later
    Ok(client
        .post("https://generativelanguage.googleapis.com/v1beta/interactions")
        .headers(headers)
        .json(&json!({
        "model":model,
        "input": input
        }))
        .send()
        .await
        .expect("Request to GEMINI failed"))
}

pub fn build_req_client() -> anyhow::Result<Client> {
    Ok(reqwest::Client::builder()
        .use_native_tls()
        .build()
        .expect("Failed to build LLM Request Client"))
}
