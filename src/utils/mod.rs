use diesel::prelude::*;
use pgvector::{Vector, VectorExpressionMethods};
use reqwest::header::{HeaderMap, HeaderValue};

use crate::{models::ChunkRow, schema::chunks};

pub fn retrive(
    connection: &mut PgConnection,
    query_vec: Vector,
) -> anyhow::Result<Vec<(ChunkRow, f64)>> {
    Ok(chunks::table
        .select((
            ChunkRow::as_select(),
            chunks::embedding.cosine_distance(query_vec.clone()),
        ))
        .order(chunks::embedding.cosine_distance(query_vec))
        .limit(3)
        .load::<(ChunkRow, f64)>(connection)?)
}

pub fn get_gemini_headers(api_key: &String) -> HeaderMap {
    let mut headers: HeaderMap<HeaderValue> = HeaderMap::new();
    headers.insert("x-goog-api-key", HeaderValue::from_str(api_key).unwrap());
    headers.insert("Content-Type", HeaderValue::from_static("application/json"));
    headers.insert("Api-Revision", HeaderValue::from_static("2026-05-20"));
    headers
}
