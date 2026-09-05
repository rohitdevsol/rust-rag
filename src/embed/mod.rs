use anyhow::{Error, Ok};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use pgvector::Vector;

use crate::models::NewChunk;

pub struct FastEmbedLocal {
    model: TextEmbedding,
}

impl FastEmbedLocal {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            model: TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::AllMiniLML6V2))?,
        })
    }

    pub fn embed_multi(self: &mut Self, chunks: &[Chunk]) -> anyhow::Result<Vec<NewChunk>> {
        let mut newchunk_vec = Vec::new();
        chunks.iter().for_each(|chunk| {
            let m = self.model.embed(vec![&chunk.text], None).unwrap().remove(0);
            let v = Vector::from(m);
            newchunk_vec.push(NewChunk {
                text: chunk.text.clone(),
                embedding: v,
            });
        });
        Ok(newchunk_vec)
    }

    pub fn embed_single(self: &mut Self, query: &String) -> anyhow::Result<Vector> {
        Ok(Vector::from(
            self.model.embed(vec![query], None).unwrap().remove(0),
        ))
    }
}

#[derive(Debug)]
pub struct Chunk {
    pub text: String,
}

impl Chunk {
    pub fn new(text: String) -> Self {
        Self { text }
    }
}

pub fn make_chunks(file: String, chunk_size: usize, overlap: usize) -> anyhow::Result<Vec<Chunk>> {
    if chunk_size <= 0 {
        return Err(Error::msg("Invalid Chunk size"));
    }

    if overlap >= chunk_size {
        return Err(Error::msg(
            "Overlap can not be more than or equal to Chunk size",
        ));
    }

    let mut vec = Vec::new();

    let chunks: Vec<String> = file.split_whitespace().map(|it| it.to_string()).collect();

    let mut i = 0;

    while i < chunks.len() {
        let words = {
            if i + chunk_size >= chunks.len() {
                chunks[i..].join(" ")
            } else {
                chunks[i..i + chunk_size].join(" ")
            }
        };

        vec.push(words);
        i += chunk_size - overlap
    }

    Ok(vec.into_iter().map(|chunk| Chunk::new(chunk)).collect())
}
