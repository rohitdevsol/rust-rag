use anyhow::Ok;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use pgvector::Vector;

use crate::{embed::Chunk, models::NewChunk};

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
