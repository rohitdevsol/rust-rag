use bm25::{Language, SearchEngine, SearchEngineBuilder};

use crate::embed::Chunk;

pub struct BM25Retriever {
    engine: SearchEngine<u32>,
    chunks: Vec<Chunk>,
}

impl BM25Retriever {
    pub fn new(chunks: Vec<Chunk>) -> Self {
        let documents = chunks
            .iter()
            .map(|chunk| chunk.text.as_str())
            .collect::<Vec<_>>();

        let engine =
            SearchEngineBuilder::<usize>::with_corpus(Language::English, documents).build();

        Self { engine, chunks }
    }

    pub fn search(&self, query: &str, k: usize) -> Vec<(&Chunk, f64)> {
        self.engine
            .search(query, k)
            .into_iter()
            .map(|result| {
                (
                    &self.chunks[result.document.id as usize],
                    result.score as f64,
                )
            })
            .collect()
    }
}
