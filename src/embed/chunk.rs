use anyhow::Error;

#[derive(Debug)]
pub struct Chunk {
    pub id: u32,
    pub text: String,
}

impl Chunk {
    pub fn new(id: u32, text: String) -> Self {
        Self { id, text }
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

    Ok(vec
        .into_iter()
        .enumerate()
        .map(|(idx, chunk)| Chunk::new(idx as u32, chunk))
        .collect())
}
