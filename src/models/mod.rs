use crate::schema::chunks;
use diesel::{Selectable, deserialize::Queryable};
use pgvector::Vector;

#[derive(Debug, Queryable, Selectable)]
#[diesel(table_name = chunks)]
pub struct ChunkRow {
    pub id: i32,
    pub text: String,
    pub embedding: Vector,
}

use diesel::Insertable;

#[derive(Debug, Insertable)]
#[diesel(table_name = chunks)]
pub struct NewChunk {
    pub text: String,
    pub embedding: Vector,
}
