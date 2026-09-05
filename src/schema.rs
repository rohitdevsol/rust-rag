// @generated automatically by Diesel CLI.

diesel::table! {
    use diesel::sql_types::*;
    use pgvector::sql_types::*;

    chunks (id) {
        id -> Int4,
        text -> Text,
        embedding -> Vector,
    }
}
