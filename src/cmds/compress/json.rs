//! SmartCrusher: deterministic JSON minification plus a columnar form for
//! arrays of uniform objects.
//!
//! Columnar encoding: an array of at least [`MIN_ROWS`] objects that all
//! carry the same key set becomes `{"_cols":[k1,…,kn],"rows":[[v1,…,vn],…]}`.
//! `_cols` follows the first object's key order; each row lists its object's
//! values in that order, and a `null` cell is written as the one-character
//! string `"∅"` ([`NULL_CELL`]). Children are rewritten bottom-up, so a nested
//! uniform array lands in a row already columnar. Everything else keeps its
//! key order and values verbatim.

use serde_json::{Map, Value};

use crate::core::utils;

/// Written in place of a `null` cell inside `rows`.
const NULL_CELL: &str = "∅";
/// Arrays smaller than this are not worth the columnar header.
const MIN_ROWS: usize = 3;

/// Parse, rewrite and re-serialize `text` minified. Unparseable input comes
/// back unchanged — [`super::classify`] should already have guaranteed JSON.
pub(crate) fn crush_kind(text: &str) -> String {
    let Ok(value) = utils::from_json_str::<Value>(text) else {
        return text.to_owned();
    };
    // serde_json::to_string on a Value cannot fail; the fallback is defensive.
    serde_json::to_string(&columnize(value)).unwrap_or_else(|_| text.to_owned())
}

/// Rewrite `value` bottom-up: children first, then the columnar check.
fn columnize(value: Value) -> Value {
    match value {
        Value::Array(items) => {
            let items: Vec<Value> = items.into_iter().map(columnize).collect();
            to_table(&items).unwrap_or(Value::Array(items))
        }
        Value::Object(map) => {
            Value::Object(map.into_iter().map(|(k, v)| (k, columnize(v))).collect())
        }
        other => other,
    }
}

/// `items` as a columnar object when every element is an object over the same
/// key set, else `None`.
fn to_table(items: &[Value]) -> Option<Value> {
    if items.len() < MIN_ROWS {
        return None;
    }
    let cols: Vec<String> = items.first()?.as_object()?.keys().cloned().collect();
    if cols.is_empty() {
        return None;
    }
    let mut rows = Vec::with_capacity(items.len());
    for item in items {
        let obj = item.as_object()?;
        if obj.len() != cols.len() || !cols.iter().all(|k| obj.contains_key(k)) {
            return None;
        }
        rows.push(Value::Array(
            cols.iter()
                .map(|k| match obj.get(k) {
                    Some(Value::Null) | None => Value::String(NULL_CELL.to_string()),
                    Some(v) => v.clone(),
                })
                .collect(),
        ));
    }
    let mut table = Map::new();
    table.insert(
        "_cols".to_string(),
        Value::Array(cols.into_iter().map(Value::String).collect()),
    );
    table.insert("rows".to_string(), Value::Array(rows));
    Some(Value::Object(table))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn minifies_without_columnar() {
        assert_eq!(
            crush_kind("{ \"a\": 1, \"b\": [1, 2] }"),
            r#"{"a":1,"b":[1,2]}"#
        );
    }

    #[test]
    fn uniform_object_arrays_become_columnar() {
        let input = r#"[{"name": "a", "v": 1}, {"name": "b", "v": 2}, {"name": "c", "v": null}]"#;
        assert_eq!(
            crush_kind(input),
            r#"{"_cols":["name","v"],"rows":[["a",1],["b",2],["c","∅"]]}"#
        );
    }

    #[test]
    fn columnar_shape_stays_readable() {
        let out = crush_kind(r#"[{"k":1},{"k":2},{"k":3}]"#);
        let value: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(value["_cols"], json!(["k"]));
        assert_eq!(value["rows"], json!([[1], [2], [3]]));
    }

    #[test]
    fn mixed_shape_arrays_stay_rows() {
        let input = r#"[{"a":1},{"a":2,"b":3},{"a":4}]"#;
        assert_eq!(crush_kind(input), input);
    }

    #[test]
    fn two_objects_are_not_enough_rows() {
        let input = r#"[{"a":1},{"a":2}]"#;
        assert_eq!(crush_kind(input), input);
    }

    #[test]
    fn nested_uniform_arrays_columnize_too() {
        let input = r#"{"data": [{"x": 1}, {"x": 2}, {"x": 3}]}"#;
        assert_eq!(
            crush_kind(input),
            r#"{"data":{"_cols":["x"],"rows":[[1],[2],[3]]}}"#
        );
    }

    #[test]
    fn same_keys_different_order_still_columnize() {
        let input = r#"[{"a":1,"b":2},{"b":3,"a":4},{"a":5,"b":6}]"#;
        assert_eq!(
            crush_kind(input),
            r#"{"_cols":["a","b"],"rows":[[1,2],[4,3],[5,6]]}"#
        );
    }

    #[test]
    fn unparseable_input_is_unchanged() {
        assert_eq!(crush_kind("{not json"), "{not json");
    }
}
