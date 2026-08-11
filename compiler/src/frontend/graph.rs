use std::collections::HashMap;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct RawGraph {
    pub format: String,
    pub model_id: String,
    pub source: String,
    /// Absent in graphs written before the exporter grew a `--mode` flag.
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub dtype: Option<String>,
    #[serde(default)]
    pub inputs: HashMap<String, Arg>,
    pub nodes: Vec<RawNode>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Op {
    Placeholder,
    CallFunction,
    CallMethod,
    CallModule,
    GetAttr,
    Output,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawNode {
    pub index: usize,
    pub name: String,
    pub op: Op,
    pub target: String,
    pub args: Vec<Arg>,
    #[serde(default)]
    pub kwargs: HashMap<String, Arg>,
    pub users: Vec<String>,
    pub tensor_meta: Option<TensorMeta>,
}

/// A reference to another node in the graph, as it appears inside `args`.
#[derive(Debug, Clone, Deserialize)]
pub struct NodeRef {
    pub kind: String,
    pub name: String,
    pub op: Op,
    pub target: String,
}

/// One positional argument. FX arguments are heterogeneous: node references,
/// scalars, and arbitrarily nested lists of both.
///
/// Variant order matters. `serde(untagged)` tries variants top to bottom, so
/// `Int` must precede `Float` or every integer would be parsed as a float.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Arg {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Node(NodeRef),
    List(Vec<Arg>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct TensorMeta {
    #[serde(default)]
    pub shape: Vec<usize>,
    pub dtype: Option<String>,
    pub requires_grad: Option<bool>,
    #[serde(default)]
    pub stride: Vec<usize>,
}

#[derive(Debug)]
pub enum ParseError {
    Json(serde_json::Error),
    UnsupportedFormat(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::Json(err) => write!(f, "malformed graph dump: {err}"),
            ParseError::UnsupportedFormat(found) => {
                write!(f, "unsupported graph format `{found}`, expected `{FORMAT}`")
            }
        }
    }
}

impl std::error::Error for ParseError {}

impl From<serde_json::Error> for ParseError {
    fn from(err: serde_json::Error) -> Self {
        ParseError::Json(err)
    }
}

pub const FORMAT: &str = "phibi.raw_fx_graph.v0";

impl RawGraph {
    pub fn from_graph_json(dump: &str) -> Result<RawGraph, ParseError> {
        let graph: RawGraph = serde_json::from_str(dump)?;

        if graph.format != FORMAT {
            return Err(ParseError::UnsupportedFormat(graph.format));
        }

        Ok(graph)
    }

    pub fn node(&self, name: &str) -> Option<&RawNode> {
        self.nodes.iter().find(|node| node.name == name)
    }

    pub fn nodes_with_op(&self, op: Op) -> impl Iterator<Item = &RawNode> {
        self.nodes.iter().filter(move |node| node.op == op)
    }
}

impl RawNode {
    /// Names of the nodes this node consumes, in argument order, flattened
    /// across nested lists.
    pub fn input_names(&self) -> Vec<&str> {
        let mut names = Vec::new();
        for arg in &self.args {
            arg.collect_node_names(&mut names);
        }
        names
    }
}

impl Arg {
    pub fn as_node(&self) -> Option<&NodeRef> {
        match self {
            Arg::Node(node) => Some(node),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Arg::Int(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_list(&self) -> Option<&[Arg]> {
        match self {
            Arg::List(items) => Some(items),
            _ => None,
        }
    }

    fn collect_node_names<'a>(&'a self, out: &mut Vec<&'a str>) {
        match self {
            Arg::Node(node) => out.push(&node.name),
            Arg::List(items) => {
                for item in items {
                    item.collect_node_names(out);
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN_PREFILL: &str =
        include_str!("../../../examples/graphs/qwen2_5_1_5b_instruct_prefill_fx_graph.json");

    fn qwen() -> RawGraph {
        RawGraph::from_graph_json(QWEN_PREFILL).expect("fixture should parse")
    }

    #[test]
    fn parses_the_qwen_prefill_fixture() {
        let graph = qwen();

        assert_eq!(graph.format, FORMAT);
        assert_eq!(graph.model_id, "Qwen/Qwen2.5-1.5B-Instruct");
        assert_eq!(graph.nodes.len(), 2151);
        assert_eq!(graph.nodes_with_op(Op::Placeholder).count(), 341);
        assert_eq!(graph.nodes_with_op(Op::CallFunction).count(), 1809);
        assert_eq!(graph.nodes_with_op(Op::Output).count(), 1);
    }

    #[test]
    fn reads_weight_placeholders() {
        let graph = qwen();
        let embed = graph
            .node("p_model_embed_tokens_weight")
            .expect("embedding weight");

        assert_eq!(embed.op, Op::Placeholder);
        assert!(embed.args.is_empty());
        let meta = embed.tensor_meta.as_ref().expect("weights carry meta");
        assert_eq!(meta.shape, vec![151936, 1536]);
        assert_eq!(meta.dtype.as_deref(), Some("torch.float16"));
    }

    #[test]
    fn handles_negative_and_large_integer_args() {
        let graph = qwen();

        // `aten.slice.Tensor` uses i64::MAX as an open-ended stop index, and
        // dim arguments are routinely negative. Neither fits in a usize.
        let ints: Vec<i64> = graph
            .nodes
            .iter()
            .flat_map(|node| node.args.iter())
            .filter_map(Arg::as_int)
            .collect();

        assert!(ints.contains(&i64::MAX));
        assert!(ints.iter().any(|&value| value < 0));
    }

    #[test]
    fn handles_sparse_and_typed_kwargs() {
        let graph = qwen();

        // Most nodes have no kwargs at all; the ones that do carry a mix of
        // strings, bools and floats rather than a fixed dtype/device/layout.
        assert!(graph.nodes.iter().any(|node| node.kwargs.is_empty()));

        let sdpa = graph
            .nodes
            .iter()
            .find(|node| node.target == "aten.scaled_dot_product_attention.default")
            .expect("attention node");
        assert!(matches!(sdpa.kwargs.get("scale"), Some(Arg::Float(_))));

        assert!(
            graph
                .nodes
                .iter()
                .any(|node| { matches!(node.kwargs.get("pin_memory"), Some(Arg::Bool(_))) })
        );
    }

    #[test]
    fn flattens_nested_node_arguments() {
        let graph = qwen();
        let output = graph
            .nodes_with_op(Op::Output)
            .next()
            .expect("graph has an output");

        // The output node's args are a list-of-list of node references.
        assert_eq!(output.input_names(), vec!["linear_196"]);
    }

    #[test]
    fn rejects_an_unknown_format() {
        let dump = r#"{"format":"something.else","model_id":"m","source":"s","nodes":[]}"#;
        assert!(matches!(
            RawGraph::from_graph_json(dump),
            Err(ParseError::UnsupportedFormat(_))
        ));
    }
}
