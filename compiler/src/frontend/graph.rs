struct RawGraph {
    nodes: Vec<RawNode>,
}

struct Args {
    kind: String,
    name: String,
    op: String,
    target: String,
}

struct RawNode {
    index: usize,
    name: String,
    op: String,
    target: String,
    args: Vec<Args>,
    kwargs: Kwargs,
    users: Vec<String>,
    tensor_meta: TensorMeta,
}

struct Kwargs {
    dtype: String,
    device: String,
    layout: String,
}

struct TensorMeta {
    shape: Vec<usize>,
    dtype: String,
    requires_grad: bool,
    stride: Vec<usize>,
}

impl RawGraph {
    fn from_graph_json(dump: &str) {}
}
