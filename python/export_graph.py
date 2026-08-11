import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading Qwen-1.5B onto your GPU...")
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="cuda", low_cpu_mem_usage=True
)

model.eval()

batch_size = 1
seq_len = 64
tokens = torch.randint(0, 151643, (batch_size, seq_len), dtype=torch.long).cuda()

# 1. Define keyword args to explicitly bypass Hugging Face's dynamic caching classes
kwargs = {"use_cache": False}

print("Tracing architecture into functional ATen FX graph...")
with torch.no_grad():
    # Pass kwargs containing 'use_cache=False' so the graph stays pure tensor-only
    exported_program = torch.export.export(
        model, args=(tokens,), kwargs=kwargs, strict=True
    )

fx_graph = exported_program.graph_module.graph

repo_root = Path(__file__).resolve().parents[1]
debug_dir = repo_root / "examples" / "fx_debug"
graph_dir = repo_root / "examples" / "graphs"

debug_dir.mkdir(parents=True, exist_ok=True)
graph_dir.mkdir(parents=True, exist_ok=True)


def jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]

    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}

    try:
        return int(value)
    except (TypeError, ValueError):
        return repr(value)


def node_ref(value):
    if isinstance(value, torch.fx.Node):
        return {
            "kind": "node",
            "name": value.name,
            "op": value.op,
            "target": str(value.target),
        }

    if isinstance(value, (list, tuple)):
        return [node_ref(v) for v in value]

    if isinstance(value, dict):
        return {str(k): node_ref(v) for k, v in value.items()}

    return jsonable(value)


def tensor_meta_to_dict(meta):
    if meta is None:
        return None

    shape = getattr(meta, "shape", None)
    dtype = getattr(meta, "dtype", None)
    requires_grad = getattr(meta, "requires_grad", None)
    stride = getattr(meta, "stride", None)

    return {
        "shape": jsonable(list(shape)) if shape is not None else None,
        "dtype": str(dtype) if dtype is not None else None,
        "requires_grad": jsonable(requires_grad),
        "stride": jsonable(list(stride)) if stride is not None else None,
    }


# 1. Human-readable node table
with (debug_dir / "nodes.txt").open("w") as f:
    for idx, node in enumerate(fx_graph.nodes):
        f.write(f"[{idx}] name={node.name}\n")
        f.write(f"    op={node.op}\n")
        f.write(f"    target={node.target}\n")
        f.write(f"    args={node_ref(node.args)}\n")
        f.write(f"    kwargs={node_ref(node.kwargs)}\n")
        f.write(f"    users={[user.name for user in node.users]}\n")
        f.write(f"    meta={tensor_meta_to_dict(node.meta.get('tensor_meta'))}\n")
        f.write("\n")


# 2. Generated FX Python code
with (debug_dir / "graph_code.py").open("w") as f:
    f.write(exported_program.graph_module.code)


# 3. Structured dump
nodes = []
for idx, node in enumerate(fx_graph.nodes):
    nodes.append(
        {
            "index": idx,
            "name": node.name,
            "op": node.op,
            "target": str(node.target),
            "args": node_ref(node.args),
            "kwargs": node_ref(node.kwargs),
            "users": [user.name for user in node.users],
            "tensor_meta": tensor_meta_to_dict(node.meta.get("tensor_meta")),
        }
    )

debug_graph = {
    "model_id": model_id,
    "batch_size": batch_size,
    "seq_len": seq_len,
    "nodes": nodes,
}

with (debug_dir / "graph_debug.json").open("w") as f:
    json.dump(debug_graph, f, indent=2)

raw_graph = {
    "format": "phibi.raw_fx_graph.v0",
    "model_id": model_id,
    "source": "torch.export",
    "inputs": {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "use_cache": kwargs["use_cache"],
    },
    "nodes": nodes,
}

raw_graph_path = graph_dir / "qwen2_5_1_5b_fx_graph.json"
with raw_graph_path.open("w") as f:
    json.dump(raw_graph, f, indent=2)


print(f"Wrote FX debug files to {debug_dir}")
print(f"Wrote raw graph to {raw_graph_path}")

print("\n[SUCCESS] Extracted graph safely within your VRAM budget!")
print(f"Total compiled nodes ready for your Custom IR: {len(fx_graph.nodes)}")


print("\n=== Inspecting First 30 Nodes ===")
for idx, node in enumerate(fx_graph.nodes):
    if idx >= 30:
        break

    # Print node name, its high-level operation type, and its inputs
    print(
        f"Node: {node.name:<15} | Op: {node.op:<13} | Target: {node.target!s:<35} | Inputs: {[str(i) for i in node.args]}"
    )
