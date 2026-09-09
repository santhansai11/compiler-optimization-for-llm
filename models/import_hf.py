"""Import real LLM computation graphs from HuggingFace transformers.

Uses ``torch.fx`` (via the HF tracer) to capture the *actual* forward graph
of a pretrained model - real architecture, real downloaded weights - and
translates the captured aten operations into this project's IR vocabulary:

    aten.mm / matmul / bmm        -> MatMul  (+ ``weight`` attr for params)
    aten.addmm / linear           -> GEMM    (+ ``weight`` / ``bias``)
    aten.embedding                -> Gather  (+ ``weight``)
    aten.native_layer_norm        -> LayerNorm (+ ``weight`` / ``bias``)
    aten.mul / div by scalar      -> Scale (factor)
    aten.where / masked_fill      -> MaskedFill (causal masks detected)
    view / reshape / transpose /
    slice / getitem / unsqueeze   -> Reshape / Transpose / Slice (metadata)
    clone / detach / dropout(eval)-> Identity

Unsupported ops fail the import with an explicit message, so every graph
we build is fully executable by ``utils.executor`` and the bit-exact
correctness check stays meaningful.
"""

import numpy as np

# hf model key -> (display label, HF id, seq len used for tracing/execution)
HF_MODELS = {
    "tiny-gpt2": ("Tiny GPT-2 (test, 2 blocks)", "sshleifer/tiny-gpt2", 16),
    "distilgpt2": ("DistilGPT-2 (real, 82M)", "distilgpt2", 16),
    "gpt2": ("GPT-2 (real, 124M)", "gpt2", 16),
}

MODEL_SEQ = {key: item[2] for key, item in HF_MODELS.items()}

# ops whose parameter inputs become attributes instead of IR nodes
_INLINE_ROLES = {
    "addmm": {0: "bias", 2: "weight"},
    "mm": {1: "weight"},
    "matmul": {1: "weight"},
    "bmm": {1: "weight"},
    "linear": {1: "weight", 2: "bias"},
    "embedding": {0: "weight"},
    "native_layer_norm": {2: "weight", 3: "bias"},
    "layer_norm": {2: "weight", 3: "bias"},
}
# pure shape bookkeeping ops (safe to absorb into fused kernels)
VIEW_OPS = {"Reshape", "Transpose", "Slice", "Cast", "Broadcast", "Identity"}


def is_available():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _fx_op_name(fx_node):
    """'aten.t.default' -> 't'; non-aten -> its __name__."""
    target_str = str(fx_node.target)
    if target_str.startswith("aten."):
        return target_str[len("aten."):].split(".")[0]
    return getattr(fx_node.target, "__name__", target_str)


def _make_traced(model, ids):
    """Trace ``model`` with concrete tensors (handles HF control flow)."""
    import torch
    from torch.fx.experimental.proxy_tensor import make_fx

    class _Wrapper(torch.nn.Module):
        """Exposes only ``input_ids`` and returns the logits tensor."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids):
            return self.inner(input_ids=input_ids, use_cache=False).logits

    return make_fx(_Wrapper(model), tracing_mode="real")(ids)


def import_hf_graph(model_key, seq=None, batch=1):
    """Trace a HF model and translate it into a ComputationGraph.

    Returns ``(graph, import_info)``. Raises ``ImportError`` when torch or
    the model weights are unavailable and ``ValueError`` when the trace
    contains ops this translator does not support.
    """
    try:
        import torch
        import transformers
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Real-model import needs torch + transformers: "
            "pip install -r requirements-real.txt"
        ) from exc

    if model_key not in HF_MODELS:
        raise ValueError(f"Unknown HF model key: {model_key!r}")
    label, hf_id, default_seq = HF_MODELS[model_key]
    seq = int(seq or default_seq)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        hf_id, attn_implementation="eager"
    )
    model.eval()

    ids = torch.ones((batch, seq), dtype=torch.long)

    # Concrete (real-tensor) tracing: HF forward code contains data-shaped
    # control flow that symbolic tracers cannot cross.
    traced = _make_traced(model, ids)

    from ir.graph import ComputationGraph

    graph = ComputationGraph(f"hf_{model_key.replace('-', '_')}")
    config = model.config
    graph.meta["config"] = {
        "batch": batch,
        "seq": seq,
        "d_model": int(config.hidden_size),
        "input_shape": [seq],
        "llm": True,
        "vocab_size": int(config.vocab_size),
        "hf_model": hf_id,
    }
    graph.meta["source"] = f"HuggingFace `{hf_id}` via torch.fx"

    tensors = {}
    for node in traced.graph.nodes:
        if node.op == "get_attr":
            tensor = getattr(traced, node.target, None)
            if tensor is not None:
                tensors[node.target] = tensor.detach().numpy()

    translator = _Translator(graph, traced.graph, tensors)
    info = translator.run(batch=batch, seq=seq)
    info.update(
        {
            "pass": "hf_import",
            "model": hf_id,
            "label": label,
            "torch.fx nodes": len(list(traced.graph.nodes)),
            "parameters": int(sum(t.size for t in tensors.values())),
        }
    )
    graph.meta["hf_import"] = info
    return graph, info


class _Unmapped(Exception):
    """Raised by op handlers when an aten op cannot be translated."""


class _Translator:
    """Walks one traced fx graph and emits IR operations."""

    def __init__(self, graph, fx_graph, tensors):
        self.graph = graph
        self.fx_graph = fx_graph
        self.tensors = tensors
        self.nodes = {n.name: n for n in fx_graph.nodes}
        self.name_map = {}
        self.tensors_map = {}
        self.shape_env = {}
        self.int_env = {}
        self.skip = set()
        self.unmapped = []
        self.op_counts = {}

    # ---------------------------------------------------------------- helpers
    def _emit(self, name, op_type, shape=None, **attrs):
        safe = name.replace(".", "_")
        self.graph.add_operation(safe, op_type, **attrs)
        if shape is not None:
            self.graph.graph.nodes[safe]["shape"] = tuple(
                int(dim) for dim in shape
            )
        self.op_counts[op_type] = self.op_counts.get(op_type, 0) + 1
        return safe

    def _resolve_int(self, value):
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return int(value)
        if hasattr(value, "name"):
            return self.int_env.get(value.name)
        return None

    def _resolve_shape(self, values):
        if not isinstance(values, (list, tuple)):
            return None
        dims = [self._resolve_int(v) for v in values]
        return dims if all(d is not None for d in dims) else None

    def _ir(self, arg):
        """IR node name for an fx arg (None for params/scalars)."""
        if hasattr(arg, "name"):
            mapped = self.name_map.get(arg.name)
            if isinstance(mapped, tuple):  # raw parameter tensor
                return None
            return mapped
        return None

    def _tensor(self, arg):
        """Numpy tensor for a parameter fx arg (or None)."""
        if hasattr(arg, "name"):
            return self.tensors_map.get(arg.name)
        return None

    def _preds(self, args):
        preds = []
        for arg in args:
            ir = self._ir(arg)
            if ir is not None:
                preds.append(ir)
        return preds

    def _shape_of(self, arg):
        if hasattr(arg, "name"):
            return self.shape_env.get(arg.name)
        return None

    # ------------------------------------------------------------------- main
    def run(self, batch, seq):
        self.shape_env["input_ids"] = (batch, seq)
        # fx graphs list their nodes in topological order already
        for node in self.fx_graph.nodes:
            self._translate(node)
        if self.unmapped:
            raise ValueError(
                "torch.fx trace contains unsupported ops: "
                + ", ".join(sorted(set(self.unmapped)))
                + f" (first failure at fx node {self.first_failure!r})"
            )
        return {
            "ops_emitted": self.graph.node_count(),
            "op_histogram": dict(sorted(self.op_counts.items())),
            "unmapped": [],
        }

    def _translate(self, node):
        kind = node.op
        name = node.name

        if kind == "placeholder":
            self.name_map[name] = self._emit(
                "input_ids", "Input", shape=self.shape_env["input_ids"]
            )
            self.shape_env[name] = self.shape_env["input_ids"]
            return
        if kind == "get_attr":
            tensor = self.tensors.get(node.target)
            if tensor is None:
                self.unmapped.append(f"attr:{node.target}")
                self.name_map[name] = None
                return
            self.name_map[name] = ("__param__", tensor)
            self.tensors_map[name] = np.asarray(tensor)
            self.shape_env[name] = tuple(tensor.shape)
            return
        if kind == "output":
            outputs = node.args[0]
            if not isinstance(outputs, (tuple, list)):
                outputs = (outputs,)
            for item in outputs:
                if hasattr(item, "name") and isinstance(
                    self.name_map.get(item.name), str
                ):
                    ir_name = self.name_map[item.name]
                    if self.graph.has_operation(ir_name):
                        self.graph.graph.nodes[ir_name]["is_output"] = True
            return
        if kind == "call_module":
            self.unmapped.append(f"module:{getattr(node, 'target', '?')}")
            self.name_map[name] = None
            return

        target_str = str(node.target)
        if target_str.startswith("aten."):
            # "aten.view.default" -> "view", "aten.pow.Tensor_Scalar" -> "pow"
            op_name = target_str[len("aten."):].split(".")[0]
        else:
            op_name = _fx_op_name(node)
        args = node.args
        # ---- pure metadata probes: resolve to ints, emit nothing --------
        if op_name == "size" and args:
            src = self._shape_of(args[0])
            if src is not None:
                self.int_env[name] = tuple(src)
                self.skip.add(name)
                self.name_map[name] = None
                return
        if op_name == "numel" and args:
            src = self._shape_of(args[0])
            if src is not None:
                self.int_env[name] = int(np.prod(src))
                self.skip.add(name)
                self.name_map[name] = None
                return
        if op_name == "getattr" and len(args) >= 2:
            base = self.int_env.get(args[0].name)
            if isinstance(base, tuple) and args[1] == "numel":
                self.int_env[name] = int(np.prod(base))
                self.skip.add(name)
                self.name_map[name] = None
                return
        if op_name == "__getitem__" and len(args) >= 2:
            base = self.int_env.get(args[0].name)
            if isinstance(base, tuple):
                index = self._resolve_int(args[1])
                if index is not None and -len(base) <= index < len(base):
                    self.int_env[name] = base[index]
                    self.skip.add(name)
                    self.name_map[name] = None
                    return

        handler = getattr(self, f"_op_{op_name}", None)
        if handler is None:
            self.unmapped.append(op_name)
            self.first_failure = getattr(self, "first_failure", f"{name} ({op_name})")
            self.name_map[name] = None
            return
        # view ops applied directly to a parameter (mask buffers etc.)
        # fold eagerly into Constant nodes
        if op_name in ("slice", "select", "t", "transpose", "permute",
                       "view", "reshape", "_unsafe_view", "unsqueeze",
                       "squeeze", "expand"):
            if node.args and self._tensor(node.args[0]) is not None:
                if self._fold_param_view(node, name, op_name):
                    return
        try:
            handler(node, name)
        except _Unmapped:
            self.unmapped.append(op_name)
            self.first_failure = getattr(self, "first_failure", f"{name} ({op_name})")
            self.name_map[name] = None

    def _fold_param_view(self, node, name, op_name):
        """Fold a view op over a parameter tensor into a Constant node."""
        arr = np.asarray(self._tensor(node.args[0]), dtype=np.float32)
        try:
            if op_name == "slice":
                dim = self._resolve_int(node.args[1])
                start = self._resolve_int(node.args[2])
                end = (
                    self._resolve_int(node.args[3])
                    if len(node.args) > 3 and node.args[3] is not None
                    else None
                )
                step = (
                    self._resolve_int(node.args[4])
                    if len(node.args) > 4 and node.args[4] is not None
                    else 1
                )
                if dim is None or start is None:
                    return False
                if end is None or end > 2 ** 40:
                    end = arr.shape[dim]
                index = [slice(None)] * arr.ndim
                index[dim] = slice(start, end, step or 1)
                arr = arr[tuple(index)]
            elif op_name == "select":
                dim = self._resolve_int(node.args[1])
                idx = self._resolve_int(node.args[2])
                if dim is None or idx is None:
                    return False
                index = [slice(None)] * arr.ndim
                index[dim] = idx
                arr = arr[tuple(index)]
            elif op_name == "t":
                arr = arr.T
            elif op_name == "transpose":
                d0 = self._resolve_int(node.args[1])
                d1 = self._resolve_int(node.args[2])
                if d0 is None or d1 is None:
                    return False
                arr = np.swapaxes(arr, d0, d1)
            elif op_name == "permute":
                dims = self._resolve_shape(node.args[1])
                if dims is None:
                    return False
                arr = np.transpose(arr, dims)
            elif op_name in ("view", "reshape", "_unsafe_view"):
                shape = self._resolve_view_shape(node.args[0], node.args[1])
                if shape is None:
                    return False
                arr = arr.reshape(shape)
            elif op_name == "unsqueeze":
                dim = self._resolve_int(node.args[1])
                if dim is None:
                    return False
                arr = np.expand_dims(arr, dim)
            elif op_name == "squeeze":
                arr = np.squeeze(arr)
            elif op_name == "expand":
                dims = []
                src = arr.shape
                for i, value in enumerate(node.args[1]):
                    resolved = self._resolve_int(value)
                    if resolved is None:
                        return False
                    dims.append(src[i] if resolved == -1 else resolved)
                arr = np.broadcast_to(arr, tuple(dims))
            else:
                return False
        except Exception:
            return False
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        ir = self._emit(name, "Constant", shape=arr.shape, value=arr)
        self.name_map[name] = ir
        self.shape_env[name] = arr.shape
        return True

    # ------------------------------------------------------------ parameters
    def _scalar_val(self, arg):
        """Float value of a python number or 0-dim constant fx node."""
        if isinstance(arg, bool):
            return None
        if isinstance(arg, (int, float)):
            return float(arg)
        if hasattr(arg, "name"):
            node = self.nodes.get(arg.name)
            if node is not None and node.op == "call_function":
                op = _fx_op_name(node)
                if op in ("full", "scalar_tensor", "tensor") and node.args:
                    raw = node.args[1] if op == "full" else node.args[0]
                    if isinstance(raw, (int, float)):
                        return float(raw)
        return None

    def _is_const_node(self, ir_name):
        if ir_name is None or not self.graph.has_operation(ir_name):
            return False
        return self.graph.op_type(ir_name) == "Constant"

    def _weight_arg(self, arg):
        """Resolve a weight operand: param, or aten.t(param) -> param.T."""
        tensor = self._tensor(arg)
        if tensor is not None:
            return np.asarray(tensor, dtype=np.float32)
        if hasattr(arg, "name"):
            node = self.nodes.get(arg.name)
            if (
                node is not None
                and node.op == "call_function"
                and _fx_op_name(node) == "t"
                and node.args
            ):
                base = self._tensor(node.args[0])
                if base is not None:
                    self.skip.add(node.name)
                    self.name_map[node.name] = None
                    return np.asarray(base, dtype=np.float32).T
        return None

    def _bcast_shape(self, shapes):
        out = ()
        for shape in shapes:
            if shape is None:
                return None
            if not out:
                out = tuple(shape)
                continue
            width = max(len(out), len(shape))
            a = (1,) * (width - len(out)) + tuple(out)
            b = (1,) * (width - len(shape)) + tuple(shape)
            out = tuple(
                max(x, y) if x == 1 or y == 1 or x == y else max(x, y)
                for x, y in zip(a, b)
            )
        return out

    # ---------------------------------------------------------- op handlers
    def _op_embedding(self, node, name):
        weight = self._weight_arg(node.args[0])
        ids_ir = self._ir(node.args[1])
        if weight is None or ids_ir is None:
            raise _Unmapped()
        ids_shape = self._shape_of(node.args[1]) or ()
        shape = tuple(ids_shape) + tuple(weight.shape[1:])
        ir = self._emit(name, "Gather", shape=shape, weight=weight)
        self.graph.add_dependency(ids_ir, ir)
        self.name_map[name] = ir
        self.shape_env[name] = shape

    def _op_mm(self, node, name):
        a_ir = self._ir(node.args[0])
        weight = self._weight_arg(node.args[1])
        if a_ir is None or weight is None:
            raise _Unmapped()
        a_shape = self._shape_of(node.args[0]) or ()
        shape = tuple(a_shape[:-1]) + tuple(weight.shape[1:])
        ir = self._emit(name, "MatMul", shape=shape, weight=weight)
        self.graph.add_dependency(a_ir, ir)
        self.name_map[name] = ir
        self.shape_env[name] = shape

    _op_bmm = _op_mm

    def _op_matmul(self, node, name):
        a, b = node.args[0], node.args[1]
        a_ir, b_ir = self._ir(a), self._ir(b)
        w = self._weight_arg(b) if b_ir is None else None
        if a_ir is None:
            raise _Unmapped()
        if w is not None:
            a_shape = self._shape_of(a) or ()
            shape = tuple(a_shape[:-1]) + tuple(w.shape[1:])
            ir = self._emit(name, "MatMul", shape=shape, weight=w)
            self.graph.add_dependency(a_ir, ir)
        elif b_ir is not None:
            shape = self._matmul_shape(self._shape_of(a),
                                       self._shape_of(b))
            ir = self._emit(name, "MatMul", shape=shape)
            self.graph.add_dependency(a_ir, ir)
            self.graph.add_dependency(b_ir, ir)
        else:
            raise _Unmapped()
        self.name_map[name] = ir
        self.shape_env[name] = shape

    def _matmul_shape(self, sa, sb):
        if sa is None or sb is None:
            return None
        if sa[-1] == sb[-2]:
            return tuple(sa[:-1]) + tuple(sb[-1:])
        if sb[-1] == sa[-2]:
            return tuple(sb[:-1]) + tuple(sa[-1:])
        if len(sa) >= 3 and len(sb) >= 3 and sa[-1] == sb[-1]:
            return tuple(sa[:-2]) + (sa[-2], sb[-2])
        return None

    _op_bmm = _op_matmul

    def _op_addmm(self, node, name):
        bias = self._tensor(node.args[0])
        a_ir = self._ir(node.args[1])
        weight = self._weight_arg(node.args[2])
        if bias is None or a_ir is None or weight is None:
            raise _Unmapped()
        a_shape = self._shape_of(node.args[1]) or ()
        shape = tuple(a_shape[:-1]) + tuple(weight.shape[1:])
        ir = self._emit(
            name, "GEMM", shape=shape,
            weight=weight, bias=np.asarray(bias, dtype=np.float32),
        )
        self.graph.add_dependency(a_ir, ir)
        self.name_map[name] = ir
        self.shape_env[name] = shape

    def _op_linear(self, node, name):
        x_ir = self._ir(node.args[0])
        raw = self._tensor(node.args[1])
        if x_ir is None or raw is None:
            raise _Unmapped()
        weight = np.asarray(raw, dtype=np.float32).T  # torch: x @ W.T
        bias = None
        if len(node.args) > 2 and node.args[2] is not None:
            bias = self._tensor(node.args[2])
            if bias is None:
                raise _Unmapped()
        x_shape = self._shape_of(node.args[0]) or ()
        shape = tuple(x_shape[:-1]) + tuple(weight.shape[1:])
        if bias is None:
            ir = self._emit(name, "MatMul", shape=shape, weight=weight)
        else:
            ir = self._emit(
                name, "GEMM", shape=shape, weight=weight,
                bias=np.asarray(bias, dtype=np.float32),
            )
        self.graph.add_dependency(x_ir, ir)
        self.name_map[name] = ir
        self.shape_env[name] = shape

    def _op_native_layer_norm(self, node, name):
        x_ir = self._ir(node.args[0])
        gamma = self._weight_arg(node.args[2]) if len(node.args) > 2 else None
        beta = self._tensor(node.args[3]) if len(node.args) > 3 else None
        eps = (
            self._scalar_val(node.args[4])
            if len(node.args) > 4
            else 1e-5
        )
        if x_ir is None or gamma is None or beta is None:
            raise _Unmapped()
        shape = self._shape_of(node.args[0])
        ir = self._emit(
            name, "LayerNorm", shape=shape,
            weight=np.asarray(gamma, dtype=np.float32),
            bias=np.asarray(beta, dtype=np.float32),
            eps=float(eps or 1e-5),
        )
        self.graph.add_dependency(x_ir, ir)
        self.name_map[name] = ir
        self.shape_env[name] = shape

    _op_layer_norm = _op_native_layer_norm

    # ------------------------------------------------------- elementwise ops
    def _op_add(self, node, name):
        preds = []
        for arg in node.args:
            ir = self._ir(arg)
            if ir is not None:
                preds.append(ir)
                continue
            scalar = self._scalar_val(arg)
            if scalar is not None:  # fold scalar operand into a Constant
                const = self._emit(
                    f"{name}_c", "Constant", value=np.float32(scalar)
                )
                preds.append(const)
                continue
            raise _Unmapped()
        shapes = [self._shape_of(a) for a in node.args
                  if hasattr(a, "name") and self._shape_of(a) is not None]
        shape = self._bcast_shape(shapes)
        ir = self._emit(name, "Add", shape=shape)
        for pred in preds:
            self.graph.add_dependency(pred, ir)
        self.name_map[name] = ir
        self.shape_env[name] = shape

    _op_sub = _op_add

    def _op_mul(self, node, name):
        # NewGELU pattern -> single GELU node
        gelu = self._try_newgelu(node, name)
        if gelu is not None:
            return
        a, b = node.args
        fa, fb = self._scalar_val(a), self._scalar_val(b)
        a_ir, b_ir = self._ir(a), self._ir(b)
        if a_ir is not None and fb is not None:
            shape = self._shape_of(a)
            ir = self._emit(name, "Scale", shape=shape, factor=fb)
            self.graph.add_dependency(a_ir, ir)
        elif b_ir is not None and fa is not None:
            shape = self._shape_of(b)
            ir = self._emit(name, "Scale", shape=shape, factor=fa)
            self.graph.add_dependency(b_ir, ir)
        elif a_ir is not None and b_ir is not None:
            shape = self._bcast_shape([self._shape_of(a), self._shape_of(b)])
            ir = self._emit(name, "Mul", shape=shape)
            self.graph.add_dependency(a_ir, ir)
            self.graph.add_dependency(b_ir, ir)
        else:
            raise _Unmapped()
        self.name_map[name] = ir
        self.shape_env[name] = shape

    def _op_div(self, node, name):
        a, b = node.args
        a_ir = self._ir(a)
        fb = self._scalar_val(b)
        if a_ir is None or fb is None:
            raise _Unmapped()
        shape = self._shape_of(a)
        ir = self._emit(name, "Scale", shape=shape, factor=1.0 / fb)
        self.graph.add_dependency(a_ir, ir)
        self.name_map[name] = ir
        self.shape_env[name] = shape

    def _op_gelu(self, node, name):
        x_ir = self._ir(node.args[0])
        if x_ir is None:
            raise _Unmapped()
        ir = self._emit(name, "GELU", shape=self._shape_of(node.args[0]))
        self.graph.add_dependency(x_ir, ir)
        self.name_map[name] = ir
        self.shape_env[name] = self._shape_of(node.args[0])

    _op_relu = _op_gelu

    def _op_pow(self, node, name):
        x_ir = self._ir(node.args[0])
        exponent = (
            self._scalar_val(node.args[1]) if len(node.args) > 1 else None
        )
        if x_ir is None or exponent is None:
            raise _Unmapped()
        shape = self._shape_of(node.args[0])
        ir = self._emit(name, "Pow", shape=shape, exponent=float(exponent))
        self.graph.add_dependency(x_ir, ir)
        self.name_map[name] = ir
        self.shape_env[name] = shape

    def _op_tanh(self, node, name):
        x_ir = self._ir(node.args[0])
        if x_ir is None:
            raise _Unmapped()
        shape = self._shape_of(node.args[0])
        ir = self._emit(name, "Tanh", shape=shape)
        self.graph.add_dependency(x_ir, ir)
        self.name_map[name] = ir
        self.shape_env[name] = shape

    def _op_softmax(self, node, name):
        x_ir = self._ir(node.args[0])
        if x_ir is None:
            raise _Unmapped()
        dim = self._resolve_int(node.args[1]) if len(node.args) > 1 else -1
        shape = self._shape_of(node.args[0])
        ir = self._emit(name, "Softmax", shape=shape, dim=int(dim or -1))
        self.graph.add_dependency(x_ir, ir)
        self.name_map[name] = ir
        self.shape_env[name] = shape

    _op__softmax = _op_softmax
    _op__log_softmax = _op_softmax

    def _op_where(self, node, name):
        cond, x, y = node.args
        cond_ir, x_ir, y_ir = self._ir(cond), self._ir(x), self._ir(y)
        if x_ir is None:
            raise _Unmapped()
        shape = self._bcast_shape(
            [self._shape_of(cond), self._shape_of(x), self._shape_of(y)]
        )
        fill = self._scalar_val(y)
        ir = self._emit(
            name, "MaskedFill", shape=shape,
            fill_value=float(fill) if fill is not None else 0.0,
            where=True,
        )
        if cond_ir is not None:
            self.graph.add_dependency(cond_ir, ir)
        self.graph.add_dependency(x_ir, ir)
        if y_ir is not None:
            self.graph.add_dependency(y_ir, ir)
        self.name_map[name] = ir
        self.shape_env[name] = shape

    def _op_masked_fill(self, node, name):
        x, mask, value = node.args
        x_ir, mask_ir = self._ir(x), self._ir(mask)
        if x_ir is None:
            raise _Unmapped()
        fill = self._scalar_val(value)
        if fill is None:
            raise _Unmapped()
        shape = self._bcast_shape([self._shape_of(x), self._shape_of(mask)])
        ir = self._emit(
            name, "MaskedFill", shape=shape, fill_value=float(fill),
        )
        self.graph.add_dependency(x_ir, ir)
        if mask_ir is not None:
            self.graph.add_dependency(mask_ir, ir)
        self.name_map[name] = ir
        self.shape_env[name] = shape

    _op_masked_fill_ = _op_masked_fill

    # ------------------------------------------------------------ shape ops
    def _op_t(self, node, name):
        x_ir = self._ir(node.args[0])
        if x_ir is None:
            raise _Unmapped()
        shape = self._shape_of(node.args[0])
        out = tuple(reversed(shape)) if shape else None
        ir = self._emit(name, "Transpose", shape=out, dims=None)
        self.graph.add_dependency(x_ir, ir)
        self.name_map[name] = ir
        self.shape_env[name] = out

    def _op_transpose(self, node, name):
        x_ir = self._ir(node.args[0])
        d0, d1 = self._resolve_int(node.args[1]), self._resolve_int(
            node.args[2]
        )
        if x_ir is None or d0 is None or d1 is None:
            raise _Unmapped()
        shape = self._shape_of(node.args[0])
        n = len(shape) if shape else 0
        dims = list(range(n))
        dims[d0], dims[d1] = dims[d1], dims[d0]
        out = tuple(shape[d] for d in dims) if shape else None
        ir = self._emit(name, "Transpose", shape=out, dims=tuple(dims))
        self.graph.add_dependency(x_ir, ir)
        self.name_map[name] = ir
        self.shape_env[name] = out

    def _op_permute(self, node, name):
        x_ir = self._ir(node.args[0])
        dims = self._resolve_shape(node.args[1])
        if x_ir is None or dims is None:
            raise _Unmapped()
        shape = self._shape_of(node.args[0])
        out = tuple(shape[d] for d in dims) if shape else None
        ir = self._emit(name, "Transpose", shape=out, dims=tuple(dims))
        self.graph.add_dependency(x_ir, ir)
        self.name_map[name] = ir
        self.shape_env[name] = out

    def _resolve_view_shape(self, x_arg, values):
        dims = [self._resolve_int(v) for v in values]
        src = self._shape_of(x_arg)
        if any(d is None for d in dims):
            return None
        if -1 in dims:
            if src is None:
                return None
            known = 1
            for d in dims:
                if d != -1:
                    known *= d
            total = int(np.prod(src))
            if known == 0 or total % known:
                return None
            dims = [total // known if d == -1 else d for d in dims]
        return tuple(int(d) for d in dims)

    def _op_view(self, node, name):
        x_ir = self._ir(node.args[0])
        shape = self._resolve_view_shape(node.args[0], node.args[1])
        if x_ir is None or shape is None:
            raise _Unmapped()
        ir = self._emit(name, "Reshape", shape=shape, shape_arg=shape)
        self.graph.add_dependency(x_ir, ir)
        self.name_map[name] = ir
        self.shape_env[name] = shape

    _op_reshape = _op_view
    _op__unsafe_view = _op_view

    def _op_slice(self, node, name):
        x, dim, start, end = node.args[0], node.args[1], node.args[2], (
            node.args[3] if len(node.args) > 3 else None
        )
        step = node.args[4] if len(node.args) > 4 else 1
        x_ir = self._ir(x)
        d = self._resolve_int(dim)
        s = self._resolve_int(start)
        e = self._resolve_int(end) if end is not None else None
        st = self._resolve_int(step) or 1
        if x_ir is None or d is None or s is None:
            raise _Unmapped()
        shape = self._shape_of(x)
        if e is None or e > 2**40:
            e = shape[d] if shape else None
        if shape and e is not None:
            if s < 0:
                s += shape[d]
            if e < 0:
                e += shape[d]
            out = list(shape)
            out[d] = len(range(s, e, st))
            out = tuple(out)
        else:
            out = None
        ir = self._emit(
            name, "Slice", shape=out, start=s, stop=e, step=st, axis=d
        )
        self.graph.add_dependency(x_ir, ir)
        self.name_map[name] = ir
        self.shape_env[name] = out

    def _op_select(self, node, name):
        x, dim, idx = node.args
        x_ir = self._ir(x)
        d, i = self._resolve_int(dim), self._resolve_int(idx)
        if x_ir is None or d is None or i is None:
            raise _Unmapped()
        shape = self._shape_of(x)
        out = (
            tuple(shape[j] for j in range(len(shape)) if j != d)
            if shape
            else None
        )
        ir = self._emit(name, "Slice", shape=out, start=i, stop=i + 1,
                        step=1, axis=d, squeeze=True)
        self.graph.add_dependency(x_ir, ir)
        self.name_map[name] = ir
        self.shape_env[name] = out

    def _op_unsqueeze(self, node, name):
        x_ir = self._ir(node.args[0])
        d = self._resolve_int(node.args[1])
        if x_ir is None or d is None:
            raise _Unmapped()
        shape = self._shape_of(node.args[0]) or ()
        if d < 0:
            d += len(shape) + 1
        out = list(shape)
        out.insert(d, 1)
        out = tuple(out)
        ir = self._emit(name, "Reshape", shape=out, shape_arg=out)
        self.graph.add_dependency(x_ir, ir)
        self.name_map[name] = ir
        self.shape_env[name] = out

    def _op_squeeze(self, node, name):
        x_ir = self._ir(node.args[0])
        if x_ir is None:
            raise _Unmapped()
        shape = self._shape_of(node.args[0]) or ()
        out = tuple(d for d in shape if d != 1)
        ir = self._emit(name, "Reshape", shape=out, shape_arg=out)
        self.graph.add_dependency(x_ir, ir)
        self.name_map[name] = ir
        self.shape_env[name] = out

    def _op_expand(self, node, name):
        x_ir = self._ir(node.args[0])
        sizes = node.args[1]
        src = self._shape_of(node.args[0]) or ()
        if x_ir is None or not isinstance(sizes, (list, tuple)):
            raise _Unmapped()
        dims = []
        for i, value in enumerate(sizes):
            resolved = self._resolve_int(value)
            if resolved is None:
                raise _Unmapped()
            if resolved == -1:
                resolved = src[i] if i < len(src) else 1
            dims.append(resolved)
        shape = tuple(dims)
        ir = self._emit(name, "Broadcast", shape=shape, target_shape=shape)
        self.graph.add_dependency(x_ir, ir)
        self.name_map[name] = ir
        self.shape_env[name] = shape

    # -------------------------------------------------- pass-through / const
    def _identity(self, node, name):
        if not node.users:  # dead clone/alias: emit nothing
            self.name_map[name] = None
            self.skip.add(name)
            return
        x_ir = self._ir(node.args[0])
        if x_ir is None:
            raise _Unmapped()
        ir = self._emit(name, "Identity",
                        shape=self._shape_of(node.args[0]))
        self.graph.add_dependency(x_ir, ir)
        self.name_map[name] = ir
        self.shape_env[name] = self._shape_of(node.args[0])

    _op_clone = _identity
    _op_detach = _identity
    _op_contiguous = _identity
    _op_dropout = _identity
    _op_to = _identity
    _op__to_copy = _identity
    _op_float = _identity
    _op_alias = _identity

    def _op_arange(self, node, name):
        start = self._resolve_int(node.args[0]) or 0
        end = (
            self._resolve_int(node.args[1])
            if len(node.args) > 1
            else None
        )
        if end is None:
            end = start
            start = 0
        arr = np.arange(start, end, dtype=np.float32)
        ir = self._emit(name, "Constant", shape=arr.shape, value=arr)
        self.name_map[name] = ir
        self.shape_env[name] = arr.shape

    def _op_full(self, node, name):
        shape = self._resolve_shape(node.args[0])
        value = self._scalar_val(node.args[1]) if len(node.args) > 1 else 0.0
        if shape is None or value is None:
            raise _Unmapped()
        arr = np.full(shape, value, dtype=np.float32)
        ir = self._emit(
            name, "Constant",
            shape=shape if shape else None,
            value=arr if shape else np.float32(value),
        )
        self.name_map[name] = ir
        self.shape_env[name] = tuple(shape)

    def _op_ones(self, node, name):
        shape = self._resolve_shape(node.args[0])
        if shape is None:
            raise _Unmapped()
        ir = self._emit(name, "Constant", shape=shape,
                        value=np.ones(shape, dtype=np.float32))
        self.name_map[name] = ir
        self.shape_env[name] = tuple(shape)

    def _op_zeros(self, node, name):
        shape = self._resolve_shape(node.args[0])
        if shape is None:
            raise _Unmapped()
        ir = self._emit(name, "Constant", shape=shape,
                        value=np.zeros(shape, dtype=np.float32))
        self.name_map[name] = ir
        self.shape_env[name] = tuple(shape)

    _op_scalar_tensor = _op_full
    _op_tensor = _op_full

    def _op_tril(self, node, name):
        shape = self._shape_of(node.args[0])
        diag = self._resolve_int(node.args[1]) if len(node.args) > 1 else 0
        if shape is None or len(shape) != 2:
            raise _Unmapped()
        arr = np.tril(np.ones(shape, dtype=np.float32), k=int(diag or 0))
        ir = self._emit(name, "Constant", shape=shape, value=arr)
        self.name_map[name] = ir
        self.shape_env[name] = shape

    def _op_split(self, node, name):
        x_ir = self._ir(node.args[0])
        size = self._resolve_int(node.args[1])
        dim = self._resolve_int(node.args[2]) if len(node.args) > 2 else 0
        shape = self._shape_of(node.args[0])
        if x_ir is None or size is None:
            raise _Unmapped()
        self.split_env = getattr(self, "split_env", {})
        self.split_env[name] = (x_ir, size, int(dim or 0), shape)
        self.name_map[name] = None  # materialised through getitem
        self.shape_env[name] = shape

    def _op_getitem(self, node, name):
        x, key = node.args[0], node.args[1]
        x_node = self.nodes.get(x.name) if hasattr(x, "name") else None
        x_op = _fx_op_name(x_node) if x_node is not None and (
            x_node.op == "call_function"
        ) else ""
        # tuple outputs: native_layer_norm(x) -> (out, mean, rstd)
        if x_op in ("native_layer_norm", "layer_norm"):
            i = self._resolve_int(key)
            if i == 0:
                self.name_map[name] = self._ir(x)
                self.shape_env[name] = self._shape_of(x)
                self.skip.add(name)
                return
            if not node.users:  # auxiliary output nobody consumes
                self.name_map[name] = None
                self.skip.add(name)
                return
            raise _Unmapped()
        # split materialisation: getitem(split(x, size, dim), i) -> Slice
        if x_op == "split":
            entry = getattr(self, "split_env", {}).get(x.name)
            i = self._resolve_int(key)
            if entry is None or i is None:
                raise _Unmapped()
            x_ir, size, dim, shape = entry
            start, stop = i * size, (i + 1) * size
            out = (
                tuple(shape[j] if j != dim else size
                      for j in range(len(shape)))
                if shape else None
            )
            ir = self._emit(name, "Slice", shape=out, start=start,
                            stop=stop, step=1, axis=dim)
            self.graph.add_dependency(x_ir, ir)
            self.name_map[name] = ir
            self.shape_env[name] = out
            return
        # parameter / constant slicing (e.g. attention-mask buffers)
        tensor = self._tensor(x)
        if tensor is not None and isinstance(key, tuple):
            arr = np.asarray(tensor)
            index = tuple(
                item if isinstance(item, (slice, int)) else slice(None)
                for item in key
            )
            arr = np.asarray(arr[index], dtype=np.float32)
            ir = self._emit(name, "Constant", shape=arr.shape, value=arr)
            self.name_map[name] = ir
            self.shape_env[name] = arr.shape
            return
        raise _Unmapped()

    def _op_eq(self, node, name):
        a, b = node.args
        ta, tb = self._tensor(a), self._tensor(b)
        ca, cb = self._ir(a), self._ir(b)
        arr_a = ta if ta is not None else (
            self.graph.graph.nodes[ca].get("value")
            if self._is_const_node(ca) else None
        )
        arr_b = tb if tb is not None else (
            self.graph.graph.nodes[cb].get("value")
            if self._is_const_node(cb) else None
        )
        if arr_a is None or arr_b is None:
            raise _Unmapped()
        arr = (
            np.asarray(arr_a) == np.asarray(arr_b)
        ).astype(np.float32)
        ir = self._emit(name, "Constant", shape=arr.shape, value=arr)
        self.name_map[name] = ir
        self.shape_env[name] = arr.shape

    _op_ne = _op_eq

    # ------------------------------------------------------- NewGELU pattern
    def _try_newgelu(self, node, name):
        """Detect HF's NewGELUActivation primitive chain and fold it.

        0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))  ->  GELU(x)
        """
        nodes = self.nodes

        def as_node(arg):
            return nodes.get(arg.name) if hasattr(arg, "name") else None

        def is_scalar(arg, expected):
            value = self._scalar_val(arg)
            return value is not None and abs(value - expected) < 1e-3

        outer = node
        args = outer.args
        if len(args) != 2:
            return None
        inner_mul, add1 = (args[0], args[1]) if self._scalar_val(
            args[0]
        ) is None else (args[1], args[0])
        n_inner_mul, n_add1 = as_node(inner_mul), as_node(add1)
        if n_inner_mul is None or n_add1 is None:
            return None
        if _fx_op_name(n_add1) != "add":
            return None
        if not any(is_scalar(a, 1.0) for a in n_add1.args):
            return None
        tanh_n = next(
            (as_node(a) for a in n_add1.args
             if as_node(a) is not None
             and _fx_op_name(as_node(a)) == "tanh"),
            None,
        )
        if tanh_n is None:
            return None
        if len(n_inner_mul.args) != 2:
            return None
        x_arg = next(
            (a for a in n_inner_mul.args if self._scalar_val(a) is None),
            None,
        )
        if not any(is_scalar(a, 0.5) for a in n_inner_mul.args):
            return None
        if x_arg is None:
            return None
        mul3 = as_node(tanh_n.args[0])
        if mul3 is None or _fx_op_name(mul3) != "mul":
            return None
        c_arg = next(
            (a for a in mul3.args if self._scalar_val(a) is not None),
            None,
        )
        if c_arg is None or abs(self._scalar_val(c_arg) - 0.797885) > 1e-3:
            return None
        add2 = next(
            (as_node(a) for a in mul3.args if as_node(a) is not None
             and _fx_op_name(as_node(a)) == "add"),
            None,
        )
        if add2 is None or len(add2.args) != 2:
            return None
        if as_node(add2.args[0]) is not None and _fx_op_name(
            as_node(add2.args[0])
        ) == "add":
            add2.args = add2.args[::-1]
        x_ir = self._ir(x_arg)
        if x_ir is None:
            return None
        if add2.args[0] is not x_arg:
            # x must be the same tensor in both branches
            if not (hasattr(add2.args[0], "name")
                    and hasattr(x_arg, "name")
                    and add2.args[0].name == x_arg.name):
                return None
        mul4 = as_node(add2.args[1])
        if mul4 is None or _fx_op_name(mul4) != "mul":
            return None
        if not any(is_scalar(a, 0.044715) for a in mul4.args):
            return None
        pow_n = next(
            (as_node(a) for a in mul4.args if as_node(a) is not None
             and _fx_op_name(as_node(a)) == "pow"),
            None,
        )
        if pow_n is None:
            return None
        if not (hasattr(pow_n.args[0], "name") and hasattr(x_arg, "name")
                and pow_n.args[0].name == x_arg.name):
            return None

        # ---- match confirmed: emit GELU, drop the primitive chain ------
        members = (outer, n_inner_mul, n_add1, tanh_n, mul3, add2,
                   mul4, pow_n)
        # capture the primitive nodes' IR (outer has none yet)
        old_irs = {}
        for fx_node in members:
            old = self.name_map.get(fx_node.name)
            if isinstance(old, str) and self.graph.has_operation(old):
                old_irs[old] = self.graph.op_type(old)
        # the pattern must be self-contained: if any primitive still feeds
        # an outside consumer we cannot remove it (bail -> plain Mul)
        member_set = set(old_irs)
        for ir_name in member_set:
            for succ in self.graph.successors(ir_name):
                if succ not in member_set:
                    return None
        ir = self._emit(name, "GELU", shape=self._shape_of(x_arg))
        self.graph.add_dependency(x_ir, ir)
        # drop the primitive-chain IR nodes and their mappings
        for ir_name, old_op in old_irs.items():
            self.graph.graph.remove_node(ir_name)
            self.op_counts[old_op] = max(
                0, self.op_counts.get(old_op, 1) - 1
            )
        for fx_node in members:
            self.name_map[fx_node.name] = None
            self.skip.add(fx_node.name)
        self.name_map[name] = ir
        self.shape_env[name] = self._shape_of(x_arg)
        return ir









