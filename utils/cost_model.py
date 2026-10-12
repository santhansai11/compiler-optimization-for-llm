"""Simulated hardware cost model.

The project runs without a real accelerator, so modeled GPU numbers come
from a deterministic per-operator cost table (datacenter-class GPU style).
Every operator node surviving into the final graph costs one kernel launch.
Measured (NumPy) numbers live in ``utils.executor`` and are reported
alongside the modeled ones.
"""

LAUNCH_OVERHEAD_US = 12.0

OP_LATENCY_US = {
    "Input": 8.0,
    "Constant": 6.0,
    "MatMul": 850.0,
    "GEMM": 920.0,
    "LinearGELU": 1010.0,
    "FusedAddLayerNorm": 210.0,
    "FusedScaleSoftmax": 290.0,
    "FusedAttention": 1650.0,
    "Scale": 65.0,
    "Softmax": 240.0,
    "Add": 95.0,
    "Mul": 70.0,
    "Sub": 70.0,
    "LayerNorm": 160.0,
    "GELU": 130.0,
    "Relu": 45.0,
    "LinearRelu": 940.0,
    "Identity": 20.0,
    "Gather": 60.0,
    "Slice": 15.0,
    "Reshape": 5.0,
    "Transpose": 110.0,
    "MaskedFill": 20.0,
    "Broadcast": 5.0,
    "Cast": 6.0,
    "Tanh": 45.0,
    "Pow": 60.0,
    "RMSNorm": 120.0,
    "SiLU": 55.0,
    "LinearSiLU": 930.0,
    "FusedAddRMSNorm": 170.0,
}

OP_MEMORY_MB = {
    "Input": 48.0,
    "Constant": 24.0,
    "MatMul": 96.0,
    "GEMM": 110.0,
    "LinearGELU": 120.0,
    "FusedAddLayerNorm": 64.0,
    "FusedScaleSoftmax": 72.0,
    "FusedAttention": 190.0,
    "Scale": 24.0,
    "Softmax": 48.0,
    "Add": 32.0,
    "Mul": 24.0,
    "Sub": 24.0,
    "LayerNorm": 40.0,
    "GELU": 32.0,
    "Relu": 16.0,
    "LinearRelu": 115.0,
    "Identity": 16.0,
    "Gather": 40.0,
    "Slice": 12.0,
    "Reshape": 2.0,
    "Transpose": 48.0,
    "MaskedFill": 12.0,
    "Broadcast": 2.0,
    "Cast": 3.0,
    "Tanh": 16.0,
    "Pow": 16.0,
    "RMSNorm": 36.0,
    "SiLU": 20.0,
    "LinearSiLU": 112.0,
    "FusedAddRMSNorm": 56.0,
}

# Pure shape bookkeeping: real runtimes execute these as views with no
# kernel launch, so they must not count toward kernel-launch metrics.
VIEW_OPS = {"Reshape", "Cast", "Broadcast", "Identity"}

DEFAULT_LATENCY_US = 150.0
DEFAULT_MEMORY_MB = 40.0


def get_op_latency(op_type):
    return OP_LATENCY_US.get(op_type, DEFAULT_LATENCY_US)


def get_op_memory(op_type):
    return OP_MEMORY_MB.get(op_type, DEFAULT_MEMORY_MB)


def kernel_launches(graph):
    """Each non-view operator surviving into the final graph = one launch."""
    return sum(
        1
        for _, data in graph.nodes(data=True)
        if data.get("op_type") not in VIEW_OPS
    )


def sequential_latency_us(graph):
    total = sum(
        get_op_latency(data.get("op_type"))
        for _, data in graph.nodes(data=True)
    )
    return total + LAUNCH_OVERHEAD_US * graph.node_count()


def scheduled_latency_us(graph):
    """Level-parallel latency when DAGS metadata exists, else sequential."""
    schedule = graph.meta.get("schedule")
    if not schedule or not schedule.get("level_latency_max_us"):
        return sequential_latency_us(graph)
    compute = sum(schedule["level_latency_max_us"])
    return compute + LAUNCH_OVERHEAD_US * graph.node_count()


def naive_peak_memory_mb(graph):
    """Baseline: every intermediate tensor stays live (no buffer reuse)."""
    return sum(
        get_op_memory(data.get("op_type"))
        for _, data in graph.nodes(data=True)
    )


def live_range_peak_memory_mb(graph):
    """Peak memory with liveness: a tensor is freed once all of its
    consumers have executed (classic buffer-reuse planning)."""
    remaining_uses = {
        node: graph.graph.out_degree(node) for node in graph.graph.nodes()
    }
    alive = 0.0
    peak = 0.0
    for node in graph.topological_order():
        alive += get_op_memory(graph.op_type(node))
        peak = max(peak, alive)
        for pred in graph.predecessors(node):
            remaining_uses[pred] -= 1
            if (
                remaining_uses[pred] == 0
                and not graph.graph.nodes[pred].get("is_output")
            ):
                alive -= get_op_memory(graph.op_type(pred))
    return peak


# ----------------------------------------------------------------------
# Multi-device deployment model (used by the learned optimizer)
# ----------------------------------------------------------------------
# Assumed cost of moving one tensor between devices (link latency + a
# small activation). This is a modeling assumption, not a measurement.
COMM_US = 150.0


def deployment_latency_us(graph):
    """Makespan of the graph on D devices (D = number of partitions).

    Each device runs its kernels one after another (large kernels saturate
    a GPU); a kernel starts once its device is free and every input has
    arrived, paying ``COMM_US`` for inputs produced on another device.
    With no partition annotation this is exactly the sequential latency,
    so parallel branches only pay off when split across devices - and
    splitting a chain only adds communication.
    """
    g = graph.graph
    order = graph.topological_order()
    if all("schedule_order" in g.nodes[n] for n in order):
        order = sorted(order, key=lambda n: g.nodes[n]["schedule_order"])
    device = {n: g.nodes[n].get("partition", 0) for n in order}
    free, finish = {}, {}
    for node in order:
        d = device[node]
        ready = max((finish[p] + (COMM_US if device[p] != d else 0.0)
                     for p in g.predecessors(node)), default=0.0)
        start = max(free.get(d, 0.0), ready)
        finish[node] = start + get_op_latency(g.nodes[node].get("op_type")) \
            + LAUNCH_OVERHEAD_US
        free[d] = finish[node]
    return max(finish.values(), default=0.0)
