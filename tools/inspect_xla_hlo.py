"""Compare saved optimized computations, excluding XLA source/debug provenance.

No timing samples or original report files are modified.
"""
import argparse
import difflib
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.xla_runtime import normalized_hlo
from utils.benchmarking import write_json


def inspect(folder):
    folder = Path(folder)
    rows = json.loads((folder / "workloads.json").read_text())
    records = []
    for row in rows:
        before = normalized_hlo((folder / "hlo" / f'{row["programs"]["original"]["graph_id"]}.optimized.hlo').read_text())
        for name, program in row["programs"].items():
            if name == "original":
                continue
            after = normalized_hlo((folder / "hlo" / f'{program["graph_id"]}.optimized.hlo').read_text())
            filename = f'{row["workload"]}-{name}.computation.diff'
            (folder / "hlo" / filename).write_text("".join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
                                                fromfile="original",tofile=name)),encoding="utf-8")
            records.append({"workload":row["workload"],"variant":name,"same_computation_text":before==after,
                            "diff_file":f"hlo/{filename}", "original_temporary_bytes":row["programs"]["original"]["memory"]["temp_size_in_bytes"],
                            "candidate_temporary_bytes":program["memory"]["temp_size_in_bytes"]})
    result = {"normalization":"Remove instruction metadata, module name and FileNames/FunctionNames/FileLocations/StackFrames debug tables; retain computation text",
              "interpretation":"Textual comparison only, not machine-code equivalence or causal timing attribution",
              "inspector_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "normalizer_source_sha256":hashlib.sha256((ROOT/'utils/xla_runtime.py').read_bytes()).hexdigest(),
              "guided_identical_computations":sum(r["same_computation_text"] for r in records if r["variant"]=="guided"),
              "records":records}
    write_json(folder / "hlo_inspection.json",result)
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder",type=Path)
    result=inspect(p.parse_args().folder)
    print(f'Guided graphs with identical optimized computation text: {result["guided_identical_computations"]}')
