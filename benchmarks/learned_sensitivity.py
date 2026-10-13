"""How much do the ASSUMED constants matter for the learned plan selector?

Re-labels the dataset and re-runs the leave-one-family-out evaluation for
different cross-device transfer costs and device penalties.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from search import learned_optimizer as lo  # noqa: E402
from utils import cost_model  # noqa: E402

CONFIGS = [("baseline", 50.0, 0.15), ("cheap transfer", 10.0, 0.15), ("expensive transfer", 150.0, 0.15),
           ("cheap devices", 50.0, 0.05), ("expensive devices", 50.0, 0.30)]


def main():
    out = []
    for name, comm, pen in CONFIGS:
        cost_model.COMM_US, lo.DEVICE_PENALTY = comm, pen
        samples = lo.build_dataset(100, use_cache=False)
        per = {}
        for fam in lo.FAMILIES:
            tr = [s for s in samples if s["family"] != fam]
            te = [s for s in samples if s["family"] == fam]
            model, _ = lo.fit(tr, 300)
            per[fam] = lo.evaluate_heldout(model, te)
        mean = lambda k: float(np.mean([per[f][k] for f in lo.FAMILIES]))
        devices = [lo.ACTIONS[int(np.argmax(s["rewards"]))][3] for s in samples]
        out.append({"config": name, "comm_us": comm, "device_penalty": pen,
                    "learned": mean("learned"), "fixed_1dev": mean("fixed_pipeline_1dev"),
                    "fixed_4dev": mean("fixed_pipeline_4dev"), "oracle": mean("oracle"),
                    "regret_pct": mean("regret_pct"),
                    "best_plan_uses_multiple_devices_pct": float(np.mean([d > 1 for d in devices]) * 100)})
        print(out[-1], flush=True)
    cost_model.COMM_US, lo.DEVICE_PENALTY = 50.0, 0.15
    (ROOT / "docs" / "results" / "learned_sensitivity.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
