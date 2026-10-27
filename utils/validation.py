"""Numerical equivalence of every declared graph output, in output order."""
import numpy as np


def compare_outputs(reference, candidate, rtol=1e-4, atol=1e-5):
    """Use |candidate-reference| <= atol + rtol*|reference| per element.

    The percentage is an element count, not task accuracy. NaN/Inf are failures,
    even when both sides contain the same nonfinite value. Unavailable error
    statistics are null so API/Excel exports remain valid JSON.
    """
    if not (np.isfinite(rtol) and np.isfinite(atol) and rtol >= 0 and atol >= 0):
        raise ValueError("Tolerances must be finite and nonnegative")
    left, right = list(reference.values()), list(candidate.values())
    shapes_match = bool(left) and len(left) == len(right)
    finite = True
    total = matched = 0
    differences = []
    details = []
    for i in range(max(len(left), len(right))):
        a = np.asarray(left[i], dtype=np.float64) if i < len(left) else None
        b = np.asarray(right[i], dtype=np.float64) if i < len(right) else None
        same = a is not None and b is not None and a.shape == b.shape
        shapes_match = shapes_match and same
        count = max(0 if a is None else a.size, 0 if b is None else b.size)
        total += count
        good = same and bool(np.isfinite(a).all() and np.isfinite(b).all())
        finite = finite and bool(good)
        passed = 0
        if same:
            mask = np.isfinite(a) & np.isfinite(b)
            # Select finite pairs first: never subtract Inf from Inf.
            delta = np.abs(b[mask] - a[mask])
            passed = int(np.count_nonzero(delta <= atol + rtol * np.abs(a[mask])))
            if good:
                differences.append(delta)
        matched += passed
        details.append({"index": i, "reference_shape": None if a is None else list(a.shape),
                        "candidate_shape": None if b is None else list(b.shape),
                        "same_shape": same, "finite": bool(good),
                        "elements": count, "within_tolerance": passed})
    valid = shapes_match and finite
    errors = np.concatenate(differences) if differences else np.array([], dtype=float)
    return {
        "preserved": bool(valid and matched == total),
        "preservation_pct": (100.0 * matched / total if total else 100.0 if valid else 0.0),
        "max_abs_diff": float(errors.max(initial=0)) if valid else None,
        "mae": float(errors.mean()) if valid and errors.size else 0.0 if valid else None,
        "rmse": float(np.sqrt(np.mean(errors ** 2))) if valid and errors.size else 0.0 if valid else None,
        "rtol": rtol, "atol": atol, "all_finite": bool(finite),
        "same_shapes": bool(shapes_match), "elements": total,
        "elements_within_tolerance": matched, "outputs": details,
        "meaning": "Element-wise output equivalence, not prediction accuracy on a labelled dataset",
    }
