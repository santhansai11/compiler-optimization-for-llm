"""Compare XLA-optimised HLO programs independently of instruction names.

``canonical(text)`` renames every computation, instruction and parameter
in order of first appearance and drops metadata (source lines, op names),
so two programs get the same ``fingerprint`` exactly when XLA produced the
same optimised program for them.
"""

import difflib
import hashlib
import re

_METADATA = re.compile(r",?\s*metadata=\{[^}]*\}")
_MODULE = re.compile(r"^HloModule\s+\S+", re.M)
# newer XLA prints source-location tables between the module line and the code
_DEBUG_TABLES = re.compile(r"^(?:FileNames|FunctionNames|FileLocations|StackFrames)\n(?:.+\n)*", re.M)
_DEFINED = [
    re.compile(r"^\s*(?:ROOT\s+)?%?([A-Za-z_][\w.\-]*)\s*=\s", re.M),      # instructions
    re.compile(r"^\s*(?:ENTRY\s+)?%?([A-Za-z_][\w.\-]*)\s*\(", re.M),      # computations
    re.compile(r"[(,]\s*%?([A-Za-z_][\w.\-]*):\s*[a-z]+\d*\["),            # parameters
]


def canonical(text):
    text = _MODULE.sub("HloModule m", _METADATA.sub("", _DEBUG_TABLES.sub("", text)))
    text = re.sub(r"\n{3,}", "\n\n", text)
    names = []
    for pattern in _DEFINED:
        for name in pattern.findall(text):
            if name not in names:
                names.append(name)
    if not names:
        return text
    index = {name: f"v{i}" for i, name in enumerate(names)}
    regex = re.compile(r"(?<![\w.\-])%?(" + "|".join(
        re.escape(n) for n in sorted(names, key=len, reverse=True)) + r")(?![\w.\-])")
    return regex.sub(lambda m: index[m.group(1)], text)


def fingerprint(text):
    return hashlib.sha1(canonical(text).encode()).hexdigest()[:16]


def summary(text):
    """Instruction / fusion counts of an optimised HLO module."""
    lines = [l for l in text.splitlines() if re.match(r"^\s*(?:ROOT\s+)?%?\S+\s*=\s", l)]
    return {"instructions": len(lines),
            "fusions": sum(1 for l in lines if re.search(r"\sfusion\(", l)),
            "fingerprint": fingerprint(text)}


def diff(before, after, label_a="before", label_b="after"):
    """Unified diff of the canonical forms (empty when programs are equal)."""
    return "".join(difflib.unified_diff(canonical(before).splitlines(True),
                                        canonical(after).splitlines(True),
                                        label_a, label_b, n=2))
