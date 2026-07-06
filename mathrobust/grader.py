"""Numeric grading for math word problems.

We never string-match the model's prose. We extract the final number from the
reply -- preferring an explicit "Answer: N" line, else the last number in the
text (the widely-used convention for grading grade-school math) -- and compare it
to the gold answer with a small tolerance.
"""
from __future__ import annotations

import re


def extract_answer(text: str) -> float | None:
    """Final number from a model reply: prefer 'Answer: N', else the last number."""
    t = text.replace(",", "")
    m = re.findall(r"[Aa]nswer\s*[:=]?\s*\$?(-?\d+(?:\.\d+)?)", t)
    if m:
        return float(m[-1])
    nums = re.findall(r"-?\d+(?:\.\d+)?", t)
    return float(nums[-1]) if nums else None


def is_correct(pred_text: str, gold: float, tol: float = 1e-4) -> bool:
    pred = extract_answer(pred_text)
    return pred is not None and abs(pred - gold) <= tol


def fmt_gold(g: float) -> str:
    return str(int(g)) if abs(g - round(g)) < 1e-9 else str(g)
