"""Semantics-preserving perturbations for math word problems.

Each perturbation is **answer-preserving**: the correct numeric answer is unchanged,
so a robust solver should be invariant to all of them. `paraphrase` is a control
that should never break a real solver; the rest are attacks in the spirit of the
math-robustness literature (irrelevant distractors, no-op clauses, name swaps,
number-format rewrites, verbose burying).
"""
from __future__ import annotations

import re

# paraphrase is the control (should not break); the rest are attacks.
ATOMS = ["paraphrase", "verbose", "distractor", "noop", "name_swap", "reformat"]

_NAME_MAP = {
    "John": "Michael", "Mary": "Sarah", "Liam": "Ethan", "Maria": "Sofia",
    "Noah": "Oliver", "Emma": "Ava", "Olivia": "Mia", "James": "Daniel",
    "Sarah": "Rachel", "Tom": "Greg", "Sam": "Alex", "Anna": "Nina",
}

NOTES = {
    "paraphrase": "faithful reword (control)",
    "verbose": "real problem buried in irrelevant-but-true context",
    "distractor": "an irrelevant numeric fact is appended",
    "noop": "a seemingly-relevant but inconsequential clause is appended",
    "name_swap": "proper names are swapped",
    "reformat": "number formatting rephrased ($5 -> 5 dollars, 60% -> 60 percent)",
}


def _name_swap(q: str) -> str:
    pattern = r"\b(" + "|".join(map(re.escape, _NAME_MAP)) + r")\b"
    return re.sub(pattern, lambda m: _NAME_MAP.get(m.group(0), m.group(0)), q)


def _reformat(q: str) -> str:
    q = re.sub(r"\$(\d+(?:\.\d+)?)", r"\1 dollars", q)
    q = re.sub(r"(\d+(?:\.\d+)?)%", r"\1 percent", q)
    return q


def _distractor(q: str) -> str:
    return (q + " Also, there were 13 birds resting on a nearby fence at the time, "
            "which has nothing to do with the question.")


def _noop(q: str) -> str:
    return q + " Note that this problem was reviewed twice for clarity before publishing."


def _verbose(q: str) -> str:
    pre = ("Background (mostly not needed): this question comes from a large practice "
           "set that is updated every semester and reviewed by several tutors. None of "
           "that affects the math. Here is the actual problem.\n\n")
    post = "\n\nThere is no trick here; just compute the requested quantity."
    return pre + q + post


def _paraphrase(q: str) -> str:  # control
    return "Please solve the following word problem carefully.\n\n" + q


_FN = {
    "name_swap": _name_swap, "reformat": _reformat, "distractor": _distractor,
    "noop": _noop, "verbose": _verbose, "paraphrase": _paraphrase,
}


def apply(atom: str, question: str) -> str:
    return _FN[atom](question) if atom else question
