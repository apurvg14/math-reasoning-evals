"""Math word-problem datasets: a normalized schema, a loader, and a fetcher.

Every item is treated uniformly regardless of source. The on-disk format is JSONL,
one object per line:

    {"id": "...", "question": "...", "answer": "18",
     "source": "gsm8k", "group": "seed-123", "perturbation": "clean"}

- `group`        ties perturbed variants back to their seed problem (defaults to id).
- `perturbation` is "clean" for seeds; datasets that ship pre-made answer-preserving
                 variants (e.g. GSM-Plus) label each variant with its type here, and
                 the runner uses those directly instead of synthesizing perturbations.

Small, verbatim slices of the MIT-licensed GSM8K and SVAMP datasets ship in
`mathrobust/data/` (see SOURCES.md). `fetch()` downloads the full public datasets
into a (gitignored) data directory; the loader prefers those over the slices.
"""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PKG = Path(__file__).resolve().parent
BUNDLED = PKG / "data"
SAMPLE_PATH = BUNDLED / "math_sample.jsonl"
DEFAULT_DATA_DIR = PKG.parent / "data" / "math"

# Public *data* sources (not paper references). Files are fetched on demand.
GSM8K_TEST_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/"
                  "master/grade_school_math/data/test.jsonl")
SVAMP_URL = "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json"


@dataclass
class MathProblem:
    id: str
    question: str
    answer: float
    source: str = "custom"
    group: str = ""
    perturbation: str = "clean"

    def __post_init__(self):
        if not self.group:
            self.group = self.id


def parse_gold(value) -> float:
    """Parse a dataset gold answer (str/int/float) into a float."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace(",", "").replace("$", "")
    if "####" in s:  # GSM8K puts the final answer after a '####' marker
        s = s.split("####")[-1].strip()
    m = re.findall(r"-?\d+\.?\d*", s)
    if not m:
        raise ValueError(f"no numeric gold in {value!r}")
    return float(m[-1])


def _row_to_problem(row: dict) -> MathProblem:
    return MathProblem(
        id=str(row["id"]),
        question=row["question"],
        answer=parse_gold(row["answer"]),
        source=row.get("source", "custom"),
        group=str(row.get("group", "") or ""),
        perturbation=row.get("perturbation", "clean"),
    )


def load_jsonl(path: Path, limit: int | None = None) -> list[MathProblem]:
    out: list[MathProblem] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(_row_to_problem(json.loads(line)))
        if limit and len(out) >= limit:
            break
    return out


def dataset_path(name: str, data_dir: Path | None = None) -> Path:
    """Resolve a dataset name to a file.

    Precedence: explicit --data-dir  >  data/math/ (fetched full)  >  bundled slice.
    Returns the first existing candidate, else the preferred path (for the error).
    """
    if name == "sample":
        return SAMPLE_PATH
    candidates: list[Path] = []
    if data_dir:
        candidates.append(Path(data_dir) / f"{name}.jsonl")
    candidates.append(DEFAULT_DATA_DIR / f"{name}.jsonl")
    candidates.append(BUNDLED / f"{name}.jsonl")
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def load_dataset(name: str, data_dir: Path | None = None,
                 limit: int | None = None) -> list[MathProblem]:
    p = dataset_path(name, data_dir)
    if not p.exists():
        raise SystemExit(
            f"dataset '{name}' not found at {p}.\n"
            f"Fetch it first:  python -m mathrobust fetch --dataset {name}")
    return load_jsonl(p, limit=limit)


# ----- fetch + normalize public datasets ------------------------------------
def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "mathrobust/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _normalize_gsm8k(raw: bytes) -> list[dict]:
    rows = []
    for i, line in enumerate(raw.decode("utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        rows.append({"id": f"gsm8k-{i}", "question": obj["question"],
                     "answer": obj["answer"], "source": "gsm8k",
                     "group": f"gsm8k-{i}", "perturbation": "clean"})
    return rows


def _normalize_svamp(raw: bytes) -> list[dict]:
    data = json.loads(raw.decode("utf-8"))
    rows = []
    for i, obj in enumerate(data):
        body = (obj.get("Body", "").strip() + " " + obj.get("Question", "").strip()).strip()
        rows.append({"id": f"svamp-{obj.get('ID', i)}", "question": body,
                     "answer": obj["Answer"], "source": "svamp",
                     "group": f"svamp-{obj.get('ID', i)}", "perturbation": "clean"})
    return rows


def fetch(name: str, data_dir: Path | None = None) -> Path:
    """Download a public dataset and write it in the normalized JSONL schema."""
    base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    base.mkdir(parents=True, exist_ok=True)
    out = base / f"{name}.jsonl"
    if name == "gsm8k":
        rows = _normalize_gsm8k(_download(GSM8K_TEST_URL))
    elif name == "svamp":
        rows = _normalize_svamp(_download(SVAMP_URL))
    elif name == "gsm-plus":
        raise SystemExit(
            "gsm-plus ships as a Hugging Face parquet dataset. Download its test "
            "split, convert each record to the normalized schema (question, answer, "
            "group=seed id, perturbation=perturbation_type) and save it as "
            f"{out}. Then run with --dataset gsm-plus.")
    else:
        raise SystemExit(f"unknown dataset '{name}' (try: gsm8k, svamp)")
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return out
