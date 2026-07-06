"""Runs the adversarial-robustness eval for a model over a set of math problems.

For each problem we:
  1. run the CLEAN problem -> can the model solve it at all? Only clean-solved
     problems are "attackable".
  2. sweep every single semantics-preserving perturbation; the problem is "broken"
     if any perturbation flips a clean-correct answer to wrong.

Two modes:
  synth     clean seeds (gsm8k / svamp / sample): perturbations are synthesized
            from perturb.ATOMS.
  provided  datasets that already ship answer-preserving variants grouped by seed
            (e.g. GSM-Plus): each provided variant is run as-is.

Records are emitted per (problem, variant) so report.py can compute clean/robust
accuracy, per-perturbation attribution, and cross-model transfer. Progress is
checkpointed to disk after every run (crash-safe); re-running resumes.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import agent, grader, perturb
from .data import MathProblem


def _combo_key(combo) -> str:
    return "+".join(combo) if combo else "clean"


def _atomic_write_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    # os.replace can transiently fail on Windows if the target is momentarily
    # locked (AV/indexer); retry briefly before giving up.
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    os.replace(tmp, path)


def atoms_for(problems: list[MathProblem]) -> list[str]:
    """Perturbation labels to report on: provided variants' types, else ATOMS."""
    provided = sorted({p.perturbation for p in problems if p.perturbation != "clean"})
    return provided if provided else list(perturb.ATOMS)


def _run_one(problem: MathProblem, model: str, atom: str, out_dir: Path,
             max_retries: int = 3, apply_perturbation: bool = True) -> dict:
    # In "provided" mode the variant text already carries the perturbation.
    question = (perturb.apply(atom, problem.question)
                if (atom and apply_perturbation) else problem.question)
    combo = [atom] if atom else []
    status, error, pred_text = "ok", "", ""
    t0 = time.time()
    for attempt in range(1, max_retries + 2):
        try:
            pred_text = agent.solve(model, question, problem.answer, atom)
            status, error = "ok", ""
            break
        except Exception as e:  # noqa: BLE001 - classify infra vs. genuine
            error = f"{type(e).__name__}: {e}"
            if agent.is_transient_error(e) and attempt <= max_retries:
                time.sleep(min(2 ** attempt, 20))
                continue
            status = "error"
            break

    passed = None if status == "error" else grader.is_correct(pred_text, problem.answer)
    elapsed = round(time.time() - t0, 2)

    tdir = out_dir / "transcripts"
    tdir.mkdir(parents=True, exist_ok=True)
    tpath = tdir / f"{model.replace('/', '-')}__{problem.group}__{_combo_key(combo)}.json"
    tpath.write_text(json.dumps({
        "task": problem.group, "model": model, "combo": combo,
        "perturbation": atom or "clean", "question": question,
        "gold": problem.answer, "prediction": pred_text,
        "extracted": grader.extract_answer(pred_text) if pred_text else None,
        "status": status, "passed": passed, "error": error,
    }, indent=2), encoding="utf-8")

    return {"task": problem.group, "title": problem.id, "model": model,
            "phase": "clean" if not combo else "search", "combo": combo,
            "note": perturb.NOTES.get(atom, atom) if atom else "clean question",
            "passed": passed, "status": status, "steps": 1, "seconds": elapsed,
            "error": error, "transcript": str(tpath)}


def _load_progress(results_path: Path | None, model: str, resume: bool):
    current, done = [], {}
    if not (results_path and Path(results_path).exists()):
        return current, done
    try:
        previous = json.loads(Path(results_path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return current, done
    in_scope_ok = {}
    for r in previous:
        if r.get("model") != model:
            current.append(r)            # other models: always preserve
            continue
        if resume and r.get("status", "ok") == "ok":
            in_scope_ok[(r["task"], _combo_key(r.get("combo")))] = r
    for (task_id, ckey), r in in_scope_ok.items():
        current.append(r)
        done.setdefault(task_id, {})[ckey] = r
    return current, done


def run(model: str, problems: list[MathProblem], out_dir: Path,
        results_path: Path | None = None, resume: bool = True) -> list[dict]:
    """Evaluate `model` on `problems`. Returns the full results list."""
    provided = any(p.perturbation != "clean" for p in problems)
    current, done = _load_progress(results_path, model, resume)

    def checkpoint(rec):
        current.append(rec)
        done.setdefault(rec["task"], {})[_combo_key(rec.get("combo"))] = rec
        if results_path:
            _atomic_write_json(results_path, current)

    resumed = sum(len(v) for v in done.values())
    if resumed:
        print(f"  (resuming: {resumed} completed run(s) reloaded; skipped, errors retried)")
    if results_path:
        _atomic_write_json(results_path, current)

    if provided:
        _run_provided(model, problems, out_dir, done, checkpoint)
    else:
        _run_synth(model, problems, out_dir, done, checkpoint)
    return current


def _run_synth(model, problems, out_dir, done, checkpoint):
    for p in problems:
        pd = done.get(p.group, {})
        clean = pd.get("clean")
        if clean is None:
            print(f"  [{model}] {p.id} :: clean ...", end="", flush=True)
            clean = _run_one(p, model, "", out_dir)
            checkpoint(clean)
            label = "PASS" if clean["passed"] else ("ERROR" if clean["status"] == "error" else "fail")
            print(f" {label}")
        else:
            print(f"  [{model}] {p.id} :: clean ... (resumed)")
        if clean["status"] == "error" or not clean["passed"]:
            continue
        for atom in perturb.ATOMS:
            if _combo_key([atom]) in pd:
                continue
            print(f"    attack {p.id} [{model}] :: {atom} ...", end="", flush=True)
            r = _run_one(p, model, atom, out_dir)
            checkpoint(r)
            out = "ERROR" if r["status"] == "error" else ("BROKE" if not r["passed"] else "SURVIVED")
            print(f" {out}")


def _run_provided(model, problems, out_dir, done, checkpoint):
    by_group = {}
    for p in problems:
        b = by_group.setdefault(p.group, {"clean": None, "variants": []})
        if p.perturbation == "clean":
            b["clean"] = p
        else:
            b["variants"].append(p)

    for group, bundle in by_group.items():
        seed = bundle["clean"]
        if seed is None:
            continue  # no clean baseline -> cannot assess robustness
        pd = done.get(group, {})
        clean = pd.get("clean")
        if clean is None:
            print(f"  [{model}] {seed.id} :: clean ...", end="", flush=True)
            clean = _run_one(seed, model, "", out_dir)
            checkpoint(clean)
            print(f" {'PASS' if clean['passed'] else 'fail'}")
        if clean["status"] == "error" or not clean["passed"]:
            continue
        for variant in bundle["variants"]:
            atom = variant.perturbation
            if _combo_key([atom]) in pd:
                continue
            r = _run_one(variant, model, atom, out_dir, apply_perturbation=False)
            r["note"] = f"provided variant: {atom}"
            checkpoint(r)
            out = "ERROR" if r["status"] == "error" else ("BROKE" if not r["passed"] else "SURVIVED")
            print(f"    attack {seed.id} [{model}] :: {atom} ... {out}")
