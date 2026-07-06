import json

from mathrobust import data, report, runner


def _run(model, dataset, out, limit=None):
    problems = data.load_dataset(dataset, limit=limit)
    results_path = out / "results.json"
    runner.run(model, problems, out, results_path=results_path, resume=True)
    return json.loads(results_path.read_text(encoding="utf-8")), runner.atoms_for(problems)


def test_reference_is_perfectly_robust(tmp_path):
    results, atoms = _run("reference", "sample", tmp_path)
    s = report.summarize(results, atoms=atoms)
    row = s["models"][0]
    assert row["model"] == "reference"
    assert row["clean_pass"] == 100.0
    assert row["robust_pass"] == 100.0
    assert row["asr"] == 0.0


def test_brittle_breaks_under_its_weaknesses(tmp_path):
    results, atoms = _run("brittle-a", "sample", tmp_path)
    s = report.summarize(results, atoms=atoms)
    row = s["models"][0]
    assert row["clean_pass"] == 100.0        # solves everything clean
    assert row["robust_pass"] < 100.0        # but not robust
    assert row["asr"] > 0.0
    broken = {fc["atom"] for fc in s["failure_classes"] if fc["counts"]["brittle-a"]}
    assert broken == {"distractor", "name_swap"}


def test_datasets_load_real_public_slices(tmp_path):
    for ds in ("gsm8k", "svamp"):
        problems = data.load_dataset(ds, limit=5)
        assert len(problems) == 5
        assert all(p.source == ds for p in problems)
        assert all(isinstance(p.answer, float) for p in problems)


def test_checkpoint_resume_reuses_prior_runs(tmp_path):
    problems = data.load_dataset("sample", limit=3)
    rp = tmp_path / "results.json"
    runner.run("reference", problems, tmp_path, results_path=rp, resume=True)
    first = json.loads(rp.read_text(encoding="utf-8"))
    # re-running with resume should not duplicate records
    runner.run("reference", problems, tmp_path, results_path=rp, resume=True)
    second = json.loads(rp.read_text(encoding="utf-8"))
    assert len(second) == len(first)


def test_transfer_matrix_partial_between_brittles(tmp_path):
    problems = data.load_dataset("sample")
    rp = tmp_path / "results.json"
    runner.run("brittle-a", problems, tmp_path, results_path=rp, resume=True)
    runner.run("brittle-b", problems, tmp_path, results_path=rp, resume=True)
    results = json.loads(rp.read_text(encoding="utf-8"))
    s = report.summarize(results, atoms=runner.atoms_for(problems))
    tm = s["transfer"]
    assert set(tm["models"]) == {"brittle-a", "brittle-b"}
    # both share the 'distractor' weakness, so transfer is > 0 but < 100
    cell = tm["matrix"]["brittle-a"]["brittle-b"]
    assert cell["pairs"] > 0
    assert 0 < cell["rate"] < 100


def test_build_report_markdown_has_headline(tmp_path):
    results, atoms = _run("reference", "sample", tmp_path)
    md = report.build_report(results, atoms=atoms)
    assert "Adversarial Robustness Scorecard" in md
    assert "robust pass@1" in md
