"""Command-line entry point for mathrobust.

  python -m mathrobust run   --model reference --dataset gsm8k
  python -m mathrobust run   --model claude-haiku-4-5-20251001 --dataset gsm8k --limit 100
  python -m mathrobust report --results results/results.json
  python -m mathrobust fetch  --dataset gsm8k
  python -m mathrobust dashboard
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import data, report, runner


def _load_env():
    """Load KEY=VALUE lines from a local .env (no dependency on python-dotenv)."""
    for envp in (Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"):
        if not envp.exists():
            continue
        import os
        for line in envp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        return


def main(argv=None):
    p = argparse.ArgumentParser(prog="mathrobust",
                                description="Adversarial robustness eval for LLM math reasoning.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the robustness eval for a model")
    r.add_argument("--model", required=True,
                   help="reference | brittle-a | brittle-b | claude-* | gpt-*")
    r.add_argument("--dataset", default="sample",
                   help="sample | gsm8k | svamp | gsm-plus (or a custom name)")
    r.add_argument("--data-dir", type=Path, default=None,
                   help="directory of fetched datasets (overrides bundled slices)")
    r.add_argument("--limit", type=int, default=None, help="cap number of problems")
    r.add_argument("--out", type=Path, default=Path("results"))
    r.add_argument("--no-resume", action="store_true",
                   help="ignore any saved progress and start fresh")
    r.add_argument("--no-report", action="store_true")

    rp = sub.add_parser("report", help="render a scorecard from a results file")
    rp.add_argument("--results", type=Path, default=Path("results/results.json"))
    rp.add_argument("--out", type=Path, default=Path("results/report.md"))

    f = sub.add_parser("fetch", help="download a public dataset (normalized JSONL)")
    f.add_argument("--dataset", required=True, help="gsm8k | svamp")
    f.add_argument("--data-dir", type=Path, default=None)

    d = sub.add_parser("dashboard", help="serve the local web dashboard")
    d.add_argument("--host", default="127.0.0.1")
    d.add_argument("--port", type=int, default=8765)
    d.add_argument("--no-open", action="store_true")

    args = p.parse_args(argv)
    _load_env()

    if args.cmd == "run":
        problems = data.load_dataset(args.dataset, data_dir=args.data_dir, limit=args.limit)
        args.out.mkdir(parents=True, exist_ok=True)
        results_path = args.out / "results.json"
        atoms = runner.atoms_for(problems)
        print(f"Running math robustness | dataset '{args.dataset}' ({len(problems)} problems) "
              f"| model '{args.model}' | resume={not args.no_resume}")
        interrupted = False
        try:
            runner.run(args.model, problems, args.out, results_path=results_path,
                       resume=not args.no_resume)
        except KeyboardInterrupt:
            interrupted = True
            print("\nInterrupted. Progress saved -- re-run the same command to continue.")
        if not args.no_report and results_path.exists():
            report.render(results_path, args.out / "report.md", atoms=atoms)
            print(f"\nScorecard -> {args.out / 'report.md'}")
        print(f"Results   -> {results_path}")
        if interrupted:
            raise SystemExit(130)

    elif args.cmd == "report":
        import json
        results = json.loads(args.results.read_text(encoding="utf-8"))
        # Report on whatever single perturbations actually appear in the file;
        # fall back to the default atom set if none are present.
        present = sorted({r["combo"][0] for r in results
                          if len(r.get("combo") or []) == 1})
        report.render(args.results, args.out, atoms=present or None)
        print(f"Scorecard -> {args.out}")

    elif args.cmd == "fetch":
        out = data.fetch(args.dataset, data_dir=args.data_dir)
        print(f"Fetched '{args.dataset}' -> {out}")

    elif args.cmd == "dashboard":
        from . import dashboard
        dashboard.serve(host=args.host, port=args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    main()
