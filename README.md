# Math Reasoning Robustness

**An adversarial-robustness benchmark for LLM math reasoning.**

Standard accuracy on a math benchmark tells you whether a model got the answer
right *once*. It doesn't tell you whether the model is *reasoning* or
*pattern-matching*. This harness measures the difference: take a word problem with
a known answer, apply a **semantics-preserving** perturbation — one that leaves the
correct answer completely unchanged — and check whether the model's answer flips.

A model that genuinely reasons should be **invariant** to these edits. A brittle
model isn't: add an irrelevant sentence, swap the names, or write `5 dollars`
instead of `$5`, and the answer changes.

The headline metric is **`robust pass@1`**: a problem counts only if the model
stayed correct under *every* perturbation — the worst case, not the lucky case.

```
| Model                | clean pass@1 | robust pass@1 | attack success rate |
|----------------------|-------------:|--------------:|--------------------:|
| reference (oracle)   |         100% |          100% |                  0% |
| brittle-a (demo)     |         100% |           60% |                 40% |
```

---

## Why this design

- **Numeric grading, never string matching.** The final number in the reply is
  extracted (preferring an explicit `Answer: N` line, else the last number) and
  compared to gold within a tolerance.
- **Answer-preserving attacks only.** Every perturbation is guaranteed not to
  change the correct answer, so a flip is unambiguously a model failure — not a
  changed problem. `paraphrase` is a built-in **control** that should never break
  a real solver.
- **Worst-case search.** Each clean-solved problem is swept across all single
  perturbations; a problem is "broken" if *any* attack flips it.
- **Cross-model transfer.** Do the attacks that break model A also break model B?
  High transfer = shared blind spots; low = model-specific fragility.
- **Infra errors are excluded.** Transient API failures (rate limits, 5xx,
  timeouts) are retried and, if they persist, dropped from every metric — they are
  never counted as a model getting the answer wrong.
- **Crash-safe.** Progress is checkpointed after every single run; re-running the
  same command resumes exactly where it left off.

---

## Quickstart (no API key needed)

The keyless backends — `reference` (an oracle) and `brittle-a` / `brittle-b`
(demo agents with deliberate, *different* weaknesses) — let you exercise the whole
pipeline offline.

```bash
python -m mathrobust run --model reference --dataset sample
python -m mathrobust run --model brittle-a --dataset sample
python -m mathrobust run --model brittle-b --dataset sample   # -> results/report.md
```

Open `results/report.md` for the scorecard, or use the dashboard (below).

## Evaluate a real model

```bash
pip install -r requirements.txt        # anthropic + openai SDKs
cp .env.example .env                    # add ANTHROPIC_API_KEY / OPENAI_API_KEY

# real public data ships in-repo (small verbatim slices), so this works offline:
python -m mathrobust run --model claude-haiku-4-5-20251001 --dataset gsm8k

# for a full-scale run, fetch the complete datasets (the loader prefers them):
python -m mathrobust fetch --dataset gsm8k                    # 1319-item test split
python -m mathrobust run --model claude-haiku-4-5-20251001 --dataset gsm8k --limit 200
```

Keys are read from `.env` (gitignored) and never printed.

---

## Datasets

Small, **verbatim** slices of two MIT-licensed public datasets are committed in
`mathrobust/data/` so everything works offline and the numbers are reproducible:

| name     | what it is                                   | bundled | fetch full |
|----------|----------------------------------------------|:-------:|:----------:|
| `sample` | 10 original hand-written problems            |   yes   |     —      |
| `gsm8k`  | grade-school math word problems              |  25/1319 | `fetch --dataset gsm8k` |
| `svamp`  | structural variations of arithmetic problems |  25/1000 | `fetch --dataset svamp` |

`fetch` downloads the full sets into `data/math/` (gitignored); the loader prefers
those over the bundled slices. See `mathrobust/data/SOURCES.md` for attribution.

Datasets that ship **pre-made** answer-preserving variants (e.g. GSM-Plus) are
supported too: normalize them to the JSONL schema with `perturbation` labels and
`group` tying variants to their seed, drop the file in `data/math/gsm-plus.jsonl`,
and the runner evaluates the provided variants directly instead of synthesizing.

## Perturbations

| atom          | what it does                                        |
|---------------|-----------------------------------------------------|
| `paraphrase`  | faithful reword — **control**, should not break     |
| `verbose`     | real problem buried in irrelevant-but-true context  |
| `distractor`  | an irrelevant numeric fact is appended              |
| `noop`        | a seemingly-relevant but inconsequential clause     |
| `name_swap`   | proper names are swapped                             |
| `reformat`    | `$5` → `5 dollars`, `60%` → `60 percent`            |

All are answer-preserving by construction.

---

## Dashboard

A zero-dependency local web UI (pure `http.server`, no JS build step): pick a
model + dataset, hit Run, watch the live log, and read the scorecard (clean/robust
accuracy, failure classes, transfer matrix, per-problem grid).

```bash
python -m mathrobust dashboard          # -> http://127.0.0.1:8765
```

### In Docker

```bash
docker compose up dashboard             # -> http://localhost:8765
# with real models:
#   uncomment env_file: .env in docker-compose.yml, or:
docker run --rm -p 8765:8765 --env-file .env mathrobust
```

---

## Tests

Three layers, runnable locally or in Docker:

- **unit** (`test_grader`, `test_perturb`, `test_runner`) — grader, answer-preserving
  perturbations, and the full run loop + report engine via keyless backends.
- **API smoke** (`test_dashboard_api`) — pure stdlib; the JSON contract and request
  validation.
- **browser E2E** (`test_dashboard_e2e`) — real Chromium via Playwright: page load,
  Advanced toggle, model/dataset selection, and a full keyless run that populates
  the scorecard.

```bash
pip install -r requirements-dev.txt
playwright install chromium             # one-time browser download
python -m pytest tests -v
```

In Docker (official Playwright image, targets the `dashboard` service):

```bash
docker compose --profile test run --rm tests
```

Tests find the dashboard via `MATHROBUST_DASHBOARD_URL` if set, otherwise they
start a throwaway server on a free port writing to a temp results dir, so your real
`results/` is untouched.

---

## Project layout

```
mathrobust/
  data.py        normalized schema + dataset loader + public-dataset fetcher
  grader.py      numeric answer extraction + tolerant comparison
  perturb.py     answer-preserving perturbations (attacks + control)
  agent.py       model backends: reference / brittle / real (claude-* / gpt-*)
  runner.py      worst-case sweep + crash-safe checkpoint/resume
  report.py      scorecard: clean/robust/ASR, failure classes, transfer
  dashboard.py   zero-dependency web UI backend
  dashboard.html single-page frontend
  data/          bundled real slices (gsm8k, svamp) + sample + SOURCES.md
tests/           unit + API + Playwright E2E
```

## License

MIT — see [LICENSE](LICENSE). Bundled dataset slices retain their original
MIT licenses; see `mathrobust/data/SOURCES.md`.
