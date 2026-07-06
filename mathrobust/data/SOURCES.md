# Bundled math data

These files back the math reasoning robustness suite (`python -m mathrobust`).

| File | Contents | Source | License |
|---|---|---|---|
| `math_sample.jsonl` | 10 original, hand-written arithmetic word problems | this repo | MIT (this repo) |
| `gsm8k.jsonl` | First 25 problems of the GSM8K test split (verbatim) | https://github.com/openai/grade-school-math | MIT |
| `svamp.jsonl` | First 25 problems of SVAMP (verbatim) | https://github.com/arkilpatel/SVAMP | MIT |

The GSM8K and SVAMP slices are small, verbatim excerpts of MIT-licensed public
datasets, redistributed here under those terms for offline/demo/test use. For a
full-scale run, download the complete datasets into the (gitignored) `data/math/`
directory — the loader prefers those over these bundled slices:

```
python -m mathrobust fetch --dataset gsm8k
python -m mathrobust fetch --dataset svamp
```

Normalized schema (one JSON object per line):
`{"id", "question", "answer", "source", "group", "perturbation"}`.
