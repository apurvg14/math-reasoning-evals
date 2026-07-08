"""A zero-dependency local dashboard for the math-robustness eval.

Pure standard library (http.server) so it runs anywhere and containerizes without
a JS build step. Pick a model + dataset, hit Run, watch the live log, and read the
scorecard (clean/robust accuracy, failure classes, transfer, per-problem grid).

Keyless backends (reference / brittle-a / brittle-b) run with no API key, so the
whole UI -- and its tests -- work offline.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import data, report, runner

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("MATHROBUST_RESULTS_DIR") or (ROOT / "results"))
HTML = Path(__file__).resolve().parent / "dashboard.html"
RESULTS = lambda: OUT / "results.json"  # noqa: E731

DATASETS = ["sample", "gsm8k", "svamp"]
KEYLESS = ["reference", "brittle-a", "brittle-b"]
SUGGESTED = ["claude-haiku-4-5-20251001", "claude-sonnet-4-5", "gpt-4o-mini"]

_state = {"running": False, "log": [], "done": False, "error": None,
          "model": None, "dataset": None}
_lock = threading.Lock()


class _LogStream(io.TextIOBase):
    """Line-buffered sink that appends completed lines to the shared log."""
    def __init__(self):
        self._buf = ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            with _lock:
                _state["log"].append(line)
        return len(s)


def _has_key(model: str) -> bool:
    if model in KEYLESS:
        return True
    if model.startswith("claude"):
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    return bool(os.environ.get("OPENAI_API_KEY"))


def _worker(model: str, dataset: str, limit, fresh: bool):
    stream = _LogStream()
    try:
        with contextlib.redirect_stdout(stream):
            problems = data.load_dataset(dataset, limit=limit)
            OUT.mkdir(parents=True, exist_ok=True)
            atoms = runner.atoms_for(problems)
            print(f"Running '{dataset}' ({len(problems)} problems) on '{model}' ...")
            runner.run(model, problems, OUT, results_path=RESULTS(), resume=not fresh)
            report.render(RESULTS(), OUT / "report.md", atoms=atoms)
            print("Done. Scorecard updated.")
    except Exception as e:  # noqa: BLE001 - surface to the UI, don't crash the server
        with _lock:
            _state["error"] = f"{type(e).__name__}: {e}"
        stream.write(f"ERROR: {type(e).__name__}: {e}\n")
    finally:
        with _lock:
            _state["running"] = False
            _state["done"] = True


def _summary():
    p = RESULTS()
    if not p.exists():
        return {"empty": True}
    results = json.loads(p.read_text(encoding="utf-8"))
    present = sorted({r["combo"][0] for r in results if len(r.get("combo") or []) == 1})
    return {"empty": False, **report.summarize(results, atoms=present or None)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the console clean
        pass

    def _send(self, code, body, ctype="application/json"):
        data_bytes = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data_bytes)))
        self.end_headers()
        self.wfile.write(data_bytes)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj))

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, HTML.read_text(encoding="utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/config":
            self._json({"datasets": DATASETS, "keyless": KEYLESS,
                        "suggested": SUGGESTED,
                        "keys": {"anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
                                 "openai": bool(os.environ.get("OPENAI_API_KEY"))}})
        elif self.path == "/api/results":
            self._json(_summary())
        elif self.path == "/api/status":
            with _lock:
                self._json({k: _state[k] for k in
                            ("running", "log", "done", "error", "model", "dataset")})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/api/run":
            self._json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "invalid JSON"}, 400)
            return
        model = (body.get("model") or "").strip()
        dataset = (body.get("dataset") or "sample").strip()
        limit = body.get("limit")
        fresh = bool(body.get("fresh"))
        if not model:
            self._json({"error": "model is required"}, 400)
            return
        if dataset not in DATASETS:
            self._json({"error": f"unknown dataset '{dataset}'"}, 400)
            return
        if limit is not None:
            try:
                limit = int(limit)
                if limit <= 0:
                    limit = None
            except (TypeError, ValueError):
                self._json({"error": "limit must be an integer"}, 400)
                return
        if not _has_key(model):
            self._json({"error": f"no API key configured for '{model}'. Add it to "
                                 f".env, or try a keyless model: {', '.join(KEYLESS)}."}, 400)
            return
        with _lock:
            if _state["running"]:
                self._json({"error": "a run is already in progress"}, 409)
                return
            _state.update({"running": True, "done": False, "error": None,
                           "log": [], "model": model, "dataset": dataset})
        threading.Thread(target=_worker, args=(model, dataset, limit, fresh),
                         daemon=True).start()
        self._json({"ok": True})


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    from .__main__ import _load_env
    _load_env()  # so real-model runs launched from the UI can see API keys
    OUT.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((host, port), Handler)
    display_host = "localhost" if host in ("0.0.0.0", "::") else host
    url = f"http://{display_host}:{port}/"
    print(f"mathrobust dashboard -> {url}  (bound to {host}:{port})")
    print("Pick a model + dataset, hit Run. Ctrl+C to stop.")
    if open_browser and host not in ("0.0.0.0", "::"):
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    serve()
