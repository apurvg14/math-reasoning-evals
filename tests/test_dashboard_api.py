"""Pure-stdlib API-contract smoke tests for the dashboard server."""
import json
import time
import urllib.error
import urllib.request


def _get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=10) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def _post(base, path, payload):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(base + path, data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def test_index_serves_html(base_url):
    status, body = _get(base_url, "/")
    assert status == 200
    assert "Math Reasoning Robustness" in body


def test_config_lists_datasets_and_keyless(base_url):
    status, body = _get(base_url, "/api/config")
    assert status == 200
    cfg = json.loads(body)
    assert "gsm8k" in cfg["datasets"]
    assert "reference" in cfg["keyless"]


def test_unknown_route_is_404(base_url):
    status, _ = _get(base_url, "/api/nope")
    assert status == 404


def test_run_requires_model(base_url):
    status, body = _post(base_url, "/api/run", {"dataset": "sample"})
    assert status == 400
    assert "model" in body["error"].lower()


def test_run_rejects_unknown_dataset(base_url):
    status, body = _post(base_url, "/api/run", {"model": "reference", "dataset": "nope"})
    assert status == 400


def test_run_rejects_real_model_without_key(base_url):
    status, body = _post(base_url, "/api/run",
                         {"model": "claude-haiku-4-5-20251001", "dataset": "sample"})
    # keyless env in CI -> should be rejected with a helpful message
    if status == 400:
        assert "key" in body["error"].lower()


def test_keyless_run_completes_and_populates_results(base_url):
    status, body = _post(base_url, "/api/run",
                         {"model": "reference", "dataset": "sample", "limit": 3})
    assert status == 200 and body.get("ok")
    # poll status until done
    for _ in range(60):
        st = json.loads(_get(base_url, "/api/status")[1])
        if st["done"]:
            break
        time.sleep(0.5)
    else:
        raise AssertionError("run did not finish in time")
    assert st["error"] is None
    res = json.loads(_get(base_url, "/api/results")[1])
    assert res["empty"] is False
    assert any(m["model"] == "reference" for m in res["models"])
