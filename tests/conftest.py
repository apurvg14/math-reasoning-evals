"""Shared fixtures. The dashboard fixtures start a throwaway server on a free port
with its own temp results dir, so the real results/ directory is never touched.
If AGENTEVAL/MATHROBUST_DASHBOARD_URL is set (e.g. in Docker), tests target that.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_ready(url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/api/config", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(0.3)
    raise RuntimeError(f"dashboard not ready at {url}: {last}")


@pytest.fixture(scope="session")
def base_url(tmp_path_factory):
    env_url = os.environ.get("MATHROBUST_DASHBOARD_URL")
    if env_url:
        env_url = env_url.rstrip("/")
        _wait_ready(env_url)
        yield env_url
        return

    port = _free_port()
    results_dir = tmp_path_factory.mktemp("results")
    env = dict(os.environ, MATHROBUST_RESULTS_DIR=str(results_dir))
    proc = subprocess.Popen(
        [sys.executable, "-m", "mathrobust", "dashboard",
         "--host", "127.0.0.1", "--port", str(port), "--no-open"],
        cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    try:
        _wait_ready(url)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="session")
def browser():
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        try:
            b = pw.chromium.launch()
        except Exception as e:  # noqa: BLE001 - browser binary not installed
            pytest.skip(f"chromium not available: {e}")
        yield b
        b.close()


@pytest.fixture
def page(browser):
    ctx = browser.new_context()
    pg = ctx.new_page()
    yield pg
    ctx.close()


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "e2e: browser-based end-to-end tests (require Playwright + a browser)")
