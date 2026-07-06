"""Browser end-to-end tests driving real Chromium via Playwright."""
import pytest

pytestmark = pytest.mark.e2e


def test_page_loads_with_title(page, base_url):
    page.goto(base_url)
    assert "Math Reasoning Robustness" in page.title()
    assert page.get_by_test_id("run-card").is_visible()


def test_advanced_settings_start_collapsed(page, base_url):
    page.goto(base_url)
    limit = page.get_by_test_id("limit-input")
    assert not limit.is_visible()
    page.get_by_test_id("advanced-toggle").click()
    assert limit.is_visible()


def test_model_and_dataset_selectable(page, base_url):
    page.goto(base_url)
    page.get_by_test_id("dataset-select").select_option("sample")
    page.get_by_test_id("model-select").select_option("reference")
    assert page.get_by_test_id("model-select").input_value() == "reference"


def test_run_pill_starts_idle(page, base_url):
    page.goto(base_url)
    assert page.get_by_test_id("run-pill").inner_text().strip() == "idle"


def test_full_reference_run_populates_scorecard(page, base_url):
    page.goto(base_url)
    page.get_by_test_id("dataset-select").select_option("sample")
    page.get_by_test_id("model-select").select_option("reference")
    page.get_by_test_id("advanced-toggle").click()
    page.get_by_test_id("limit-input").fill("3")
    page.get_by_test_id("run-button").click()

    # pill should reach "done" and the model table should appear
    page.wait_for_selector('[data-testid="run-pill"].ok', timeout=60000)
    assert page.get_by_test_id("run-pill").inner_text().strip() == "done"
    page.wait_for_selector('[data-testid="model-table"]', timeout=10000)
    rows = page.get_by_test_id("model-row")
    assert rows.count() >= 1
    assert "reference" in page.get_by_test_id("model-table").inner_text()
