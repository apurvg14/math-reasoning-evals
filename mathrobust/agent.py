"""Model backends for answering math word problems.

  reference          oracle; emits the gold answer (validates the plumbing, and is
                     an unbreakable ceiling in the scorecard)
  brittle-a/-b       keyless demo agents with fixed, *different* weaknesses, so the
                     transfer table shows partial (not total) transfer offline
  claude-* / gpt-*   real LLMs, single-turn chain-of-thought (need an API key)

Infrastructure failures (API connection / rate-limit / 5xx) are raised so the
runner can retry and exclude them -- they must never be counted as a model error.
"""
from __future__ import annotations

from .grader import fmt_gold

SDK_MAX_RETRIES = 5

_TRANSIENT_NAMES = {
    "APIConnectionError", "APITimeoutError", "RateLimitError",
    "InternalServerError", "OverloadedError", "ServiceUnavailableError", "APIError",
}

# Keyless demo backends: which perturbation atoms deterministically break them.
# Both share `distractor` (transfers); the rest differ (do NOT transfer).
BRITTLE_WEAKNESSES = {
    "brittle-a": {"distractor", "name_swap"},
    "brittle-b": {"distractor", "verbose"},
}

SYSTEM = (
    "You are a careful math tutor. Solve the word problem step by step, then state "
    "the final answer on its own last line in the form 'Answer: <number>'. Use only "
    "the information needed; ignore irrelevant details."
)


def is_transient_error(exc: BaseException) -> bool:
    """True if `exc` looks like a transient infrastructure/networking failure."""
    if type(exc).__name__ in _TRANSIENT_NAMES:
        code = getattr(exc, "status_code", None)
        if code is None:
            return True
        return code == 429 or code >= 500
    code = getattr(exc, "status_code", None)
    return code is not None and (code == 429 or code >= 500)


def is_keyless(model: str) -> bool:
    return model == "reference" or model in BRITTLE_WEAKNESSES


def answer_question(model: str, question: str, provider: str | None = None) -> str:
    """Single-turn chain-of-thought answer from a real LLM (returns raw text)."""
    provider = provider or ("anthropic" if model.startswith("claude") else "openai")
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(max_retries=SDK_MAX_RETRIES)
        resp = client.messages.create(
            model=model, max_tokens=1024, system=SYSTEM,
            messages=[{"role": "user", "content": question}])
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    from openai import OpenAI
    client = OpenAI(max_retries=SDK_MAX_RETRIES)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": question}])
    return resp.choices[0].message.content or ""


def solve(model: str, question: str, gold: float, atom: str) -> str:
    """Dispatch to the right backend and return the model's reply text."""
    if model == "reference":
        return f"Answer: {fmt_gold(gold)}"
    if model in BRITTLE_WEAKNESSES:
        if atom and atom in BRITTLE_WEAKNESSES[model]:
            return f"Answer: {fmt_gold(gold + 1)}"  # deterministically wrong
        return f"Answer: {fmt_gold(gold)}"
    return answer_question(model, question)
