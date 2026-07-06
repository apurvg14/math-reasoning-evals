"""mathrobust: an adversarial-robustness benchmark for LLM math reasoning.

Take a math word problem with a known answer, apply a *semantics-preserving*
perturbation (the correct answer never changes), and measure whether the model's
answer flips. A model that truly reasons should be invariant; a pattern-matcher
is not. Reports clean vs. worst-case (robust) accuracy, per-perturbation
attribution, and cross-model transfer.
"""
__version__ = "0.1.0"
