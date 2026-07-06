import pytest

from mathrobust import perturb

Q = "John had $5 and 60% of a pie. Mary gave him 3 more. How many dollars?"


@pytest.mark.parametrize("atom", perturb.ATOMS)
def test_every_atom_returns_nonempty_string(atom):
    out = perturb.apply(atom, Q)
    assert isinstance(out, str) and out.strip()


@pytest.mark.parametrize("atom", perturb.ATOMS)
def test_perturbations_change_or_keep_text_meaningfully(atom):
    out = perturb.apply(atom, Q)
    # each atom must do *something* (except that all preserve the answer, which
    # the runner tests cover); paraphrase prepends, others alter/append.
    assert out != "" and (out != Q or atom == "identity")


def test_name_swap_replaces_known_names():
    out = perturb.apply("name_swap", Q)
    assert "John" not in out and "Mary" not in out


def test_reformat_rewrites_symbols():
    out = perturb.apply("reformat", Q)
    assert "$" not in out and "%" not in out
    assert "dollars" in out and "percent" in out


def test_distractor_adds_irrelevant_number():
    out = perturb.apply("distractor", Q)
    assert "13" in out and len(out) > len(Q)


def test_empty_atom_is_identity():
    assert perturb.apply("", Q) == Q
