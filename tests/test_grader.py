from mathrobust.grader import extract_answer, is_correct, fmt_gold


def test_prefers_explicit_answer_line():
    assert extract_answer("Work: 2+2=4. Answer: 12") == 12.0


def test_falls_back_to_last_number():
    assert extract_answer("First 5, then 7, so the result is 12") == 12.0


def test_handles_commas_and_dollars():
    assert extract_answer("Answer: $1,234") == 1234.0


def test_handles_negatives_and_decimals():
    assert extract_answer("Answer: -3.5") == -3.5


def test_none_when_no_number():
    assert extract_answer("no digits here") is None


def test_is_correct_within_tolerance():
    assert is_correct("Answer: 12", 12.0)
    assert is_correct("Answer: 12.00001", 12.0)
    assert not is_correct("Answer: 13", 12.0)


def test_is_correct_false_when_unparseable():
    assert not is_correct("I am not sure", 12.0)


def test_fmt_gold_integers_have_no_decimal():
    assert fmt_gold(12.0) == "12"
    assert fmt_gold(3.5) == "3.5"
