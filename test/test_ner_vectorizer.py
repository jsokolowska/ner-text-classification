import pytest

from src.representations import DoubleTfIdfVectorizer


def test_fail():
    assert 2 == 1


def test_success():
    assert 1 == 1


@pytest.mark.parametrize(
    "tokenized, tags, raw",
    [
        ([["a", ":"]], [["B-PER", "z"]], None),
        ([["a"]], [["O", "O"]], None),
        ([["a"]], None, None),
        (None, [["O"]], None),
        (None, ["a"], ["aaa", "bbb"]),
        (["A"], None, ["a"]),
    ],
)
def test_input_validation_for_fit(tokenized, tags, raw):
    vect = DoubleTfIdfVectorizer()
    with pytest.raises(ValueError):
        vect.fit(tokenized=tokenized, bio_tags=tags, raw_documents=raw)
