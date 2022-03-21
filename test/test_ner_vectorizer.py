import pandas as pd
import pytest
import lorem
from sklearn.feature_extraction.text import TfidfVectorizer
from src.representations.preprocessing import *

from src.representations import DoubleTfIdfVectorizer, MockNoNamedEntityClassifier


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


def test_equality_to_sklearn():
    # given ne classifier that returns no entities
    ner = MockNoNamedEntityClassifier()
    # and random sentences as input
    raw_documents = [
        lorem.sentence(),
        lorem.sentence(),
        lorem.sentence(),
        lorem.sentence(),
    ]
    # and vectorizer with params compliant with sklearn
    double_vect = DoubleTfIdfVectorizer(ner_clf=ner)
    # and sklearn tfidf vectorizer
    tfidf_vect = TfidfVectorizer(
        analyzer="word",
        tokenizer=double_vect.build_preprocessor_and_tokenizer(),
        token_pattern=None,
    )
    # log input text
    print(raw_documents)

    # then when both vectorizer are run
    res_double = double_vect.fit_transform(raw_documents)
    res_sklearn = pd.DataFrame(
        tfidf_vect.fit_transform(raw_documents).toarray(),
        columns=tfidf_vect.get_feature_names(),
    )

    # their results are the same
    assert res_sklearn.shape[0] == res_double.shape[0]
    assert res_sklearn.shape[1] == res_double.shape[1]
    assert all(res_double == res_sklearn)


def test_fit_transform_equal_to_fit_and_transform():
    # given ne classifier that returns no entities
    ner = MockNoNamedEntityClassifier()
    # and random sentences as input
    raw_documents = [
        lorem.sentence(),
        lorem.sentence(),
        lorem.sentence(),
        lorem.sentence(),
    ]
    # and two vectorizers
    vect1 = DoubleTfIdfVectorizer(ner_clf=ner)
    vect2 = DoubleTfIdfVectorizer(ner_clf=ner)

    # log input text
    print(raw_documents)

    # then when both vectorizer are run
    res1 = vect1.fit_transform(raw_documents)

    vect2.fit(raw_documents)
    res2 = vect2.transform(raw_documents)

    # their results are the same
    assert res1.shape[0] == res2.shape[0]
    assert res1.shape[1] == res2.shape[1]
    assert all(res1 == res2)


def test_build_preprocessor_and_tokenizer():
    # given ne classifier that returns no entities
    ner = MockNoNamedEntityClassifier()
    # and input sentences
    raw_documents = [
        "This sentence has an url http://www.google.com and @mention and a number 12 12.3 12,5",
        lorem.sentence(),
        lorem.sentence(),
        lorem.sentence(),
    ]
    # and vectorizer with params compliant with sklearn
    double_vect = DoubleTfIdfVectorizer(
        ner_clf=ner, preprocessor=text_preprocessing, token_filter=token_filter
    )
    # and sklearn tfidf vectorizer with custom preprocesor
    tfidf_vect = TfidfVectorizer(
        analyzer="word",
        tokenizer=double_vect.build_preprocessor_and_tokenizer(),
        token_pattern=None,
    )

    # then when both vectorizer are run
    res_double = double_vect.fit_transform(raw_documents)
    res_sklearn = pd.DataFrame(
        tfidf_vect.fit_transform(raw_documents).toarray(),
        columns=tfidf_vect.get_feature_names(),
    )

    # resulting tokenization is identical
    assert all(res_double.columns == res_sklearn.columns)
