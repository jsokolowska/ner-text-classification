from abc import ABC, abstractmethod
from .ner_classifier import NamedEntityClassifier
from typing import Callable, Iterable, Dict
from nltk.corpus import stopwords
import nltk
import pandas as pd


class NamedEntityVectorizer(ABC):
    """Provides common interface and methods for named entity vectorizers"""

    def __init__(
        self,
        ner_clf: NamedEntityClassifier = None,
        max_df=1.0,
        min_df=1.0,
        filter_stopwords: bool = True,
        lemmatize: bool = True,
        normalize: bool = True,
        filter_punctuation: bool = True,
        lower: bool = True,
        fix_contractions: bool = True,
        preprocessor: Callable[[str], str] = None,
        token_filter: Callable[[Iterable[str]], Iterable[str]] = None,
    ):
        # Download needed nltk libraries
        nltk.download("stopwords")

        self.clf = ner_clf

        self.min_df = min_df
        self.max_df = max_df

        self.filter_stopwords = filter_stopwords
        self.lemmatize = lemmatize
        self.normalize = normalize
        self.filter_punctuation = filter_punctuation
        self.lower = lower
        self.fix_contractions = fix_contractions

        self.preprocessor = preprocessor
        self.token_filter = token_filter

        self._idf = None
        self._token_count = None
        self._df = None
        self._STOPWORDS = stopwords
        self._feature_names = []

    @abstractmethod
    def fit(
        self,
        raw_documents: Iterable[str] = None,
        tokenized: pd.Series = None,
        tags: pd.Series = None,
    ):
        pass

    @abstractmethod
    def transform(
        self,
        raw_documents: Iterable[str] = None,
        tokenized: pd.Series = None,
        tags: pd.Series = None,
        use_idf: bool = True,
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def fit_transform(
        self,
        raw_documents: Iterable[str] = None,
        tokenized: pd.Series = None,
        tags: pd.Series = None,
        use_idf: bool = True,
    ) -> pd.DataFrame:
        pass

    def get_params(self) -> Dict[str, any]:
        params = {}
        for name in dir(self):
            value = getattr(self, name)
            if not name.startswith("_") and (
                isinstance(value, bool)
                or isinstance(value, str)
                or isinstance(value, int)
                or isinstance(value, float)
            ):
                params[name] = value
        return params

    def _check_params(
        self, raw_documents: Iterable[str], tokenized: pd.Series, tags: pd.Series
    ):
        if raw_documents is not None and (tokenized is not None or tags is not None):
            raise TypeError(
                f"Please provide either raw documents or tokenized documents and tags"
            )
        elif (tokenized is not None and tags is None) or (
            tokenized is None and tags is not None
        ):
            raise TypeError(
                f"You need to provide tokenized sentences and tags associated with them"
            )

        if len(tokenized) != len(tags):
            raise TypeError(f"Input length mismatch")

    def _preprocess(self):
        # call user-defined preprocessor
        # todo clean html tags
        # todo fix contractions
        pass

    def _tag(self):
        # todo tag ner classifier
        pass

    def _calculate_idf(self):
        pass


def _validate_params(raw_documents, tokenized, bio_tags):
    if not (
        raw_documents is not None and tokenized is None and bio_tags is None
    ) and not (
        tokenized is not None and bio_tags is not None and raw_documents is not None
    ):
        raise ValueError(
            "Either raw documents or tokenized and bio_tags must be provided"
        )
    if tokenized is not None and bio_tags is not None:
        if not len(tokenized) == len(bio_tags):
            raise ValueError(
                "Expected tokenized sentences and tag list to have the same length"
            )
        for token_list, bio_list in zip(tokenized, bio_tags):
            if not len(token_list) == len(bio_list):
                raise ValueError(
                    "Token list and tag list for each sentence must be the same length"
                )
    if bio_tags:
        _validate_bio_tags(bio_tags)


def _validate_bio_tags(tag_list: [[str]]):
    for lst in tag_list:
        for tag in lst:
            if not (tag == "O" or tag.startswith("B-") or tag.startswith("I-")):
                raise ValueError("Noncompliant tagging scheme")


class DoubleTfIdfVectorizer(NamedEntityVectorizer):
    """
    Text vectorizer built on top of tf-idf implementation.
    Data preprocessing steps:
        - removing contractions (for raw documents only)
        - tokenizing (raw documents only)
        - predicting bio tags with user-provided ner_classifier
        - filtering stopwords and punctuation tokens
        - pos tagging
        - lemmatization
    Computing tf-idf scores:
    Each term can have a maximum of two tf-idf values. One for all term occurences in which it is a part of any type
    of named entity and one for all the other term occurences. Tf-idf scores are computed with formula used by
    TfidfVectorizer from sci-kit learn in order to avoid zero division. Tf-idf vectors are then optionally normalized
    with euclidean norm.
    """

    def __init__(
        self,
        ner_clf: NamedEntityClassifier = None,
        max_df=1.0,
        min_df=1,
        filter_stopwords: bool = True,
        lemmatize: bool = True,
        normalize: bool = True,
        filter_punctuation: bool = True,
        lower: bool = True,
        fix_contractions: bool = True,
        preprocessor: Callable[[str], str] = None,
        token_filter: Callable[[str], str] = None,
    ):
        """
        :param max_df
        int or float, ignore words that have higher document
        frequency, if float between 0.0 and 1.0 then it refers to the proportion of documents, if integer - to
        document count
        :param min_df
        int or float, ignore words that have lower document frequency, if float between
        0.0 and 1.0 then it refers to the proportion of documents, if integer - to document count
        :param filter_stopwords
        bool, defaults to true, if false then stopwords won't be filtered
        :param lemmatize
        bool, defaults to true, when true lemmatization with wordnet is performed before calculating tf-idf
        representation
        :param ner_clf:
        named entity classifier that will be used for bio tags prediction if none are provided in fit and transform
        methods
        :param filter_punctuation:
        bool, defaults to true, filters out punctuation tokens
        :param lower:
        bool, defaults to true, lowers all characters in strings
        :param normalize:
        bool, defaults  to true, resulting tf-idf vectors will be  normalized so that their  measure is equal to one
        :param fix_contractions:
        bool, defaults to true, attempts to replace contractions with full forms
        :param preprocessor:
        function with additional preprocessing steps for text documents (before tokenization), takes in unprocessed
        string and returns text after cleaning
        :param token_filter:
        function with additional token filtering or replacing, should take list of tokens as parameter, and return
        a list of filtered token
        """
        super().__init__(
            ner_clf,
            max_df,
            min_df,
            filter_stopwords,
            lemmatize,
            normalize,
            filter_punctuation,
            lower,
            fix_contractions,
            preprocessor,
            token_filter,
        )
        self._tfidf = None

    def fit(
        self,
        raw_documents: Iterable[str] = None,
        tokenized: pd.Series = None,
        bio_tags: pd.Series = None,
        n_iter: int = 10,
    ) -> NamedEntityVectorizer:
        """Calculate idf frequencies of words
        :param raw_documents
        iterable yielding strings, representing raw, unprocessed collection of documents
        :param tokenized: pd.Series, containing lists of strings (tokens)
        :param bio_tags: pd.Series, containing lists of enitity tags, this uses BIO tagging schema, assuming 'O' to mean
        no named entity is present
        :param n_iter
        Number of iterations to pass to ner_classifier if tune_classifier was set to true
        :returns fitted vectorizer
        """
        _validate_params(raw_documents, tokenized, bio_tags)
        if raw_documents:
            self._preprocess()
            self._tag()
        else:
            self._df = pd.DataFrame({"Tokens": tokenized, "Tags": bio_tags})
        self._calculate_idf()
        return self

    def _preprocess(self):
        pass

    def _tag(self):
        pass

    def _calculate_idf(self):
        pass

    def fit_transform(
        self,
        raw_documents: Iterable[str] = None,
        tokenized: pd.Series = None,
        bio_tags: pd.Series = None,
        use_idf=True,
    ) -> pd.DataFrame:
        pass

    def transform(
        self,
        raw_documents: Iterable[str] = None,
        tokenized: pd.Series = None,
        bio_tags: pd.Series = None,
        use_idf: bool = True,
    ) -> pd.DataFrame:
        pass
