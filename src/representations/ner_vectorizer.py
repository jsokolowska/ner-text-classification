from abc import ABC, abstractmethod

import contractions
import numpy as np
from nltk import WordNetLemmatizer, pos_tag
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer

from .common import PreprocessingPipeline
from .ner_classifier import NamedEntityClassifier
from typing import Callable, Iterable, Dict, Tuple
from nltk.corpus import stopwords, wordnet
import pandas as pd
import time

__all__ = ["DoubleTfIdfVectorizer", "NamedEntityVectorizer"]


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
        token_filter: Callable[
            [Iterable[Tuple[str, str]]], Iterable[Tuple[str, str]]
        ] = None,
    ):
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
        self.tokenized = None
        self.tagged = None
        self._use_idf = None
        self.corpus = None
        self._time_count = []
        self.__lemmatizer = WordNetLemmatizer()
        self.tfidf = None

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
            raise ValueError(
                f"Please provide either raw documents or tokenized documents and tags"
            )
        elif raw_documents is not None and self.clf is None:
            raise ValueError(f"When using raw_documents please provide ne classifier")
        elif (tokenized is not None and tags is None) or (
            tokenized is None and tags is not None
        ):
            raise TypeError(
                f"You need to provide tokenized sentences and tags associated with them"
            )
        if tokenized is not None and tags is not None and len(tokenized) != len(tags):
            raise TypeError(f"Input length mismatch")

    def _preprocess(self, documents: [str]):
        pipe = self._preprocess_pipeline()
        return [pipe.run(text) for text in documents]

    def _preprocess_pipeline(self) -> PreprocessingPipeline:
        pipeline = PreprocessingPipeline()
        if self.preprocessor:
            pipeline.add(self.preprocessor)
        pipeline.add(contractions.fix)
        return pipeline

    def _tag(self, preprocessed: [str]):
        self._df = self.clf.predict(preprocessed)

    def _calculate_idf(self):
        self._time_count.append(("Before count", time.time()))
        self._count_df()
        self._time_count.append(("After count", time.time()))
        self._narrow_down_vocab()
        self._time_count.append(("Before invert", time.time()))
        self._invert_df()
        self._time_count.append(("After invert", time.time()))

    def _group(self):
        if "sentence #" in self._df.columns:
            df = pd.DataFrame()
            df["tokens"] = self._df.groupby("sentence #").apply(
                lambda sent: [w for w in sent["tokens"].values.tolist()]
            )
            df["tags"] = self._df.groupby("sentence #").apply(
                lambda sent: [t for t in sent["tags"].values.tolist()]
            )
            self._df = df

    def _degroup(self):
        if "sentence #" not in self._df.columns:
            self._df["sentence #"] = [i for i in range(0, len(self._df))]
        elif not self._df["sentence #"].is_unique:
            return
        self._df = self._df.explode(["tokens", "tags"])

    def _count_df(self):
        def add(x):
            keys = self._get_tokens(x[1], x[2])
            tokens.extend(keys)

        self._degroup()
        tokens = []
        self._df.apply(lambda x: add(x), axis=1)
        self._token_count = pd.Series(tokens).value_counts()

    @abstractmethod
    def _get_tokens(self, word: str, tag: str) -> [str]:
        pass

    def _build_preprocessor(self):
        def preprocess_for_sklearn(sentence):
            results = pipe.run(sentence)
            return [w for w, _ in results]

        def lemmatize_no_tag(tokenlist):
            pos_tagged = pos_tag(tokenlist)
            return [lemmatizer.lemmatize(w, get_wnet_tag(t)) for w, t in pos_tagged]

        lemmatizer = self.__lemmatizer
        pipe = PreprocessingPipeline()

        if self.lemmatize:
            pipe.add(lemmatize_no_tag)
        if self.lower:
            pipe.add(lambda lst: [t.lower() for t in lst])
        if self.filter_stopwords:
            pipe.add(
                lambda lst: [
                    token for token in lst if token not in stopwords.words("english")
                ]
            )
        if self.token_filter:
            pipe.add(lambda lst: self.token_filter([(token, "O") for token in lst]))
        return preprocess_for_sklearn

    def build_preprocessor_and_tokenizer(self):
        def tokenize_and_preprocess(sentence):
            cleaned = pipe.run(sentence)
            tokenlist = [t.text for t in tokenizer(cleaned)]
            return preprocessor(tokenlist)

        preprocessor = self._build_preprocessor()
        pipe = self._preprocess_pipeline()
        nlp = English()
        tokenizer = Tokenizer(nlp.vocab)
        return tokenize_and_preprocess

    def _calculate_tf_idf(self):
        if self._use_idf:
            self._calculate_tfidf()
        else:
            self._calculate_tf()

    def _calculate_tfidf(self):
        self._group()
        self._calculate_tf()
        self.tfidf = self.tfidf.multiply(self._idf, axis=1)

    def _narrow_down_vocab(self):
        doc_num = len(self.corpus)
        lower_bound = self.min_df
        upper_bound = self.max_df
        if isinstance(self.min_df, float):
            lower_bound *= doc_num
        if isinstance(self.max_df, float):
            upper_bound *= doc_num

        self._idf = self._token_count.where(
            lambda x: [upper_bound >= item >= lower_bound for item in x]
        ).dropna()

    def _invert_df(self):
        doc_num = len(self.corpus)
        self._idf = np.log((1 + doc_num) / (1 + self._idf)) + 1

    def _calculate_tf(self):
        self._time_count.append(("Before tf count", time.time()))
        self._idf = self._idf.sort_index()
        self.feature_names = self._idf.index.values.tolist()

        self.tfidf = pd.DataFrame(columns=self.feature_names, index=self._df.index)
        for idx, row in self._df.iterrows():
            terms = []
            for word, tag in zip(row["tokens"], row["tags"]):
                keys = self._get_tokens(word, tag)
                terms.extend(keys)
            doc_len = len(row["tokens"])
            tf = pd.Series(terms).value_counts()
            tf = tf / doc_len
            for word, tf_val in tf.iteritems():
                self.tfidf.loc[idx][word] = tf_val
        self.tfidf = self.tfidf.fillna(0)
        self._time_count.append(("After tf count", time.time()))

    def _filter_tokens(self):
        self._group()
        pipe = PreprocessingPipeline()
        if self.lemmatize:
            pipe.add(self._lemmatize)
        if self.lower:
            pipe.add(lambda lst: [(z.lower(), y) for z, y in lst])
        if self.filter_stopwords:
            pipe.add(
                lambda lst: [
                    (x, y) for x, y in lst if x not in stopwords.words("english")
                ]
            )
        if self.token_filter:
            pipe.add(lambda lst: self.token_filter(lst))
        tokens = []
        tags = []
        for token_list, tag_list in zip(self._df["tokens"], self._df["tags"]):
            res = pipe.run([(w, t) for w, t in zip(token_list, tag_list)])
            tokens.append([w for w, _ in res])
            tags.append([t for _, t in res])
        self._df = pd.DataFrame(
            {
                "sentence #": [i for i in range(0, len(tokens))],
                "tokens": tokens,
                "tags": tags,
            }
        )

    def _lemmatize(self, lst) -> [str]:
        pos_tagged = pos_tag([w for w, _ in lst])
        lemmatized = [
            self.__lemmatizer.lemmatize(w, get_wnet_tag(t)) for w, t in pos_tagged
        ]
        return [(l, t) for l, t in zip(lemmatized, [t for _, t in lst])]

    def _is_fitted(self):
        if self._use_idf:
            return True
        elif self._idf is not None:
            return True
        return False


def get_wnet_tag(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def _normalize(tfidf: pd.DataFrame) -> pd.DataFrame:
    squared = tfidf**2
    norms = np.sqrt(squared.sum(axis=1))
    norms[norms == 0] = 1  # prevent zero division
    return tfidf.div(norms, axis=0)


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
            self._time_count.append(("Before preprocessing [ms]", time.time()))
            self.corpus = raw_documents
            preprocessed = self._preprocess(raw_documents)
            self._tag(preprocessed)
            self._time_count.append(("After preprocessing [ms]", time.time()))
        else:
            self.corpus = tokenized
            self._df = pd.DataFrame({"Tokens": tokenized, "Tags": bio_tags})
        self._filter_tokens()
        self._calculate_idf()
        return self

    def get_idf(self):
        return self._idf

    def get_time_counts(self):
        return self._time_count

    def fit_transform(
        self,
        raw_documents: Iterable[str] = None,
        tokenized: pd.Series = None,
        bio_tags: pd.Series = None,
        use_idf=True,
    ) -> pd.DataFrame:
        self._use_idf = use_idf
        _validate_params(raw_documents, tokenized, bio_tags)
        if raw_documents:
            self._time_count.append(("Before preprocessing [ms]", time.time()))
            self.corpus = raw_documents
            preprocessed = self._preprocess(raw_documents)
            self._tag(preprocessed)
            self._time_count.append(("After preprocessing [ms]", time.time()))
            self._group()
        else:
            self.corpus = tokenized
            self._df = pd.DataFrame({"Tokens": tokenized, "Tags": bio_tags})
        self._filter_tokens()
        self._degroup()
        self._calculate_idf()
        self._calculate_tf_idf()
        if self.normalize:
            self.tfidf = _normalize(self.tfidf)
        return self.tfidf

    def transform(
        self,
        raw_documents: Iterable[str] = None,
        tokenized: pd.Series = None,
        bio_tags: pd.Series = None,
        use_idf: bool = True,
    ) -> pd.DataFrame:
        self._use_idf = use_idf
        if not self._is_fitted():
            raise ValueError("Vectorizer not fitted")
        _validate_params(raw_documents, tokenized, bio_tags)
        if raw_documents:
            self._time_count.append(("Before preprocessing [ms]", time.time()))
            self.corpus = raw_documents
            preprocessed = self._preprocess(raw_documents)
            self._tag(preprocessed)
            self._time_count.append(("After preprocessing [ms]", time.time()))
        else:
            self.corpus = tokenized
            self._df = pd.DataFrame({"Tokens": tokenized, "Tags": bio_tags})
        self._filter_tokens()
        self._calculate_tf()
        if self.normalize:
            self.tfidf = _normalize(self.tfidf)
        return self.tfidf

    def preprocessing_only(self, raw_documents: Iterable[str]):
        self._time_count.append(("Before preprocessing [ms]", time.time()))
        self.corpus = raw_documents
        preprocessed = self._preprocess(raw_documents)
        self._tag(preprocessed)
        self._time_count.append(("After preprocessing [ms]", time.time()))
        self._group()
        self._filter_tokens()
        self._degroup()
        return self._df

    def _get_tokens(self, word: str, tag: str) -> [str]:
        if tag == "O":
            return [word]
        else:
            return [word + "_NE"]
