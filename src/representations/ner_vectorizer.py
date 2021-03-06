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
from collections import defaultdict
from scipy.sparse import dok_matrix, spmatrix
from sklearn.preprocessing import normalize as sklearn_norm
import logging


__all__ = ["DoubleTfIdfVectorizer", "NamedEntityVectorizer", "BioTfIdfVectorizer"]


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
                [Iterable[str], Iterable[str]], Tuple[Iterable[str], Iterable[str]]
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
        self._use_idf = True
        self.__lemmatizer = WordNetLemmatizer()
        self._tfidf = None
        self._feature_counts = defaultdict()
        self._feature_counts.default_factory = lambda: 0
        self._feature_idx = defaultdict()
        self._feature_idx.default_factory = self._feature_idx.__len__
        self._doc_num = 0
        self._log = logging.getLogger("Ha")

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
    ) -> spmatrix:
        pass

    @abstractmethod
    def fit_transform(
            self,
            raw_documents: Iterable[str] = None,
            tokenized: pd.Series = None,
            tags: pd.Series = None,
            use_idf: bool = True,
    ) -> spmatrix:
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

    def get_feature_names(self):
        return self._feature_names

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
        self._count_vocab()
        self._invert()

    def _group(self):
        if "sentence #" in self._df.columns and not self._df["sentence #"].is_unique:
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

    def _count_vocab(self):
        self._group()
        for _, row in self._df.iterrows():
            tokens, tags = self.filter_tokens_sent(row['tokens'], row['tags'])
            keys = self._get_tokens(tokens, tags)

            for k in keys:
                self._feature_counts[k] += 1

        lower_bound = self.min_df
        upper_bound = self.max_df
        if isinstance(self.min_df, float):
            lower_bound *= self._doc_num
        if isinstance(self.max_df, float):
            upper_bound *= self._doc_num

        self._feature_counts = {k: v for k, v in self._feature_counts.items() if upper_bound >= v >= lower_bound}
        self._idf = pd.Series(data=self._feature_counts.values(), index=self._feature_counts.keys())
        self._log.error(f"idf shape = {self._idf.shape}")
        self._log.error(f"Feature counts = {self._feature_counts}")

    @abstractmethod
    def _get_tokens(self, words: [str], tags: [str]) -> [str]:
        pass

    def _build_preprocessor(self):
        def preprocess_for_sklearn(sentence):
            results = pipe.run(sentence)
            if results and type(results[0]) is tuple:
                return [w for w, _ in results]
            elif results and isinstance(results, tuple):
                return results[0]
            else:
                return results

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
            pipe.add(lambda lst: self.token_filter(lst, ["O" for _ in lst]))
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

    def _narrow_down_vocab(self):
        lower_bound = self.min_df
        upper_bound = self.max_df
        if isinstance(self.min_df, float):
            lower_bound *= self._doc_num
        if isinstance(self.max_df, float):
            upper_bound *= self._doc_num

        self._idf = self._token_count.where(
            lambda x: [upper_bound >= item >= lower_bound for item in x]
        ).dropna()

    def _invert(self):
        self._idf = np.log((1 + self._doc_num) / (1 + self._idf)) + 1

    def _calculate_tf_idf(self):
        self._idf = self._idf
        self._group()
        shape = (len(self._df), len(self._idf.index.values.tolist()))
        tfidf = dok_matrix(shape)
        self._log.error(f"Starting shape of sparse array = {shape}")

        for idx, row in self._df.iterrows():
            tokens, tags = self.filter_tokens_sent(row['tokens'], row['tags'])
            terms = self._get_tokens(tokens, tags)
            tf = pd.Series(terms).value_counts()
            tf = tf / len(row["tokens"])

            for word, tf_val in tf.iteritems():
                if word in self._feature_counts:
                    col_idx = self._feature_idx[word]
                    if self._use_idf:
                        idf = self._idf[word]
                        tf_val = tf_val / idf
                    tfidf[idx, col_idx] = tf_val

        self._feature_names = [*self._feature_idx.keys()]
        self._log.error(f"End shape of sparse array = {tfidf.shape}")
        self._log.error(tfidf.values())
        if self.normalize:
            tfidf = sklearn_norm(tfidf)
        self._tfidf = tfidf

    """
    filter and preprocess tokens for given input document
    """

    def filter_tokens_sent(self, tokens, tags):
        if self.lemmatize:
            tokens, tags = self._lemmatize(tokens, tags)
        if self.lower:
            tokens = [word.lower() for word in tokens]
        if self.filter_stopwords:
            filtered_tokens = []
            filtered_tags = []
            for token, tag in zip(tokens, tags):
                if token not in stopwords.words("english"):
                    filtered_tags.append(tag)
                    filtered_tokens.append(token)
            tokens = filtered_tokens
            tags = filtered_tags
        if self.token_filter:
            tokens, tags = self.token_filter(tokens, tags)
        return tokens, tags

    def _lemmatize(self, token_list, tag_list) -> [str]:
        pos_tagged = pos_tag(token_list)
        lemmatized = [
            self.__lemmatizer.lemmatize(w, get_wnet_tag(t)) for w, t in pos_tagged
        ]
        return lemmatized, tag_list

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


def _strip_prefix_if_bio_tag(tag: str):
    if tag.startswith("B-") or tag.startswith("I-"):
        return tag[2:]
    else:
        return tag


def _validate_params(raw_documents, tokenized, bio_tags):
    if not (
            raw_documents is not None and tokenized is None and bio_tags is None
    ) and not (
            tokenized is not None and bio_tags is not None and raw_documents is None
    ):
        raise ValueError(
            "Either raw documents or tokenized and bio_tags must be provided"
        )
    if tokenized is not None and bio_tags is not None:
        if len(tokenized) != len(bio_tags):
            raise ValueError(
                "Expected tokenized sentences and tag list to have the same length"
            )

        for token_list, bio_list in zip(tokenized, bio_tags):
            if not len(token_list) == len(bio_list):
                raise ValueError(
                    f"Token list and tag list for each sentence must be the same length. {token_list} and {bio_list} are not equal"
                )

    if bio_tags is not None:
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
        :returns fitted vectorizer
        """
        _validate_params(raw_documents, tokenized, bio_tags)
        if raw_documents is not None:
            self._doc_num = len(raw_documents)
            preprocessed = self._preprocess(raw_documents)
            self._tag(preprocessed)
        else:
            self._df = pd.DataFrame({"tokens": tokenized, "tags": bio_tags})
            self._doc_num = len(tokenized)
        if self._use_idf:
            self._calculate_idf()
        return self

    def get_idf(self):
        return self._idf

    def fit_transform(
            self,
            raw_documents: [str] = None,
            tokenized: pd.Series = None,
            bio_tags: pd.Series = None,
            use_idf=True,
    ) -> spmatrix:
        self._use_idf = use_idf
        self.fit(raw_documents, tokenized, bio_tags)
        self._calculate_tf_idf()
        return self._tfidf

    def transform(
            self,
            raw_documents: Iterable[str] = None,
            tokenized: pd.Series = None,
            bio_tags: pd.Series = None,
            use_idf: bool = True,
    ) -> spmatrix:
        self._use_idf = use_idf
        if not self._is_fitted():
            raise ValueError("Vectorizer not fitted")
        _validate_params(raw_documents, tokenized, bio_tags)
        if raw_documents is not None:
            preprocessed = self._preprocess(raw_documents)
            self._tag(preprocessed)
            self._doc_num = len(raw_documents)
        else:
            self._df = pd.DataFrame({"tokens": tokenized, "tags": bio_tags})
            self._doc_num = len(tokenized)

        self._calculate_tf_idf()
        return self._tfidf

    def _get_tokens(self, words: [str], tags: [str]) -> [str]:
        final_tokens = []
        for w, t in zip(words, tags):
            if t == "O":
                final_tokens.append(w)
            else:
                final_tokens.append(w + "_NE")
        return final_tokens


class BioTfIdfVectorizer(DoubleTfIdfVectorizer):
    def _get_tokens(self, words: [str], tags: [str]) -> [str]:
        final_tokens = []
        for w, t in zip(words, tags):
            final_tokens.append(w)
            if t != "O":
                final_tokens.append(_strip_prefix_if_bio_tag(t))
        return final_tokens
