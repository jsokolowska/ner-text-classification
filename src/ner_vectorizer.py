from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from string import punctuation
from abc import ABC, abstractmethod
from typing import Iterable, Dict
import numpy as np
import pandas as pd
import contractions

from ner_classifier import NamedEntityClassifier

__all__ = ["DoubleTfIdfVectorizer"]


class NamedEntityVectorizer(ABC):
    """Provides common interface and methods for named entity vectorizers"""

    def __init__(self, ner_classifier: NamedEntityClassifier = None, max_df=1.0, min_df=1,
                 tune_classifier: bool = False, filter_stopwords: bool = True, lemmatize: bool = True,
                 normalize: bool = True, filter_punctuation: bool = True, filter_whitespaces: bool = True,
                 lower: bool = True):

        self._idf = None
        self._token_count = None
        self._df = None

        self.ner_classifier = ner_classifier
        self.corpus = [[]]
        self.max_df = max_df
        self.min_df = min_df
        self._STOPWORDS = ENGLISH_STOP_WORDS
        self._PUNCTUATION = list(punctuation)
        self._lemmatizer = WordNetLemmatizer()
        self._bio_tags = []
        self._tokenized = []
        self._preprocessed = []
        self.filter_stopwords = filter_stopwords
        self.filter_punctuation = filter_punctuation
        self.lower = lower
        self.filter_whitespaces = filter_whitespaces
        self.lemmatize = lemmatize
        self.norm = normalize
        self.tune_ner = tune_classifier
        self.feature_names = []
        self._text_getter = None
        self.n_iter = 10

    @abstractmethod
    def fit(self, raw_documents: Iterable[str] = None, tokenized: pd.Series = None, bio_tags: pd.Series = None,
            n_iter: int = 10):
        pass

    @abstractmethod
    def transform(self, raw_documents: Iterable[str] = None,
                  tokenized: pd.Series = None, bio_tags: pd.Series = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def fit_transform(self, raw_documents: Iterable[str] = None, tokenized: pd.Series = None,
                      bio_tags: pd.Series = None, use_idf=True) -> pd.DataFrame:
        pass

    def get_params(self) -> Dict[str, any]:
        params = {}
        for name in dir(self):
            value = getattr(self, name)
            if not name.startswith("_") and (isinstance(value, bool) or isinstance(value, str) or isinstance(value, int)
                                             or isinstance(value, float)):
                params[name] = value
        return params

    def _check_ner_classifier(self) -> bool:
        if self.ner_classifier is not None and isinstance(self.ner_classifier, NamedEntityClassifier.__class__):
            return True
        raise TypeError("Ner classifier must be provided to use ner tags")

    def is_fitted(self):
        return len(self._idf) > 0

    def _invert_df(self):
        doc_num = len(self.corpus)
        self._idf = np.log((1 + doc_num) / (1 + self._idf)) + 1

    def _narrow_down_vocab(self):
        doc_num = len(self.corpus)
        lower_bound = self.min_df
        upper_bound = self.max_df
        if isinstance(self.min_df, float):
            lower_bound *= doc_num
        if isinstance(self.max_df, float):
            upper_bound *= doc_num
        temp = self._token_count[self._token_count <= upper_bound]
        self._idf = temp[temp >= lower_bound]

    def tokenize_and_tag(self, raw_documents):
        raw_documents = [contractions.fix(doc) for doc in raw_documents]
        self._tokenized = [_tokenize(doc) for doc in raw_documents]
        self._bio_tags = self._add_bio_tags(self._tokenized)

    def _add_bio_tags(self, tokenized_docs):
        predictions = self.ner_classifier.predict(tokenized_docs)
        return predictions.bio_tags

    def _tune_classifier(self):
        # todo fit ner classifier - add niter as param
        self.ner_classifier.train(self._text_getter.labelled_sentences, self._text_getter.tags, self.n_iter)

    def _filter(self):
        tokens_to_filter_out = []
        if self.filter_punctuation:
            tokens_to_filter_out.extend(self._PUNCTUATION)
        if self.filter_stopwords:
            tokens_to_filter_out.extend(list(self._STOPWORDS))
        if self.filter_whitespaces and len(tokens_to_filter_out) > 0:
            for _, row in self._df.iterrows():
                filtered = [(word, bio) for word, bio in zip(row["Tokens"], row["Tags"]) if word not
                            in tokens_to_filter_out and not word.isspace()]
                row["Tokens"] = [word for word, _ in filtered]
                row["Tags"] = [tag for _, tag in filtered]
        elif self.filter_whitespaces:
            for _, row in self._df.iterrows():
                filtered = [(word, bio) for word, bio in zip(row["Tokens"], row["Tags"]) if not word.isspace()]
                row["Tokens"] = [word for word, _ in filtered]
                row["Tags"] = [tag for _, tag in filtered]
        elif len(tokens_to_filter_out) > 0:
            for _, row in self._df.iterrows():
                filtered = [(word, bio) for word, bio in zip(row["Tokens"], row["Tags"]) if
                            word not in tokens_to_filter_out]
                row["Tokens"] = [word for word, _ in filtered]
                row["Tags"] = [tag for _, tag in filtered]

    def _lower(self):
        self._df["Tokens"] = [[word.lower() for word in row] for row in self._df["Tokens"]]

    def _lemmatize(self):
        for _, row in self._df.iterrows():
            pos_tagged = _add_pos_tags(row["Tokens"])
            row["Tokens"] = [self._lemmatizer.lemmatize(w, t) for w, t in pos_tagged]

    def build_preprocessor(self):
        steps = []
        if self.lower:
            steps.append(lambda tokenlist: [e.lower() for e in tokenlist])
        filter_out = []
        if self.filter_punctuation:
            filter_out.extend(self._PUNCTUATION)
        if self.filter_stopwords:
            filter_out.extend(list(self._STOPWORDS))
        if len(filter_out) > 0:
            steps.append(lambda tokenlist: [e for e in tokenlist if e not in filter_out])
        if self.filter_whitespaces:
            steps.append(lambda tokenlist: [e for e in tokenlist if not e.isspace()])
        if self.lemmatize:
            steps.append(lambda tokenlist: [self._lemmatizer.lemmatize(w,t) for w,t in _add_pos_tags(tokenlist)])


        def preprocessor(x):
            lst = x
            for step in steps:
                lst = step(lst)
            return lst

        return preprocessor


def _normalize(tfidf: pd.DataFrame) -> pd.DataFrame:
    squared = tfidf ** 2
    norms = np.sqrt(squared.sum(axis=1))
    norms[norms == 0] = 1  # prevent zero division
    return tfidf.div(norms, axis=0)


def _get_wnet_tag(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    if tag.startswith('V'):
        return wordnet.VERB
    if tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def _add_pos_tags(document):
    tokenized_sent = [word for word in document]
    pos_tagged_sent = pos_tag(tokenized_sent)
    return [(word, _get_wnet_tag(pos)) for word, pos in pos_tagged_sent]


def _tokenize(document):
    return [x.lower() for x in word_tokenize(document)]


def _strip_prefix_if_bio_tag(tag: str):
    if tag.startswith("B-") or tag.startswith("I-"):
        return tag[2:]
    else:
        return tag


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
        - filtering whitespace-only tokens
    Computing tf-idf scores:
    Each term can have a maximum of two tf-idf values. One for all term occurences in which it is a part of any type
    of named entity and one for all the other term occurences. Tf-idf scores are computed with formula used by
    TfidfVectorizer from sci-kit learn in order to avoid zero division. Tf-idf vectors are then normalized with
    euclidean norm.
    """

    def __init__(self, ner_classifier: NamedEntityClassifier = None, max_df=1.0, min_df=1,
                 tune_classifier: bool = False, filter_stopwords: bool = True, lemmatize: bool = True,
                 normalize: bool = True, filter_punctuation: bool = True, filter_whitespaces: bool = True,
                 lower: bool = True):
        """
        :param max_df
        int or float, ignore words that have higher document
        frequency, if float between 0.0 and 1.0 then it refers to the proportion of documents, if integer - to
        document count
        :param min_df
        int or float, ignore words that have lower document frequency, if float between
        0.0 and 1.0 then it refers to the proportion of documents, if integer - to document count
        :param tune_classifier
        bool, If true then underlying ner classifier will be tuned using provided tags. Can be only set to true
        if preprocessed_text is provided
        :param filter_stopwords
        bool, defaults to true, if false then stopwords won't be filtered
        :param lemmatize
        bool, defaults to true, when true lemmatization with wordnet is performed before calculating tf-idf
        representation
        :param ner_classifier:
        named entity classifier that will be used for bio tags prediction if none are provided in fit and transform
        methods
        :param filter_whitespaces:
        bool, defaults to true, filters out whitespace only tokens (and associated ner tags)
        :param filter_punctuation:
        bool, defaults to true, filters out punctuation tokens
        :param lower:
        bool, defaults to true, lowers all characters in strings
        :param normalize:
        bool, defaults  to true, resulting tf-idf vectors will be  normalized so that their  measure is equal to one
        """
        super().__init__(ner_classifier, max_df, min_df, tune_classifier, filter_stopwords, lemmatize, normalize,
                         filter_punctuation, filter_whitespaces, lower)
        self.tfidf = None

    def fit(self, raw_documents: Iterable[str] = None, tokenized: pd.Series = None, bio_tags: pd.Series = None,
            n_iter: int = 10) -> NamedEntityVectorizer:
        """Calculate idf frequencies of words and optionally tune underlying ner classifier

        :param raw_documents
        iterable yielding strings, representing raw, unprocessed collection of documents
        :param tokenized: pd.Series, containing lists of strings (tokens)
        :param bio_tags: pd.Series, containing lists of bio_tags
        :param n_iter
        Number of iterations to pass to ner_classifier if tune_classifier was set to true
        :returns fitted vectorizer
        """
        self._validate_params(raw_documents, tokenized, bio_tags)
        self.n_iter = n_iter

        if raw_documents:
            self.corpus = raw_documents
            self.tokenize_and_tag(raw_documents)
            # todo make it work with df, don t tokenize raw documents - use spacy!
        elif tokenized is not None and bio_tags is not None:
            self._df = pd.DataFrame({"Tokens": tokenized, "Tags": bio_tags})
            self.corpus = tokenized
            if self.tune_ner:
                self._tune_classifier()

        self._preprocess()

        self._count_df()
        self._narrow_down_vocab()
        self._invert_df()
        return self

    def transform(self, raw_documents: Iterable[str] = None,
                  tokenized: pd.Series = None, bio_tags: pd.Series = None) -> pd.DataFrame:
        """
        Transform documents into double tf-idf representation using idf scores learned from fit method.
        Must be run after class has been fitted
        :param tokenized: pd.Series, containing lists of strings (tokens)
        :param bio_tags: pd.Series, containing lists of bio_tags
        :param raw_documents: iterable yielding strings of documents to be transformed
        :return: np.ndarray of tf-idf scores
        """
        if not self.is_fitted():
            raise ValueError("Cannot transform text, vectorizer has not been fitted")
        if raw_documents:
            self.tokenize_and_tag(raw_documents)
            # todo make it work with df
        elif tokenized is not None and bio_tags is not None:
            self._df = pd.DataFrame({"Tokens": tokenized, "Tags": bio_tags})
            if self.tune_ner:
                # todo make it work with df
                self._tune_classifier()

        self._preprocess()
        self._calculate_tf_idf()
        return self.tfidf

    def fit_transform(self, raw_documents: Iterable[str] = None, tokenized: pd.Series = None,
                      bio_tags: pd.Series = None, use_idf=True) -> pd.DataFrame:
        """
        Calculate idf frequencies of words  and transform documents into double tf-idf representation.
        This is equal to fit followed by transform, but a bit more effecient

        :param tokenized: pd.Series, containing lists of strings (tokens)
        :param bio_tags: pd.Series, containing lists of bio_tags
        :param use_idf: bool, defaults to true, if set to false this vectorizer will only compute term frequencies
        :param raw_documents: iterable yielding strings of documents to be transformed
        :return: pd.Dataframe of tf-idf scores
        """
        self._validate_params(raw_documents, tokenized, bio_tags)
        if raw_documents:
            self.corpus = raw_documents
            self.tokenize_and_tag(raw_documents)
            # todo make it work with df, don t tokenize raw documents - use spacy?
        elif tokenized is not None and bio_tags is not None:
            self._df = pd.DataFrame({"Tokens": tokenized, "Tags": bio_tags})
            self.corpus = tokenized

        self._preprocess()
        if use_idf:
            self._count_df()
            self._narrow_down_vocab()
            self._invert_df()
            self._calculate_tf_idf()
        else:
            self._calculate_tf()

        if self.norm:
            self.tfidf = _normalize(self.tfidf)
        return self.tfidf

    def _validate_params(self, raw_documents, tokenized, bio_tags):
        if not (tokenized is not None and bio_tags is not None) and self.tune_ner:
            raise ValueError("Bio tags must be provided to do name entity classificator tuning")
        if not raw_documents and not (tokenized is not None and bio_tags is not None):
            raise ValueError("Either raw documents or tokenized and bio_tags must be provided")
        if tokenized is not None and bio_tags is not None:
            if not len(tokenized) == len(bio_tags):
                raise ValueError("Expected tokenized sentences and tag list to have the same length")

    def _preprocess(self):
        if self.lower:
            self._lower()
        self._filter()
        if self.lemmatize:
            self._lemmatize()

    def _count_df(self):
        df_degrouped = pd.DataFrame(columns=["Term"])
        for _, row in self._df.iterrows():
            terms = set()
            for word, tag in zip(row["Tokens"], row["Tags"]):
                if tag != "O":
                    terms.add(word + "_NE")
                terms.add(word)
            df_degrouped = df_degrouped.append(pd.DataFrame({"Term": list(terms)}))
        self._token_count = df_degrouped["Term"].value_counts()

    def _calculate_tf_idf(self):
        self._calculate_tf()
        self.tfidf = self.tfidf.multiply(self._idf, axis=1)

    def _calculate_tf(self):
        self._idf = self._idf.sort_index()
        self.feature_names = self._idf.index.values.tolist()

        self.tfidf = pd.DataFrame(columns=self.feature_names, index=self._df.index)
        for idx, row in self._df.iterrows():
            terms = []
            for word, tag in zip(row["Tokens"], row["Tags"]):
                if tag != "O":
                    key = word + "_NE"
                    if key in self.feature_names:
                        terms.append(word + "_NE")
                if word in self.feature_names:
                    terms.append(word)
            doc_len = len(row["Tokens"])
            tf = pd.Series(terms).value_counts()
            tf = tf / doc_len
            for word, tf_val in tf.iteritems():
                self.tfidf.loc[idx][word] = tf_val
        self.tfidf = self.tfidf.fillna(0)


class BioTfIdfVectorizer(DoubleTfIdfVectorizer):
    """
    Text vectorizer built on top of tf-idf implementation.
    Data preprocessing steps:
        - removing contractions
        - tokenizing
        - predicting bio tags with user-provided ner_classifier
        - filtering stopwords and punctuation tokens
        - pos tagging
        - lemmatization
    Computing tf-idf scores:
    All tokens and all tag (with the exception of tag 'O') are considered to be terms. Thus resulting implementation
    gives information about frequency of lemmas but also how frequent are respective types of named entities.
    """

    def __init__(self, ner_classifier: NamedEntityClassifier = None, max_df: float = 1.0, min_df: float = 1,
                 tune_classifier: bool = False, filter_stopwords: bool = True, lemmatize: bool = True,
                 normalize: bool = True, filter_punctuation: bool = True, filter_whitespaces: bool = True,
                 lower: bool = True):
        """
        :param max_df
        int or float, ignore words that have higher document
        frequency, if float between 0.0 and 1.0 then it refers to the proportion of documents, if integer - to
        document count
        :param min_df
        int or float, ignore words that have lower document frequency, if float between
        0.0 and 1.0 then it refers to the proportion of documents, if integer - to document count
        :param tune_classifier
        bool, If true then underlying ner classifier will be tuned using provided tags. Can be only set to true
        if preprocessed_text is provided
        :param filter_stopwords
        bool, defaults to true, if false then stopwords won't be filtered
        :param lemmatize
        bool, defaults to true, when true lemmatization with wordnet is performed before calculating tf-idf r
        epresentation
        :param ner_classifier:
        named entity classifier that will be used for bio tags prediction if none are provided in fit and transform
        methods
        :param filter_whitespaces:
        bool, defaults to true, filters out whitespace only tokens (and associated ner tags)
        :param filter_punctuation:
        bool, defaults to true, filters out punctuation tokens
        :param lower:
        bool, defaults to true, lowers all characters in strings
        :param normalize:
        bool, defaults  to true, resulting tf-idf vectors will be  normalized so that their  measure is equal to one
        """
        super().__init__(ner_classifier, max_df, min_df, tune_classifier, filter_stopwords, lemmatize, normalize,
                         filter_punctuation, filter_whitespaces, lower)
        self.tfidf = None

    def _count_df(self):
        df_degrouped = pd.DataFrame(columns=["Term"])
        for _, row in self._df.iterrows():
            terms = set()
            for word, tag in zip(row["Tokens"], row["Tags"]):
                if tag != "O":
                    key = _strip_prefix_if_bio_tag(tag)
                    terms.add(key)
                terms.add(word)
            df_degrouped = df_degrouped.append(pd.DataFrame({"Term": list(terms)}))
        self._token_count = df_degrouped["Term"].value_counts()

    def _calculate_tf(self):
        self._idf = self._idf.sort_index()
        self.feature_names = self._idf.index.values.tolist()

        self.tfidf = pd.DataFrame(columns=self.feature_names, index=self._df.index)
        for idx, row in self._df.iterrows():
            terms = []
            for word, tag in zip(row["Tokens"], row["Tags"]):
                if tag != "O":
                    key = _strip_prefix_if_bio_tag(tag)
                    if key in self.feature_names:
                        terms.append(key)
                if word in self.feature_names:
                    terms.append(word)
            doc_len = len(row["Tokens"])
            tf = pd.Series(terms).value_counts()
            tf = tf / doc_len
            for word, tf_val in tf.iteritems():
                self.tfidf.loc[idx][word] = tf_val
        self.tfidf = self.tfidf.fillna(0)


if __name__ == "__main__":
    sent = ["jumped", "right", "into", "a", "average",",","but", "spirited", "young", "woman","."]
    tags = [["0" for t in sent]]
    tfidf = DoubleTfIdfVectorizer()
    tfidf.fit(tokenized= [sent], bio_tags = tags)
    pass

#%%
