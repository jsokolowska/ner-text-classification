from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from string import punctuation
from frozenlist import FrozenList
from abc import ABC, abstractmethod
from typing import List, Iterable, Dict, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
import contractions

from util import TextGetter
from ner_classifier import NamedEntityClassifier, SpacyNEClassifier

__all__ = ["DoubleTfIdfVectorizer"]

TokenList = Iterable[Iterable[str]]


class NamedEntityVectorizer(ABC):
    """Provides common interface and methods for named entity vectorizers"""

    def __init__(self, ner_classifier: NamedEntityClassifier = None, max_df=1.0, min_df=1,
                 tune_classifier: bool = False, filter_stopwords: bool = True, lemmatize: bool = True,
                 normalize: bool = True):

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
        self.lemmatize = lemmatize
        self.norm = normalize
        self.tune_ner = tune_classifier
        self.feature_names = []
        self._text_getter = None
        self.n_iter = 10

    @abstractmethod
    def fit(self, raw_documents: Iterable[str] = None, preprocessed_text: TextGetter = None, n_iter: int = 10):
        pass

    @abstractmethod
    def transform(self, raw_documents: Iterable[str],
                  preprocessed_text: TextGetter = None) -> np.ndarray:  # returns array-like
        pass

    def get_feature_names(self) -> FrozenList[str]:
        return FrozenList(self.feature_names)

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

    def _filter_stopwords(self):
        for _, row in self._df.iterrows():
            filtered = [(word, bio) for word, bio in zip(row["Tokens"], row["Tags"]) if
                        word not in self._STOPWORDS and word not
                        in self._PUNCTUATION]
            row["Tokens"] = [word for word, _ in filtered]
            row["Tags"] = [tag for _, tag in filtered]

    def _lower_and_filter_punctuation(self):
        for _, row in self._df.iterrows():
            filtered = [(word, bio) for word, bio in zip(row["Tokens"], row["Tags"]) if word not
                        in self._PUNCTUATION]
            row["Tokens"] = [word.lower() for word, _ in filtered]
            row["Tags"] = [tag for _, tag in filtered]

    def _lemmatize(self):
        for _, row in self._df.iterrows():
            pos_tagged = _add_pos_tags(row["Tokens"])
            row["Tokens"] = [self._lemmatizer.lemmatize(w, t) for w, t in pos_tagged]


def _normalize(tfidf: pd.DataFrame) -> np.ndarray:
    squared = tfidf**2
    norms = np.sqrt(squared.sum(axis=1))
    norms[norms == 0] = 1   # prevent zero division
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


class DoubleTfIdfVectorizer(NamedEntityVectorizer):
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
    Each term can have a maximum of two tf-idf values. One for all term occurences in which it is a part of any type
    of named entity and one for all the other term occurences. Tf-idf scores are computed with formula used by
    TfidfVectorizer from sci-kit learn in order to avoid zero division. Tf-idf vectors are then normalized with
    euclidean norm.
    """

    def __init__(self, ner_classifier: NamedEntityClassifier = None, max_df=1.0, min_df=1,
                 tune_classifier: bool = False, filter_stopwords: bool = True, lemmatize: bool = True,
                 normalize: bool = True):
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
        """
        super().__init__(ner_classifier, max_df, min_df, tune_classifier, filter_stopwords, lemmatize, normalize)
        self.tfidf = None

    def fit(self, raw_documents: Iterable[str] = None, tokenized: pd.Series = None, bio_tags: pd.Series = None,
            n_iter: int = 10) -> NamedEntityVectorizer:
        """Calculate idf frequencies of words and optionally tune underlying ner classifier

        :param raw_documents
        iterable yielding strings, representing raw, unprocessed collection of documents
        :param preprocessed_text
        TextGetter with tokenized sentences and BIO tags
        :param n_iter
        Number of iterations to pass to ner_classifier if tune_classifier was set to true
        :returns fitted vectorizer
        """
        self._validate_params(raw_documents, tokenized, bio_tags)
        self.n_iter = n_iter

        if raw_documents:
            self.corpus = raw_documents
            self.tokenize_and_tag(raw_documents)
            # todo make it work with df
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
                  tokenized:pd.Series = None, bio_tags:pd.Series= None, use_idf=True) -> np.ndarray:
        """
        Transform documents into double tf-idf representation using idf scores learned from fit method.
        Must be run after class has been fitted
        :param raw_documents: iterable yielding strings of documents to be transformed
        :param preprocessed_text: TextGetter class containing tokenized documents and bio tags
        :param normalize: default true, if true tf-idf scores for each document will be normalized using euclidean norm
        :return: np.ndarray of tf-idf scores
        """
        self._use_idf = use_idf
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

    def _validate_params(self, raw_documents, tokenized, bio_tags):
        if not (tokenized is not None and bio_tags is not None) and self.tune_ner:
            raise ValueError("Bio tags must be provided to do name entity classificator tuning")
        if not raw_documents and not (tokenized is not None and bio_tags is not None):
            raise ValueError("Either raw documents or tokenized and bio_tags must be provided")
        if tokenized is not None and bio_tags is not None:
            if not len(tokenized) == len(bio_tags):
                raise ValueError("Expected tokenized sentences and tokenlist to have the same length")

    def _preprocess(self):
        if self.filter_stopwords:
            self._filter_stopwords()
        else:
            self._lower_and_filter_punctuation()
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
        self._idf = self._idf.sort_index()
        self.feature_names = self._idf.index.values.tolist()

        self.tfidf = pd.DataFrame(columns=self.feature_names, index = self._df.index)

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
            tf = tf/doc_len
            for word, tf_val in tf.iteritems():
                self.tfidf.loc[idx][word] = tf_val

        self.tfidf = self.tfidf.fillna(0)
        self.tfidf = self.tfidf.multiply(self._idf, axis=1)
        if self.norm:
            self.tfidf = _normalize(self.tfidf)


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
                 normalize: bool = True):
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
        """
        super().__init__(ner_classifier, max_df, min_df, tune_classifier, filter_stopwords, lemmatize, normalize)
        self.tfidf = None

    #todo make work with df
    def _count_df(self):
        for sentence in self._preprocessed:
            words = set()
            for word, tag in sentence:
                if tag != "O":
                    words.add(tag)
                words.add(word)
            for word in words:
                self._idf[word] += 1

    #todo make work with df
    def _calculate_tf_idf(self):
        vocab_size = len(self._idf)
        doc_num = len(self._preprocessed)
        self.tfidf = np.zeros((doc_num, vocab_size))
        self.feature_names = [key for key, _ in self._idf.items()]
        self.feature_names.sort()
        for i in range(0, doc_num):
            for word, tag in self._preprocessed[i]:
                if tag != "O":
                    key = tag[2:]
                    if key in self.feature_names:
                        col_num = self.feature_names.index(key)
                        self.tfidf[i, col_num] += 1
                if word in self.feature_names:
                    col_num = self.feature_names.index(word)
                    self.tfidf[i, col_num] += 1
            self.tfidf[i, :] /= len(self._preprocessed[i]) + len(
                [tag for _, tag in self._preprocessed[i] if tag != "O"])

        for col_num in range(0, vocab_size):
            feature_name = self.feature_names[col_num]
            idf_score = self._idf[feature_name]
            self.tfidf[:, col_num] *= idf_score
        if self.norm:
            self.tfidf = _normalize(self.tfidf)


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd

if __name__ == "__main__":
    sentences = [["This", "is", "the", "first", "document", "here"],
                 ["And", "here", "we", "have", "the", "second", "one"]]
    tags = [['O' for t in s] for s in sentences]
    sentence_concat = [" ".join(s) for s in sentences]

    df = pd.DataFrame({"Tokens": sentences, "Tags": tags})

    mine = DoubleTfIdfVectorizer(filter_stopwords=False,lemmatize=False)
    tfidf = TfidfVectorizer()
    count = CountVectorizer()
    print("--- sklearn ----")
    tf_idf = tfidf.fit_transform(sentence_concat)
    sklearn_tfidf = pd.DataFrame(data=tf_idf.toarray(),columns = tfidf.get_feature_names())
    #sklearn_count = pd.DataFrame(data=count.fit_transform(sentence_concat).toarray(), columns=count.get_feature_names())
    #print("Count: ")
    #print(sklearn_count)
    print("Tfidf: ")
    print(sklearn_tfidf)
    print("---- custom ----")
    res = mine.fit(tokenized=df["Tokens"], bio_tags=df["Tags"])
    res2 = mine.transform(tokenized=df["Tokens"], bio_tags=df["Tags"])
    df = pd.DataFrame(res2, columns=mine.get_feature_names())
    print("Tfidf: ")
    print(df)

