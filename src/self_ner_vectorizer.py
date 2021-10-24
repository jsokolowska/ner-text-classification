from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
import warnings
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
import numpy as np
import contractions
from frozenlist import FrozenList

__all__ = ["DoubleTfIdfVectorizer"]

from ner_classifier import NamedEntityClassifier
from abc import ABC, abstractmethod
from typing import List, Iterable, Dict, Tuple
from util import TextGetter
from collections import defaultdict


class NamedEntityVectorizer(ABC):
    """Provides common interface and methods for named entity vectorizers"""

    def __init__(self, ner_classifier: NamedEntityClassifier = None, max_df=1.0, min_df=1,
                 tune_classifier: bool = False, filter_stopwords: bool = True, lemmatize: bool = True,
                 normalize: bool = True):
        def default_val():
            return 0

        self._idf = defaultdict(default_val)

        self.ner_classifier = ner_classifier
        self.documents = [[]]
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
        doc_num = len(self.documents)
        self._idf = {k: np.log((1 + doc_num) / (1 + v)) + 1 for k, v in self._idf.items()}

    def _narrow_down_vocab(self):
        doc_num = len(self.documents)
        lower_bound = self.min_df
        upper_bound = self.max_df
        if isinstance(self.min_df, float):
            lower_bound *= doc_num
        if isinstance(self.max_df, float):
            upper_bound *= doc_num
        self._idf = {k: v for k, v in self._idf.items() if lower_bound <= v <= upper_bound}

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

    def _filter_stopwords(self) -> Tuple[List[List[str]], List[List[str]]]:
        combined = [[(word, bio) for word, bio in zip(sentence, entities)] for sentence, entities in
                    zip(self._tokenized, self._bio_tags)]
        filtered_combined = [[(word, bio) for word, bio in sentence if word not in self._STOPWORDS and word not
                              in self._PUNCTUATION] for sentence in combined]
        bio_tags = [[bio for _, bio in sentence] for sentence in filtered_combined]
        sentences = [[word for word, _ in sentence] for sentence in filtered_combined]
        return sentences, bio_tags

    def _lemmatize(self, pos_tagged_sentence) -> List[str]:
        return [self._lemmatizer.lemmatize(w, t) for w, t in pos_tagged_sentence]


def _normalize(tfidf: np.ndarray) -> np.ndarray:
    vect_norms = np.sqrt(np.sum(tfidf ** 2, axis=1))
    vect_norms = vect_norms.reshape((vect_norms.shape[0], 1))
    return tfidf / vect_norms


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

    def fit(self, raw_documents: Iterable[str] = None, preprocessed_text: TextGetter = None, n_iter: int = 10) -> NamedEntityVectorizer:
        """Calculate idf frequencies of words and optionally tune underlying ner classifier

        :param raw_documents
        iterable yielding strings, representing raw, unprocessed collection of documents
        :param preprocessed_text
        TextGetter with tokenized sentences and BIO tags
        :param n_iter
        Number of iterations to pass to ner_classifier if tune_classifier was set to true
        :returns fitted vectorizer
        """
        self.__validate_params(raw_documents, preprocessed_text)
        self._text_getter = preprocessed_text
        self.n_iter = n_iter

        if raw_documents:
            self.documents = raw_documents
            self.tokenize_and_tag(raw_documents)
        elif preprocessed_text:
            self.documents = preprocessed_text.sentences
            if self.tune_ner:
                self._tune_classifier()
            self._bio_tags = preprocessed_text.tags
            self._tokenized = preprocessed_text.sentences

        self.__preprocess()
        self.__count_df()
        self._narrow_down_vocab()
        self._invert_df()
        return self

    # todo add bio tag handling
    def transform(self, raw_documents: Iterable[str],
                  preprocessed_text: TextGetter = None) -> np.ndarray:  # returns array-like
        """
        Transform documents into double tf-idf representation using idf scores learned from fit method.
        Must be run after class has been fitted
        :param raw_documents: iterable yielding strings of documents to be transformed
        :param preprocessed_text: TextGetter class containing tokenized documents and bio tags
        :param normalize: default true, if true tf-idf scores for each document will be normalized using euclidean norm
        :return: np.ndarray of tf-idf scores
        """
        if not self.is_fitted():
            raise ValueError("Cannot transform text, vectorizer has not been fitted")

        self.tokenize_and_tag(raw_documents)
        self.__preprocess()
        self.__calculate_tf_idf()
        return self.tfidf

    def __validate_params(self, raw_documents, preprocessed_text):
        if not preprocessed_text and self.tune_ner:
            raise ValueError("Bio tags must be provided to do classificator tuning")
        if not raw_documents and not preprocessed_text:
            raise ValueError("Either raw documents or preprocessed_text must be provided")

    def __preprocess(self):
        if self.filter_stopwords:
            sentences, bio_tags = self._filter_stopwords()
        else:
            sentences = self._tokenized
            bio_tags = self._bio_tags
        pos_tagged = [_add_pos_tags(doc) for doc in sentences]
        if self.lemmatize:
            lemmatized = [self._lemmatize(doc) for doc in pos_tagged]
        else:
            lemmatized = [[word for word, _ in doc] for doc in pos_tagged]
        self.__preprocessed = [[(word, bio) for word, bio in zip(sentence, bio_tags)] for sentence, bio_tags in
                               zip(lemmatized, bio_tags)]

    def __count_df(self):
        for sentence in self.__preprocessed:
            words = set()
            for word, tag in sentence:
                if tag != "O":
                    words.add(word + "_NE")
                words.add(word)
            for word in words:
                self._idf[word] += 1

    def __calculate_tf_idf(self):
        vocab_size = len(self._idf)
        doc_num = len(self.__preprocessed)
        self.tfidf = np.zeros((doc_num, vocab_size))
        self.feature_names = [key for key, _ in self._idf.items()]
        self.feature_names.sort()
        for i in range(0, doc_num):
            for word, tag in self.__preprocessed[i]:
                if tag != "O":
                    key = word + "_NE"
                    if key in self.feature_names:
                        col_num = self.feature_names.index(key)
                        self.tfidf[i, col_num] += 1
                if word in self.feature_names:
                    col_num = self.feature_names.index(word)
                    self.tfidf[i, col_num] += 1
            self.tfidf[i, :] /= len(self.__preprocessed[i])

        for col_num in range(0, vocab_size):
            feature_name = self.feature_names[col_num]
            idf_score = self._idf[feature_name]
            self.tfidf[:, col_num] *= idf_score
        if self.norm:
            self.tfidf = _normalize(self.tfidf)

# documents = ['Topic sentences aren\'t similar to mini thesis statements.',
#              'Like a thesis statement, a topic sentence has a specific \
#               main point.']
# stopwords_for_sklearn = list(ENGLISH_STOP_WORDS) + list(punctuation)
#
# vect = DoubleTfIdfVectorizer()
# res = vect.fit(raw_documents=documents)
#
# tf_idf = vect.transform(raw_documents=documents)
# tfvect_sklearn = TfidfVectorizer(stop_words=stopwords_for_sklearn)
# transformed = tfvect_sklearn.fit_transform(raw_documents=documents).toarray()
# print(tf_idf[0])
# print(vect.feature_names)
# print(vect.get_params())
#
# print(transformed[0])
# print(tfvect_sklearn.get_feature_names())
