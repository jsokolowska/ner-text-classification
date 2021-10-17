from typing import Dict, List, Tuple
from ner_classifier import NamedEntityClassifier
from abc import ABC, abstractmethod


class NamedEntityVectorizer(ABC):
    """Provides common interface and methods for named entity vectorizers"""
    def __init__(self,
                 ner_classifier: NamedEntityClassifier):
        if ner_classifier is None:
            raise TypeError("ner_classifier must be a class inheriting from NamedEntityClassifier")
        self.ner_classifier = ner_classifier

    @abstractmethod
    def fit(self, raw_documents: List[List[str]], bio_tags: List[List[str]]):
        pass

    @abstractmethod
    def transform(self, raw_documents):  # returns array-like
        pass

    @abstractmethod
    def get_feature_names_out(self):  # returns array of strings
        pass

    @abstractmethod
    def get_params(self, deep: bool) -> Dict[str, str]:
        pass

    @abstractmethod
    def get_named_entities(self):  # returns array of tuples? or maybe json-like representation of named entities like in spacy
        pass

    def __get_bio_tags(self, data: List[List[str]]) -> List[List[Tuple[str, str]]]:
        """
        Predict BIO tags
        :param data: list of tokenized documents
        :return: tagged_data: iterable yielding lists of tuples (token, tag)
        """

        # todo write implementation based on ner classifier interface

        pass

    def __check_ner_classifier(self) -> bool:
        if self.ner_classifier is not None and isinstance(self.ner_classifier, NamedEntityClassifier.__class__):
            return True
        raise TypeError("Ner classifier must be provided to use ner tags")


class DoubleTfIdfVectorizer(NamedEntityVectorizer):
    """ Text representation based on tf-idf vectorizer where two values are computed
    for each term - one tf-idf value for those occurrences of the term in which it is
    a part of named entity and one tf-idf for other occurrences"""

    def __init__(self, ner_classifier: NamedEntityClassifier):
        super(DoubleTfIdfVectorizer, self).__init__(
            ner_classifier
        )

    def fit(self, raw_documents: List[List[str]], /, bio_tags: List[List[str]]):
        """Learn idf and vocabulary from training set. If bio_tags are provided underlying
        NER classifier will be tuned too.

        Parameters
        ----------
        raw_documents : iterable
            An iterable yielding lists of terms

        bio_tags : iterable
            An iterable yielding list of tags, dimensions should match raw_documents
        """
        # todo predict tags
        # todo concatenate word + NER if bio tag != 'O'
        # todo create tf-idf representation
        pass

    def transform(self, raw_documents):     # returns array-like
        """Learn vocabulary and idf, return document-term matrix.

        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which generates either str, unicode or file objects.

        y : None
            This parameter is ignored.

        Returns
        -------
        X : sparse matrix of (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """
        pass

    def get_feature_names_out(self):    # returns array of strings
        pass

    def get_params(self, deep: bool) -> Dict[str, str]:
        pass

    def get_named_entities(self):   # returns array of tuples? or maybe json-like representation of named entities as in spacy
        pass
