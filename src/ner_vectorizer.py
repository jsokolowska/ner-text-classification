from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

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
    def transform(self, raw_documents, bio_tags):  # returns array-like
        pass

    @abstractmethod
    def get_feature_names_out(self):  # returns array of strings
        pass

    @abstractmethod
    def get_params(self, deep: bool) -> Dict[str, str]:
        pass

    @abstractmethod
    def get_named_entities(
            self):  # returns array of tuples? or maybe json-like representation of named entities like in spacy
        pass

    def __get_bio_tags(self, data: List[List[str]]) -> List[List[Tuple[str, str]]]:
        """
        Predict BIO tags
        :param data: list of tokenized documents
        :return: tagged_data: iterable yielding lists of tuples (token, tag)
        """

        pass

    def __check_ner_classifier(self) -> bool:
        if self.ner_classifier is not None and isinstance(self.ner_classifier, NamedEntityClassifier.__class__):
            return True
        raise TypeError("Ner classifier must be provided to use ner tags")


# todo cleanup double tf-idf, especially auto ner predictions
class DoubleTfIdfVectorizer(NamedEntityVectorizer):
    """ Text representation based on tf-idf vectorizer where two values are computed
    for each term - one tf-idf value for those occurences of the term in which it is
    a part of named entity and one tf-idf for other occurences"""

    def __init__(self, ner_classifier: NamedEntityClassifier):
        super(DoubleTfIdfVectorizer, self).__init__(
            ner_classifier
        )
        self._bio_tags = None
        self._tagged_documents = []
        self._tfidf = TfidfVectorizer()

    def fit(self, raw_documents: List[List[str]], /, bio_tags: List[List[str]] = None):
        """Learn idf and vocabulary from training set. If bio_tags are provided underlying
        NER classifier will be tuned too.

        Parameters
        ----------
        raw_documents : iterable
            An iterable yielding lists of terms

        bio_tags : iterable
            An iterable yielding list of tags, dimensions should match raw_documents
        """
        self._tagged_documents = []
        if bio_tags is None:
            self._predict_bio_tags(raw_documents)
        else:
            self._bio_tags = bio_tags

        for document, tag_list in zip(raw_documents, self._bio_tags):
            tagged_doc = []
            for word, tag in zip(document, tag_list):
                if tag != 'O':
                    tagged_doc.append(word + "_NER")
                else:
                    tagged_doc.append(word)
            self._tagged_documents.append(tagged_doc)

        self.doc_list = [" ".join(tagged_doc) for tagged_doc in self._tagged_documents]
        self._tfidf.fit(self.doc_list)

    def fit_transform(self, raw_documents, bio_tags: List[List[str]] = None):  # returns array-like
        """Learn vocabulary and idf, return document-term matrix.

        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which generates either str, unicode or file objects.

        bio_tags : None
            This parameter is ignored.

        Returns
        -------
        X : sparse matrix of (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """
        self._tagged_documents = []
        if bio_tags is None:
            self._bio_tags = self._predict_bio_tags(raw_documents)
        else:
            self._bio_tags = bio_tags

        for document, tag_list in zip(raw_documents, self._bio_tags):
            tagged_doc = []
            for word, tag in zip(document, tag_list):
                if tag != 'O':
                    tagged_doc.append(word + "_NER")
                else:
                    tagged_doc.append(word)
            self._tagged_documents.append(tagged_doc)

        self.doc_list = [" ".join(tagged_doc) for tagged_doc in self._tagged_documents]
        return self._tfidf.fit_transform(self.doc_list)

    def transform(self, raw_documents, bio_tags):
        pass

    def get_feature_names_out(self):  # returns array of strings
        pass

    def get_params(self, deep: bool) -> Dict[str, str]:
        pass

    def get_named_entities(
            self):  # returns array of tuples? or maybe json-like representation of named entities as in spacy
        pass

    def _predict_bio_tags(self, raw_documents):
        # todo predict tags
        pass
