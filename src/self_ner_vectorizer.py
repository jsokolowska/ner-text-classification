from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords, wordnet
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
import warnings

__all__ = ["DoubleTfIdfVectorizer"]

from ner_classifier import NamedEntityClassifier
from abc import ABC, abstractmethod
from typing import List, Iterable, Dict, Tuple
from util import TextGetter


class NamedEntityVectorizer(ABC):
    """Provides common interface and methods for named entity vectorizers"""

    def __init__(self,
                 ner_classifier: NamedEntityClassifier):
        if ner_classifier is None:
            raise TypeError("ner_classifier must be a class inheriting from NamedEntityClassifier")
        self.ner_classifier = ner_classifier

    @abstractmethod
    def fit(self, raw_documents: Iterable[str], labelled_data: TextGetter):
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


class DoubleTfIdfVectorizer():
    def __init__(self, ner_classifier):
        self.__STOPWORDS = set(list(punctuation) + stopwords.words("english"))
        self.__lemmatizer = WordNetLemmatizer()
        self.sentences = [[]]
        self.bio_tags = []
        self.tokenized = []
        self.preprocessed = []
        self.max_df = 1.0
        self.min_df = 1
        self.vocab = {}
        self.ner_classifier = ner_classifier
        self.tune_ner = False

    def fit(self, raw_documents: Iterable[str] = None, preprocessed_text: TextGetter = None, max_df=1.0, min_df=1,
            tune_classifier: bool = False
            ):
        """Calculate idf frequencies of words

        :param raw_documents
        iterable yielding strings, representing raw, unprocessed collection of documents
        :param preprocessed_text
        TextGetter with tokenized sentences and BIO tags
        :param max_df
        int or float, ignore words that have higher document
        frequency, if float between 0.0 and 1.0 then it refers to the proportion of documents, if integer - to
        document count
        :param min_df
        int or float, ignore words that have lower document frequency, if float between
        0.0 and 1.0 then it refers to the proportion of documents, if integer - to document count
        :param tune_classifier
        bool, If true then underlying ner classifier will be tuned using provided tags. Can be only set to true if preprocessed_text is provided
        """
        if not preprocessed_text and tune_classifier:
            raise ValueError("Bio tags must be provided to do classificator tuning")
        if raw_documents:
            self.tokenize_and_tag(raw_documents)
        elif preprocessed_text:
            if tune_classifier:
                self.__tune_classifier()
            self.bio_tags = preprocessed_text.tags
            self.tokenized = preprocessed_text.sentences

        else:
            raise ValueError("Either raw documents or preprocessed_text must be provided")

        self.tune_ner = tune_classifier
        self.preprocess()
        self.__count_vocab()
        pass

    def __count_vocab(self):
        pass

    def __tune_classifier(self):
        # todo fit ner classifier
        warnings.warn("Tuning classifiers not implemented")
        pass

    def __filter_stopwords(self) -> Tuple[List[List[str]], List[List[str]]]:
        combined = [[(word, bio) for word, bio in zip(sentence, entities)] for sentence, entities in
                    zip(self.tokenized, self.bio_tags)]
        filtered_combined = [[(word, bio) for word, bio in sentence if word not in self.__STOPWORDS] for sentence in
                             combined]
        bio_tags = [[bio for _, bio in sentence] for sentence in filtered_combined]
        sentences = [[word for word, _ in sentence] for sentence in filtered_combined]
        return sentences, bio_tags

    def __lemmatize(self, pos_tagged_sentence) -> List[str]:
        return [self.__lemmatizer.lemmatize(w, t) for w, t in pos_tagged_sentence]

    def tokenize_and_tag(self, raw_documents):
        self.bio_tags = self.__add_bio_tags(raw_documents)
        self.tokenized = [self.__tokenize(doc) for doc in raw_documents]

    def preprocess(self):
        sentences, bio_tags = self.__filter_stopwords()
        pos_tagged = [self.__add_pos_tags(doc) for doc in sentences]
        lemmatized = [self.__lemmatize(doc) for doc in pos_tagged]
        self.preprocessed = [[(word, bio) for word, bio in zip(sentence, bio_tags)] for sentence, bio_tags in
                             zip(lemmatized, bio_tags)]

    def __get_wnet_tag(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        if tag.startswith('V'):
            return wordnet.VERB
        if tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def __add_pos_tags(self, document):
        tokenized_sent = [word for word in document]  # maybe python has more effecient function for that
        pos_tagged_sent = pos_tag(tokenized_sent)
        return [(word, self.__get_wnet_tag(pos)) for word, pos in pos_tagged_sent]

    def __tokenize(self, document):
        return [x.lower() for x in word_tokenize(document)]

    def __add_bio_tags(self, documents):
        # todo handle that with nerclassifier
        warnings.warn("Adding bio tags not implemented")
        return [
            ["a" for word in sentence] for sentence in documents
        ]
        pass


document = ["This is the first sentence.", "This was not only but also walked the second one."]
vect = DoubleTfIdfVectorizer()
print(vect.preprocess(document))
