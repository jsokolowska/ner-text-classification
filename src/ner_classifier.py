from abc import ABC, abstractmethod
from typing import List, Tuple, Iterable

import pandas as pd
import spacy
from spacy.util import minibatch
from tqdm import tqdm
from spacy.training import Example
from spacy.tokens import Doc
import random

__all__ = ["NamedEntityClassifier", "SpacyNEClassifier"]


class NamedEntityClassifier(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def _load_from_file(self, filepath):
        pass

    @abstractmethod
    def _load_from_library(self):
        pass

    @abstractmethod
    def train(self, train_data: List[List[Tuple[str, str]]], tags: List[str], n_iter: int):
        pass

    @abstractmethod
    def predict(self, test_data: List[List[str]], raw_data:Iterable[str]):
        pass


class SpacyNEClassifier(NamedEntityClassifier):
    def __init__(self, filepath: str = None):
        """

        Parameters
        -----
        filepath: str
            Path to file containing pretrained model
        train_data: list[list[tuple(str, str)]]
            Iterable yielding iterables of tuples (word, ner tag)
        """
        super().__init__()
        self.reformatted_data = None
        self.losses = []
        self.ner = None
        if filepath is not None:
            self._load_from_file(filepath)
        else:
            self._load_from_library()

    def _load_from_file(self, filepath: str):
        self.nlp = spacy.blank('en')
        self.nlp.add_pipe('ner')
        self.nlp.from_disk(filepath)
        self.ner = self.nlp.get_pipe('ner')

    def _load_from_library(self):
        self.nlp = spacy.blank('en')
        self.nlp.add_pipe('ner')
        self.ner = self.nlp.get_pipe('ner')

    # todo redo for df
    def train(self, train_data: List[List[Tuple[str, str]]], tags: List[str], n_iter: int):
        self.reformatted_data = _to_spacy_format(train_data)
        self._add_tags(tags)

        optimizer = self.nlp.begin_training()
        pipes = [p for p in self.nlp.pipe_names if p != 'ner']
        with self.nlp.disable_pipes(*pipes):
            examples = self._make_training_examples()
            self.nlp.initialize(lambda: examples)
            self.losses = []
            for _ in tqdm(range(n_iter)):
                random.shuffle(examples)
                for batch in minibatch(examples, size=8):
                    loss = {}
                    self.nlp.update(batch, sgd=optimizer, losses=loss)
                    self.losses.append(loss['ner'])

    def predict(self, test_data: List[List[str]] = None, raw_data: Iterable[str] = None):
        test_docs = self._make_doc_test(test_data, raw_data)

        scores = self.ner.predict(test_docs)
        self.ner.set_annotations(test_docs, scores)
        return _from_spacy_format(test_docs)

    def get_losses(self) -> List[float]:
        return self.losses

    def _make_training_examples(self):
        examples = []
        for text, annotation in self.reformatted_data:
            doc = self.nlp.make_doc(text)
            examples.append(Example.from_dict(doc, annotation))
        return examples

    def _make_doc_test(self, test_data: List[List[str]] = None, raw_data: Iterable[str] = None) -> Iterable[Doc]:
        if test_data is not None:
            return [self.nlp.make_doc(" ".join(words)) for words in test_data]
        elif raw_data is not None:
            return [self.nlp.make_doc(document) for document in raw_data]
        else:
            raise TypeError("No test_data or raw_data provided")

    def _add_tags(self, tags: List[str]):
        self.tags = tags
        for tag in tags:
            self.ner.add_label(tag)

    def save(self, filepath: str):
        self.nlp.to_disk(filepath)


def _to_spacy_format(labeled_sentences: List[List[Tuple[str, str]]]):
    json_data = []
    for labeled_sentence in labeled_sentences:
        entity_list = []
        idx = 0
        for token, label in labeled_sentence:
            if label != 'O':
                entity_list.append((idx, idx + len(token), label))
            idx += len(token) + 1

        text = " ".join([word for (word, _) in labeled_sentence][:-1])
        json_data.append((text, {"entities": entity_list}))
    return json_data


def _from_spacy_format(processed_examples: Iterable[Doc]) -> pd.DataFrame:
    sentences = []
    tags = []
    for elem in processed_examples:
        tokens = [token for token in elem]
        labels = ['0' for i in range(len(tokens))]
        for entity in elem.ents:
            for idx in range(entity.start, entity.end):
                labels[idx]= entity.label_

        sentences.append(tokens)
        tags.append(labels)
    return pd.DataFrame({"tokens": sentences, "tags": tags})


if __name__ == "__main__":
    df = pd.read_csv("../preprocessed-data/bbc/whole_raw.csv")

    ner = SpacyNEClassifier(filepath ="../pretrained-models/kaggle-ner-train-spacy")
    res = ner.predict(raw_data=df["raw_text"][:10])
    print(res.labelled_sentences)
