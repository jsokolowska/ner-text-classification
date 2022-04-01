from abc import ABC, abstractmethod
import pandas as pd
from typing import Iterable, Collection
import spacy
import time
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer

__all__ = ["NamedEntityClassifier", "SpacyNEClassifier", "MockNoNamedEntityClassifier"]

from spacy.tokens import Doc


class NamedEntityClassifier(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, sentences: Collection[str]):
        pass


class MockNoNamedEntityClassifier(NamedEntityClassifier):
    def predict(self, sentences: Collection[str]):
        tokenizer = Tokenizer(English().vocab)
        tokenized = []
        sentence_num = []
        counter = 0
        for sent in sentences:
            tokenized.append([t.text for t in tokenizer(sent)])
            sentence_num.append(counter)
            counter += 1
        df = pd.DataFrame({"sentence #": sentence_num, "tokens": tokenized}).explode(
            "tokens"
        )
        df["tags"] = "O"
        return df


class SpacyNEClassifier(NamedEntityClassifier):
    def __init__(self, pretrained=None, model_name="en_core_web_md"):
        super().__init__()
        self.ner = None
        self.nlp = None
        self.loses = []
        self.__load(pretrained, model_name)

    def __load(self, pretrained, model_name):
        if not pretrained:
            self.nlp = spacy.load(model_name)
        else:
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("ner")
            self.nlp.from_disk(pretrained)
        self.ner = self.nlp.get_pipe("ner")

    def predict(self, sentences: Collection[str]) -> pd.DataFrame:
        if sentences is None or len(sentences) == 0:
            raise TypeError("sentences need to be provided")

        docs = [self.nlp(doc) for doc in sentences]
        return _from_spacy_format(docs)


def _from_spacy_format(processed_examples: Iterable[Doc]) -> pd.DataFrame:
    sentence_num = []
    sentences = []
    tags = []
    counter = 0
    for elem in processed_examples:
        tokens = [token.text for token in elem]
        labels = ["O" for _ in range(len(tokens))]
        for entity in elem.ents:
            is_first = True
            for idx in range(entity.start, entity.end):
                if is_first:
                    labels[idx] = "B-" + entity.label_
                else:
                    labels[idx] = "I-" + entity.label_
                is_first = False

        sentences += tokens
        tags += labels
        sentence_num += [counter for _ in range(len(labels))]
        counter += 1
    return pd.DataFrame({"sentence #": sentence_num, "tokens": sentences, "tags": tags})


if __name__ == "__main__":
    start = time.time()
    df = pd.read_csv("../../data/bbc/raw.csv")
    ner = SpacyNEClassifier()
    res = ner.predict(df["raw_text"])
    end = time.time()
    print(f"Executed in {end - start} s")
