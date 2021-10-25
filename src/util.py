import pandas as pd
from typing import List, Tuple, Set
from sklearn.model_selection import train_test_split
import csv

__all__ = ["load_kaggle_ner", "LabelledSentence", "TokenList", "TextData", "save", "data_split", "load_preprocessed",
           "TextGetter"]

LabelledSentence = List[Tuple[str, str]]
TokenList = List[List[str]]
TextData = (List[LabelledSentence], List[str])


class TextGetter:
    def __init__(self, /, labelled_sentences: List[LabelledSentence] = None, sentences: TokenList = None,
                 bio_tags: TokenList = None,
                 tags: Set[str] = None):
        if labelled_sentences:
            self.labelled_sentences = labelled_sentences
            self.bio_tags = [[tag for _, tag in labelled_sentence] for labelled_sentence in labelled_sentences]
            self.sentences = [[word for word, _ in labelled_sentence] for labelled_sentence in labelled_sentences]
        elif sentences and bio_tags:
            self.labelled_sentences = [[(word, tag) for word, tag in zip(sentence, labels)] for sentence, labels in
                                       zip(sentences, bio_tags)]
            self.sentences = sentences
            self.bio_tags = bio_tags
        else:
            raise ValueError(
                "Either labelled_sentences or sentences and tokens must be provided"
            )
        if tags:
            self.tags = tags
        else:
            self.tags = set()
            for tag_list in self.bio_tags:
                self.tags.update(tag_list)


def load_kaggle_ner(data_path: str = "../data/kaggle-ner/ner_dataset.csv") \
        -> TextGetter:
    df = pd.read_csv(data_path, encoding="latin1").fillna(method="ffill")
    grouped_sentences = _get_sentences(df)
    return TextGetter(labelled_sentences=[[(str(s[0]), str(s[1])) for s in sentence] for sentence in grouped_sentences],
                      tags=df["Tag"].unique().tolist())


def _get_sentences(data: pd.DataFrame):
    return data.groupby("Sentence #").apply(
        lambda sentence: [(word, tag) for word, tag
                          in zip(sentence["Word"].values.tolist(),
                                 sentence["Tag"].values.tolist())]
    )


def data_split(data: TextGetter) -> (TextGetter, TextGetter):
    sentences = data.sentences
    tags = data.sentences
    train_s, test_s, train_l, test_l = train_test_split(sentences, tags)
    return TextGetter(sentences=train_s, bio_tags=train_l), TextGetter(sentences=test_s, bio_tags=test_l)


def save(filepath: str, labelled_sentences: List[LabelledSentence] = None, sentences: TokenList = None,
         labels: TokenList = None, text_getter: TextGetter = None):
    if labelled_sentences or text_getter:
        if text_getter:
            labelled_sentences = text_getter.labelled_sentences
        with open(filepath, "w") as file:
            writer = csv.writer(file)
            writer.writerow(["Sentence #", "Word", "Tag"])
            i = 0
            for sentence in labelled_sentences:
                header = f"Sentence: {i}"
                for word, token in sentence:
                    writer.writerow([header, word, token])
                    header = None
                i += 1

    elif sentences and labels:
        with open(filepath, "w") as file:
            writer = csv.writer(file)
            writer.writerow(["Sentence #", "Word", "Tag"])
            i = 0
            for sentence, label_list in zip(sentences, labels):
                header = f"Sentence: {i}"
                for word, token in zip(sentence, label_list):
                    writer.writerow([header, word, token])
                    header = None
                i += 1
    else:
        raise ValueError("Either labelled_sentences or text_getter or sentences and labels must be provided")


def load_preprocessed(data_dir: str) -> (TextGetter, TextGetter):
    train_sentences = []
    train_labels = []
    test_sentences = []
    test_labels = []
    with open(data_dir + "train.csv", "r") as file:
        reader = csv.reader(file)
        for sentence, labels in reader:
            train_sentences.append(sentence)
            train_labels.append(labels)

    with open(data_dir + "test.csv", "r") as file:
        reader = csv.reader(file)
        for sentence, labels in reader:
            test_sentences.append(sentence)
            test_labels.append(labels)
    return TextGetter(sentences=train_sentences, bio_tags=train_labels), TextGetter(sentences=test_sentences,
                                                                                    bio_tags=test_labels)
