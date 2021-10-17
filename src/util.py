import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split
import csv

__all__ = ["load_kaggle_ner", "LabelledSentence", "save", "data_split", "load_preprocessed"]

LabelledSentence = List[Tuple[str, str]]
TokenList = List[List[str]]
TextData = (List[LabelledSentence], List[str])


def load_kaggle_ner(data_path: str = "../data/kaggle-ner/ner_dataset.csv") \
        -> TextData:
    df = pd.read_csv(data_path, encoding="latin1").fillna(method="ffill")
    grouped_sentences = _get_sentences(df)
    return [[(s[0], s[1]) for s in sentence] for sentence in grouped_sentences], df["Tag"].unique().tolist()


def _get_sentences(data: pd.DataFrame):
    return data.groupby("Sentence #").apply(
        lambda sentence: [(word, tag) for word, tag
                          in zip(sentence["Word"].values.tolist(),
                                 sentence["Tag"].values.tolist())]
    )


def data_split(data: TextData) -> (TokenList, TokenList, TokenList, TokenList):
    sentences = [[word for word, _ in sentence] for sentence in data[0]]
    tags = [[tag for _, tag in sentence] for sentence in data[0]]
    return train_test_split(sentences, tags)


def save(filepath: str, labeled_sentences: List[LabelledSentence] = None, sentences: TokenList = None,
         labels: TokenList = None):
    # argchecks
    if labeled_sentences:
        with open(filepath, "w") as file:
            writer = csv.writer(file)
            writer.writerow(["Sentence #", "Word", "Tag"])
            i = 0
            for sentence in labeled_sentences:
                header = f"Sentence: {i}"
                for word, token in sentence:
                    writer.writerow([header, word, token])
                    header = None
                i += 1

    if sentences:
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

# todo maybe read and save that data with pandas
def load_preprocessed(data_dir: str) -> (TextData, TextData):
    train_data = ([], [])
    test_data = ([], [])
    with open(data_dir + "train.csv", "r") as file:
        reader = csv.reader(file)
        for sentence, labels in reader:
            train_data[0].append(sentence)
            train_data[1].append(labels)

    with open(data_dir + "test.csv", "r") as file:
        reader = csv.reader(file)
        for sentence, labels in reader:
            test_data[0].append(sentence)
            test_data[1].append(labels)
    return train_data, test_data
