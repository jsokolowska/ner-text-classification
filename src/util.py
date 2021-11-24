import pandas as pd
from typing import List, Tuple, Set
from sklearn.model_selection import train_test_split
import csv

__all__ = ["load_kaggle_ner", "LabelledSentence", "TokenList", "TextData", "save", "data_split", "load_preprocessed",
           "TextGetter"]

LabelledSentence = List[Tuple[str, str]]
TokenList = List[List[str]]
TextData = (List[LabelledSentence], List[str])


# todo refactor to use dataframes instead of this textgetter
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
    df2 = pd.DataFrame({"sentences": [[tupl[0] for tupl in sentence] for sentence in grouped_sentences],
                        "tokens": [[tupl[1] for tupl in sentence] for sentence in grouped_sentences]})
    return TextGetter(labelled_sentences=[[(str(s[0]), str(s[1])) for s in sentence] for sentence in grouped_sentences],
                      tags=df["Tag"].unique().tolist())


def kaggle_ner2csv_split(data_path: str = "../data/kaggle-ner/ner_dataset.csv"):
    df = pd.read_csv(data_path, encoding="latin1").fillna(method="ffill")
    grouped_sentences = _get_sentences(df)
    df2 = pd.DataFrame({"sentences": [[tupl[0] for tupl in sentence] for sentence in grouped_sentences],
                        "tokens": [[tupl[1] for tupl in sentence] for sentence in grouped_sentences]})
    data_split(df2)


def _get_sentences(data: pd.DataFrame):
    return data.groupby("Sentence #").apply(
        lambda sentence: [(word, tag) for word, tag
                          in zip(sentence["Word"].values.tolist(),
                                 sentence["Tag"].values.tolist())]
    )


def data_split(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    sentences = df["sentences"]
    tags = df["tokens"]
    target_dir = "../preprocessed_data/kaggle-ner/"
    train_s, test_dev_s, train_l, test_dev_l = train_test_split(sentences, tags, test_size=0.3)
    test_s, dev_s, test_l, dev_l = train_test_split(test_dev_s, test_dev_l, test_size=0.5)
    df_test = pd.DataFrame({"sentences": test_s, "tokens": test_l})
    df_test.to_csv(target_dir + "test.csv", index=False)
    df_train = pd.DataFrame({"sentences": train_s, "tokens": train_s})
    df_train.to_csv(target_dir + "train.csv", index=False)
    df_dev = pd.DataFrame({"sentences": dev_s, "tokens": dev_l})
    df_dev.to_csv(target_dir + "dev.csv", index=False)
    return df_train, df_dev, df_test


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


def conll2csv(data_dir: str, out_dir: str, datasets: [], dataset_extension="conll", strip_prefix: str = ""):
    for dataset in datasets:
        dataset_with_extension = dataset
        if dataset_extension is not None:
            dataset_with_extension += "." + dataset_extension
        with open(data_dir + dataset_with_extension) as f:
            data = f.read()
        elems = [elem.split("\t") for elem in data.split("\n")]

        sentence = []
        tokens = []
        df_dict = {"sentences": [], "tokens": []}
        for elem in elems:
            if 2 > len(elem) > 0 and (len(elem[0]) == 0 or elem[0].isspace()):
                if len(sentence):
                    df_dict["tokens"].append(tokens)
                    df_dict["sentences"].append(sentence)
                sentence = []
                tokens = []
            elif len(elem) == 2:
                word = elem[0]
                if len(strip_prefix) > 0:
                    if word.startswith(strip_prefix):
                        word = word[3:]
                    else:
                        raise TypeError(f"word {word} does not start with expected prefix \"{strip_prefix}\"")
                sentence.append(word)
                tokens.append(elem[1])
            else:
                print(f"Unexpected elem length {elem} for sentence {sentence} in corpus {dataset_with_extension}, "
                      f"skipping")
        df = pd.DataFrame(df_dict)
        df.to_csv(out_dir + dataset + ".csv", index=False)


#conll2csv("../data/panx_dataset/en/", "../preprocessed_data/panx_dataset/en/", ["test", "train", "dev"], None, "en:")
conll2csv("../data/broad-twitter-corpus/", "../preprocessed_data/btc/", ["a", "b", "e", "f", "g", "h"], "conll")
#kaggle_ner2csv_split()
