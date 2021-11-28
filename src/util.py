import pandas as pd
from typing import List, Tuple, Set
from sklearn.model_selection import train_test_split
import csv

__all__ = ["load_preprocessed_ner_data", "LabelledSentence", "TokenList", "TextData", "save_ner", "data_split",
           "load_preprocessed_ner", "TextGetter", "load_tagged_classification", "save_tagged_classification"]

LabelledSentence = List[Tuple[str, str]]
TokenList = List[List[str]]
TextData = (List[LabelledSentence], List[str])


# todo refactor to use dataframes instead of this textgetter ?s
class TextGetter:
    def __init__(self, labelled_sentences: List[LabelledSentence] = None, sentences: TokenList = None,
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


def load_preprocessed_ner_data(data_path: str = "../data/kaggle-ner/ner_dataset.csv") \
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


def data_split(data: TextGetter, test_size) -> (TextGetter, TextGetter):
    sentences = data.sentences
    tags = data.bio_tags
    train_s, test_s, train_l, test_l = train_test_split(sentences, tags, test_size=test_size)
    return TextGetter(sentences=train_s, bio_tags=train_l), TextGetter(sentences=test_s, bio_tags=test_l)


def save_ner(filepath: str, labelled_sentences: List[LabelledSentence] = None, sentences: TokenList = None,
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


def load_preprocessed_ner(data_dir: str) -> (TextGetter, TextGetter):
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

        df_dict = {"Sentence #": [], "Word": [], "Tag": []}
        idx = 0
        for elem in elems:
            if 2 > len(elem) > 0 and (len(elem[0]) == 0 or elem[0].isspace()):
                idx += 1
            elif len(elem) == 2:
                word = elem[0]
                if len(strip_prefix) > 0:
                    if word.startswith(strip_prefix):
                        word = word[3:]
                    else:
                        raise TypeError(f"word {word} does not start with expected prefix \"{strip_prefix}\"")
                df_dict["Word"].append(word)
                df_dict["Tag"].append(elem[1])
                df_dict["Sentence #"].append("Sentence " + str(idx))
            else:
                print(f"Unexpected elem length {elem} in corpus {dataset_with_extension}, "
                      f"skipping")
        df = pd.DataFrame(df_dict)
        df.to_csv(out_dir + dataset + ".csv", index=False)


def load_tagged_classification(filepath:str) -> pd.DataFrame:
    df = pd.read_csv(filepath, index_col=0)
    return _group(df)


def _group(df):
    grouped = df.groupby("Sentence #").apply(
        lambda sentence: sentence["Token"].values.tolist())
    df_grouped = pd.DataFrame(grouped, columns=["Tokens"])
    df_grouped["Tags"] =  df.groupby("Sentence #").apply(
        lambda sentence: sentence["Tag"].values.tolist())
    df_grouped["Class"] =  df.groupby("Sentence #").apply(
        _check_and_group_class)
    return df_grouped


def _check_and_group_class(sentence_data):
    all_classes = sentence_data["Class"].unique()
    if len(all_classes) != 1:
        raise ValueError("Expected only one unique class value per sentence")
    return all_classes[0]


def save_tagged_classification(df,filepath):
    df_degrouped = pd.DataFrame(columns=["Sentence #", "Tokens","Tags", "Class"])
    for idx,row in  df.iterrows():
        df_temp = pd.DataFrame({"Token":row["Tokens"], "Tag":row["Tags"],"Class": row["Class"], "Sentence #" : idx})
        df_degrouped = df_degrouped.append(df_temp)
    df_degrouped.to_csv(filepath)


def load_and_split_all_data():
    conll2csv("../data/panx_dataset/en/", "../preprocessed_data/panx_dataset/en/", ["train", "test", "dev"], None, "en:")
    conll2csv("../data/broad-twitter-corpus/", "../preprocessed_data/btc/", ["a", "b", "e", "f", "g", "h"], "conll")

    data = load_preprocessed_ner_data("../data/kaggle-ner/ner_dataset.csv")
    res_train, res_test_dev = data_split(data, test_size=0.3)
    res_dev, res_test = data_split(res_test_dev, test_size=0.5)
    save_ner("../preprocessed_data/kaggle-ner/test.csv", text_getter=res_test)
    save_ner("../preprocessed_data/kaggle-ner/dev.csv", text_getter=res_dev)
    save_ner("../preprocessed_data/kaggle-ner/train.csv", text_getter=res_train)



if __name__ == "__main__":
    # load_and_split_all_data()
    loaded = load_preprocessed_ner_data("../preprocessed_data/panx_dataset/en/train.csv")
    a = 2
