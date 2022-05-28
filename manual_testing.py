import pandas as pd
from tqdm import tqdm

from src.representations.preprocessing import *
from src.scripts.util.common import *

from src.representations import (
    DoubleTfIdfVectorizer,
    SpacyNEClassifier,
    BioTfIdfVectorizer,
)


def load_train_test(dataset, state):
    base_dir = DATA_DIR + dataset.value + "/" + state.value + "/"
    train = pd.read_csv(base_dir + "train.csv",nrows=40)
    test = pd.read_csv(base_dir + "test.csv", nrows=20)
    return train, test


def save_df(df, dataset, state, name):
    df_path = DATA_DIR + dataset.value + "/" + state.value + "/" + name + ".csv"
    df.to_csv(df_path, index=False)


if __name__ == "__main__":

    for data in tqdm(Dataset):
        df_train, df_test = load_train_test(data, State.RAW)
        ner = SpacyNEClassifier()
        #
        # # -- bio --
        vectorizer = BioTfIdfVectorizer(
            ner_clf=ner,
            max_df=0.95,
            min_df=5,
            preprocessor=text_preprocessing,
            token_filter=token_filter)

        tag_train = vectorizer.tag_only(df_train["text"])
        res_train = vectorizer.fit_transform(
            tokenized=tag_train.groupby("sentence #").apply(
                lambda sent: [w for w in sent["tokens"].values.tolist()]
            ),
            bio_tags=tag_train.groupby("sentence #").apply(
                lambda sent: [t for t in sent["tags"].values.tolist()]
            ),
        )
        save_df(res_train, data, State.BIO, "train")
        #
        tag_test = vectorizer.tag_only(df_test["text"])
        res_test = vectorizer.transform(
            tokenized=tag_test.groupby("sentence #").apply(
                lambda sent: [w for w in sent["tokens"].values.tolist()]
            ),
            bio_tags=tag_test.groupby("sentence #").apply(
                lambda sent: [t for t in sent["tags"].values.tolist()]
            ),
        )
        save_df(res_test, data, State.BIO, "test")

        # -- double --
        vectorizer = DoubleTfIdfVectorizer(
            ner_clf=ner,
            max_df=1.0,
            min_df=1,
            preprocessor=text_preprocessing,
            token_filter=token_filter)

        tag_train = vectorizer.tag_only(df_train["text"])
        res_train = vectorizer.fit_transform(
            tokenized=tag_train.groupby("sentence #").apply(
                lambda sent: [w for w in sent["tokens"].values.tolist()]
            ),
            bio_tags=tag_train.groupby("sentence #").apply(
                lambda sent: [t for t in sent["tags"].values.tolist()]
            ),
        )
        save_df(res_train, data, State.DOUBLE, "train")

        tag_test = vectorizer.tag_only(df_test["text"])
        res_test = vectorizer.transform(
            tokenized=tag_test.groupby("sentence #").apply(
                lambda sent: [w for w in sent["tokens"].values.tolist()]
            ),
            bio_tags=tag_test.groupby("sentence #").apply(
                lambda sent: [t for t in sent["tags"].values.tolist()]
            ),
        )
        save_df(res_test, data, State.DOUBLE, "test")
