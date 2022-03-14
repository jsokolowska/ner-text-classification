import math

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

from src.scripts.common import *

SEED = 1289
TEST_PERCENT = 0.2


def load_and_split_datasets():
    # ----- AG NEWS ------
    print(f"Start loading {Dataset.AG_NEWS.value}")
    load_and_split_ag_news()
    # ----- IMDB ---------
    print(f"Start loading {Dataset.IMDB.value}")
    load_and_split_imdb()
    # ----- BBC ----------
    print(f"Start loading {Dataset.BBC.value}")
    load_and_split_bbc()
    # ----- Disasters ----
    print(f"Start loading {Dataset.DISASTERS.value}")
    load_and_split_disasters()
    # ----- Fine Foods ---
    print(f"Start loading {Dataset.FINE_FOODS.value}")
    load_and_split_fine_foods()


def load_and_split_ag_news():
    base_ag_dir = DATA_DIR + Dataset.AG_NEWS.value + "\\"
    ensure_dirs_for_representations(base_ag_dir)
    df_train = pd.read_csv(base_ag_dir + "train.csv")
    df_test = pd.read_csv(base_ag_dir + "test.csv")

    class_size_test = 700
    class_size_train = math.ceil(class_size_test / TEST_PERCENT)

    df_train_sampled = (
        df_train.groupby("Class Index")
        .apply(lambda x: x.sample(n=class_size_train))
        .reset_index(drop=True)
    )
    df_test_sampled = (
        df_test.groupby("Class Index")
        .apply(lambda x: x.sample(n=class_size_test))
        .reset_index(drop=True)
    )

    def helper(row):
        return row["Title"] + " \n" + row["Description"]

    df_test_sampled[TEXT_COL] = df_test_sampled.apply(lambda row: helper(row), axis=1)
    df_train_sampled[TEXT_COL] = df_train_sampled.apply(
        lambda row: row["Title"] + " \n" + row["Description"], axis=1
    )

    df_train_sampled = df_train_sampled.rename(columns={"Class Index": TARGET_COL})
    df_test_sampled = df_test_sampled.rename(columns={"Class Index": TARGET_COL})

    df_test_sampled[[TARGET_COL, TEXT_COL]].to_csv(
        base_ag_dir + State.RAW.value + "\\test.csv", index=False
    )
    df_train_sampled[[TARGET_COL, TEXT_COL]].to_csv(
        base_ag_dir + State.RAW.value + "\\train.csv", index=False
    )


def load_and_split_imdb():
    base_dir = DATA_DIR + Dataset.IMDB.value + "\\"
    ensure_dirs_for_representations(base_dir)
    df = pd.read_csv(base_dir + "IMDB Dataset.csv", nrows=10000)
    X = df["review"]
    y = df["sentiment"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_PERCENT, random_state=SEED
    )

    df_train = pd.concat([x_train, y_train], axis=1).rename(
        columns={"review": TEXT_COL, "sentiment": TARGET_COL}
    )
    df_test = pd.concat([x_test, y_test], axis=1).rename(
        columns={"review": TEXT_COL, "sentiment": TARGET_COL}
    )

    df_train.to_csv(base_dir + State.RAW.value + "\\train.csv", index=False)
    df_test.to_csv(base_dir + State.RAW.value + "\\test.csv", index=False)


def load_and_split_bbc():
    base_dir = DATA_DIR + Dataset.BBC.value + "\\"
    ensure_dirs_for_representations(base_dir)
    df = pd.read_csv(base_dir + "raw.csv")
    X = df["raw_text"]
    y = df["class"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_PERCENT, random_state=SEED
    )

    df_train = pd.concat([x_train, y_train], axis=1).rename(
        columns={"raw_text": TEXT_COL, "class": TARGET_COL}
    )
    df_test = pd.concat([x_test, y_test], axis=1).rename(
        columns={"raw_text": TEXT_COL, "class": TARGET_COL}
    )

    df_train[[TARGET_COL, TEXT_COL]].to_csv(
        base_dir + State.RAW.value + "\\train.csv", index=False
    )
    df_test[[TARGET_COL, TEXT_COL]].to_csv(
        base_dir + State.RAW.value + "\\test.csv", index=False
    )


def load_and_split_disasters():
    base_dir = DATA_DIR + Dataset.DISASTERS.value + "\\"
    ensure_dirs_for_representations(base_dir)
    df = pd.read_csv(
        base_dir + "socialmedia-disaster-tweets-DFE.csv", encoding="iso-8859-1"
    )
    X = df["text"]
    y = df["choose_one"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_PERCENT, random_state=SEED
    )

    df_train = pd.concat([x_train, y_train], axis=1).rename(
        columns={"text": TEXT_COL, "choose_one": TARGET_COL}
    )
    df_test = pd.concat([x_test, y_test], axis=1).rename(
        columns={"text": TEXT_COL, "choose_one": TARGET_COL}
    )

    df_train[[TARGET_COL, TEXT_COL]].to_csv(
        base_dir + State.RAW.value + "\\train.csv", index=False
    )
    df_test[[TARGET_COL, TEXT_COL]].to_csv(
        base_dir + State.RAW.value + "\\test.csv", index=False
    )


def load_and_split_fine_foods():
    base_dir = DATA_DIR + Dataset.FINE_FOODS.value + "\\"
    ensure_dirs_for_representations(base_dir)
    df = pd.read_csv(base_dir + "Reviews.csv", nrows=10000)

    def combine(row):
        if type(row["Summary"]) != str and row["Summary"].isnan():
            return row["Text"]
        return row["Summary"] + "\n " + row["Text"]

    X = df[["Summary", "Text"]].apply(lambda row: combine(row), axis=1)
    y = df["Score"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_PERCENT, random_state=SEED
    )

    df_train = pd.concat([x_train, y_train], axis=1).rename(
        columns={0: TEXT_COL, "Score": TARGET_COL}
    )
    df_test = pd.concat([x_test, y_test], axis=1).rename(
        columns={0: TEXT_COL, "Score": TARGET_COL}
    )

    df_train[[TARGET_COL, TEXT_COL]].to_csv(
        base_dir + State.RAW.value + "\\train.csv", index=False
    )
    df_test[[TARGET_COL, TEXT_COL]].to_csv(
        base_dir + State.RAW.value + "\\test.csv", index=False
    )


def ensure_dirs_for_representations(dataset_dir):
    for r in State:
        if not os.path.exists(dataset_dir + r.value):
            os.makedirs(dataset_dir + r.value)


if __name__ == "__main__":
    np.random.seed(SEED)
    load_and_split_datasets()
