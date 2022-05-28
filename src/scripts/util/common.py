from enum import Enum
import pandas as pd
from scipy.sparse import save_npz, load_npz
import os

__all__ = ["Dataset", "DATA_DIR", "State", "TEXT_COL", "TARGET_COL", "save_as_npz", "validate_or_save_columns",
           "read_as_dataframe", "get_train_test", "load_test", "df_to_latex", "load_raw"]

from sklearn.utils import shuffle


class Dataset(Enum):
    AG_NEWS = "ag-news"
    DISASTERS = "disasters"
    BBC = "bbc"
    FINE_FOODS = "fine-foods"
    IMDB = "imdb"


class State(Enum):
    BIO = "bio"
    DOUBLE = "double"
    STD = "std"
    RAW = "raw"


TEXT_COL = "text"
TARGET_COL = "TARGET"
DATA_DIR = "C:\\Users\\Asia\\Documents\\Projekty\\PyCharm Projects\\text-classification\\data\\"
SEED = 19178


def save_as_npz(dataset: Dataset, state: State, name, vectorizer, df_raw: pd.DataFrame, sp_array):
    save_npz(DATA_DIR + dataset.value + "/" + state.value + "/" + "array-" + name + ".npz", sp_array)
    df_raw["target"].to_csv(DATA_DIR + dataset.value + "/" + state.value + "/" + "target-" + name + ".csv",
                            index=False)
    validate_or_save_columns(dataset, state, vectorizer)


def validate_or_save_columns(dataset: Dataset, state: State, vectorizer):
    name = DATA_DIR + dataset.value + "/" + state.value + "/columns.csv"
    col_df = pd.DataFrame({"columns": vectorizer.get_feature_names()})
    if os.path.exists(name):
        cols = pd.read_csv(name)
        assert cols.shape == col_df.shape
    else:
        col_df.to_csv(name, index=False)


def read_as_dataframe(dataset: Dataset, state: State, name):
    data_dir = DATA_DIR + dataset.value + "/" + state.value + "/"
    sp_array = load_npz(data_dir + "array-" + name + ".npz")
    cols = pd.read_csv(data_dir + "columns.csv")
    target = pd.read_csv(data_dir + "target-" + name + ".csv")

    # sanity check
    assert len(cols) == sp_array.shape[1]
    assert len(target) == sp_array.shape[0]
    df = pd.DataFrame(sp_array.toarray(), columns=cols['columns'])
    df['TARGET'] = target['target']
    return df


def load_raw(data: Dataset, name):
    return pd.read_csv(DATA_DIR + data.value + "/raw/" + name + ".csv")


def get_train_test(dataset, state):
    df_train = read_as_dataframe(dataset, state, "train")
    df_train = shuffle(df_train, random_state=SEED)
    if dataset == Dataset.DISASTERS:
        df_train = df_train.drop(df_train[df_train[TARGET_COL] == "Can't Decide"].index)

    X_train = df_train.drop(TARGET_COL, axis=1)
    y_train = df_train[TARGET_COL]

    df_test = read_as_dataframe(dataset, state, "test")
    df_test = shuffle(df_test, random_state=SEED)
    if dataset == Dataset.DISASTERS:
        df_test = df_test.drop(df_test[df_test[TARGET_COL] == "Can't Decide"].index)
    X_test = df_test.drop(TARGET_COL, axis=1)
    y_test = df_test[TARGET_COL]
    return X_train, y_train, X_test, y_test


def load_test(dataset: Dataset, state: State) -> pd.DataFrame:
    df_test = read_as_dataframe(dataset, state, "test")
    df_test = shuffle(df_test, random_state=SEED)
    if dataset == Dataset.DISASTERS:
        df_test = df_test.drop(df_test[df_test[TARGET_COL] == "Can't Decide"].index)
    X_test = df_test.drop(TARGET_COL, axis=1)
    y_test = df_test[TARGET_COL]
    return X_test, y_test

def df_to_latex (df: pd.DataFrame, label:str, caption):
    latex = "\\begin{table}[!h] \label{" +label+ "} \centering\n" \
            "\caption{" + caption + "}" \
            "\\begin{tabular}{"
    cols = df.columns.tolist()
    latex += '| c ' * (len(cols) + 1)
    latex += '|}\hline \n'
    latex += "Zbiór danych & " + " & ".join(cols)  + "\\\\ \hline \n"

    for index,row in df.iterrows():
        latex += str(index) + " & "
        for c in cols:
            latex +=  "{0:.2f}".format(row[c]) + " & "
        latex = latex[:-2]  #delete last &
        latex += "\\\\ \hline\n"

    latex += "\end{tabular}\n\end{table}\n"
    return latex


