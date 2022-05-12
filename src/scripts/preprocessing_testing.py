from common import *
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from scipy.sparse import save_npz


def load(data: Dataset, state: State, name):
    return pd.read_csv(DATA_DIR + data.value + "\\" + state.value + "\\" + name + ".csv")


def save_df(df, dataset, state, name):
    df_path = DATA_DIR + dataset.value + "/" + state.value + "/" + name + ".csv"
    df.to_csv(df_path, index=False)


def time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return current_time

if __name__ == "__main__":


     # ag news bio train & test
     # ag news std train & test (diff feature sizes?)
     # all ag news smaller than raw?

     # disasters - std sus
     #bbc - std sus
     # fine foods std sus
     # todo redo bio vectorizer keys :/

     #todo
     # - redo bio vectorizer keys
     # - look and test fit, fit transform and transform methods
     # - Test wtf with std (???)

    print(f"Start time: {time()}")
    for d in tqdm(Dataset):
        for state in tqdm(State):
            df = load(d, state, "train")
            print(f"[{time()}] {d.value} - {state.value} train: {df.shape}")

            df = load(d, state, "test")
            print(f"[{time()}] {d.value} - {state.value} test: {df.shape}")

