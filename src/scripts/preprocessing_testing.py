from src.representations import SpacyNEClassifier, DoubleTfIdfVectorizer
from common import *
import pandas as pd
from datetime import datetime
from src.representations.preprocessing import *
from tqdm import tqdm


def load_train(data: Dataset):
    return pd.read_csv(DATA_DIR + data.value + "\\" + State.RAW.value + "\\train.csv", nrows=400)


def time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return current_time


if __name__ == "__main__":
    print(f"Start time: {time()}")
    for d in tqdm(Dataset):
        train = load_train(d)
        ner = SpacyNEClassifier()
        vectorizer = DoubleTfIdfVectorizer(
            ner_clf=ner,
            preprocessor=text_preprocessing,
            token_filter=token_filter,
            min_df=0,
            max_df=1.0,
        )

        res = vectorizer.fit_transform(
            train['text']
        )

        col = res.columns.values
        print(res.columns)
        a = 2


