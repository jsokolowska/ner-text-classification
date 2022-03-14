import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.representations.preprocessing import *

from src.representations import (
    DoubleTfIdfVectorizer,
    SpacyNEClassifier,
    MockNoNamedEntityClassifier,
)

if __name__ == "__main__":
    text = "@bbcmtd Wholesale Markets 12 ablaze https://url.url"
    documents = [
        text,
        "Something new set our country ablaze yet again",
        "Let's figure this out tomorrow",
    ]
    ner = MockNoNamedEntityClassifier()
    vect = DoubleTfIdfVectorizer(
        ner_clf=ner,
        max_df=1.0,
        min_df=1,
        preprocessor=text_preprocessing,
        token_filter=token_filter,
    )
    res = vect.fit_transform(documents)


    def sum_them(row):
        sum = 0
        max_idx = len(row)
        for idx in range(0, max_idx):
            sum += row[idx] ** 2
        return sum


    res["sum"] = res.apply(lambda row: sum_them(row), axis=1)
    print(res)
    print("----------")
    print(res.columns.values)
    print("----------")
    tfidf = TfidfVectorizer(
        analyzer="word",
        tokenizer=vect.build_preprocessor_and_tokenizer(),
        token_pattern=None,
    )
    train_transformed = pd.DataFrame(
        tfidf.fit_transform(documents).toarray(),
        columns=tfidf.get_feature_names(),
    )

    print(train_transformed)
    print("--------")
    print(train_transformed.columns.values)

    # todo MASTER TODOLIST
    # [DONE] todo add token filtering - stopwords, lemmatization etc. -debug _lemmatize
    # [DONE] todo apply appropriate grouping and degrouping of dataframes -done
    # [NO NEED] todo unit test implementation against sklearn tf-idf
    # [DONE] todo validate implementation of different expected params
    # [DONE] todo -investigate suspicious tokens
    # [DONE] todo replace numeric tokens with number
    # [DONE] todo find out why we have whitespace tokens in ag news
    # [DONE] todo investigate nan tokens in bbc '\x01' and '\x10own' and in imdb
    # [DONE] todo special tags removed? f.e. no user tags in disasters - and they should be there

    # todo validate fit followed by transform [Wtorek rano]
    # todo create and run representation scripts for all representations - with asserts for expected dimensions [Środa]
    # todo validate script output against previous outputs [Czwartek]
    # todo check multiple parameters combos [Future]
