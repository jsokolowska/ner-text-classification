from src.representations import DoubleTfIdfVectorizer, SpacyNEClassifier

if __name__ == "__main__":
    ner = SpacyNEClassifier()
    vectorizer = DoubleTfIdfVectorizer(ner_clf=ner)
    # todo understand where nans for only sent 1 come in
    res = vectorizer.preprocessing_only(["this has a contraction - no it doesn't!"])
    print(res)
    counts = vectorizer.get_time_counts()
    start = counts[0][1]
    for l, c in counts:
        print(f"[{c - start}] {l}")

    # todo MASTER TODOLIST
    # todo add token filtering - stopwords, lemmatization etc. -debug _lemmatize
    # todo apply appropriate grouping and degrouping of dataframes -done
    # todo unit test implementation against sklearn tf-idf
    # todo validate implementation of different expected tokens
