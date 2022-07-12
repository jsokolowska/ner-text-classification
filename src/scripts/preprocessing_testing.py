from src.representations import DoubleTfIdfVectorizer, SpacyNEClassifier, BioTfIdfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
if __name__ == "__main__":
    t3 = "He has worked in Mexico and Uruguay."
    t4 = "He is the UK Independence Party's weapon"
    t2 = "Rick Morris is a winner."
    t1 = "UKIP's secret weapon?"
    texts = [t2, t4]
    docs = [
        ["Rick", "Morris", "attends", "every", "party", "."],
        ["He", "is", "the", "UK", "Independence", "Party", "'s", "weapon", "."]
    ]
    tags = [
        ["B-PERSON", "I-PERSON", "O", "O", "O", "O"],
        ["O", "O", "O", "B-GPE", "B-NORP", "I-NORP", "I-NORP", "O", "O"]
    ]
    ner = SpacyNEClassifier()
    vect = DoubleTfIdfVectorizer(ner,
                              filter_stopwords=False,
                              filter_punctuation=True,
                              lemmatize=False,
                              normalize=False,
                              fix_contractions=False)

    df = pd.DataFrame(vect.fit_transform(tokenized=pd.Series(docs), bio_tags=pd.Series(tags)).toarray(),
                      columns=vect.get_feature_names())


    def df_to_latex(df: pd.DataFrame, label: str, caption):
        latex = "\\begin{table}[!h] \label{" + label + "} \centering\n" \
                                                       "\caption{" + caption + "}" \
                                                                               "\\begin{tabular}{"
        headers = ['Dokument 1', 'Dokument 2']
        latex += '| c ' * (len(headers) + 1)
        latex += '|}\hline \n'
        latex += "Tokeny & " + " & ".join(headers) + "\\\\ \hline \n"

        cols = [c for c in df.columns]
        cols.sort()
        for col in cols:
            col_latex_safe = col.replace("_", "\\textunderscore ")
            latex += col_latex_safe + "&"
            for h_idx in range(len(headers)):
                latex += "{0:.3f}".format(df.iloc[h_idx].loc[col]) + " & "
            latex = latex[:-2]  # delete last &
            latex += "\\\\ \hline\n"

        latex += "\end{tabular}\n\end{table}\n"
        return latex

    print(df_to_latex(df, "tab:tfidf-double", "Wartości TF-IDF dla reprezentacji Double"))

    vect = BioTfIdfVectorizer(ner,
                                 filter_stopwords=False,
                                 filter_punctuation=True,
                                 lemmatize=False,
                                 normalize=False,
                                 fix_contractions=False)

    df = pd.DataFrame(vect.fit_transform(tokenized=pd.Series(docs), bio_tags=pd.Series(tags)).toarray(),
                      columns=vect.get_feature_names())
    print(df_to_latex(df, "tab:tfidf-bio", "Wartości TF-IDF dla reprezentacji BIO"))

    tfidf = TfidfVectorizer(
        analyzer='word',
        tokenizer=lambda x: x,
        preprocessor=vect.build_preprocessor_and_tokenizer(),
        min_df=1, max_df=1.0,
        token_pattern=None)
    docs_r = [" ".join(d) for d in docs

    ]

    df = pd.DataFrame(tfidf.fit_transform(docs_r).toarray(), columns =tfidf.get_feature_names())
    print(df_to_latex(df, "tab:tfidf-std", "Wartości TF-IDF bez modyfikacji"))
