import pandas as pd

def df_to_latex (df: pd.DataFrame, label:str, caption):
    latex = "\\begin{table}[!h] \label{" +label+ "} \centering\n" \
            "\caption{" + caption + "}" \
            "\\begin{tabular}{"
    cols = df.columns.tolist()
    latex += '| c ' * len(cols)
    latex += '|}\hline \n'

    for _,row in df.iterrows():
        for c in cols:
            latex += str(row[c]) + "&"
        latex = latex[:-1]  #delete last &
        latex += "\\\\ \hline\n"

    latex += "\end{tabular}\n\end{table}\n"
    return latex

if __name__ == "__main__":
    df = pd.DataFrame(data={"bio": [1, 2, 3, 4], "std": [1, 3, 0, 1]}, index=["ag", "imdbb", "bbc", "disasters"])

    print(df_to_latex(df, "ref:label", "some smart caption"))

