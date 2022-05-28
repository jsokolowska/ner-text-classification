import re

from tqdm import tqdm
from yellowbrick.classifier import ROCAUC
import matplotlib.pyplot as plt
from src.scripts.util.common import *
from src.scripts.util.experiments_util import *

AUC_SCORE_RE = re.compile('AUC = [0-9.]+')
DIAGRAMS_DIR = "C:\\Users\\Asia\\Documents\\Projekty\\PyCharm Projects\\text-classification\\results\\diagrams\\"


def draw_plots(X_train, X_test, y_train, y_test, model, state):
    fig, ax = plt.subplots()
    visualizer = ROCAUC(model, ax=ax, per_class=False, micro=False)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.finalize()
    # get data from plotted macro curve
    lines = ax.get_lines()
    micro_line = lines[0]
    legend = ax.get_legend()
    auc = AUC_SCORE_RE.search(legend.get_texts()[0].get_text()).group(0)
    label = f"Reprezentacja {STATE_LABELS[state]} ({auc})"
    return micro_line.get_xdata(), micro_line.get_ydata(), label


states = [State.STD, State.BIO, State.DOUBLE]
colors = ["red", "green", "yellow"]
reference_color = "blue"

if __name__ == "__main__":
    for d in Dataset:
        for params in [LogisticRegressionParams(), KnnParams(), RandomForestParams()]:
            lines = []
            for s in tqdm([state for state in State if state != State.RAW]):
                X_train, y_train, X_test, y_test = get_train_test(d, s)
                model = params.get_classifier(d, s)
                line = draw_plots(X_train, X_test, y_train, y_test, model, s)
                lines.append(line)
            fig, ax = plt.subplots()
            for idx in range(len(lines)):
                ln = lines[idx]
                ax.plot(ln[0], ln[1], linestyle="solid", label=ln[2], color=colors[idx])
            ax.plot([0, 1], [0, 1], linestyle="dotted", color=reference_color)
            ax.set_xlabel("Swoistość")
            ax.set_ylabel("Czułość")
            ax.set_title(f"Krzywe ROC dla klasyfikatora {params.clf_name()} i zbioru {DATASET_LABELS[d]}")
            plt.savefig(f"{DIAGRAMS_DIR}{d}-{params.clf_name()}.png")
