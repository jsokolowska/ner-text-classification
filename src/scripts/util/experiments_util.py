__all__ = ["CLASS_LABELS", "KnnParams", "RandomForestParams", "SVCParams", "LogisticRegressionParams",
           "STATE_LABELS", "DATASET_LABELS", "save_score_dfs", "get_plot_path", "plt_clear", "RESULTS_DIR"]

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.scripts.util.common import Dataset, State
import matplotlib.pyplot as plt

SEED = 12890

# labels
CLASS_LABELS = {
    Dataset.AG_NEWS: {1: 'World', 2: 'Sports', 3: 'Buisness', 4: 'Sci/Tech'},
    Dataset.BBC: {0: 'politics', 1: 'tech', 2: 'entertainment', 3: 'sport', 4: 'business'},
    Dataset.FINE_FOODS: {5: '5*', 4: '4*', 1: '1*', 3: '3*', 2: '2*'}
}
STATE_LABELS = {
    State.STD: "Standard",
    State.BIO: "BIO",
    State.DOUBLE: "Double"
}
DATASET_LABELS = {
    Dataset.AG_NEWS: "Ag's news",
    Dataset.IMDB: "IMDB",
    Dataset.FINE_FOODS: "Fine Foods",
    Dataset.DISASTERS: "Disasters",
    Dataset.BBC: "BBC"
}


# Classifier params from tuning
class ClassifierParams:
    def __init__(self, clf_class):
        self._clf_class = clf_class
        self.default_params = {}
        self.param_override = {}

    def get_params(self, dataset: Dataset, state: State) -> dict:
        params = self.default_params[dataset]
        if dataset in self.param_override and state in self.param_override[dataset]:
            for key, item in self.param_override[dataset][state].items():
                params[key] = item
        return params

    def get_classifier(self, dataset: Dataset, state: State):
        params = {'clf__' + k: v for k, v in self.get_params(dataset, state).items()}
        pipe = Pipeline([("std", StandardScaler()), ("clf", self._clf_class())])
        pipe.set_params(**params)
        return pipe

    def clf_name(self):
        pass


class KnnParams(ClassifierParams):
    def __init__(self):
        super().__init__(KNeighborsClassifier)
        self.default_params = {
            Dataset.AG_NEWS: {
                'n_neighbors': 200,
                'weights': 'distance',
                'p': 2
            },
            Dataset.FINE_FOODS: {
                'n_neighbors': 200,
                'weights': 'distance',
                'p': 1
            },
            Dataset.IMDB: {
                'n_neighbors': 150,
                'weights': 'distance',
                'p': 1
            },
            Dataset.BBC: {
                'n_neighbors': 200,
                'weights': 'distance',
                'p': 1
            },
            Dataset.DISASTERS: {
                'n_neighbors': 150,
                'weights': 'uniform',
                'p': 2
            }
        }
        self.param_override = {
            Dataset.AG_NEWS: {
                State.STD: {'p': 1}
            },
            Dataset.FINE_FOODS: {
                State.STD: {'n_neighbors': 150}
            },
            Dataset.IMDB: {
                State.DOUBLE: {'n_neighbors': 200}
            },
            Dataset.DISASTERS: {
                State.BIO: {'n_neighbors': 200, 'p': 1},
                State.DOUBLE: {'n_neighbors': 50, 'p': 2}
            }
        }

    def clf_name(self):
        return "knn"


class RandomForestParams(ClassifierParams):
    def __init__(self):
        super().__init__(RandomForestClassifier)
        self.default_params = {
            Dataset.AG_NEWS: {
                'n_estimators': 1000,
                'max_features': 'log2',
                'random_state': SEED
            },
            Dataset.FINE_FOODS: {
                'n_estimators': 1000,
                'max_features': 'log2',
                'random_state': SEED
            },
            Dataset.IMDB: {
                'n_estimators': 1000,
                'max_features': 'log2',
                'random_state': SEED
            },
            Dataset.BBC: {
                'n_estimators': 1000,
                'max_features': 'log2',
                'random_state': SEED
            },
            Dataset.DISASTERS: {'random_state': SEED}
        }
        self.param_override = {
            Dataset.DISASTERS: {
                State.STD: {'n_estimators': 100, 'max_features': 'log2'},
                State.BIO: {'n_estimators': 200, 'max_features': 0.01},
                State.DOUBLE: {'n_estimators': 50, 'max_features': 'sqrt'}
            },

        }

    def clf_name(self):
        return "rf"


class LogisticRegressionParams(ClassifierParams):
    def __init__(self):
        super().__init__(LogisticRegression)
        self.default_params = {
            Dataset.AG_NEWS: {'C': 0.01, 'random_state': SEED},
            Dataset.FINE_FOODS: {'C': 0.01, 'random_state': SEED},
            Dataset.IMDB: {'C': 0.01, 'random_state': SEED},
            Dataset.BBC: {'C': 0.01, 'random_state': SEED},
            Dataset.DISASTERS: {'C': 0.01, 'random_state': SEED}
        }

    def clf_name(self):
        return "log"


class SVCParams(ClassifierParams):
    def __init__(self):
        super().__init__(SVC)
        self.default_params = {
            Dataset.AG_NEWS: {
                'kernel': 'sigmoid',
                'gamma': 'scale',
                'C': 0.01,
                'random_state': SEED,
                'class_weight': 'balanced'
            },
            Dataset.FINE_FOODS: {
                'kernel': 'sigmoid',
                'gamma': 'scale',
                'C': 0.01,
                'random_state': SEED,
                'class_weight': 'balanced'
            },
            Dataset.IMDB: {
                'kernel': 'sigmoid',
                'gamma': 'scale',
                'C': 0.01,
                'random_state': SEED,
                'class_weight': 'balanced'
            },
            Dataset.BBC: {
                'kernel': 'sigmoid',
                'gamma': 0.01,
                'C': 0.01,
                'random_state': SEED,
                'class_weight': 'balanced'
            },
            Dataset.DISASTERS: {
                'kernel': 'sigmoid',
                'gamma': 'auto',
                'C': 0.1,
                'random_state': SEED,
                'class_weight': 'balanced'
            }

        }
        self.param_override = {
            Dataset.DISASTERS: {
                State.DOUBLE: {
                    'kernel': 'rbf',
                    'gamma': 'auto',
                    'C': 0.01,
                },
                State.STD: {
                    'kernel': 'rbf',
                    'gamma': 'scale',
                    'C': 0.1,
                }
            },
            Dataset.BBC: {
                State.STD: {'gamma': 'scale', 'C': 1, }
            },
            Dataset.IMDB: {
                State.STD: {'kernel': 'rbf'}
            },
            Dataset.FINE_FOODS: {
                State.STD: {'kernel': 'rbf'},
                State.DOUBLE: {'kernel': 'rbf'}
            }
        }

    def clf_name(self):
        return "svc"


# scores util
RESULTS_DIR = "/results\\"


def save_score_dfs(clf_name, roc_auc_df, avg_pr_df):
    roc_auc_df.to_csv(f"{RESULTS_DIR}/{clf_name}-roc_auc.csv", index=False)
    avg_pr_df.to_csv(f"{RESULTS_DIR}/{clf_name}_acg_pr.csv", index=False)


def get_plot_path(clf_name, dataset, state):
    return f"{RESULTS_DIR}/{clf_name}-{dataset.value}-{state.value}.png"


def plt_clear():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
