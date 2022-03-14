from enum import Enum

__all__ = ["Dataset", "DATA_DIR", "State", "TEXT_COL", "TARGET_COL"]


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
TARGET_COL = "target"

DATA_DIR = "C:\\Users\\Asia\\Documents\\Projekty\\PyCharm Projects\\text-classification\\data\\"
