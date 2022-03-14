# todo maybe implement here code for loading and tagging classification datasets
from src.representations import SpacyNEClassifier, DoubleTfIdfVectorizer



raw_dir = 'C:\\Users\\Asia\\Documents\\Projekty\\PyCharm Projects\\text-classification\\data\\'
raw_data = ['bbc\\raw.csv', ]
ner = SpacyNEClassifier()
vectorizer = DoubleTfIdfVectorizer(ner_clf=ner)
