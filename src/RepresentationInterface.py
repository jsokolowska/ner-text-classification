from typing import Dict, List


class RepresentationInterface:
    def fit(self, raw_documents: List[List[str]]):
        pass

    def transform(self, raw_documents):     # returns array-like
        pass

    def fit_transform(self, raw_documents):     # returns array-like
        pass

    def get_feature_names_out(self):    # returns array of strings
        pass

    def get_params(self, deep: bool) -> Dict[str, str]:
        pass

    def get_named_entities(self):   # returns array of tuples? or maybe json-like representation of named entities as in spacy
        pass

    pass
