import ner_vectorizer


class BioTfIdfVectorizer(RepresentationInterface.NamedEntityVectorizer):
    """Tf-idf based vectorization in which words as well as their BIO tags
    are considered to be terms. May need excluding 'O' tag in order to eliminate redundant information.
    Gives information about frequency of term occurences and also of named entities occurences and types of
    named entities
    """
    pass
