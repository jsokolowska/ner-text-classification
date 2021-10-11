import RepresentationInterface


class BioTfIdfVectorizer(RepresentationInterface.RepresentationInterface):
    """Biorący pod uwagę rodzaje nazw własnych i liczący oddzielnie wartości tf-idf
    dla słów i dla tagów - tj. traktujący zarówno słowa jak i tagi BIO jako termy,
    dla których wyliczana jest wartość tf-idf. Być może będzie to wymagało wykluczenia
    z takiej reprezentacji tagów 'O', ze względu na ich częstość występowania,
    ale to prawdopodobnie należy zweryfikować w praktyce."""
    pass
