from src.representations import SpacyNEClassifier


def test_result_format():
    ner = SpacyNEClassifier()
    sentences = ["Important Person went to the White House", "aaaa"]
    res = ner.predict(sentences)
    for lst in res["tags"]:
        all(
            item == "O" or item.startswith("B-") or item.startswith("I-")
            for item in lst
        )
