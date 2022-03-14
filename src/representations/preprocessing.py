import re
import html

__all__ = ["text_preprocessing", "token_filter"]

MISSING_AMP_GENERAL = re.compile(r"(\s)(#[0-9]+;)")
MISSING_AMP = re.compile(r"\samp;")
MISSING_AMP_QUOT = re.compile(r"\squot;")
HTML_TAGS = re.compile(r"<.*?>")
HASHTAG = re.compile(r"#[^\s]+")
WORD_IN_HASH = re.compile("(?:[A-Z][A-Z]+)|(?:[a-z][a-z]+)|(?:[0-9]+)|(?:[A-Z][a-z]+)")
EMOTES = [
    re.compile("^" + e + "$")
    for e in [":\(", ":\)", ":-\)", ":p", ";\)", "<3", "\) :", "\):"]
]
URL = re.compile("^http[^\s]+$")
USER_MENTION = re.compile("^@[^\s]+$")
NUMBER_WITH_SEPARATOR = re.compile("^[0-9]+(,|.)[0-9]+$")


def add_missing_amps(text: str) -> str:
    text = re.sub(MISSING_AMP_GENERAL, r"&\2", text)
    text = re.sub(MISSING_AMP, "&amp;", text)
    text = re.sub(MISSING_AMP_QUOT, "&quot;", text)
    return text


def clean_html_tags(text: str) -> str:
    clean_text = html.unescape(text)
    return re.sub(HTML_TAGS, "", clean_text)


def split_hashtags(text: str) -> str:
    def hashtag_splitter(hashtag: str) -> [str]:
        hashtag = hashtag[1:]  # ditch starting #
        if hashtag.isupper() or hashtag.islower():
            return [hashtag]
        # split by words
        words = re.findall(WORD_IN_HASH, hashtag)
        return words

    match = re.search(HASHTAG, text)
    while match:
        start = match.span()[0]
        end = match.span()[1]
        split = hashtag_splitter(text[start:end])
        text = text[0:start] + " ".join(split) + text[end:]
        match = re.search(HASHTAG, text)
    return text


def remove_nonascii(text: str) -> str:
    text = text.encode("ascii", errors="ignore")
    return text.decode()


def replace_token(token: str) -> str:
    for emote in EMOTES:
        token = re.sub(emote, "<EMOTE>", token)
    token = re.sub(URL, "<URL>", token)
    token = re.sub(USER_MENTION, "<USER>", token)
    if token.isnumeric():
        token = "<NUMBER>"
    token = re.sub(NUMBER_WITH_SEPARATOR, "<NUMBER>", token)
    if not token.isalnum() and token not in ["<EMOTE>", "<URL>", "<USER>", "<NUMBER>"]:
        return None
    return token


def token_filter(token_lst):
    result = []
    for token, tag in token_lst:
        temp = replace_token(token)
        if temp:
            result.append((temp, tag))
    return result


def text_preprocessing(text: str) -> str:
    # normalize whitespaces
    text = re.sub("\s+", " ", text)
    # ag news has some ampersands missing
    text = add_missing_amps(text)
    # escape html codes and remove html tags
    text = clean_html_tags(text)
    # split existing hashtags
    text = split_hashtags(text)
    # remove any remaining non-ascii characters
    text = remove_nonascii(text)
    return text
