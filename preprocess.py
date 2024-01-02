import re


def normalize_number(word: str) -> str:
    pattern = r"^(\d{1,3}(,\d{3})*(\.\d+)?|\d+\.\d+|\d+)$"
    if bool(re.match(pattern, word)):
        return "0"
    else:
        return word
