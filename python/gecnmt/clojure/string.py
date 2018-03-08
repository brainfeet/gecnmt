import re


def replace(s, match, replacement):
    return re.sub(match, replacement, s)
