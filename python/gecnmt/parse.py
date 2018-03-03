import argparse
import json
import builtins
import re

from funcy import *
import spacy

nlp = spacy.load("en")


def get_combined_path():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    return parser.parse_args().path


def get_token_map(token):
    return {"is_sent_start": token.is_sent_start,
            "lemma_": token.lemma_,
            "lower_": token.lower_,
            "tag_": token.tag_,
            "text_with_ws": token.text_with_ws}


parse_stringify = compose(json.dumps,
                          tuple,
                          partial(map, get_token_map),
                          nlp)


def replace(s, match, replacement):
    return re.sub(match, replacement, s)


def get_parsed_path(path):
    return replace(path, "combined.txt", "parsed.txt")


def if_(test, then, else_):
    if test:
        return then
    return else_


def spit(f, content, append=False):
    with open(f, if_(append,
                     "a",
                     "w")) as file:
        file.write(content)


def appending_spit(f, content):
    spit(f, content, append=True)


def str(*more):
    return str_join("", walk(builtins.str, more))


def apply(f, *more):
    return f(*butlast(more), *last(more))


def flip(f):
    def g(x, *more):
        if empty(more):
            def h(y, *more_):
                apply(f, y, x, more_)
            return h
        return apply(f, first(more), x, rest(more))
    return g


append_newline = partial(flip(str), "\n")


def line_seq(file):
    for line in file:
        yield line


def dorun(coll):
    for _ in coll:
        pass


run_ = compose(dorun, map)


def parse():
    with open(get_combined_path()) as f:
        return run_(compose(partial(appending_spit,
                                    get_parsed_path(get_combined_path())),
                            append_newline,
                            parse_stringify),
                    line_seq(f))


if __name__ == "__main__":
    print(parse())
