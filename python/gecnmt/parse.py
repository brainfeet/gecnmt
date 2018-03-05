import argparse
import json
import re

import spacy

from gecnmt.clojure import *
import gecnmt.aid as aid

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
            "text": token.text}


parse_stringify = comp(json.dumps,
                       tuple,
                       partial(map, get_token_map),
                       nlp)


def replace(s, match, replacement):
    return re.sub(match, replacement, s)


def get_parsed_path(path):
    return replace(path, "combined.txt", "parsed.txt")


def spit(f, content, append=False):
    with open(f, if_(append,
                     "a",
                     "w")) as file:
        file.write(content)


def appending_spit(f, content):
    spit(f, content, append=True)


append_newline = partial(aid.flip(str), "\n")


def dorun(coll):
    for _ in coll:
        pass


run_ = comp(dorun, map)


def parse():
    with open(get_combined_path()) as f:
        return run_(comp(partial(appending_spit,
                                 get_parsed_path(get_combined_path())),
                         append_newline,
                         parse_stringify),
                    line_seq(f))


if __name__ == "__main__":
    parse()
