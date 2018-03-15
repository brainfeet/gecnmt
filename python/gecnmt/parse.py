import argparse
import json

import spacy

from gecnmt.clojure.core import *
import gecnmt.clojure.string as string
import gecnmt.helpers as helpers

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


def get_parsed_path(path):
    return string.replace(path, "combined.txt", "parsed.txt")


def parse():
    with open(get_combined_path()) as f:
        return run_(comp(partial(helpers.appending_spit,
                                 get_parsed_path(get_combined_path())),
                         helpers.append_newline,
                         parse_stringify),
                    line_seq(f))


if __name__ == "__main__":
    parse()
