import argparse
import glob
import json
import os
import re

from funcy import *
import spacy

from gecnmt.clojure import *

nlp = spacy.load("en")


def get_isolated_paths():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    return glob.glob(parser.parse_args().path + "/*")


def get_token_map(token):
    return {"is_sent_start": token.is_sent_start,
            "lemma_": token.lemma_,
            "lower_": token.lower_,
            "tag_": token.tag_,
            "text": token.text}


parse_stringify = compose(json.dumps,
                          tuple,
                          partial(map, get_token_map),
                          nlp)

get_parsed_path = partial(re.sub, r"isolated/([^\/]+).txt$", r"parsed/\1.json")


def spit(path, s):
    with open(path, "w") as file:
        file.write(s)


def mkdirs(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))


def spit_parents(path, s):
    mkdirs(path)
    spit(path, s)


def parse():
    tuple(map(spit_parents,
              map(get_parsed_path, get_isolated_paths()),
              map(compose(parse_stringify, slurp),
                  get_isolated_paths())))


if __name__ == "__main__":
    print(parse())
