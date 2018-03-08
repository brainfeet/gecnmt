import builtins

from funcy import *


def if_(test, then, else_):
    if test:
        return then
    return else_


def line_seq(file):
    for line in file:
        yield line


def apply(f, *more):
    return f(*butlast(more), *last(more))


comp = compose


def str(*more):
    return str_join("", walk(builtins.str, more))


def get_items(coll):
    if isinstance(coll, dict):
        return coll.items()
    return coll


def map(f, *colls):
    return builtins.map(f, *builtins.map(get_items, colls))


def spit(f, content, append=False):
    with open(f, if_(append,
                     "a",
                     "w")) as file:
        file.write(content)
