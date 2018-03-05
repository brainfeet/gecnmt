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
