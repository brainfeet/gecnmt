from gecnmt.clojure.core import *


def flip(f):
    def g(x, *more):
        if empty(more):
            def h(y, *more_):
                apply(f, y, x, more_)
            return h
        return apply(f, first(more), x, rest(more))
    return g
