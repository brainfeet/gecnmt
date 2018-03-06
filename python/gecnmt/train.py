import functools
import json
import os.path as path

import funcy
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils.rnn as rnn
import torchtext.vocab as vocab

from gecnmt.clojure import *
import gecnmt.aid as aid


def slurp(path):
    with open(path) as file:
        return file.read()


hyperparameter = json.loads(slurp("hyperparameter/hyperparameter.json"))
embedding = vocab.GloVe("6B", 50)
vocabulary_size = first(embedding.vectors.size())
embedding_vectors = torch.cat(
    # TODO initialize <UNK> with zeros
    (embedding.vectors, init.kaiming_normal(torch.zeros(1, embedding.dim))))
bag_size = 128


def get_encoder(m):
    model = nn.Module()
    model.gru = nn.GRU(bag_size,
                       m["hidden_size"],
                       m["num_layers"],
                       bidirectional=True,
                       dropout=m["dropout"])
    return model


count = len


def and_(*more):
    if count(more) == 2:
        return first(more) and last(more)
    return and_(first(more), and_(*rest(more)))


def equal(*more):
    if count(more) == 2:
        return first(more) == last(more)
    return equal(first(more), equal(*rest(more)))


def multiply(*more):
    if equal(count(more), 2):
        return first(more) * last(more)
    return multiply(first(more), multiply(*rest(more)))


get_bidirectional_size = partial(multiply, 2)


def get_hidden(m):
    return autograd.Variable(init.kaiming_normal(
        torch.zeros(get_bidirectional_size(m["num_layers"]),
                    if_(equal(m["split"], "training"),
                        m["batch_size"],
                        1),
                    m["hidden_size"])))


def encode(m):
    output = m["encoder"].gru(rnn.pack_padded_sequence(m["bag"],
                                                       m["lengths"]),
                              get_hidden(m))
    return {"embedding": rnn.pad_packed_sequence(first(output)),
            "hidden": last(output)}


dataset_path = "../resources/dataset"


def get_sorted_path(m):
    return path.join(dataset_path,
                     m["dataset"],
                     m["split"],
                     "sorted.txt")


def reduce(f, *more):
    if count(more) == 1:
        functools.reduce(f, last(more))
    return functools.reduce(f, last(more), first(more))


def get_continuation(continuation, element):
    def continuation_(structure):
        if isinstance(element, builtins.str):
            return update_in(structure, [element], continuation)
        if isinstance(element, RichNavigator):
            return element.transform_(continuation, structure)
    return continuation_


class RichNavigator:
    def __init__(self, transform__):
        self.transform_ = transform__


def update_first(continuation, structure):
    if isinstance(structure, builtins.str):
        return str(continuation(first(structure)), *rest(structure))
    return (continuation(first(structure)), *rest(structure))


def append(continuation, structure):
    return (*structure, *continuation(()))


MAP_VALS = RichNavigator(walk_values)
ALL = RichNavigator(walk)
FIRST = RichNavigator(update_first)
END = RichNavigator(append)
nth = aid.flip(funcy.nth)


def nth_path(n):
    def update_nth(continuation, structure):
        return (*take(n, structure),
                continuation(nth(structure, n)),
                *drop(inc(n), structure))
    return RichNavigator(update_nth)


def multi_path(*paths):
    def continuation_(continuation, structure):
        return reduce(get_continuation, continuation,
                      coerce_path(first(paths)))(
            reduce(get_continuation, continuation, coerce_path(second(paths)))(
                structure))
    if count(paths) == 2:
        return RichNavigator(continuation_)
    return reduce(multi_path, RichNavigator(continuation_), drop(2, paths))


reverse = reversed


def vector(*more):
    return tuple(more)


def coerce_path(path):
    if isinstance(path, tuple):
        return path
    return vector(path)


def transform_(path, transform_fn, structure):
    return reduce(get_continuation, transform_fn, reverse(coerce_path(path)))(
        structure)


def set_val_(path, val, structure):
    return transform_(path, constantly(val), structure)


def greater_than(x, y):
    return x > y


def get(m, k):
    return m[k]


def partition(n, *more):
    if count(more) == 1:
        return funcy.partition(n, n, last(more))


partition_by = comp(partial(map, tuple), funcy.partition_by)


def contains_(coll, k):
    return k in coll


determiner_ = partial(equal, "DT")
prepositions = {"with",
                "at",
                "from",
                "into",
                "during",
                "including",
                "until",
                "against",
                "among",
                "throughout",
                "despite",
                "towards",
                "upon",
                "concerning",
                "of",
                "to",
                "in",
                "for",
                "on",
                "by",
                "about",
                "like",
                "through",
                "over",
                "before",
                "between",
                "after",
                "since",
                "without",
                "under",
                "within",
                "along",
                "following",
                "across",
                "behind",
                "beyond",
                "plus",
                "except",
                "but",
                "up",
                "out",
                "around",
                "down",
                "off",
                "above",
                "near"}


def build(f, *more):
    return comp(partial(apply, f), apply(juxt, more))


def or_(*more):
    if count(more) == 2:
        return first(more) or last(more)
    return or_(first(more), or_(*rest(more)))


preposition_ = partial(contains_, prepositions)
remove_tokens = partial(transform_,
                        "tokens",
                        # if tuple isn't called, tokens don't persist
                        compose(tuple,
                                # TODO make remove persistent
                                partial(remove,
                                        build(or_,
                                              comp(determiner_,
                                                   partial(aid.flip(get),
                                                           "tag_")),
                                              # TODO check tag_
                                              comp(preposition_,
                                                   partial(aid.flip(get),
                                                           "lower_"))))))

inflecteds = {"BES",
              "HVS",
              "JJR",
              "JJS",
              "NNS",
              "RBR",
              "RBS",
              "VBD",
              "VBG",
              "VBN",
              "VBZ"}


def lemmatize(token):
    return if_(contains_(inflecteds, token["tag_"]),
               token["lemma_"],
               token["text"])


def make_set(k, f):
    return build(partial(set_val_, k),
                 comp(f,
                      partial(aid.flip(get), "tokens")),
                 identity)


def lower_case(s):
    return s.lower()


int = ord


def increment_vector(reduction, c):
    return transform_(nth_path(int(c)), inc, reduction)


# TODO make repeat persistent
repeat = aid.flip(funcy.repeat)

# if tuple isn't called, repeat doesn't persist
zero_bag = tuple(repeat(bag_size, 0))
bag_ = partial(reduce, increment_vector, zero_bag)
# if tuple isn't called, map doesn't persist
bag = comp(tuple,
           partial(map, bag_),
           partial(transform_, (FIRST, FIRST), lower_case),
           partial(map, lemmatize))

# TODO implement this function
convert = comp(apply(comp,
                     map(partial(apply, make_set),
                         (("bag", bag),
                          ("lengths", count)))),
               remove_tokens)


def sort_by(comp_, key_fn, coll):
    return sorted(coll, key=key_fn, reverse=if_(equal(comp_, greater_than),
                                                True,
                                                False))


def merge_with(f, *more):
    if equal(count(more), 2):
        return funcy.merge_with(partial(apply, f), *more)
    return merge_with(f, first(more), merge_with(f, *rest(more)))


def subtract(*more):
    if equal(count(more), 2):
        return first(more) - last(more)
    return subtract(first(more), subtract(*rest(more)))


def make_pad(n, placeholder):
    def pad(coll):
        return set_val_(END,
                        repeat(subtract(n, count(coll)), placeholder),
                        coll)
    return pad


def pad_step(m):
    return transform_(("bag", ALL), make_pad(first(m["lengths"]), zero_bag), m)


def batch_transpose(input):
    return torch.transpose(input, 0, 1)


get_variable = comp(batch_transpose,
                    autograd.Variable,
                    torch.FloatTensor)


def get_steps(m):
    # TODO implement this function
    return map(comp(partial(transform_,
                            multi_path("bag",
                                       "input-bpes",
                                       "output-bpes"),
                            get_variable),
                    pad_step,
                    partial(apply, merge_with, vector),
                    partial(sort_by,
                            greater_than,
                            partial(aid.flip(get), "lengths"))),
               apply(concat,
                     map(partial(partition, m["batch_size"]),
                         partition_by(partial(aid.flip(get), "length"),
                                      map(convert,
                                          filter(comp(
                                              partial(greater_than,
                                                      m["max_length"]),
                                              partial(aid.flip(get), "length")),
                                              map(json.loads,
                                                  (line_seq(m["file"])))))))))


def make_run_step(m):
    # TODO implement this function
    def run_step(reduction, element):
        m["encoder"].zero_grad()
        encode(merge(m, element, {"split": "training"}))
        return transform_("step_count", inc, reduction)
    return run_step


def load(m):
    # TODO implement this function
    return merge(m, {"encoder": get_encoder(m),
                     "step_count": 0})


def train():
    # TODO reduce steps
    with open(get_sorted_path(merge(hyperparameter,
                                    {"dataset": "simple",
                                     "split": "training"}))) as file:
        build(reduce,
              make_run_step,
              identity,
              get_steps)(set_val_("file", file, load(hyperparameter)))
