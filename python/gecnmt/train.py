import functools
import json
import os.path as path

from funcy import *
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils.rnn as rnn
import torchtext.vocab as vocab

from gecnmt.clojure import *


def slurp(path):
    with open(path) as file:
        return file.read()


hyperparameter = json.loads(slurp("hyperparameter/hyperparameter.json"))
embedding = vocab.GloVe("6B", 50)
vocabulary_size = first(embedding.vectors.size())
embedding_vectors = torch.cat(
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
        if isinstance(element, str):
            return update_in(structure, [element], continuation)
        if isinstance(element, RichNavigator):
            return element.transform_(continuation, structure)
    return continuation_


class RichNavigator:
    def __init__(self, transform_):
        self.transform_ = transform_

    def transform_(self, *args):
        return self.transform_(*args)


MAP_VALS = RichNavigator(walk_values)
ALL = RichNavigator(walk)

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


def get_steps(m):
    # TODO implement this function
    return m["file"]


def train():
    # TODO reduce steps
    with open(get_sorted_path(merge(hyperparameter,
                                    {"dataset": "simple",
                                     "split": "training"}))) as file:
        get_steps(set_val_("file", file, hyperparameter))
