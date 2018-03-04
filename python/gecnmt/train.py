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


def get_steps(file):
    return line_seq(file)


def train():
    with open(get_sorted_path(merge(hyperparameter,
                                    {"dataset": "simple",
                                     "split": "training"}))) as file:
        get_steps(file)
