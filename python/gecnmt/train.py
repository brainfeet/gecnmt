import glob
import json
import os.path

from funcy import *
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.init as init
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


def multiply(*more):
    if equal(count(more), 2):
        return first(more) * second(more)
    return multiply(first(more), multiply(*rest(more)))


get_bidirectional_size = partial(multiply, 2)

count = len


def and_(*more):
    if count(more) == 2:
        return first(more) and second(more)
    return and_(first(more), and_(*rest(more)))


def equal(*more):
    if count(more) == 2:
        return first(more) == last(more)
    return equal(first(more), equal(*rest(more)))


def get_hidden(m):
    return autograd.Variable(init.kaiming_normal(
        torch.zeros(get_bidirectional_size(m["num_layers"]),
                    if_(equal(m["split"], "training"),
                        m["batch_size"],
                        1),
                    m["hidden_size"])))


dataset_path = "../resources/dataset"


def get_glob(m):
    return os.path.join(dataset_path, m["dataset"], m["split"], "*")
