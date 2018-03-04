import glob
import json
import os.path

from funcy import *
import torch
import torch.nn as nn
import torch.nn.init as init
import torchtext.vocab as vocab

embedding = vocab.GloVe("6B", 50)
vocabulary_size = first(embedding.vectors.size())
embedding_vectors = torch.cat(
    (embedding.vectors, init.kaiming_normal(torch.zeros(1, embedding.dim))))


def slurp(path):
    with open(path) as file:
        return file.read()


hyperparameter = json.loads(slurp("hyperparameter/hyperparameter.json"))

dataset_path = "../resources/dataset"


def get_glob(m):
    return os.path.join(dataset_path,
                        m["dataset"],
                        m["split"],
                        "*")
