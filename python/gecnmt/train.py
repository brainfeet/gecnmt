import glob
import json
import os.path

from funcy import *


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
