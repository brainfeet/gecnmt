import os.path as path

from gecnmt.clojure.core import *

dataset_path = "../resources/dataset"
replaced_filename = "replaced.txt"


def get_replaced_path(dataset):
    return path.join(dataset_path, dataset, replaced_filename)


def appending_spit(f, content):
    spit(f, content, append=True)
