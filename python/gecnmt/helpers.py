import os.path as path

dataset_path = "../resources/dataset"
replaced_filename = "replaced.txt"


def get_replaced_path(dataset):
    return path.join(dataset_path, dataset, replaced_filename)
