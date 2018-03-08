import os.path as path

dataset_path = "../resources/dataset"
inferred_filename = "inferred.txt"


def get_inferred_path(dataset):
    return path.join(dataset_path, dataset, inferred_filename)
