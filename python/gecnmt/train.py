import json
import os.path


def slurp(path):
    with open(path) as file:
        return file.read()

hyperparameter = json.loads(slurp("hyperparameter/template.json"))
