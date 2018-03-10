import glob

from funcy import *

import gecnmt.jfleg.eval.gleu as gleu
import gecnmt.helpers as helpers

read_string = float
source = "gecnmt/jfleg/dev/dev.src"
gleu_calculator = gleu.GLEU(4)
gleu_calculator.load_sources(source)
gleu_calculator.load_references(glob.glob("gecnmt/jfleg/dev/*ref*"))


def get_score():
    return read_string(first(first(gleu_calculator.run_iterations(
        num_iterations=500,
        source=source,
        hypothesis=helpers.get_replaced_path("jfleg"),
        per_sent=False))))
