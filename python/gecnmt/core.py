import argparse
import functools
import json
import math
import os.path as path
import random
import subprocess

import funcy
import numpy
import torch
import torch.autograd as autograd
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.utils.rnn as rnn
import torch.optim as optim
import torchtext.vocab as vocab

from gecnmt.clojure.core import *
import gecnmt.clojure.string as string
import gecnmt.aid as aid
import gecnmt.helpers as helpers
import jfleg.jfleg as jfleg
import nucle.nucle as nucle


def slurp(path):
    with open(path) as f:
        return f.read()


hyperparameter = json.loads(slurp("hyperparameter/hyperparameter.json"))
dim = 50
glove = vocab.GloVe("6B", dim)
count = len
vocabulary_size = count(glove.vectors)


def get_cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x


embedding_vectors = get_cuda(torch.cat((glove.vectors,
                                        torch.zeros(1, glove.dim))))
bag_size = 128
bpe_path = path.join(helpers.dataset_path, "simple/bpe.json")
bpe = json.loads(slurp(bpe_path))
bpe_size = count(bpe)


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


get_concatenated_size = partial(multiply, 2)


def add(*more):
    if equal(count(more), 2):
        return first(more) + last(more)
    return add(first(more), add(*rest(more)))


def get_model(m):
    model = nn.Module()
    model.encoder_gru = nn.GRU(bag_size,
                               m["hidden_size"],
                               m["num_layers"],
                               bidirectional=True,
                               dropout=m["dropout"])
    model.encoder_linear = nn.Linear(get_bidirectional_size(m["hidden_size"]),
                                     dim)
    model.embedding = nn.Embedding(bpe_size, m["hidden_size"])
    model.attention = nn.Linear(get_concatenated_size(m["hidden_size"]),
                                m["max_length"])
    model.attention_combiner = nn.Linear(
        add(get_bidirectional_size(m["hidden_size"]),
            dim,
            m["hidden_size"]),
        m["hidden_size"])
    model.decoder_gru = nn.GRU(m["hidden_size"], m["hidden_size"])
    model.out = nn.Linear(m["hidden_size"], bpe_size)
    return get_cuda(model)


get_bidirectional_size = partial(multiply, 2)


def get_hidden(m):
    return get_cuda_variable(
        if_(m["encoder"],
            init.kaiming_normal,
            identity)(torch.zeros(if_(m["encoder"],
                                      get_bidirectional_size,
                                      identity)(m["num_layers"]),
                                  if_(equal(m["split"], "training"),
                                      m["batch_size"],
                                      1),
                                  m["hidden_size"])))


get_mse = nn.MSELoss()


def encode(m):
    outputs = m["model"].encoder_gru(rnn.pack_padded_sequence(m["bag"],
                                                              m["lengths"]),
                                     get_hidden(set_val_("encoder", True, m)))
    gru_embedding = first(rnn.pad_packed_sequence(first(outputs)))
    linear_embedding = m["model"].encoder_linear(gru_embedding)
    return {"encoder_embedding": torch.cat((gru_embedding, linear_embedding),
                                           2),
            "loss": get_mse(torch.mul(linear_embedding, m["embedded"]),
                            m["pretrained_embedding"])}


def get_sorted_path(m):
    return path.join(helpers.dataset_path,
                     m["dataset"],
                     m["split"],
                     "sorted.txt")


def reduce(f, *more):
    if count(more) == 1:
        functools.reduce(f, last(more))
    return functools.reduce(f, last(more), first(more))


def get_continuation(continuation, element):
    def continuation_(structure):
        if isinstance(element, builtins.str):
            return update_in(structure, [element], continuation)
        if isinstance(element, RichNavigator):
            return element.transform_(continuation, structure)
    return continuation_


class RichNavigator:
    def __init__(self, transform__):
        self.transform_ = transform__


def update_first(continuation, structure):
    if isinstance(structure, builtins.str):
        return str(continuation(first(structure)), *rest(structure))
    return (continuation(first(structure)), *rest(structure))


def append(continuation, structure):
    return (*structure, *continuation(()))


MAP_VALS = RichNavigator(walk_values)
ALL = RichNavigator(walk)
FIRST = RichNavigator(update_first)
END = RichNavigator(append)
nth = aid.flip(funcy.nth)


def nth_path(n):
    def update_nth(continuation, structure):
        return (*take(n, structure),
                continuation(nth(structure, n)),
                *drop(inc(n), structure))
    return RichNavigator(update_nth)


def multi_path(*paths):
    def continuation_(continuation, structure):
        return reduce(get_continuation, continuation,
                      coerce_path(first(paths)))(
            reduce(get_continuation, continuation, coerce_path(second(paths)))(
                structure))
    if count(paths) == 2:
        return RichNavigator(continuation_)
    return reduce(multi_path, RichNavigator(continuation_), drop(2, paths))


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


def get(m, *more):
    if equal(count(more), 1):
        return m[first(more)]
    return m.get(first(more), second(more))


def partition(n, *more):
    if count(more) == 1:
        return funcy.partition(n, n, last(more))


partition_by = comp(partial(map, tuple),
                    funcy.partition_by)


def contains_(coll, k):
    return k in coll


determiner_ = comp(partial(equal, "DT"),
                   partial(aid.flip(get),
                           "tag_"))
prepositions = {"with",
                "at",
                "from",
                "into",
                "during",
                "including",
                "until",
                "against",
                "among",
                "throughout",
                "despite",
                "towards",
                "upon",
                "concerning",
                "of",
                "to",
                "in",
                "for",
                "on",
                "by",
                "about",
                "like",
                "through",
                "over",
                "before",
                "between",
                "after",
                "since",
                "without",
                "under",
                "within",
                "along",
                "following",
                "across",
                "behind",
                "beyond",
                "plus",
                "except",
                "but",
                "up",
                "out",
                "around",
                "down",
                "off",
                "above",
                "near"}


def build(f, *more):
    return comp(partial(apply, f), apply(juxt, more))


def or_(*more):
    if count(more) == 2:
        return first(more) or last(more)
    return or_(first(more), or_(*rest(more)))


preposition_ = build(and_,
                     comp(partial(equal, "IN"),
                          partial(aid.flip(get), "tag_")),
                     comp(partial(contains_, prepositions),
                          partial(aid.flip(get), "lower_")))
rand = random.random
noise_ = partial(greater_than, hyperparameter["noise_probability"])


def to_remove_(x):
    return and_(or_(determiner_(x), preposition_(x)), noise_(rand()))


remove_tokens = partial(transform_,
                        "tokens",
                        # if tuple isn't called, tokens don't persist
                        compose(tuple,
                                # TODO make remove persistent
                                partial(remove, to_remove_)))
inflecteds = {"BES",
              "HVS",
              "JJR",
              "JJS",
              "NNS",
              "RBR",
              "RBS",
              "VBD",
              "VBG",
              "VBN",
              "VBZ"}


def lemmatize(token):
    return if_(and_(contains_(inflecteds, token["tag_"]), noise_(rand())),
               token["lemma_"],
               token["text"])


def make_set(x):
    k, f = x
    return build(partial(set_val_, k),
                 comp(f,
                      partial(aid.flip(get), "tokens")),
                 identity)


def lower_case(s):
    return s.lower()


int = ord


def increment_vector(reduction, c):
    return transform_(nth_path(int(c)), inc, reduction)


# TODO make repeat persistent
repeat = aid.flip(funcy.repeat)
# if tuple isn't called, repeat doesn't persist
zero_bag = tuple(repeat(bag_size, 0))
bag_ = partial(reduce, increment_vector, zero_bag)
# if tuple isn't called, map doesn't persist
bag = comp(tuple,
           partial(map, bag_),
           partial(transform_, (FIRST, FIRST), lower_case),
           partial(map, lemmatize))


def get_index(s):
    return get(glove.stoi, s, vocabulary_size)


get_pretrained_embedding = comp(tuple,
                                partial(map, comp(get_index,
                                                  partial(aid.flip(get),
                                                          "lower_"))))


def keys(map):
    return tuple(map.keys())


def get_embedded_vector(x):
    return tuple(repeat(dim, if_(x, 1, 0)))


glove_words = keys(glove.stoi)
get_embedded = comp(tuple,
                    partial(map, comp(get_embedded_vector,
                                      partial(contains_, glove_words),
                                      partial(aid.flip(get),
                                              "lower_"))))
convert_from_tokens = comp(
    apply(comp,
          map(make_set,
              {"bag": bag,
               "embedded": get_embedded,
               "lengths": count,
               "pretrained_embedding": get_pretrained_embedding})),
    remove_tokens)


def sort_by(comp, key_fn, coll):
    return sorted(coll, key=key_fn, reverse=if_(equal(comp, greater_than),
                                                True,
                                                False))


def merge_with(f, *more):
    return funcy.merge_with(partial(apply, f), *more)


def subtract(*more):
    if equal(count(more), 2):
        return first(more) - last(more)
    return subtract(first(more), subtract(*rest(more)))


def make_pad(n, placeholder):
    def pad(coll):
        return set_val_(END,
                        repeat(subtract(n, count(coll)), placeholder),
                        coll)
    return pad


def make_pad_all(x):
    k, placeholder = x
    def pad_all(m):
        return transform_((k, ALL),
                          make_pad(first(m["lengths"]), placeholder),
                          m)
    return pad_all


pad_step = apply(comp,
                 map(make_pad_all, {"bag": zero_bag,
                                    "embedded": get_embedded_vector(False),
                                    "pretrained_embedding": 0}))


def batch_transpose(input):
    if equal(input.dim(), 1):
        return input
    return torch.transpose(input, 0, 1)


def get_input_output(input, output):
    return {"input_reference_bpe": input,
            "output_reference_bpe": output}


def pair(m):
    return set_val_("reference_bpes",
                    apply(map,
                          get_input_output,
                          map(tuple, (m["input-reference-bpes"],
                                      m["output-reference-bpes"]))),
                    m)


get_cuda_variable = comp(get_cuda,
                         autograd.Variable)
get_transposed_variable = comp(batch_transpose,
                               get_cuda_variable)
embedding = get_cuda(nn.Embedding(count(embedding_vectors),
                                  count(first(embedding_vectors))))
embedding.weight = nn.Parameter(embedding_vectors)
embedding.weight.requires_grad = False
convert_to_variables = comp(pair,
                            partial(transform_, "length", first),
                            partial(transform_,
                                    "pretrained_embedding",
                                    embedding),
                            partial(transform_,
                                    multi_path("bag",
                                               "pretrained_embedding",
                                               "embedded",
                                               "input-reference-bpes",
                                               "output-reference-bpes",
                                               "decoder-bpe"),
                                    get_transposed_variable),
                            partial(transform_,
                                    multi_path("bag", "embedded"),
                                    torch.FloatTensor),
                            partial(transform_,
                                    multi_path("pretrained_embedding",
                                               "input-reference-bpes",
                                               "output-reference-bpes",
                                               "decoder-bpe"),
                                    torch.LongTensor),
                            pad_step)


def get_steps(m):
    # TODO cycle
    return map(
        comp(convert_to_variables,
             if_(equal(m["split"], "training"),
                 identity,
                 partial(transform_, MAP_VALS, vector)),
             partial(apply, merge_with, vector),
             partial(sort_by,
                     greater_than,
                     partial(aid.flip(get), "lengths"))),
        mapcat(
            partial(partition,
                    if_(equal(m["split"], "training"),
                        m["batch_size"],
                        1)),
            partition_by(
                partial(aid.flip(get),
                        "length"),
                map(convert_from_tokens,
                    remove(comp(partial(equal,
                                        0),
                                partial(aid.flip(get),
                                        "lengths")),
                           map(comp(make_set(("lengths",
                                              count)),
                                    remove_tokens),
                               filter(if_(equal(m["dataset"],
                                                "simple"),
                                          comp(partial(greater_than,
                                                       m["max_length"]),
                                               partial(aid.flip(get),
                                                       "length")),
                                          constantly(True)),
                                      map(json.loads,
                                          (line_seq(m["file"]))))))))))


def pad_embedding(m):
    if equal(count(m["encoder_embedding"]), m["max_length"]):
        return m["encoder_embedding"]
    return torch.cat(
        (m["encoder_embedding"],
         get_cuda_variable(
             torch.zeros(*transform_(FIRST,
                                     partial(subtract,
                                             m["max_length"]),
                                     m["encoder_embedding"].size())))))


get_nll = nn.NLLLoss()


def get_first_data(variable):
    return first(variable.data)


def decode_token(reduction, element):
    if equal(reduction["split"], "training"):
        input_bpe = element["input_reference_bpe"]
    else:
        input_bpe = reduction["decoder-bpe"]
    decoder_embedding = torch.unsqueeze(reduction["model"].embedding(input_bpe),
                                        0)
    gru_output, hidden = reduction["model"].decoder_gru(
        F.relu(
            reduction["model"].attention_combiner(
                torch.cat(
                    (decoder_embedding,
                     batch_transpose(torch.bmm(
                         batch_transpose(
                             F.softmax(
                                 reduction["model"].attention(
                                     torch.cat((decoder_embedding,
                                                reduction["hidden"]),
                                               2)),
                                 2)),
                         batch_transpose(
                             reduction["padded_embedding"])))),
                    2))),
        reduction["hidden"])
    log_softmax_output = F.log_softmax(
        reduction["model"].out(first(gru_output)),
        1)
    decoder_bpe = torch.squeeze(second(torch.topk(log_softmax_output, 1)), 1)
    if equal(reduction["dataset"], "simple"):
        add_loss = partial(add, get_nll(log_softmax_output,
                                        element["output_reference_bpe"]))
    else:
        add_loss = identity
    return merge(
        transform_("decoder_bpes",
                   if_(equal(reduction["dataset"], "simple"),
                       identity,
                       partial(set_val_,
                               END,
                               (bpe[str(get_first_data(decoder_bpe))],))),
                   transform_("loss", add_loss, reduction)),
        {"hidden": hidden,
         "decoder-bpe": decoder_bpe})


def decode_tokens(m):
    return reduce(decode_token,
                  merge(m, {"decoder_bpes": (),
                            "hidden": get_hidden(set_val_("encoder", False, m)),
                            "padded_embedding": pad_embedding(m)}),
                  if_(equal(m["dataset"], "simple"),
                      m["reference_bpes"],
                      repeat(m["max_length"], nothing)))


class Maybe:
    def __init__(self, *more):
        if equal(count(more), 1):
            self.v = first(more)
    def __repr__(self):
        if hasattr(self, "v"):
            return str("#<Just ", self.v, ">")
        return "#<Nothing>"


nothing = Maybe()


def mod(num, div):
    return num % div


def make_run_non_training_step(m):
    def run_internal_step(step):
        return decode_tokens(merge(set_val_("split", "non_training", m),
                                   step,
                                   encode(merge(set_val_("split",
                                                         "non_training",
                                                         m), step))))
    return run_internal_step


def validate_internally(m):
    with open(get_sorted_path(merge(m, {"dataset": "simple",
                                        "split": "non_training"}))) as f:
        return numpy.mean(
            tuple(map(comp(get_first_data,
                           partial(aid.flip(get), "loss"),
                           make_run_non_training_step(set_val_("dataset",
                                                               "simple",
                                                               m))),
                      get_steps(merge(m, {"dataset": "simple",
                                          "split": "non_training",
                                          "file": f})))))


join = str_join


def delete_eos(sentence):
    return string.replace(sentence, r" ?<EOS>.*", "")


def infer(m):
    with open(get_sorted_path(set_val_("split", "non_training", m))) as f:
        return join(map(comp(partial(aid.flip(str), "\n"),
                             delete_eos,
                             partial(join, " "),
                             partial(aid.flip(get), "decoder_bpes"),
                             make_run_non_training_step(m)),
                        get_steps(merge(m, {"split": "non_training",
                                            "file": f}))))


def get_inferred_path(dataset):
    return path.join(helpers.dataset_path, dataset, "inferred.txt")


def validate_externally(m):
    spit(get_inferred_path(m["dataset"]), infer(m))
    with open(helpers.get_replaced_path(m["dataset"]), "w") as f:
        subprocess.Popen(["sed",
                          "-r",
                          "s/(@@ )|(@@ ?$)//g",
                          get_inferred_path(m["dataset"])],
                         stdout=f).wait()
    if equal(m["dataset"], "jfleg"):
        return jfleg.get_score()
    return nucle.get_score()


def validate(m):
    m["model"].eval()
    result = {"simple": validate_internally(m),
              "jfleg": validate_externally(set_val_("dataset", "jfleg", m)),
              "nucle": validate_externally(set_val_("dataset", "nucle", m))}
    m["model"].train()
    return result


def divide(*more):
    if equal(count(more), 2):
        return first(more) / last(more)
    return divide(first(more), divide(*rest(more)))


def learn(m):
    m["model"].zero_grad()
    loss = decode_tokens(merge(m,
                               encode(set_val_("split", "training", m)),
                               {"dataset": "simple",
                                "split": "training"}))["loss"]
    loss.backward()
    m["optimizer"].step()
    return divide(loss, m["length"])


def select_keys(m, ks):
    return funcy.select_keys(partial(contains_, ks), m)


max = comp(builtins.max,
           vector)
min = comp(builtins.min,
           vector)


def get_checkpoint_path(s):
    return path.join("checkpoints", str(s, ".pth.tar"))


def less_than(x, y):
    return x < y


def state_dict(x):
    return x.state_dict()


def save_(m):
    torch.save(transform_(multi_path("model", "optimizer"), state_dict, m),
               get_checkpoint_path(m["checkpoint"]))


def make_compare_save(before, after):
    def compare_save(m):
        if m["comparator"](before[m["checkpoint"]], after[m["checkpoint"]]):
            save_(set_val_("checkpoint", m["checkpoint"], after))
    return compare_save


def save(before, after):
    save_(set_val_("checkpoint", "recent", after))
    run_(make_compare_save(before, after), ({"checkpoint": "simple",
                                             "comparator": greater_than},
                                            {"checkpoint": "jfleg",
                                             "comparator": less_than},
                                            {"checkpoint": "nucle",
                                             "comparator": less_than}))


def validation_step_(m):
    return equal(mod(m["step_count"], m["validation_interval"]), 0)


def run_training_step(reduction, step):
    learn(merge(reduction, step))
    if validation_step_(reduction):
        validated = validate(reduction)
    else:
        validated = {}
    after = merge_with(max,
                       merge_with(min,
                                  transform_("step_count", inc, reduction),
                                  select_keys(validated, ("simple",))),
                       select_keys(validated, ("jfleg", "nucle")))
    # TODO print loss
    if validation_step_(reduction):
        save(reduction, after)
    return after


POSITIVE_INFINITY = math.inf
test_parameter = json.loads(slurp("test_parameter/test_parameter.json"))


def load(checkpoint):
    model = get_model(hyperparameter)
    optimizer = optim.Adam(model.parameters(), hyperparameter["lr"])
    if path.exists(get_checkpoint_path("recent")):
        checkpoint_ = torch.load(get_checkpoint_path(checkpoint))
        model.load_state_dict(checkpoint_["model"])
        optimizer.load_state_dict(checkpoint_["optimizer"])
    else:
        checkpoint_ = {}
    return merge(hyperparameter,
                 test_parameter,
                 {"step_count": 0,
                  "simple": POSITIVE_INFINITY,
                  "jfleg": 0,
                  "nucle": 0},
                 checkpoint_,
                 {"model": model,
                  "optimizer": optimizer})


def train():
    with open(get_sorted_path(merge(hyperparameter,
                                    {"dataset": "simple",
                                     "split": "training"}))) as f:
        loaded = load("recent")
        reduce(run_training_step,
               loaded,
               get_steps(merge(loaded, {"dataset": "simple",
                                        "file": f,
                                        "split": "training"})))


def get_corrected_path(m):
    return path.join(helpers.dataset_path, m["dataset"], "corrected.txt")


def test():
    with open(get_sorted_path(set_val_("split",
                                       "non_training",
                                       test_parameter))) as f:
        spit(get_corrected_path(test_parameter),
             infer(set_val_("file", f, load(test_parameter["checkpoint"]))))


parser = argparse.ArgumentParser()
parser.add_argument("command", type=builtins.str)

if equal(__name__, "__main__"):
    get({"train": train,
         "test": test},
        parser.parse_args().command)()
