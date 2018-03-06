import functools
import json
import os.path as path

import funcy
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils.rnn as rnn
import torchtext.vocab as vocab

from gecnmt.clojure import *
import gecnmt.aid as aid


def slurp(path):
    with open(path) as file:
        return file.read()


hyperparameter = json.loads(slurp("hyperparameter/hyperparameter.json"))
dim = 50
glove = vocab.GloVe("6B", dim)
vocabulary_size = first(glove.vectors.size())
embedding_vectors = torch.cat((glove.vectors, torch.zeros(1, glove.dim)))
bag_size = 128
dataset_path = "../resources/dataset"
bpe_path = path.join(dataset_path, "simple/bpe.json")
bpe = json.loads(slurp(bpe_path))
count = len
bpe_size = count(bpe)


def get_model(m):
    model = nn.Module()
    model.encoder_gru = nn.GRU(bag_size,
                               m["hidden_size"],
                               m["num_layers"],
                               bidirectional=True,
                               dropout=m["dropout"])
    model.encoder_linear = nn.Linear(get_bidirectional_size(m["hidden_size"]),
                                     dim)
    model.attention = nn.Linear(get_bidirectional_size(m["hidden_size"]),
                                m["max_length"])
    model.attention_combiner = nn.Linear(
        get_bidirectional_size(m["hidden_size"]),
        m["hidden_size"])
    model.decoder_gru = nn.GRU(m["hidden_size"], m["hidden_size"])
    model.out = nn.Linear(m["hidden_size"], bpe_size)
    return model


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
    outputs = m["model"].encoder_gru(rnn.pack_padded_sequence(m["bag"],
                                                              m["lengths"]),
                                     get_hidden(m))
    gru_embedding = first(rnn.pad_packed_sequence(first(outputs)))
    linear_embedding = m["model"].encoder_linear(gru_embedding)
    return {"encoder_embedding": torch.cat((gru_embedding, linear_embedding),
                                           2),
            "linear_embedding": linear_embedding,
            "hidden": last(outputs)}


def get_sorted_path(m):
    return path.join(dataset_path,
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


determiner_ = partial(equal, "DT")
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


preposition_ = partial(contains_, prepositions)
remove_tokens = partial(transform_,
                        "tokens",
                        # if tuple isn't called, tokens don't persist
                        compose(tuple,
                                # TODO make remove persistent
                                partial(remove,
                                        build(or_,
                                              comp(determiner_,
                                                   partial(aid.flip(get),
                                                           "tag_")),
                                              # TODO check tag_
                                              comp(preposition_,
                                                   partial(aid.flip(get),
                                                           "lower_"))))))

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
    return if_(contains_(inflecteds, token["tag_"]),
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
# TODO implement this function
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
    if equal(count(more), 2):
        return funcy.merge_with(partial(apply, f), *more)
    return merge_with(f, first(more), merge_with(f, *rest(more)))


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
    return torch.transpose(input, 0, 1)


def get_input_output(input_output):
    input, output = input_output
    return {"input": input,
            "output": output}


def pair(m):
    return set_val_("bpes",
                    map(comp(get_input_output,
                             partial(map, tuple),
                             vector),
                        m["input-bpes"],
                        m["output-bpes"]),
                    m)


get_variable = comp(batch_transpose,
                    autograd.Variable)
embedding = nn.Embedding(embedding_vectors.size(0), embedding_vectors.size(1))
embedding.weight = nn.Parameter(embedding_vectors)
embedding.weight.requires_grad = False
convert_to_variables = comp(pair,
                            partial(transform_,
                                    "pretrained_embedding",
                                    embedding),
                            partial(transform_,
                                    multi_path("bag",
                                               "pretrained_embedding",
                                               "embedded",
                                               "input-bpes",
                                               "output-bpes"),
                                    get_variable),
                            partial(transform_,
                                    multi_path("bag", "embedded"),
                                    torch.FloatTensor),
                            partial(transform_,
                                    multi_path("pretrained_embedding",
                                               "input-bpes",
                                               "output-bpes"),
                                    torch.LongTensor),
                            pad_step)


def get_steps(m):
    # TODO implement this function
    return map(comp(convert_to_variables,
                    partial(apply, merge_with, vector),
                    partial(sort_by,
                            greater_than,
                            partial(aid.flip(get), "lengths"))),
               apply(concat,
                     map(partial(partition, m["batch_size"]),
                         partition_by(partial(aid.flip(get), "length"),
                                      map(convert_from_tokens,
                                          filter(comp(
                                              partial(greater_than,
                                                      m["max_length"]),
                                              partial(aid.flip(get), "length")),
                                              map(json.loads,
                                                  (line_seq(m["file"])))))))))


get_mse = nn.MSELoss()


def get_encoder_loss(m):
    return get_mse(torch.mul(m["linear_embedding"], m["embedded"]),
                   m["pretrained_embedding"])


def pad_embedding(m):
    if equal(first(m["encoder_embedding"].size()), m["max_length"]):
        return m["encoder_embedding"]
    return torch.cat(
        (m["encoder_embedding"],
         autograd.Variable(
             torch.zeros(*transform_(FIRST,
                                     partial(subtract,
                                             m["max_length"]),
                                     m["encoder_embedding"].size())))))


def decode_tokens(m):
    # TODO implement this function
    return pad_embedding(m)


def make_run_step(m):
    # TODO implement this function
    def run_step(reduction, element):
        m["model"].zero_grad()
        encoder_output = encode(merge(m, element, {"split": "training"}))
        get_encoder_loss(merge(element, encoder_output))
        decode_tokens(merge(m, element, encoder_output))
        return transform_("step_count", inc, reduction)
    return run_step


def initialize(m):
    m["model"].encoder_linear.weight = nn.Parameter(
        init.kaiming_normal(
            torch.zeros(dim, get_bidirectional_size(m["hidden_size"]))))
    # TODO initialize the decoder
    return m["model"]


def load(m):
    # TODO implement this function
    return merge(m, {"model": initialize(set_val_("model", get_model(m), m)),
                     "step_count": 0})


def train():
    # TODO reduce steps
    with open(get_sorted_path(merge(hyperparameter,
                                    {"dataset": "simple",
                                     "split": "training"}))) as file:
        build(reduce,
              make_run_step,
              identity,
              get_steps)(set_val_("file", file, load(hyperparameter)))
