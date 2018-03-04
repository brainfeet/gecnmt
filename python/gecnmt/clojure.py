def if_(test, then, else_):
    if test:
        return then
    return else_


def line_seq(file):
    for line in file:
        yield line
